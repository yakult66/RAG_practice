import hashlib
import os
import random
import re
import time

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "models/gemini-embedding-2")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "my_rag_collection")
DOCS_DIR = os.getenv("DOCS_DIR", "docs")

EMBED_BATCH_SIZE = 100
MAX_RETRIES = 6
RETRYABLE_CODES = {429, 500, 502, 503, 504}

CHUNK_MAX_CHARS = 500
CHUNK_OVERLAP_RATIO = 0.15
CHUNK_VERSION = f"punct-v2-{CHUNK_MAX_CHARS}-{CHUNK_OVERLAP_RATIO}|docling-v2"

PLAIN_TEXT_EXTS = {".txt"}
DOCLING_EXTS = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".md", ".csv"}
SUPPORTED_EXTS = PLAIN_TEXT_EXTS | DOCLING_EXTS

HISTORY_TURNS = 3
NOT_FOUND_ANSWER = "資料中找不到相關內容，無法回答。"
SOURCES_HEADER = "\n\n來源："

_docling = None


def get_docling():
    global _docling
    if _docling is None:
        from docling.chunking import HybridChunker
        from docling.document_converter import DocumentConverter
        from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer

        class CharTokenizer(BaseTokenizer):
            max_tokens: int = 1_000_000

            def count_tokens(self, text):
                return len(text)

            def get_max_tokens(self):
                return self.max_tokens

            def get_tokenizer(self):
                return len

        _docling = (DocumentConverter(), HybridChunker(tokenizer=CharTokenizer()))
    return _docling


def default_collection_name(model_name=EMBED_MODEL, embed_dim=EMBED_DIM, base=COLLECTION_NAME):
    model_slug = re.sub(r"[^a-zA-Z0-9]+", "-", model_name.split("/")[-1]).strip("-")
    return f"{base}_{model_slug}_d{embed_dim}"


def is_daily_quota_error(e):
    return getattr(e, "code", None) == 429 and "PerDay" in str(e)


def call_with_retry(fn, label="API"):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn()
        except errors.APIError as e:
            if is_daily_quota_error(e):
                raise RuntimeError(f"Gemini 每日配額已用完（{label}），請明天再試或升級付費方案。") from e
            if e.code not in RETRYABLE_CODES or attempt == MAX_RETRIES:
                raise
            delay = min(2 ** (attempt - 1), 60) + random.uniform(0, 1)
            print(f"  {label} 失敗（{e.code}），{delay:.1f} 秒後重試（第 {attempt}/{MAX_RETRIES - 1} 次）")
            time.sleep(delay)


_END = "。！？!?；;…"
_CLOSE = "」』”’)）】"
SENTENCE_SPLIT = re.compile(
    rf"(?<=[{_END}])(?![{_CLOSE}{_END}])"
    rf"|(?<=[{_END}][{_CLOSE}])"
    r"|(?<=\.)(?=\s)"
    r"|(?<=\n)(?!\n)"
)


class Embedding:
    def __init__(self, model_name=EMBED_MODEL, llm_model=LLM_MODEL, embed_dim=EMBED_DIM,
                 chroma_path=CHROMA_PATH, collection_name=None, client=None, chroma_client=None):
        self.model_name = model_name
        self.llm_model = llm_model
        self.embed_dim = embed_dim
        self.client = client or genai.Client()
        self.chroma_client = chroma_client or chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name or default_collection_name(model_name, embed_dim),
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def split_sentences(text):
        sentences = []
        for p in SENTENCE_SPLIT.split(text):
            if not p:
                continue
            if not p.strip() and sentences:
                sentences[-1] += p
            else:
                sentences.append(p)
        return sentences

    @staticmethod
    def _tail_overlap(sentences, max_overlap_chars):
        overlap, length = [], 0
        for s in reversed(sentences[1:]):
            if length + len(s) > max_overlap_chars:
                break
            overlap.insert(0, s)
            length += len(s)
        return overlap

    @staticmethod
    def chunk_text(text, max_chars=CHUNK_MAX_CHARS, overlap_ratio=CHUNK_OVERLAP_RATIO):
        max_overlap_chars = int(max_chars * overlap_ratio)
        chunks, cur, cur_len = [], [], 0
        for sent in Embedding.split_sentences(text):
            pieces = [sent[i:i + max_chars] for i in range(0, len(sent), max_chars)]
            for piece in pieces:
                if cur and cur_len + len(piece) > max_chars:
                    chunks.append("".join(cur).strip())
                    cur = Embedding._tail_overlap(cur, max_overlap_chars)
                    cur_len = sum(len(s) for s in cur)
                    if cur_len + len(piece) > max_chars:
                        cur, cur_len = [], 0
                cur.append(piece)
                cur_len += len(piece)
        if cur:
            chunks.append("".join(cur).strip())
        return chunks

    @staticmethod
    def normalize_path(path):
        return os.path.realpath(path)

    @staticmethod
    def source_label(meta):
        label = meta.get("name") or os.path.basename(meta.get("source", ""))
        if meta.get("headings"):
            label += f" > {meta['headings']}"
        if meta.get("page"):
            label += f"（第 {meta['page']} 頁）"
        return label

    def get_embeddings(self, texts, is_query=False, titles=None):
        if is_query:
            texts = [f"task: search result | query: {t}" for t in texts]
        else:
            titles = titles or ["none"] * len(texts)
            texts = [f"title: {title or 'none'} | text: {t}" for title, t in zip(titles, texts)]

        vectors = []
        total_batches = (len(texts) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
        for b, start in enumerate(range(0, len(texts), EMBED_BATCH_SIZE), 1):
            batch = texts[start:start + EMBED_BATCH_SIZE]
            vectors.extend(self._embed_batch(batch))
            if total_batches > 1:
                print(f"  embedding 批次 {b}/{total_batches} 完成（{len(vectors)}/{len(texts)}）")
        return vectors

    def _embed_batch(self, batch):
        resp = call_with_retry(
            lambda: self.client.models.embed_content(
                model=self.model_name,
                contents=[types.Content(parts=[types.Part.from_text(text=t)]) for t in batch],
                config=types.EmbedContentConfig(output_dimensionality=self.embed_dim),
            ),
            "embedding",
        )
        if len(resp.embeddings) != len(batch):
            raise RuntimeError(f"送出 {len(batch)} 段文字，卻收到 {len(resp.embeddings)} 個向量")
        return [e.values for e in resp.embeddings]

    def _generate(self, prompt):
        resp = call_with_retry(
            lambda: self.client.models.generate_content(
                model=self.llm_model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0),
            ),
            "生成回答",
        )
        return (resp.text or "").strip()

    def load_chunks(self, path):
        name = os.path.basename(path)
        if os.path.splitext(path)[1].lower() in PLAIN_TEXT_EXTS:
            with open(path, encoding="utf-8") as f:
                return [{"text": c, "title": name, "document": c, "meta": {}}
                        for c in self.chunk_text(f.read())]

        converter, chunker = get_docling()
        doc = converter.convert(path).document
        chunks = []
        for chunk in chunker.chunk(dl_doc=doc):
            if not chunk.text.strip():
                continue
            headings = chunk.meta.headings or []
            pages = sorted({p.page_no for item in chunk.meta.doc_items for p in item.prov})
            meta = {"headings": " > ".join(headings)}
            if pages:
                meta["page"] = pages[0]
            heading_prefix = "\n".join(headings)
            for piece in self.chunk_text(chunk.text):
                chunks.append({
                    "text": piece,
                    "title": " > ".join([name, *headings]),
                    "document": f"{heading_prefix}\n{piece}" if heading_prefix else piece,
                    "meta": meta,
                })
        return chunks

    def index_file(self, path):
        path = self.normalize_path(path)
        name = os.path.basename(path)
        with open(path, "rb") as f:
            file_hash = hashlib.sha256(CHUNK_VERSION.encode() + b"|" + f.read()).hexdigest()

        existing = self.collection.get(where={"source": path}, limit=1, include=["metadatas"])
        if existing["metadatas"] and existing["metadatas"][0].get("hash") == file_hash:
            print(f"{name}：內容未變更，略過")
            return False

        items = self.load_chunks(path)
        if not items:
            print(f"{name}：沒有可用的文字，略過")
            return False
        chunks = [it["document"] for it in items]
        print(f"{name}：切成 {len(chunks)} 塊，開始 embedding")
        embeddings = self.get_embeddings([it["text"] for it in items],
                                         titles=[it["title"] for it in items])
        self.collection.delete(where={"source": path})

        ids = [f"{path}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": path, "name": name, "chunk": i, "hash": file_hash, **it["meta"]}
                     for i, it in enumerate(items)]
        step = self.chroma_client.get_max_batch_size()
        for s in range(0, len(chunks), step):
            self.collection.upsert(
                ids=ids[s:s + step],
                embeddings=embeddings[s:s + step],
                documents=chunks[s:s + step],
                metadatas=metadatas[s:s + step],
            )
        print(f"{name}：已寫入 {len(chunks)} 塊")
        return True

    def indexed_sources(self):
        sources, offset, page = set(), 0, 1000
        while True:
            r = self.collection.get(include=["metadatas"], limit=page, offset=offset)
            if not r["ids"]:
                return sources
            sources.update(m["source"] for m in r["metadatas"])
            offset += page

    @staticmethod
    def find_documents(folder):
        files = []
        for root, dirs, names in os.walk(folder):
            dirs[:] = sorted(d for d in dirs if not d.startswith("."))
            for n in sorted(names):
                if n.startswith((".", "~$")):
                    continue
                if os.path.splitext(n)[1].lower() in SUPPORTED_EXTS:
                    files.append(os.path.join(root, n))
        return files

    def index_folder(self, folder=DOCS_DIR, remove_missing=True):
        folder = self.normalize_path(folder)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"找不到資料夾：{folder}")

        files = self.find_documents(folder)
        updated, failed = [], []
        for path in files:
            try:
                if self.index_file(path):
                    updated.append(path)
            except (errors.APIError, RuntimeError):
                raise
            except Exception as e:
                print(f"{os.path.basename(path)}：處理失敗，略過（{type(e).__name__}: {e}）")
                failed.append(path)

        removed = []
        if remove_missing:
            current, prefix = set(files), folder + os.sep
            for src in self.indexed_sources():
                if src.startswith(prefix) and src not in current:
                    self.collection.delete(where={"source": src})
                    removed.append(src)
                    print(f"{os.path.basename(src)}：檔案已不存在，移除索引")

        print(f"索引完成：共 {len(files)} 個檔案，更新 {len(updated)}，失敗 {len(failed)}，移除 {len(removed)}")
        return {"files": files, "updated": updated, "failed": failed, "removed": removed}

    def retrieve(self, query, top_k=3, max_distance=0.5, verbose=False):
        count = self.collection.count()
        if count == 0:
            return []
        q_emb = self.get_embeddings([query], is_query=True)[0]
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        hits = [{"document": doc, "meta": meta, "distance": dist}
                for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]
        if verbose:
            for h in hits:
                mark = "✓" if h["distance"] <= max_distance else "✗"
                print(f"  {mark} 距離 {h['distance']:.3f}｜{self.source_label(h['meta'])}｜"
                      f"{h['document'][:30].replace(chr(10), ' ')}")
        return [h for h in hits if h["distance"] <= max_distance]

    def rewrite_query(self, query, history=None):
        if not history:
            return query
        convo = "\n".join(f"使用者：{q}\n助理：{a}" for q, a in history[-HISTORY_TURNS:])
        prompt = (
            "根據對話紀錄，把使用者最新的問題改寫成「不需要上下文也看得懂」的完整問題，用來搜尋資料庫。\n"
            "只輸出改寫後的問題，不要任何說明。如果問題本身已經完整，就原樣輸出。\n\n"
            f"對話紀錄：\n{convo}\n\n最新問題：{query}"
        )
        return self._generate(prompt) or query

    def answer(self, query, history=None, top_k=3, max_distance=0.5, verbose=False):
        search_query = self.rewrite_query(query, history)
        if verbose and search_query != query:
            print(f"  改寫後的問題：{search_query}")

        hits = self.retrieve(search_query, top_k, max_distance, verbose)
        if not hits:
            return NOT_FOUND_ANSWER

        context = "\n\n".join(f"[{i}] 來源：{self.source_label(h['meta'])}\n{h['document']}"
                              for i, h in enumerate(hits, 1))
        prompt = (
            "你只能根據下方「資料」回答問題，不可以使用你自己的知識。\n"
            "如果資料中沒有足夠資訊回答，請只回答：「資料中沒有提到。」\n"
            "不要猜測，也不要補充資料以外的內容。\n"
            "引用資料時，請在該句句末標註資料編號，例如 [1]。\n\n"
            f"資料：\n{context}\n\n問題：{search_query}"
        )
        text = self._generate(prompt) or "資料中沒有提到。"

        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(hits)})
        if cited:
            text += SOURCES_HEADER + "".join(
                f"\n[{i}] {self.source_label(hits[i - 1]['meta'])}" for i in cited)
        return text

    @staticmethod
    def strip_sources(answer_text):
        return answer_text.split(SOURCES_HEADER)[0]


if __name__ == "__main__":
    emb = Embedding()
    emb.index_folder(DOCS_DIR)
    for q in ["請問我今天吃了什麼?", "今天天氣如何?", "台灣的首都是哪裡?"]:
        print(f"問：{q}")
        print(f"答：{emb.answer(q, verbose=True)}\n")
