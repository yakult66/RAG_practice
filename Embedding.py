import hashlib
import os
import random
import re
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
import chromadb
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer

load_dotenv()
client = genai.Client()
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="my_rag_collection",
    metadata={"hnsw:space": "cosine"},
)

EMBED_BATCH_SIZE = 100
EMBED_MAX_RETRIES = 6
RETRYABLE_CODES = {429, 500, 502, 503, 504}

CHUNK_MAX_CHARS = 500
CHUNK_OVERLAP_RATIO = 0.15
CHUNK_VERSION = f"punct-v2-{CHUNK_MAX_CHARS}-{CHUNK_OVERLAP_RATIO}|docling-v2"

PLAIN_TEXT_EXTS = {".txt"}

_docling = None


def get_docling():
    global _docling
    if _docling is None:

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


_END = "。！？!?；;…"
_CLOSE = "」』”’)）】"
SENTENCE_SPLIT = re.compile(
    rf"(?<=[{_END}])(?![{_CLOSE}{_END}])"
    rf"|(?<=[{_END}][{_CLOSE}])"
    r"|(?<=\.)(?=\s)"
    r"|(?<=\n)(?!\n)"
)


class Embedding:
    def __init__(self, model_name="models/gemini-embedding-2", llm_model="gemini-3.6-flash"):
        self.model_name = model_name
        self.llm_model = llm_model

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
            vectors.extend(self._embed_batch_with_retry(batch))
            if total_batches > 1:
                print(f"  embedding 批次 {b}/{total_batches} 完成（{len(vectors)}/{len(texts)}）")
        return vectors

    def _embed_batch_with_retry(self, batch):
        for attempt in range(1, EMBED_MAX_RETRIES + 1):
            try:
                resp = client.models.embed_content(
                    model=self.model_name,
                    contents=[types.Content(parts=[types.Part.from_text(text=t)]) for t in batch],
                )
                if len(resp.embeddings) != len(batch):
                    raise RuntimeError(f"送出 {len(batch)} 段文字，卻收到 {len(resp.embeddings)} 個向量")
                return [e.values for e in resp.embeddings]
            except errors.APIError as e:
                if e.code not in RETRYABLE_CODES or attempt == EMBED_MAX_RETRIES:
                    raise
                delay = min(2 ** (attempt - 1), 60) + random.uniform(0, 1)
                print(f"  embedding 失敗（{e.code}），{delay:.1f} 秒後重試（第 {attempt}/{EMBED_MAX_RETRIES - 1} 次）")
                time.sleep(delay)

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
        with open(path, "rb") as f:
            file_hash = hashlib.sha256(CHUNK_VERSION.encode() + b"|" + f.read()).hexdigest()

        existing = collection.get(where={"source": path}, limit=1, include=["metadatas"])
        if existing["metadatas"] and existing["metadatas"][0].get("hash") == file_hash:
            print(f"{path}：內容未變更，略過")
            return False

        items = self.load_chunks(path)
        if not items:
            print(f"{path}：沒有可用的文字，略過")
            return False
        chunks = [it["document"] for it in items]
        print(f"{path}：切成 {len(chunks)} 塊，開始 embedding")
        embeddings = self.get_embeddings([it["text"] for it in items],
                                         titles=[it["title"] for it in items])
        collection.delete(where={"source": path})

        ids = [f"{path}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": path, "chunk": i, "hash": file_hash, **it["meta"]}
                     for i, it in enumerate(items)]
        step = chroma_client.get_max_batch_size()
        for s in range(0, len(chunks), step):
            collection.upsert(
                ids=ids[s:s + step],
                embeddings=embeddings[s:s + step],
                documents=chunks[s:s + step],
                metadatas=metadatas[s:s + step],
            )
        print(f"{path}：已寫入 {len(chunks)} 塊")
        return True

    def retrieve(self, query, top_k=3, max_distance=0.5, verbose=False):
        count = collection.count()
        if count == 0:
            return []
        q_emb = self.get_embeddings([query], is_query=True)[0]
        res = collection.query(
            query_embeddings=[q_emb],
            n_results=min(top_k, count),
        )
        print(f"retrieve res: {res}")
        docs, dists = res["documents"][0], res["distances"][0]
        print(f"retrieve docs: {docs}")
        print(f"retrieve dists: {dists}")
        if verbose:
            for doc, dist in zip(docs, dists):
                print(f"  距離 {dist:.3f}｜{doc[:40].replace(chr(10), ' ')}")
        return [doc for doc, dist in zip(docs, dists) if dist <= max_distance]

    def answer(self, query, top_k=3, max_distance=0.5, verbose=False):
        docs = self.retrieve(query, top_k, max_distance, verbose)
        if not docs:
            return "資料中找不到相關內容，無法回答。"

        context = "\n---\n".join(docs)
        prompt = (
            "你只能根據下方「資料」回答問題，不可以使用你自己的知識。\n"
            "如果資料中沒有足夠資訊回答，請只回答：「資料中沒有提到。」\n"
            "不要猜測，也不要補充資料以外的內容。\n\n"
            f"資料：\n{context}\n\n問題：{query}"
        )
        resp = client.models.generate_content(
            model=self.llm_model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0),
        )
        print(f"answer resp: {resp}")
        return resp.text


if __name__ == "__main__":
    emb = Embedding()
    emb.index_file("sample.txt")
    for q in ["請問我今天吃了什麼?", "今天天氣如何?", "台灣的首都是哪裡?"]:
        print(f"問：{q}")
        print(f"答：{emb.answer(q, verbose=True)}\n")
