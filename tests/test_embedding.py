import uuid
from types import SimpleNamespace

import chromadb
import pytest
from google.genai import errors

import Embedding as M
from Embedding import Embedding


def test_split_sentences_keeps_all_text():
    text = '他說：「今天好熱。」我點頭。版本 3.14 已發布. Next one. 真的嗎？是的…\n\n第二段\n第三行'
    assert "".join(Embedding.split_sentences(text)) == text


def test_split_sentences_boundaries():
    s = Embedding.split_sentences('他說：「今天好熱。」我點頭。版本 3.14 已發布. Next one.')
    assert s[0] == "他說：「今天好熱。」"
    assert s[1] == "我點頭。"
    assert "3.14" in s[2]


def test_split_sentences_keeps_blank_line_with_previous():
    s = Embedding.split_sentences("是的…\n\n第二段")
    assert s == ["是的…\n\n", "第二段"]


def test_chunk_text_respects_max_and_sentence_boundaries():
    text = "".join(f"第{i}項：員工應妥善保管帳號密碼，不得與他人共用。" for i in range(1, 40))
    chunks = Embedding.chunk_text(text, max_chars=100, overlap_ratio=0.15)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    assert all(c.startswith("第") and c.endswith("。") for c in chunks)


def test_chunk_text_overlap_is_whole_sentences_within_ratio():
    sentences = [f"短{i}。" for i in range(6)] + [f"第{i}句一二三四五六七八九十一二三四五。" for i in range(6)]
    chunks = Embedding.chunk_text("".join(sentences), max_chars=60, overlap_ratio=0.15)
    for prev, nxt in zip(chunks, chunks[1:]):
        overlap = next((k for k in range(len(nxt), 0, -1) if prev.endswith(nxt[:k])), 0)
        assert overlap <= 9
        assert overlap == 0 or nxt[:overlap].endswith("。")


def test_chunk_text_hard_splits_long_sentence_without_punctuation():
    chunks = Embedding.chunk_text("長" * 250, max_chars=100)
    assert [len(c) for c in chunks] == [100, 100, 50]


def test_chunk_text_empty():
    assert Embedding.chunk_text("") == []


def test_tail_overlap_never_takes_every_sentence():
    assert Embedding._tail_overlap(["a。", "b。"], 100) == ["b。"]
    assert Embedding._tail_overlap(["only。"], 100) == []


def test_source_label():
    assert Embedding.source_label({"name": "a.pdf", "headings": "第一章", "page": 3}) == "a.pdf > 第一章（第 3 頁）"
    assert Embedding.source_label({"source": "/x/sample.txt"}) == "sample.txt"


def test_default_collection_name_depends_on_model_and_dim():
    assert M.default_collection_name("models/gemini-embedding-2", 768, "kb") == "kb_gemini-embedding-2_d768"
    assert M.default_collection_name("models/x", 768) != M.default_collection_name("models/x", 3072)


def api_error(code, message="fake", quota_id=None):
    details = {"error": {"code": code, "message": message}}
    if quota_id:
        details["error"]["details"] = [{"violations": [{"quotaId": quota_id}]}]
    return errors.APIError(code, details)


def test_retry_recovers_from_transient_errors(monkeypatch):
    monkeypatch.setattr(M.time, "sleep", lambda s: None)
    calls = iter([api_error(429), api_error(503), "ok"])

    def fn():
        r = next(calls)
        if isinstance(r, Exception):
            raise r
        return r

    assert M.call_with_retry(fn) == "ok"


def test_retry_does_not_retry_client_errors(monkeypatch):
    monkeypatch.setattr(M.time, "sleep", lambda s: None)
    n = {"c": 0}

    def fn():
        n["c"] += 1
        raise api_error(400)

    with pytest.raises(errors.APIError):
        M.call_with_retry(fn)
    assert n["c"] == 1


def test_retry_stops_immediately_on_daily_quota(monkeypatch):
    monkeypatch.setattr(M.time, "sleep", lambda s: None)
    n = {"c": 0}

    def fn():
        n["c"] += 1
        raise api_error(429, quota_id="GenerateRequestsPerDayPerProjectPerModel-FreeTier")

    with pytest.raises(RuntimeError, match="每日配額"):
        M.call_with_retry(fn)
    assert n["c"] == 1


class FakeModels:
    def __init__(self, answer="答案 [1]"):
        self.answer = answer
        self.prompts = []

    def embed_content(self, model, contents, config):
        dim = config.output_dimensionality
        return SimpleNamespace(embeddings=[
            SimpleNamespace(values=[float(len(c.parts[0].text) % 7 + 1)] + [1.0] * (dim - 1))
            for c in contents
        ])

    def generate_content(self, model, contents, config):
        self.prompts.append(contents)
        return SimpleNamespace(text=self.answer)


def make_rag(answer="答案 [1]"):
    models = FakeModels(answer)
    rag = Embedding(
        embed_dim=8,
        client=SimpleNamespace(models=models),
        chroma_client=chromadb.EphemeralClient(),
        collection_name=f"test-{uuid.uuid4().hex}",
    )
    return rag, models


def test_index_folder_skips_unchanged_and_removes_deleted(tmp_path):
    (tmp_path / "a.txt").write_text("蘋果很好吃。", encoding="utf-8")
    (tmp_path / "b.txt").write_text("香蕉是黃色的。", encoding="utf-8")
    (tmp_path / "ignore.py").write_text("print(1)", encoding="utf-8")
    (tmp_path / "~$temp.docx").write_text("x", encoding="utf-8")
    rag, _ = make_rag()

    r1 = rag.index_folder(tmp_path)
    assert len(r1["files"]) == 2 and len(r1["updated"]) == 2

    r2 = rag.index_folder(tmp_path)
    assert r2["updated"] == []

    (tmp_path / "b.txt").unlink()
    r3 = rag.index_folder(tmp_path)
    assert len(r3["removed"]) == 1
    assert {m["name"] for m in rag.collection.get(include=["metadatas"])["metadatas"]} == {"a.txt"}


def test_same_file_via_different_paths_is_indexed_once(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("蘋果很好吃。", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    rag, _ = make_rag()
    assert rag.index_file("a.txt") is True
    assert rag.index_file("./a.txt") is False
    assert rag.collection.count() == 1


def test_answer_appends_cited_sources(tmp_path):
    (tmp_path / "a.txt").write_text("蘋果很好吃。", encoding="utf-8")
    rag, models = make_rag(answer="蘋果很好吃 [1]")
    rag.index_folder(tmp_path)
    result = rag.answer("蘋果好吃嗎", max_distance=2)
    assert result.startswith("蘋果很好吃 [1]")
    assert "來源：\n[1] a.txt" in result
    assert "[1] 來源：a.txt" in models.prompts[-1]
    assert Embedding.strip_sources(result) == "蘋果很好吃 [1]"


def test_answer_uses_history_to_rewrite_query(tmp_path):
    (tmp_path / "a.txt").write_text("蘋果很好吃。", encoding="utf-8")
    rag, models = make_rag(answer="改寫後的問題")
    rag.index_folder(tmp_path)
    rag.answer("那它呢", history=[("蘋果好吃嗎", "很好吃")], max_distance=2)
    assert "對話紀錄" in models.prompts[0] and "蘋果好吃嗎" in models.prompts[0]
    assert "問題：改寫後的問題" in models.prompts[1]


def test_answer_without_history_makes_single_llm_call(tmp_path):
    (tmp_path / "a.txt").write_text("蘋果很好吃。", encoding="utf-8")
    rag, models = make_rag()
    rag.index_folder(tmp_path)
    rag.answer("蘋果好吃嗎", max_distance=2)
    assert len(models.prompts) == 1
