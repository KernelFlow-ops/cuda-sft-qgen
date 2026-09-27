"""BM25 doc index (task T34)."""
import json


def _write(dirp, name, text, url):
    (dirp / f"{name}.txt").write_text(text, encoding="utf-8")
    (dirp / f"{name}.meta.json").write_text(json.dumps({"url": url, "title": name, "toolkit_version": "12.x", "sha256": "x"}), encoding="utf-8")


def test_tokenize_mixed():
    from cuda_sft_qgen.knowledge.doc_index import tokenize
    assert tokenize("TMA 对齐 cp.async.bulk 16") == ["tma", "对", "齐", "cp.async.bulk", "16"]


def test_split_chunks():
    from cuda_sft_qgen.knowledge.doc_index import split_chunks
    text = "\n\n".join(["para %d " % i + "word " * 100 for i in range(5)])
    chunks = split_chunks("d", text, {"url": "u", "title": "t", "toolkit_version": "v"}, min_chars=800)
    assert chunks[0].chunk_id == "d#0" and all(c.url == "u" for c in chunks)
    assert "".join(c.text for c in chunks).replace("\n", "").replace(" ", "") == text.replace("\n", "").replace(" ", "")


def test_long_document_line_is_bounded():
    from cuda_sft_qgen.knowledge.doc_index import split_chunks
    text = "CUDA TMA descriptor alignment requirement. " * 500
    chunks = split_chunks("large", text, {"url": "https://docs.nvidia.com/a"})
    assert len(chunks) > 1
    assert max(len(chunk.text) for chunk in chunks) <= 2400
    assert "".join(chunk.text for chunk in chunks).replace(" ", "") == text.replace(" ", "")


def test_search_ranks_relevant(tmp_path):
    from cuda_sft_qgen.knowledge.doc_index import DocIndex
    _write(tmp_path, "tma", "The global address must be aligned to 16 bytes for cuTensorMapEncodeTiled. " * 20, "https://docs.nvidia.com/a")
    _write(tmp_path, "nccl", "NCCL all_reduce bus bandwidth equals algbw times 2(n-1)/n. " * 20, "https://docs.nvidia.com/b")
    (tmp_path / "orphan.txt").write_text("no meta file", encoding="utf-8")
    idx = DocIndex.from_dir(tmp_path)
    hits = idx.search("cuTensorMapEncodeTiled global address alignment", k=2)
    assert hits and hits[0][0].url == "https://docs.nvidia.com/a" and hits[0][1] >= hits[-1][1]
    assert all("orphan" not in c.chunk_id for c, _ in idx.search("meta file", k=5))
    assert DocIndex([]).search("x") == [] and idx.search("") == []
