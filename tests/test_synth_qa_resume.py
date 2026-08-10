"""
Unit tests for --resume support in scripts/synth_qa.py.

These test the pure file-I/O / filtering logic only: no network calls,
no LLM backend involved, no real_kb.jsonl required.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.synth_qa import _filter_chunks_for_resume, _read_existing_output_rows

CHUNKS = [
    {"id": "real_00000", "answer": "chunk 0 text", "source": "3GPP TS 36.300"},
    {"id": "real_00001", "answer": "chunk 1 text", "source": "3GPP TS 36.300"},
    {"id": "real_00002", "answer": "chunk 2 text", "source": "3GPP TS 36.300"},
]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


class TestReadExistingOutputRows:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        rows, n_bad = _read_existing_output_rows(tmp_path / "does_not_exist.jsonl")
        assert rows == []
        assert n_bad == 0

    def test_reads_valid_rows(self, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        _write_jsonl(
            out,
            [{"source_chunk_id": "real_00000"}, {"source_chunk_id": "real_00001"}],
        )
        rows, n_bad = _read_existing_output_rows(out)
        assert len(rows) == 2
        assert n_bad == 0

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        with out.open("w") as fh:
            fh.write(json.dumps({"source_chunk_id": "real_00000"}) + "\n")
            fh.write("\n")
            fh.write("   \n")
            fh.write(json.dumps({"source_chunk_id": "real_00001"}) + "\n")
        rows, n_bad = _read_existing_output_rows(out)
        assert len(rows) == 2
        assert n_bad == 0

    def test_skips_malformed_trailing_line_without_crashing(self, tmp_path: Path) -> None:
        """Simulates a process killed mid-write: the last line is a
        truncated, invalid JSON fragment with no trailing newline."""
        out = tmp_path / "out.jsonl"
        with out.open("w") as fh:
            fh.write(json.dumps({"source_chunk_id": "real_00000"}) + "\n")
            fh.write(json.dumps({"source_chunk_id": "real_00001"}) + "\n")
            fh.write('{"source_chunk_id": "real_00002", "question": "trunc')
        rows, n_bad = _read_existing_output_rows(out)
        assert len(rows) == 2
        assert n_bad == 1

    def test_multiple_pairs_per_chunk_all_read(self, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        _write_jsonl(
            out,
            [
                {"source_chunk_id": "real_00000", "question": "q0"},
                {"source_chunk_id": "real_00000", "question": "q0b"},
            ],
        )
        rows, n_bad = _read_existing_output_rows(out)
        assert len(rows) == 2
        assert n_bad == 0


class TestFilterChunksForResume:
    def test_resume_false_returns_all_chunks_unchanged(self, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        _write_jsonl(out, [{"source_chunk_id": "real_00000"}])
        remaining, n_skipped, n_bad = _filter_chunks_for_resume(CHUNKS, out, resume=False)
        assert remaining == CHUNKS
        assert n_skipped == 0
        assert n_bad == 0

    def test_resume_true_no_existing_file(self, tmp_path: Path) -> None:
        out = tmp_path / "does_not_exist.jsonl"
        remaining, n_skipped, n_bad = _filter_chunks_for_resume(CHUNKS, out, resume=True)
        assert remaining == CHUNKS
        assert n_skipped == 0
        assert n_bad == 0

    def test_resume_true_skips_already_processed_chunks(self, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        _write_jsonl(
            out,
            [
                # real_00000 produced two pairs — still only skips once
                {"source_chunk_id": "real_00000", "question": "q0"},
                {"source_chunk_id": "real_00000", "question": "q0b"},
                {"source_chunk_id": "real_00001", "question": "q1"},
            ],
        )
        remaining, n_skipped, n_bad = _filter_chunks_for_resume(CHUNKS, out, resume=True)
        assert [c["id"] for c in remaining] == ["real_00002"]
        assert n_skipped == 2
        assert n_bad == 0

    def test_resume_true_all_chunks_already_processed(self, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        _write_jsonl(out, [{"source_chunk_id": c["id"]} for c in CHUNKS])
        remaining, n_skipped, n_bad = _filter_chunks_for_resume(CHUNKS, out, resume=True)
        assert remaining == []
        assert n_skipped == 3

    def test_resume_true_no_chunks_already_processed(self, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        _write_jsonl(out, [{"source_chunk_id": "unrelated_chunk"}])
        remaining, n_skipped, n_bad = _filter_chunks_for_resume(CHUNKS, out, resume=True)
        assert remaining == CHUNKS
        assert n_skipped == 0

    def test_resume_true_tolerates_malformed_trailing_line(self, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        with out.open("w") as fh:
            fh.write(json.dumps({"source_chunk_id": "real_00000"}) + "\n")
            fh.write('{"source_chunk_id": "real_00001", "answer": "cut off mid')
        remaining, n_skipped, n_bad = _filter_chunks_for_resume(CHUNKS, out, resume=True)
        # real_00000 correctly recognized as done. real_00001's row never
        # successfully parsed, so it's NOT recognized as done and gets
        # reprocessed rather than silently lost.
        assert [c["id"] for c in remaining] == ["real_00001", "real_00002"]
        assert n_skipped == 1
        assert n_bad == 1

    def test_resume_true_rows_without_source_chunk_id_are_ignored(self, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        _write_jsonl(out, [{"question": "no source_chunk_id field here"}])
        remaining, n_skipped, n_bad = _filter_chunks_for_resume(CHUNKS, out, resume=True)
        assert remaining == CHUNKS
        assert n_skipped == 0
        assert n_bad == 0

    def test_resume_true_preserves_chunk_order(self, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        _write_jsonl(out, [{"source_chunk_id": "real_00001"}])
        remaining, n_skipped, n_bad = _filter_chunks_for_resume(CHUNKS, out, resume=True)
        assert [c["id"] for c in remaining] == ["real_00000", "real_00002"]
