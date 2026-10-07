# Post-verifier smoke evaluation (17 cases)

**What this is:** a small regression check run after the claim-to-cited-source check was added to the
verifier (2026-10-06). It is **not** the main benchmark. The full 42-case and 27-case runs are one level up
in `evaluation/results/`, and they predate this verifier change.

- **Cases:** the 17 cases in categories `direct_document`, `citation`, `absent_answer` and `multi_tool` from
  `evaluation/dataset.json`. The selection was chosen to exercise the verifier, not sampled at random.
- **System:** the enhanced agent only, `qwen3:8b`, CPU-only Ollama.
- **Run history:** the run was interrupted by a machine shutdown. Two cases (`cite-03`, `multi-01`) first
  crashed in the harness (Windows console encoding, since fixed in `evaluation/__init__.py`). They were re-run
  with `--resume`, which retries only harness crashes; the newer record supersedes the older line in
  `runs/enhanced.jsonl` (latest record per case wins).
- **Regenerate:**
  `python -m evaluation.run_evaluation --systems enhanced --cases direct_document,citation,absent_answer,multi_tool --results-dir evaluation/results/smoke_post_verifier --report-only`
- **Finding:** `cite-03` passed but was marked verification `FAILED`, a false positive of the new claim check
  on a provenance sentence ("mentioned on page 10 of the report"). It was fixed afterwards; see
  `../recheck_cite03/`.
