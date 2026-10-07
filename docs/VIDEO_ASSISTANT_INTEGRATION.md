# AI Video Assistant → Nova integration

**Source:** [`risitasutar/AI-Video-Assistant`](https://github.com/risitasutar/AI-Video-Assistant), commit `9b86583`
("first commit"). Audited 2026-10-05 by reading every source file. The repository README declares the MIT
License, but **the repository contains no `LICENSE` file**; Nova reproduces the MIT notice in
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) on that basis.

## 1. Audit of the source project (verified from code)

| Module | What it actually does | Observations |
|---|---|---|
| `utils/audio_processor.py` | `download_youtube_audio`: yt-dlp `bestaudio` with the FFmpeg WAV post-processor, output template `downloades/%(title)s.%(ext)s`. `convert_to_wav`: pydub → mono 16 kHz WAV. `chunk_audio`: 10-minute WAV pieces. `process_input`: any `http(s)://` string is treated as YouTube. | Needs the FFmpeg binary. Output path is built from the **video title** (path injection and collision risk). The `.webm/.m4a → .wav` filename rewrite is a string hack. No size, duration or URL validation. Files are never cleaned up. |
| `core/transcriber.py` | `english` → local `openai-whisper` (`WHISPER_MODEL`, default `small`), returning **only `result["text"]`**. `hinglish` → Sarvam `speech-to-text-translate` (`SARVAM_STT_MODEL`, default `saaras:v2.5`) on 25 s WAV pieces because the sync API rejects audio over 30 s. | **Timestamps are discarded:** Whisper's segments are dropped and the transcript becomes one string. Sarvam output is English (translate endpoint). No caching. |
| `core/summarizer.py` | Map-reduce summary: split at 3000/200 chars, Mistral (`mistral-small-latest`, T=0.3) summarises each part, then a combine prompt returns bullet points. `generate_title` from the first 2000 chars. | Hard-wired to Mistral. |
| `core/extractor.py` | Three prompts over the **whole transcript**: action items (task / owner / deadline, "Not specified" if absent), key decisions, open questions. | Free-text output; nothing checks that owners and deadlines actually appear in the transcript. A long transcript is sent in one call. |
| `core/vector_store.py` | Chroma at `vector_db/`, **single global collection `meeting_transcript`**, `all-MiniLM-L6-v2` via sentence-transformers (torch). 500/50 character chunks with metadata `{chunk_index}` only. | No thread or media isolation: every processed video lands in the same collection. No timestamps in chunk metadata. |
| `core/rag_engine.py` | LCEL chain: retriever(k=4) → "answer ONLY from context" prompt → Mistral. | `load_rag_chain()` calls `get_retriever()` without the required `vector_store` argument, so it raises `TypeError` if used. Retrieved text has no source or time attribution. |
| `app.py` | Streamlit UI (544 lines) with a step bar. | UI only, not reused (Nova has one UI). |
| `main.py`, `test.py` | CLI pipeline and a manual script that calls a live YouTube URL. | No automated tests. |
| `Requirements.txt` | yt-dlp, pydub, ffmpeg-python, openai-whisper, torch, torchaudio, deep-translator, langchain-mistralai, mistralai, chromadb, sentence-transformers, langchain-huggingface, huggingface-hub, tiktoken, streamlit-extras, watchdog, reportlab, fpdf2, … | `deep-translator`, `ffmpeg-python`, `torchaudio`, `tiktoken`, `reportlab`/`fpdf2` are not imported by `core/`, `utils/` or `main.py`. `core/vector_store.py` imports `langchain_chroma`, which is **missing** from the requirements. |

## 2. Migration decisions

| Capability | Decision | Reason |
|---|---|---|
| Whisper transcription | **Reused, with timestamps kept.** `nova/video/transcription.py` keeps Whisper's segment `start`/`end`. The default backend is `faster-whisper` (same Whisper model family via CTranslate2); `openai-whisper` remains selectable (`NOVA_WHISPER_BACKEND=openai`). | Timestamps are the core requirement. faster-whisper needs neither torch nor an FFmpeg binary (PyAV bundles the FFmpeg libraries) and runs Whisper faster on CPU. The dev machine has no FFmpeg, no torch and no GPU. |
| Sarvam (Hinglish) | **Reused:** same endpoint, model variable, 25 s piece size and translate-to-English behaviour. Each piece becomes a segment with its real offset (`start = 25·i`). | Sarvam's sync API returns no word timings, so piece boundaries are the most precise honest timestamps. |
| Audio extraction | **Rewritten** with PyAV (decode any container → mono 16 kHz PCM WAV). | Removes the FFmpeg-binary and pydub dependency; one decoder for MP4/MOV/AVI/MKV/WEBM/MP3/WAV. |
| YouTube | **Reused** (yt-dlp), hardened: host allow-list, metadata-only duration check *before* download, fixed output filename, no FFmpeg post-processing (PyAV decodes the downloaded audio). | Safety and fewer dependencies. |
| 10-minute audio chunking | **Dropped.** | Whisper/faster-whisper stream long audio themselves; Sarvam pieces are produced in-process. |
| Summary + decisions + action items + open questions | **Reused** prompt intent (map → reduce, same four outputs), run through **Nova's LLM abstraction** (Ollama by default, `NOVA_MEDIA_LLM_PROVIDER=mistral` to keep Mistral). Upgraded to structured output with **timestamp evidence per item**, plus a verifier that blanks owners and deadlines not found in the cited transcript text ("Not identified in the transcript"). | Required: one provider abstraction, no fabricated owners or deadlines, timestamped citations. |
| Chroma + MiniLM vector store | **Replaced by Nova's existing FAISS store** (`nova/rag/store.py`), reused for media with a separate root, per-thread / per-media collections and timestamp metadata. | A second vector database adds no capability here; Chroma's global collection was the isolation bug. One store, one embedding model (`nomic-embed-text`), and the same SHA-256 trust boundary for both documents and media. Removes torch and sentence-transformers. |
| RAG Q&A chain | **Replaced** by Nova's agent (`video_search` tool → evidence gate → grounded generation → verifier). | Nova already provides routing, grounding and verification; the old chain had no citations. |
| Streamlit UI, PDF/TXT export | **Not copied.** Media features were added to Nova's single UI. Export is covered by Nova's Markdown chat export. | One application. |

## 3. Dependencies

| Added to Nova | Why |
|---|---|
| `faster-whisper` (brings `ctranslate2`, `av`, `onnxruntime`, `tokenizers`) | Whisper transcription with timestamps; PyAV media decoding |
| `yt-dlp` | YouTube ingestion |

**Not added:** torch, torchaudio, openai-whisper (optional), pydub, ffmpeg-python, chromadb, sentence-transformers,
langchain-huggingface, deep-translator, tiktoken, reportlab, fpdf2, streamlit-extras, watchdog. They are either
unused by the source backend or replaced as described above. `langchain-mistralai` and `openai-whisper` are
listed in `requirements-optional.txt` for anyone who wants those providers.

## 4. Compatibility issues found and how they were handled

- Timestamps lost in transcription → segments preserved end-to-end (transcript → chunks → vectors → evidence → citations).
- Global vector collection → per-thread FAISS collections with ownership checks (tested).
- `load_rag_chain` `TypeError` → that code path is not reused.
- Title-derived download paths → fixed filenames under a validated per-thread directory.
- Unverified extraction → structured items checked against cited transcript text.
- Mistral-only → Nova's provider abstraction (Ollama by default, Mistral optional).

## 5. Status

The implementation, tests, measurements and evaluation results are described in the sections below, which are
updated as the integration is validated.
