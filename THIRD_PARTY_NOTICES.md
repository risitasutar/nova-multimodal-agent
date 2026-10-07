# Third-party notices

## AI Video Assistant

Nova's video/audio capability (`nova/video/`) was derived from
[`risitasutar/AI-Video-Assistant`](https://github.com/risitasutar/AI-Video-Assistant) (commit `9b86583`).
That repository's README states it is licensed under the **MIT License**; the repository itself does not
include a `LICENSE` file. The standard MIT terms are reproduced below on the basis of that declaration.

What was reused, and how it was changed, is documented in
[`docs/VIDEO_ASSISTANT_INTEGRATION.md`](docs/VIDEO_ASSISTANT_INTEGRATION.md). In summary: the Whisper/Sarvam
transcription routing (including the Sarvam endpoint, model variable and 25-second piece size), the YouTube
ingestion approach, and the meeting-intelligence design (map-reduce summary, key decisions, action items with
owner/deadline, open questions) originate from that project and were refactored into Nova's architecture.

```
MIT License

Copyright (c) risitasutar (AI-Video-Assistant)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## CampusX — chatbot-in-langgraph

Nova's original foundation is the CampusX
[`chatbot-in-langgraph`](https://github.com/campusx-official/chatbot-in-langgraph) learning repository. That
repository publishes **no license**; its code (preserved in `examples/upstream_tutorial/`) remains the
property of its authors and is not re-licensed by this repository.

## Libraries

Runtime dependencies are listed with pinned versions in `requirements.txt`; each is used under its own
license (LangGraph/LangChain: MIT; FAISS: MIT; faster-whisper: MIT; yt-dlp: Unlicense; PyAV: BSD; Streamlit:
Apache-2.0; FastAPI: MIT; pypdf: BSD).
