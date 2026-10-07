"""
Nova — Streamlit product UI.

Thin presentation layer over `NovaService` (the same LangGraph agent the API uses).
Shows streaming answers, an execution timeline (actions, never model reasoning),
validated citations with previews, evidence/verification badges, latency, finance
panels and the human-in-the-loop approval card.

Run with:  streamlit run streamlit_app_pro.py
"""

from __future__ import annotations

import html
import uuid
from datetime import datetime
from typing import Any

import streamlit as st

from nova.config import PROJECT_ROOT
from nova.demo import DEMO_QUERIES, sample_meeting_transcript, sample_report, sample_strategy
from nova.errors import NovaError
from nova.evidence import format_span, format_timestamp
from nova.service import NovaService, TurnResult, get_service

st.set_page_config(page_title="Nova · AI Research Assistant", page_icon="✨", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(f"<style>{(PROJECT_ROOT / 'ui' / 'style.css').read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True)

esc = html.escape


@st.cache_resource(show_spinner="Starting Nova…")
def load_service() -> NovaService:
    return get_service()


@st.cache_data(ttl=15, show_spinner=False)
def readiness() -> dict[str, Any]:
    return load_service().readiness()


# ============================== state ==============================
ss = st.session_state
ss.setdefault("thread_id", str(uuid.uuid4()))
ss.setdefault("pending_prompt", None)
ss.setdefault("approval_required", False)
ss.setdefault("flash", None)
ss.setdefault("player", None)  # (media_id, start_seconds) for timestamp seeking

try:
    svc = load_service()
except Exception as exc:  # noqa: BLE001 - show a friendly startup error, details in logs
    st.error("Nova could not start. Check the configuration in `.env` and the logs.")
    st.caption(type(exc).__name__)
    st.stop()

ready = readiness()
thread_id: str = ss["thread_id"]

if not ready["checks"]["llm"]["ok"]:
    st.markdown(
        f"""<div class="hero"><div class="hero-badge" style="border-color:rgba(239,68,68,.5);color:#fca5a5;
        background:rgba(239,68,68,.12)">⚠️ Model server unavailable</div><h1>Nova isn't ready</h1>
        <p>{esc(ready['checks']['llm']['detail'])}</p></div>""",
        unsafe_allow_html=True,
    )
    if st.button("🔄  Check again"):
        readiness.clear()
        st.rerun()
    st.stop()


# ============================== helpers ==============================
def badge(text: str, kind: str = "") -> str:
    return f'<span class="badge {kind}">{esc(text)}</span>'


VERIFICATION_BADGE = {
    "VERIFIED": ("✓ Evidence verified", "ok"),
    "INSUFFICIENT_EVIDENCE": ("⚠ Insufficient evidence", "warn"),
    "FAILED": ("⚠ Partially verified", "warn"),
    "SKIPPED": ("General answer", ""),
}


def jump_button(media_id: str, start: float, key: str, label: str | None = None) -> None:
    """▶ seek button — only offered when the recording can actually be played back."""
    try:
        playable = svc.media is not None and (
            svc.media.source_path(media_id, thread_id) is not None
            or svc.media.get(media_id, thread_id).source_type == "youtube")
    except NovaError:
        playable = False
    if playable and st.button(label or f"▶ Jump to {format_timestamp(start)}", key=key):
        ss["player"] = (media_id, int(start))
        st.rerun()


def render_meta(meta: dict[str, Any], key: str = "m") -> None:
    """Badges, sources, finance panel and timeline under an assistant message."""
    if not meta:
        return
    label, kind = VERIFICATION_BADGE.get(meta.get("verification_status") or "", ("", ""))
    parts = [badge(f"route: {meta.get('route')}")]
    if label:
        parts.append(badge(label, kind))
    if meta.get("plan"):
        parts.append(badge(f"evidence {meta.get('evidence_score', 0):.2f}"))
    if meta.get("latency_ms"):
        parts.append(badge(f"{meta['latency_ms'] / 1000:.1f}s"))
    st.markdown(f'<div class="meta-row">{"".join(parts)}</div>', unsafe_allow_html=True)

    for panel in meta.get("finance") or []:
        render_finance(panel)

    citations = meta.get("citations") or []
    if citations:
        with st.expander(f"📚 Sources ({len(citations)})"):
            for c in citations:
                if c["type"] == "document":
                    st.markdown(
                        f'<div class="source-card"><b>📄 {esc(c["filename"])}</b> · page {c["page"]}'
                        f'<div class="preview">{esc((c.get("preview") or "")[:300])}…</div></div>',
                        unsafe_allow_html=True,
                    )
                elif c["type"] == "video":
                    span = format_span(c["start_time"], c["end_time"])
                    icon = "🎧" if c.get("source_type") == "audio" else "🎬"
                    st.markdown(
                        f'<div class="source-card"><b>{icon} {esc(c["source_name"])}</b> · {span}'
                        f'<div class="preview">{esc((c.get("preview") or "")[:300])}…</div></div>',
                        unsafe_allow_html=True,
                    )
                    jump_button(c["media_id"], c["start_time"], key=f"{key}-{c['id']}")
                elif c["type"] == "web":
                    st.markdown(f"🌐 [{c.get('title') or c['domain']}]({c['url']}) — `{c['domain']}`")
                else:
                    st.markdown(f"📈 {c.get('label')}")
    if meta.get("timeline"):
        with st.expander("🧭 Execution timeline"):
            st.markdown("\n".join(f"- {esc(step)}" for step in meta["timeline"]))
            if meta.get("latency_breakdown"):
                st.caption(" · ".join(f"{k} {v / 1000:.2f}s" for k, v in meta["latency_breakdown"].items()))


def render_finance(panel: dict[str, Any]) -> None:
    d = panel["data"]
    if panel["tool"] == "stock_quote":
        cols = st.columns(3)
        cols[0].metric(f"{d['symbol']} price", f"{d['price']:,.2f} {d.get('currency') or ''}",
                       f"{d['change_percent']:+.2f}%" if d.get("change_percent") is not None else None)
        if d.get("day_low") is not None:
            cols[1].metric("Day range", f"{d['day_low']:,.2f} – {d['day_high']:,.2f}")
        if d.get("fifty_two_week_low") is not None:
            cols[2].metric("52-week range", f"{d['fifty_two_week_low']:,.2f} – {d['fifty_two_week_high']:,.2f}")
        st.caption(f"Source: {panel['source']} · market data is informational, not investment advice")
    elif panel["tool"] == "price_history":
        a = d["analytics"]
        cols = st.columns(4)
        cols[0].metric(f"{d['symbol']} return ({d['period']})", f"{a['period_return_pct']:+.2f}%")
        cols[1].metric("Volatility (ann.)", f"{a['annualized_volatility_pct']:.1f}%" if a["annualized_volatility_pct"] is not None else "–")
        cols[2].metric("Max drawdown", f"{a['max_drawdown_pct']:.2f}%" if a["max_drawdown_pct"] is not None else "–")
        cols[3].metric("Last close", f"{a['end_price']:,.2f}")
        series = d.get("series") or {}
        if series.get("dates"):
            st.line_chart({"close": dict(zip(series["dates"], series["closes"]))}, height=180)
        st.caption(f"Source: {panel['source']}")


def render_approval(approval: dict[str, Any]) -> None:
    ops = "".join(
        f"<li><b>{esc(o['tool'])}</b>: {esc(o['input'])}{' · ' + esc(o['period']) if o.get('period') else ''}"
        f" — {esc(o['description'])}</li>"
        for o in approval.get("operations", [])
    )
    st.markdown(f'<div class="approval-card">🛡️ <b>{esc(approval.get("message", "Approval required"))}</b>'
                f"<ul>{ops}</ul></div>", unsafe_allow_html=True)
    c1, c2, _ = st.columns([1, 1, 4])
    if c1.button("✅ Approve", type="primary", key="approve"):
        run_turn(None, resume={"approved": True})
    if c2.button("✖ Reject", key="reject"):
        run_turn(None, resume={"approved": False})


def run_turn(prompt: str | None, resume: dict[str, Any] | None = None) -> None:
    """Stream one agent turn into the chat, then rerun to render it from the checkpoint."""
    if prompt:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
    with st.chat_message("assistant", avatar="✨"):
        status = st.status("Working…", expanded=False)
        placeholder = st.empty()
        buffer = ""
        result: TurnResult | None = None
        try:
            for event in svc.stream(prompt, thread_id, approval_required=ss["approval_required"], resume=resume):
                if event["type"] == "step":
                    status.update(label=f"{event['label']}…", state="running")
                    status.write(f"✓ {event['label']}")
                elif event["type"] == "token":
                    buffer += event["text"]
                    placeholder.markdown(buffer + "▌")
                elif event["type"] == "retry":
                    buffer = ""
                    placeholder.markdown("_Refining the answer to match the evidence…_")
                elif event["type"] == "result":
                    result = event["result"]
        except NovaError as exc:
            status.update(label="Could not complete", state="error")
            placeholder.empty()
            st.error(exc.user_message)
            return
        if result and result.status == "awaiting_approval":
            status.update(label="Waiting for approval", state="complete")
        else:
            status.update(label="Done", state="complete")
    st.rerun()


def export_markdown(messages: list[dict[str, Any]]) -> str:
    lines = [f"# Nova chat export — {datetime.now():%Y-%m-%d %H:%M}\n"]
    for m in messages:
        lines.append(f"**{'🧑 You' if m['role'] == 'user' else '✨ Nova'}:**\n\n{m['content']}\n")
    return "\n".join(lines)


try:
    thread = svc.get_thread(thread_id)
except NovaError:
    thread = {"messages": [], "documents": svc.list_documents(thread_id), "pending_approval": None, "title": None,
              "media": [a.model_dump() for a in svc.media.list_assets(thread_id)] if svc.media else []}
messages: list[dict[str, Any]] = thread["messages"]
pending = thread.get("pending_approval")

# ============================== sidebar ==============================
with st.sidebar:
    st.markdown(
        """<div class="brand"><div class="brand-logo">✨</div><div><div class="brand-name">Nova</div>
        <div class="brand-sub">Financial &amp; document intelligence</div></div></div>""",
        unsafe_allow_html=True,
    )
    if st.button("＋  New chat", type="primary", use_container_width=True):
        ss["thread_id"] = str(uuid.uuid4())
        st.rerun()

    # ---- documents ----
    st.markdown('<div class="side-label">Documents</div>', unsafe_allow_html=True)
    docs = thread["documents"]
    for doc in docs:
        st.markdown(
            f'<div class="doc-card">📄 <b>{esc(doc["filename"])}</b><div class="doc-stats">'
            f'<span>{doc["pages"]} pages</span><span>{doc["chunks"]} chunks</span><span>● indexed</span></div></div>',
            unsafe_allow_html=True,
        )
    if not docs:
        st.markdown('<div class="doc-empty">No PDF attached to this chat yet.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed", key=f"up-{thread_id}")
    if uploaded is not None and uploaded.name not in {d["filename"] for d in docs}:
        with st.status("Indexing PDF locally…", expanded=True) as box:
            try:
                info = svc.ingest_document(thread_id, uploaded.name, uploaded.getvalue())
                box.update(label=f"✅ Indexed {info['chunks']} chunks", state="complete", expanded=False)
                if info.get("injection_flagged_chunks"):
                    ss["flash"] = ("warning", f"{info['injection_flagged_chunks']} passage(s) in this PDF contain "
                                   "instruction-like text. Nova treats them as quoted data only.")
                st.rerun()
            except NovaError as exc:
                box.update(label="❌ Indexing failed", state="error")
                st.error(exc.user_message)
    for label, sample in (("📎 Load sample report (demo)", sample_report()),
                          ("📎 Load sample strategy (demo)", sample_strategy())):
        if sample and sample[0] not in {d["filename"] for d in docs}:
            if st.button(label, use_container_width=True):
                try:
                    svc.ingest_document(thread_id, *sample)
                    st.rerun()
                except NovaError as exc:
                    st.error(exc.user_message)

    # ---- media (video / audio) ----
    if svc.media is not None:
        st.markdown('<div class="side-label">Video &amp; audio</div>', unsafe_allow_html=True)
        assets = thread.get("media") or []
        for a in assets:
            status = a["status"]
            icon = {"COMPLETED": "✅", "FAILED": "❌", "PROCESSING": "⏳", "QUEUED": "🕓"}.get(status, "•")
            dur = format_timestamp(a["duration_seconds"]) if a.get("duration_seconds") else "–"
            st.markdown(
                f'<div class="doc-card">{icon} <b>{esc(a["filename"])}</b><div class="doc-stats">'
                f'<span>{dur}</span><span>{a.get("segments") or 0} segments</span>'
                f'<span>{a.get("chunks") or 0} chunks</span><span>{esc(status.lower())}</span></div></div>',
                unsafe_allow_html=True,
            )
            if status == "FAILED":
                st.caption(f"⚠ {a.get('error') or 'Processing failed.'}")
            c1, c2 = st.columns(2)
            if status in ("FAILED", "QUEUED") and c1.button("↻ Process", key=f"proc-{a['media_id']}",
                                                             use_container_width=True):
                ss["process_media"] = a["media_id"]
                st.rerun()
            if c2.button("🗑 Remove", key=f"del-{a['media_id']}", use_container_width=True):
                svc.media.delete(a["media_id"], thread_id)
                if ss.get("player") and ss["player"][0] == a["media_id"]:
                    ss["player"] = None
                st.rerun()
        if not assets:
            st.markdown('<div class="doc-empty">No recording attached. Upload a meeting, call or lecture.</div>',
                        unsafe_allow_html=True)
        language = st.selectbox("Language", ["auto", "english", "hinglish"], key=f"lang-{thread_id}",
                                format_func=lambda x: {"auto": "Auto-detect (Whisper)", "english": "English (Whisper)",
                                                       "hinglish": "Hinglish → English (Sarvam)"}[x])
        with_insights = st.checkbox("Generate summary & insights", value=True, key=f"ins-{thread_id}",
                                    help="Summary, key decisions, action items and open questions (one LLM pass).")
        media_file = st.file_uploader(
            "Upload video/audio", label_visibility="collapsed", key=f"media-{thread_id}",
            type=["mp4", "mov", "avi", "mkv", "webm", "mp3", "wav", "m4a", "srt", "vtt"])
        if media_file is not None and media_file.name not in {a["filename"] for a in assets}:
            try:
                if media_file.name.lower().endswith((".srt", ".vtt")):
                    new = svc.media.create_subtitle_import(thread_id, media_file.name,
                                                           media_file.getvalue().decode("utf-8", "replace"))
                else:
                    new = svc.media.create_upload(thread_id, media_file.name, media_file.getvalue(), language)
                svc.registry.touch(thread_id, f"🎬 {new.filename}", count_turn=False)
                ss["process_media"] = new.media_id
                ss["process_insights"] = with_insights
                st.rerun()
            except NovaError as exc:
                st.error(exc.user_message)
        yt = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=…", key=f"yt-{thread_id}")
        if yt and st.button("Add YouTube video", use_container_width=True, key=f"ytb-{thread_id}"):
            try:
                new = svc.media.create_youtube(thread_id, yt, language)
                svc.registry.touch(thread_id, f"🎬 {new.source_uri}", count_turn=False)
                ss["process_media"] = new.media_id
                ss["process_insights"] = with_insights
                st.rerun()
            except NovaError as exc:
                st.error(exc.user_message)
        sample_media = sample_meeting_transcript()
        if sample_media and sample_media[0] not in {a["filename"] for a in assets}:
            if st.button("🎬 Load sample meeting (demo)", use_container_width=True):
                new = svc.media.create_from_transcript(thread_id, *sample_media)
                svc.registry.touch(thread_id, f"🎬 {new.filename}", count_turn=False)
                ss["process_media"] = new.media_id
                ss["process_insights"] = True
                st.rerun()

    # ---- settings ----
    st.markdown('<div class="side-label">Controls</div>', unsafe_allow_html=True)
    ss["approval_required"] = st.toggle(
        "Require approval for external data", value=ss["approval_required"],
        help="Pause before web search or market-data calls and ask you to approve them (LangGraph interrupt).",
    )

    # ---- conversations ----
    st.markdown('<div class="side-label">Conversations</div>', unsafe_allow_html=True)
    threads = svc.list_threads(50)
    if not threads:
        st.markdown('<div class="doc-empty">Your chats will appear here.</div>', unsafe_allow_html=True)
    for t in threads:
        active = t["thread_id"] == thread_id
        title = t["title"] if len(t["title"]) <= 34 else t["title"][:32] + "…"
        if st.button(("● " if active else "💬 ") + title, key=f"thread-{t['thread_id']}",
                     use_container_width=True, disabled=active):
            ss["thread_id"] = t["thread_id"]
            st.rerun()

    if messages:
        st.markdown('<div class="side-label">This chat</div>', unsafe_allow_html=True)
        st.download_button("⬇  Download chat (.md)", data=export_markdown(messages),
                           file_name=f"nova-chat-{thread_id[:8]}.md", mime="text/markdown", use_container_width=True)
    if thread.get("title") is not None:
        if st.button("🗑  Delete this chat", use_container_width=True):
            svc.delete_thread(thread_id)
            ss["thread_id"] = str(uuid.uuid4())
            st.rerun()

    # ---- system status ----
    st.markdown('<div class="side-label">System</div>', unsafe_allow_html=True)
    s = svc.settings
    tracing = "on" if (s.langchain_tracing_v2 and s.langchain_api_key) else "off"
    finance_src = "Alpha Vantage → Yahoo" if s.alphavantage_api_key and s.finance_provider == "auto" else (
        "Alpha Vantage" if s.finance_provider == "alphavantage" else "Yahoo Finance (keyless)")
    db_ok = ready["checks"]["database"]["ok"]
    st.markdown(
        f'<div class="status-line"><span class="dot-ok">●</span> {esc(ready["checks"]["llm"]["detail"])}<br>'
        f'<span class="{"dot-ok" if db_ok else "dot-bad"}">●</span> SQLite memory · embeddings {esc(s.embedding_model_name)}<br>'
        f"● Market data: {esc(finance_src)}<br>● LangSmith tracing: {tracing}</div>",
        unsafe_allow_html=True,
    )

# ============================== main ==============================
if ss.get("flash"):
    kind, text = ss.pop("flash")
    getattr(st, kind)(text)

if svc.media is not None and ss.get("process_media"):
    media_id = ss.pop("process_media")
    want_insights = ss.pop("process_insights", True)
    with st.status("Processing media…", expanded=True) as box:
        def _progress(step: str, label: str) -> None:
            box.write(f"✓ {label}")
            box.update(label=f"{label}…")

        done = svc.media.process(media_id, insights=want_insights, progress=_progress)
        if done.status.value == "COMPLETED":
            box.update(label=f"✅ {done.filename} ready ({done.segments} segments, {done.chunks} chunks)",
                       state="complete", expanded=False)
        else:
            box.update(label="❌ Processing failed", state="error")
            st.error(done.error or "Processing failed.")
    st.rerun()

if svc.media is not None and ss.get("player"):
    p_id, p_start = ss["player"]
    try:
        p_asset = svc.media.get(p_id, thread_id)
        st.markdown(f"**▶ {esc(p_asset.filename)}** from {format_timestamp(p_start)}")
        if p_asset.source_type == "youtube":
            st.video(p_asset.source_uri, start_time=p_start)
        else:
            path = svc.media.source_path(p_id, thread_id)
            if path is not None:
                (st.audio if p_asset.source_type == "audio" else st.video)(str(path), start_time=p_start)
        if st.button("✖ Close player"):
            ss["player"] = None
            st.rerun()
    except NovaError:
        ss["player"] = None

for a in thread.get("media") or []:
    if a["status"] != "COMPLETED" or svc.media is None:
        continue
    ins = svc.media.insights(a["media_id"], thread_id)
    if ins is None:
        continue
    with st.expander(f"🧠 Meeting intelligence · {a['filename']}" + (f" — {ins.title}" if ins.title else "")):
        for section, items in (("Summary", ins.summary), ("Key decisions", ins.decisions),
                               ("Action items", ins.action_items), ("Open questions", ins.open_questions)):
            st.markdown(f"**{section}**")
            if not items:
                st.caption("Not identified in the transcript.")
            for n, it in enumerate(items):
                span = format_span(it.start_time or 0, it.end_time or 0, a.get("duration_seconds"))
                line = f"- {esc(it.text)} `[{span}]`"
                if section == "Action items":
                    line += (f"  \n  Owner: {esc(it.owner or 'Not identified in the transcript.')} · "
                             f"Deadline: {esc(it.deadline or 'Not identified in the transcript.')}")
                st.markdown(line)
                if it.start_time is not None:
                    jump_button(a["media_id"], it.start_time, key=f"ins-{a['media_id']}-{section}-{n}")

user_input = st.chat_input("Ask about your documents, recordings, markets, or anything else…", disabled=bool(pending))
if ss["pending_prompt"] and not user_input:
    user_input = ss["pending_prompt"]
ss["pending_prompt"] = None

if not messages and not user_input:
    st.markdown(
        f"""<div class="hero"><div class="hero-badge">⚡ LangGraph agent · {esc(svc.settings.chat_model_name)}</div>
        <h1>How can I help you today?</h1>
        <p>Grounded answers from your documents, meetings and videos, live market data and the web — with timestamped citations and verification.</p></div>""",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, (icon, title, prompt) in enumerate(DEMO_QUERIES):
        with cols[i % 2]:
            if st.button(f"{icon}  **{title}**\n{prompt}", key=f"demo-{i}"):
                ss["pending_prompt"] = prompt
                st.rerun()
else:
    turns = sum(1 for m in messages if m["role"] == "user")
    st.markdown(
        f"""<div class="chat-header"><div><div class="chat-title">{esc(thread.get('title') or 'New conversation')}</div>
        <div class="chat-meta">Thread · {thread_id[:8]} · {turns} message{'s' if turns != 1 else ''}</div></div>
        <div class="chat-meta"><span class="live-dot"></span>Online</div></div>""",
        unsafe_allow_html=True,
    )

for idx, m in enumerate(messages):
    with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "✨"):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_meta(m.get("metadata") or {}, key=f"msg{idx}")

if pending:
    render_approval(pending)
elif user_input:
    run_turn(user_input)
