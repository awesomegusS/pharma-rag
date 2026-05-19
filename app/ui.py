"""
ui.py — Gradio interface: 3-column layout, streaming chat, doc info panel.

Layout:
  Column 1 (left)   — PDF upload, Process / Clear buttons
  Column 2 (middle) — Document Info panel + Retrieval Settings
  Column 3 (right)  — Streaming chat + example quick-ask buttons
  Bottom             — Status bar + Export chat history
"""

import os
import logging
from typing import Generator

import gradio as gr

from app.store import EnhancedDocumentStore

from app.config import CHAT_HISTORY_PATH, MISTRAL_MAX_REPLY, MISTRAL_N_CTX, MISTRAL_TEMPERATURE

logger = logging.getLogger(__name__)

PFIZER_LOGO = "https://upload.wikimedia.org/wikipedia/commons/0/0b/Pfizer_logo.svg"

# Session-scoped document store (one instance per process)
doc_store = EnhancedDocumentStore()


# ── Callback helpers ──────────────────────────────────────────────────────────

def _status_bar_text() -> str:
    if doc_store.is_ready:
        s = doc_store.stats
        return (
            f"**Status:** Ready | "
            f"**Documents:** {s.get('documents_found', 0)} | "
            f"**Chunks:** {s.get('total_chunks', 0)}"
        )
    return "**Status:** Ready | **Documents:** 0 | **Chunks:** 0"


def process_pdf_handler(pdf_files):
    """Triggered when the user clicks 'Process Document'."""
    if not pdf_files:
        return (
            "Waiting for PDF upload...", "",
            gr.update(choices=["All"], value="All"),
            _status_bar_text(),
        )

    if isinstance(pdf_files, str):
        pdf_files = [pdf_files]

    success, stats = doc_store.process_pdf(pdf_files)

    if success:
        status_md = (
            f"**Successfully Processed:**\n"
            f"- Files: {stats['filename']}\n"
            f"- Pages: {stats['total_pages']}\n"
            f"- Documents Found: {stats['documents_found']}\n"
            f"- Chunks Created: {stats['total_chunks']}\n"
            f"- Types: {', '.join(stats['document_types'])}\n"
            f"- Time: {stats['processing_time']}"
        )

        structure     = doc_store.get_document_structure()
        grouped: dict = {}
        for d in structure:
            grouped.setdefault(d["source_file"], []).append(d)

        md_lines = []
        for fname, docs in grouped.items():
            md_lines.append(f"\n**📄 {fname}**")
            for d in docs:
                md_lines.append(f"  - **{d['type']}** (Pages {d['pages']}) — {d['chunks']} chunks")

        type_choices = ["All"] + stats["document_types"]
        return (
            status_md,
            "\n".join(md_lines),
            gr.update(choices=type_choices, value="All"),
            _status_bar_text(),
        )

    return (
        f"❌ Error: {stats.get('error', 'Unknown error')}", "",
        gr.update(choices=["All"], value="All"),
        _status_bar_text(),
    )


def chat_handler(
    message: str, history, doc_filter: str, auto_route: bool, num_chunks: int
) -> Generator:
    """Streaming chat handler — yields (history, status_bar) on each token."""
    if history is None:
        history = []

    if not doc_store.is_ready:
        reply = (
            "Please upload and process a pharmaceutical PDF document first. "
            "Use the **Upload Pharmaceutical PDF(s)** panel on the left, "
            "then click **Process Document**."
        )
        history = history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": reply},
        ]
        yield history, _status_bar_text()
        return

    if not message or not message.strip():
        yield history, _status_bar_text()
        return

    filter_type = None if (not doc_filter or doc_filter == "All") else doc_filter

    history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": ""},
    ]
    yield history, _status_bar_text()

    acc               = ""
    final_sources     = []
    final_filter_used = "global"

    for partial in doc_store.query_stream(
        message,
        filter_doc_type=filter_type,
        auto_route=bool(auto_route) and filter_type is None,
        k=int(num_chunks),
        max_new_tokens=MISTRAL_MAX_REPLY,
        temperature=MISTRAL_TEMPERATURE,
        max_ctx_tokens=MISTRAL_N_CTX,
    ):
        acc               = partial["answer_partial"]
        final_sources     = partial.get("sources", [])
        final_filter_used = partial.get("filter_used", "global")
        history[-1]["content"] = acc
        yield history, _status_bar_text()

    # Append source citations once streaming is complete
    if final_sources:
        citations = "\n\n**Sources:**\n" + "".join(
            f"- 📄 `{s['source_file']}` | {s['doc_type']} "
            f"(Pages {s['pages']}) — Relevance: {s['relevance']}\n"
            for s in final_sources
        )
        history[-1]["content"] = acc.strip() + citations + f"\n\n*Filter: {final_filter_used}*"
        yield history, _status_bar_text()

    # Append to chat history file
    with open(CHAT_HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(f"User: {message}\n")
        f.write(f"Assistant: {acc.strip()}\n")
        f.write("-" * 60 + "\n")


def clear_all():
    global doc_store
    doc_store = EnhancedDocumentStore()
    open(CHAT_HISTORY_PATH, "w").close()
    return (
        None,
        "Waiting for PDF upload...",
        "",
        gr.update(choices=["All"], value="All"),
        [],
        "",
        _status_bar_text(),
    )


def export_chat_history():
    if not os.path.exists(CHAT_HISTORY_PATH):
        open(CHAT_HISTORY_PATH, "w").close()
    return gr.update(value=CHAT_HISTORY_PATH, visible=True)


def make_example_handler(question: str):
    def _handler(history, doc_filter, auto_route, num_chunks):
        yield from chat_handler(question, history, doc_filter, auto_route, num_chunks)
    return _handler


# ── Interface builder ─────────────────────────────────────────────────────────

def create_interface() -> gr.Blocks:
    with gr.Blocks(title="Pfizer Pharmaceutical Document RAG Chatbot", theme=gr.themes.Soft()) as demo:

        gr.HTML(f"""
        <div style="display:flex;align-items:center;margin-bottom:8px;">
            <img src="{PFIZER_LOGO}" style="height:56px;margin-right:18px;" alt="Pfizer">
            <div>
                <h2 style="margin:0;font-size:1.5rem;">
                    Pfizer Pharmaceutical Document RAG Chatbot
                </h2>
                <p style="margin:2px 0 0;color:#555;font-size:0.85rem;">
                    Intelligent Multi-Document Analysis · Mistral-7B · BGE Embeddings · LlamaIndex
                </p>
            </div>
        </div>
        <p style="color:#666;font-size:0.8rem;margin:0 0 4px;">
            Upload one or more pharmaceutical PDFs to identify document types,
            build a searchable index, and ask questions in natural language.
        </p>
        """)

        with gr.Row():
            # ── Column 1: Upload ─────────────────────────────────────────────
            with gr.Column(scale=2):
                pdf_input = gr.File(
                    label="Upload Pharmaceutical PDF(s)",
                    file_types=[".pdf"],
                    file_count="multiple",
                    type="filepath",
                )
                with gr.Row():
                    process_btn   = gr.Button("Process Document", variant="primary",   size="lg", scale=2)
                    clear_all_btn = gr.Button("Clear All",         variant="secondary", size="lg", scale=1)

            # ── Column 2: Doc Info + Settings ────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Document Info")
                status_output    = gr.Markdown("Waiting for PDF upload...")
                structure_output = gr.Markdown("")

                gr.Markdown("### Retrieval Settings")
                doc_filter = gr.Dropdown(
                    choices=["All"], value="All",
                    label="Document Type Filter",
                    info="Filter search to a specific pharmaceutical document type",
                )
                auto_route = gr.Checkbox(
                    value=True, label="Auto-Route Queries",
                    info="Automatically detect the most relevant document type",
                )
                num_chunks = gr.Slider(minimum=1, maximum=10, value=4, step=1, label="Chunks to Retrieve")

            # ── Column 3: Chat ───────────────────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### Ask Questions")
                chatbot = gr.Chatbot(height=440, show_label=False, type="messages")

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="e.g., What is the lot number? What sterilization method was used?",
                        show_label=False, scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    clear_chat_btn = gr.Button("Clear Chat",               size="sm", scale=1)
                    sum_btn        = gr.Button("Summarise this document",   size="sm", scale=1)
                    lot_btn        = gr.Button("Find lot numbers",          size="sm", scale=1)

        with gr.Row():
            status_bar = gr.Markdown(_status_bar_text())

        with gr.Row():
            export_btn  = gr.Button("📥 Export Chat History", scale=1)
            export_file = gr.File(label="Download", visible=False, scale=1)

        # ── Wiring ───────────────────────────────────────────────────────────
        proc_outputs = [status_output, structure_output, doc_filter, status_bar]

        process_btn.click(fn=process_pdf_handler, inputs=[pdf_input], outputs=proc_outputs)
        clear_all_btn.click(
            fn=clear_all,
            outputs=[pdf_input, status_output, structure_output, doc_filter, chatbot, msg_input, status_bar],
        )

        chat_inputs  = [msg_input, chatbot, doc_filter, auto_route, num_chunks]
        chat_outputs = [chatbot, status_bar]

        send_btn.click(chat_handler,  inputs=chat_inputs, outputs=chat_outputs).then(lambda: "", outputs=[msg_input])
        msg_input.submit(chat_handler, inputs=chat_inputs, outputs=chat_outputs).then(lambda: "", outputs=[msg_input])
        clear_chat_btn.click(lambda: [], outputs=[chatbot])

        sum_btn.click(
            make_example_handler("Can you provide a summary of the main points in this document?"),
            inputs=[chatbot, doc_filter, auto_route, num_chunks], outputs=chat_outputs,
        )
        lot_btn.click(
            make_example_handler("What lot numbers or batch numbers are mentioned in these documents?"),
            inputs=[chatbot, doc_filter, auto_route, num_chunks], outputs=chat_outputs,
        )
        export_btn.click(fn=export_chat_history, outputs=[export_file])

    return demo
