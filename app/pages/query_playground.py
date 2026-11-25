"""Page 4: ask a question in one retrieval mode, or compare all of them."""

from datetime import date

import streamlit as st

from app.components.common import get_lab, require_graph, require_vectors, table
from backend.evaluation.runner import load_questions
from backend.retrieval.answer import AnswerUnavailableError
from backend.retrieval.engine import MODES
from backend.schemas import RetrievalResult

ICONS = {"chunk": "📄", "fact": "🔗", "conflict": "⚠️", "context": "🧩"}


def show(result: RetrievalResult, answer: str | None, compact: bool = False) -> None:
    st.metric("Retrieval time", f"{result.latency_ms:.0f} ms")
    if not result.sufficient:
        st.warning(f"Insufficient evidence. {result.note}")
    elif result.note:
        st.caption(result.note)

    if answer is not None:
        with st.container(border=True):
            st.markdown("#### Generated conclusion (LLM)")
            st.caption("Written by a language model from the evidence below. Verify it.")
            # Plain text: model output can be steered by document content, so it
            # must not render links or remote images.
            st.text(answer)

    st.markdown(f"#### Retrieved evidence ({len(result.evidence)})")
    st.caption("Text taken directly from the indexed documents and graph. Not model-generated.")
    if not result.evidence:
        st.caption("Nothing was retrieved.")
    for number, item in enumerate(result.evidence, start=1):
        with st.container(border=True):
            st.markdown(f"{ICONS.get(item.kind, '•')} **[{number}] {item.kind}**")
            st.text(item.text)
            score = "" if item.score is None else f" · score {item.score:.3f}"
            st.caption(
                f"Source: {', '.join(item.filenames) or 'not identified'} · "
                f"{item.location or 'no location'}{score}"
            )
    if compact:
        return
    st.markdown("#### Source documents")
    st.write(", ".join(result.source_files) or "None")
    if result.paths:
        st.markdown("#### Graph paths between named entities")
        for path in result.paths:
            st.code(path, language="text")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Retrieved entities")
        table(
            ({"Entity": e["name"], "Type": e["type"]} for e in result.entities),
            empty="This mode returns no entities.",
        )
    with right:
        st.markdown("#### Relevant relationships")
        names = {e["id"]: e["name"] for e in st.session_state.get("graph_entities", [])}
        table(
            (
                {
                    "Source": names.get(r["source"], r["source"]),
                    "Relationship": r["type"],
                    "Target": names.get(r["target"], r["target"]),
                }
                for r in result.relationships
            ),
            empty="This mode returns no relationships.",
        )


lab = get_lab()
st.title("Query Playground")
graph = require_graph(lab)
chunks = require_vectors(lab)
if not chunks:
    st.info("Nothing is indexed yet. Load or upload documents first.")
    st.stop()
st.session_state["graph_entities"] = graph["entities"]

samples = [q["question"] for q in load_questions(lab.settings)]


def use_sample() -> None:
    if st.session_state.sample:
        st.session_state.question = st.session_state.sample


st.selectbox(
    "Sample questions", ["", *samples], key="sample", on_change=use_sample,
    format_func=lambda q: q or "(write your own below)",
)
question = st.text_input("Question", key="question", max_chars=500)

settings_row = st.columns([2, 1, 1, 1])
mode = settings_row[0].selectbox("Retrieval mode", list(MODES), index=2, format_func=MODES.get)
top_k = settings_row[1].slider("Chunks (top k)", 1, 8, 4)
max_facts = settings_row[2].slider("Max graph facts", 4, 30, 12)
max_hops = settings_row[3].slider("Max hops", 1, 4, 3)

options_row = st.columns([1, 1, 2])
use_date = options_row[0].checkbox("Point in time")
as_of = options_row[1].date_input("Valid on", date.today(), disabled=not use_date)
generate = options_row[2].checkbox(
    "Generate an answer with Groq",
    disabled=not lab.settings.llm_available,
    help="Needs GROQ_API_KEY. Sends the question and evidence to Groq.",
)

run, compare = st.columns(2)
single = run.button("Run query", type="primary", disabled=not question.strip())
side_by_side = compare.button("Compare all modes", disabled=not question.strip())

if single or side_by_side:
    engine = lab.engine()
    options = {
        "top_k": top_k,
        "max_facts": max_facts,
        "max_hops": max_hops,
        "as_of": as_of.isoformat() if use_date else None,
    }
    modes = list(MODES) if side_by_side else [mode]
    columns = st.columns(len(modes)) if side_by_side else [st.container()]
    for column, selected in zip(columns, modes):
        with column:
            if side_by_side:
                st.subheader(MODES[selected])
            with st.spinner("Retrieving…"):
                try:
                    result, answer = lab.query(
                        question, selected, generate=generate, engine=engine, **options
                    )
                except AnswerUnavailableError as exc:
                    st.error(str(exc))
                    result, answer = lab.query(question, selected, engine=engine, **options)
            show(result, answer, compact=side_by_side)
