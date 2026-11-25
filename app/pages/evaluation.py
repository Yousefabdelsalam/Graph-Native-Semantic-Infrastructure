"""Page 6: run and inspect the retrieval and extraction comparison."""

import json

import streamlit as st

from app.components.common import get_lab, require_graph, require_vectors, table
from backend.evaluation.runner import corpus_mismatch, render_results, run_evaluation
from backend.retrieval.engine import MODES

lab = get_lab()
st.title("Evaluation")
require_graph(lab)
require_vectors(lab)

st.markdown(
    "Every mode answers the same manually labelled questions. Metrics are deterministic: they "
    "check whether the expected items and sources appear in the retrieved evidence. They do not "
    "show that a reader or model would pick the right answer out of that evidence."
)
if warning := corpus_mismatch(lab):
    st.warning(warning)

generate = st.checkbox(
    "Also generate and score LLM answers",
    disabled=not lab.settings.llm_available,
    help="Needs GROQ_API_KEY.",
)
if st.button("Run evaluation", type="primary", disabled=not lab.registry.list_documents()):
    with st.spinner("Running all questions in all modes…"):
        run_evaluation(lab, generate=generate)

path = lab.settings.exports_dir / "evaluation.json"
if not path.exists():
    st.info("No evaluation has been run yet.")
    st.stop()
report = json.loads(path.read_text(encoding="utf-8"))
st.caption(f"Last run {report['generated_at']} · Semantica {report['semantica_version']}")
if report["warning"]:
    st.warning(report["warning"])
retrieval = report["retrieval"]

st.subheader("Overall")
table(
    {
        "Mode": MODES[mode],
        "Correct": f"{s['correct']}/{s['questions']}",
        "Coverage": s["coverage"],
        "Source recall": s["source_recall"],
        "Source precision": s["source_precision"],
        "Traceability": s["traceability"],
        "Latency ms": s["latency_ms"],
        "Evidence chars": s["evidence_chars"],
        "LLM answer correct": s["answer_correct"],
    }
    for mode, s in retrieval["by_mode"].items()
)

left, right = st.columns(2)
with left:
    st.markdown("**Mean retrieval latency (ms)**")
    st.bar_chart(
        {"mode": list(MODES), "ms": [retrieval["by_mode"][m]["latency_ms"] for m in MODES]},
        x="mode",
        y="ms",
    )
with right:
    st.markdown("**Correct answers by category**")
    table(
        {
            "Category": category,
            **{m: f"{s[m]['correct']}/{s[m]['questions']}" for m in MODES},
        }
        for category, s in retrieval["by_category"].items()
    )

st.subheader("Evidence budget sensitivity")
st.caption("Coverage at different sizes. Compare rows with a similar number of evidence characters.")
table(retrieval["sensitivity"])

st.subheader("Per question")
questions = {q["id"]: q for q in report["questions"]}
chosen = st.selectbox(
    "Question", list(questions), format_func=lambda i: f"{i} · {questions[i]['question']}"
)
question = questions[chosen]
st.write(
    "Expected:",
    "; ".join(" / ".join(group) for group in question["expected"]) or "insufficient evidence",
)
if question.get("notes"):
    st.caption(question["notes"])
table(
    {
        "Mode": MODES[r["mode"]],
        "Correct": r["correct"],
        "Coverage": r["coverage"],
        "Declared sufficient": r["sufficient"],
        "Wrongly included": ", ".join(r["excluded_leaked"]),
        "Source recall": r["source_recall"],
        "Latency ms": r["latency_ms"],
        "Answer": r["answer"] or "",
    }
    for r in retrieval["rows"]
    if r["question_id"] == chosen
)

st.subheader("Extraction quality")
for title, key in (("Sample corpus", "extraction"), ("Paraphrase probe", "paraphrases")):
    st.markdown(f"**{title}**")
    table(
        {
            "Method": method,
            "Predicted": s["predicted"],
            "Gold": s["gold"],
            "Correct": s["correct"],
            "Precision": s["precision"],
            "Recall": s["recall"],
            "F1": s["f1"],
        }
        for method, s in report[key].items()
    )
st.caption(
    "The pattern pack was written with the sample corpus in view, so corpus scores are optimistic. "
    "The paraphrase probe shows how far it generalises."
)

st.download_button(
    "Download results (Markdown)", render_results(report), file_name="evaluation_results.md"
)
