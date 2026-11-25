"""Streamlit entry point:  streamlit run app/main.py"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="Semantica Knowledge Graph Lab", page_icon="🕸️", layout="wide")

PAGES = Path(__file__).parent / "pages"
navigation = st.navigation(
    [
        st.Page(PAGES / "overview.py", title="Overview", icon="📊", default=True),
        st.Page(PAGES / "documents.py", title="Documents", icon="📄"),
        st.Page(PAGES / "graph_explorer.py", title="Knowledge Graph", icon="🕸️"),
        st.Page(PAGES / "query_playground.py", title="Query Playground", icon="🔎"),
        st.Page(PAGES / "provenance.py", title="Provenance & Conflicts", icon="🧾"),
        st.Page(PAGES / "evaluation.py", title="Evaluation", icon="⚖️"),
    ]
)

with st.sidebar:
    st.caption("Semantica 0.7.0 · Neo4j · FAISS · SQLite")
    if st.button("Reload resources", help="Use after changing data from a script."):
        st.cache_resource.clear()
        st.rerun()

navigation.run()
