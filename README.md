# Semantica Knowledge Graph Lab

A small local application for finding out what
[Semantica](https://github.com/semantica-agi/semantica) actually does when it turns
documents into a connected, traceable knowledge graph. It is an experiment, not a
chatbot: upload documents, watch them become entities, facts, conflicts and
provenance, then compare vector search with graph-based retrieval on the same
questions.

Built against **Semantica 0.7.0**, Neo4j 5.26, FAISS, SQLite and Streamlit. Runs
offline after the first model download; no paid API is needed.

## What to read

| Document | Content |
|---|---|
| [docs/semantica_integration.md](docs/semantica_integration.md) | what is Semantica and what is ours, plus 18 surprises found along the way |
| [docs/evaluation_results.md](docs/evaluation_results.md) | measured comparison and what it does and does not show |
| [docs/architecture.md](docs/architecture.md) | diagram, processing flow, design decisions |
| [docs/database_design.md](docs/database_design.md) | what each store owns |

## Headline results (sample corpus)

* All four retrieval modes answered **13 of 15** questions correctly, failing on
  different ones. GraphRAG was **not** better overall.
* At a similar evidence size, plain vector search covered **more** of the expected
  answers than graph retrieval. The corpus is tiny (11 chunks), which favours it.
* Graph retrieval was the only mode that handled the "who led this team in 2024"
  question without also returning the later lead, and it adds paths and conflicts.
* The offline pattern extraction scores 0.98 recall on the sample corpus but only
  **0.27** on reworded sentences. Semantica's unmodified `pattern` method found no
  domain facts at all.

## Requirements

* Windows 10/11 (commands below are PowerShell; the code is not Windows-specific)
* Python 3.12 recommended (3.11 or 3.13 should work; **3.14 is not supported** by Semantica)
* Docker Desktop, for Neo4j
* About 2.5 GB of disk space (Python environment, Neo4j image, embedding model) and
  internet access for the first install

## Setup

Run these from the project folder.

```powershell
py -3.12 -m venv .venv
```

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

For an exact copy of the tested dependency tree use `requirements.lock.txt` instead.

Create your settings file, then set `NEO4J_PASSWORD` in it (at least 8 characters).
This command leaves an existing `.env` alone.

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

```powershell
docker compose up -d --wait neo4j
```

## Run

Load the sample corpus and record the sample decision (the first run downloads the
embedding model, about 70 MB):

```powershell
.\.venv\Scripts\python.exe scripts\load_sample_data.py
```

Start the application, then open <http://localhost:8501>:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app\main.py
```

The Neo4j browser is at <http://localhost:7474> (user `neo4j`, your password). Try
`MATCH (n:Entity)-[r]->(m) RETURN n, r, m`.

Other commands:

```powershell
.\.venv\Scripts\python.exe scripts\run_evaluation.py
```

```powershell
.\.venv\Scripts\python.exe scripts\load_sample_data.py --reset
```

```powershell
docker compose down
```

Run scripts while the app is idle, then press **Reload resources** in the sidebar.

## Tests

Unit tests need no database and no API key:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Integration tests (Neo4j and the UI) use a separate, disposable container, because
they delete everything in the database they run against:

```powershell
docker compose --profile test up -d --wait neo4j-test
```

```powershell
.\.venv\Scripts\python.exe -m pytest -m integration
```

To check that everything survives a restart, compare two fingerprints of the stored
state (the two outputs should be identical):

```powershell
.\.venv\Scripts\python.exe scripts\snapshot.py
```

```powershell
docker compose down; docker compose up -d --wait neo4j
```

```powershell
.\.venv\Scripts\python.exe scripts\snapshot.py
```

## The pages

| Page | What you can do |
|---|---|
| Overview | totals, types, recent runs; load the sample data |
| Documents | upload (PDF, DOCX, TXT, HTML), inspect extraction and discards, reprocess, delete |
| Knowledge Graph | explore, search, click nodes, read provenance, shortest paths, OWL / SHACL, reasoning, point-in-time |
| Query Playground | vector, graph, hybrid and native retrieval, alone or side by side |
| Provenance & Conflicts | facts and lineage, contradictions, merges and duplicate candidates, hash-chain check, decision trace |
| Evaluation | run and inspect the comparison |

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` | `bolt://localhost:7687` / `neo4j` / – | graph database |
| `NEO4J_TEST_URI` | `bolt://localhost:7688` | database for integration tests |
| `GROQ_API_KEY` | empty | enables LLM extraction and answers |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | check that Groq still serves it |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | any FastEmbed model; delete `data\vector_index` after changing it |
| `DATA_DIR` | `./data` | runtime state |
| `MAX_UPLOAD_MB` | `10` | upload limit (also set in `.streamlit/config.toml`) |
| `LOG_LEVEL` | `INFO` | application logging |

### Optional: Groq

Put `GROQ_API_KEY` in `.env` and restart. The Documents page then offers LLM
extraction and the Query Playground can write an answer, shown apart from the
evidence. Both send document text to Groq.
`scripts\run_evaluation.py --generate` scores the generated answers.

> The Groq paths are covered only by tests with a stand-in client. They have **not**
> been run against the real service.

## Limitations

**Extraction**
* The default method is a rule-based pattern pack written for this domain. It does not
  resolve pronouns, passive or nominal phrasings it has no rule for, lists of
  subjects, or people mentioned away from a recognised verb.
* It produces false positives, for example the heading "Project Register" becomes a
  project.
* Confidence values are fixed constants, not probabilities.
* Scanned PDFs are rejected (no OCR). Tables are not extracted.

**Resolution and conflicts**
* Merging relies on a small alias table and a first-name rule. Candidates for review
  are listed, but there is no button to accept or reject them.
* Conflicts are detected only for `launch_year`, `primary_database` and a team's lead.
* Validity dates are read only from `from … until …` / `since …` with ISO dates.

**Retrieval**
* Graph and hybrid ranking is a heuristic written here. It misses a four-hop question
  in the evaluation.
* "Insufficient evidence" thresholds were tuned on the same small question set.
* Retrieval returns evidence; without an LLM no answer is composed.

**Operations**
* Single user, no authentication; both servers listen on `127.0.0.1` only.
* Writes across Neo4j, FAISS, provenance and SQLite are not one transaction.
  Reprocess a document if a run was interrupted.
* The whole graph is re-derived on each ingestion and loaded for querying, which will
  not scale far.
* Two files with the same name but different content are both kept.
* Semantica's precedent search does not work on Neo4j in this release.
* Decisions survive re-processing, but lose their link to any entity that disappears
  (for example after deleting its documents or running `--reset`).

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Cannot connect to Neo4j" | `docker compose up -d --wait neo4j` |
| Authentication fails after editing the password | Neo4j keeps the first password in its volume. `docker compose down -v` (deletes the graph), then start and reload |
| Port 7474, 7687 or 8501 in use | stop the other service or change the mapping |
| "Embedding model did not load" | the first run needs internet access |
| "openai / litellm library not installed" messages | harmless |
