# Phase 2 plan: advanced extraction and GraphRAG evaluation

Status: **paused**. On 2026-11-26 the model provider was changed from Groq to OpenAI;
the Groq model, price and rate-limit details below are historical.
When written, no application code had been changed.
Inspected on 2026-11-25: codebase, tests, evaluation report, installed Semantica 0.7.0.
Baseline: 90 unit tests pass; no Groq key is configured.

Figures are marked **measured** (observed here) or **estimate**.

## 1. What the inspection found

### Defects in what was delivered in Phase 1

| # | Finding | Evidence | Fix |
|---|---|---|---|
| D1 | The configured model `llama-3.3-70b-versatile` was shut down on 2026-08-16 for free and developer accounts. It is also the default in `.env.example`, `config.py` and your `.env`. | Groq [deprecations page](https://console.groq.com/docs/deprecations) | default to `openai/gpt-oss-120b` |
| D2 | With `llm` extraction, if every API call fails, ingestion **reports success with 0 entities and 0 facts**. Reprocessing would then remove that document's existing facts from the graph and mark their provenance invalid. | measured offline: all calls forced to raise; the run returned normally. The graph removal follows from `lab.py` | fail the run, keep previous data |
| D3 | A failing call is retried inside Semantica so that 7 sentences produced 42 attempts (6 per sentence). Under rate limiting this multiplies load. | measured offline | bounded retries with back-off |
| D4 | The README and evaluation report say graph retrieval missed a **four**-hop question. Question q11 needs **three** hops. | read from the sample facts | correct both documents |

### Limits that shape the design

| Area | Finding | Kind |
|---|---|---|
| LLM extraction | Runs per sentence (2 calls each). A 450-character chunk needs prompts of about 1,600 characters per call. | measured |
| LLM output | Entities come back with offsets `0, 0`; locations must be recovered by the app. | measured, canned response |
| Token usage | Semantica's Groq provider discards `response.usage`. `provider_registry` allows registering a metered provider. | source inspection |
| Model settings | The provider forwards no reasoning-effort setting, which gpt-oss models are likely to need. | source inspection; effect unverified |
| Traversal | On a synthetic graph (800 entities, 2,400 edges) a 3-hop expansion reaches 388 to 782 entities, 4 hops 760 to 790. "Expand then rank" stops selecting anything. | measured |
| Neo4j | Traversal under 0.2 s; first sync 13.9 s; unchanged re-sync 0.3 s. | measured |
| Native retriever | 1.5 s per query at 300 graph elements, 4.9 s at 1,000. About 15 s at benchmark size. | measured / estimate |
| Resolution | 630 synthetic entities yield 21,417 "review" pairs (synthetic names exaggerate this). | measured |
| Ingestion | Every document triggers a full graph rebuild. | source inspection |
| Machine | 15.7 GB RAM, about 2 GB free: run one extra Neo4j at a time. | measured |

### Weaknesses of the Phase 1 evaluation

* Coverage is string matching; chunks and facts are not compared on equal terms.
* Evidence is limited by item count, not size.
* Refusal thresholds were tuned on the test questions.
* The graph planner gives a distance bonus only for hops 0 and 1, consistent with its
  miss on the three-hop question; temporal handling reads only "in &lt;year&gt;".
* Conflicts are checked for three properties only.

## 2. Design

### 2.1 Isolation (architectural change)

A `bench` workspace: its own data folder and its own Neo4j container (`neo4j-bench`,
port 7689, persistent volume). Community Neo4j has one database, so this is the way to
keep 100+ documents out of the Phase 1 graph. The default workspace is untouched.

### 2.2 Dataset (`data/benchmarks/v2`)

* Generated from a seeded **world model**: about 8 organisations, 150 people, 30 teams,
  40 products, 60 projects, 40 technologies, with dated changes.
* About 120 documents (TXT, PDF, DOCX, HTML), targeting 600 to 800 chunks.
* Injected, labelled contradictions, name variants, a few homonyms, distractors, and
  chains that only exist across documents.
* Several phrasing families per fact type, one held out for test documents.
* Gold data by construction: entity mentions with spans, facts with document, span and
  validity, alias groups, contradictions.
* About 160 questions, 20 per category: direct, 2-hop, 3-hop, 4-hop, temporal,
  conflict, provenance, unanswerable (including near misses). Split 40 development /
  120 test. Each question lists answers, forbidden answers and the supporting
  statements.
* **"Manually verified" needs a person.** The generator can check consistency; it
  cannot verify manually. A review sheet will be produced; reports state how many items
  a person confirmed and call the rest "derived from the generator".

Original files in `data/sample_documents` stay byte-identical, guarded by a checksum
test. New entity labels for that corpus go in a new file.

**Bias control.** I write both the generator and the system. So: the regex pack is
frozen (hash recorded) before the generator exists; names include forms the pack does
not expect; all tuning uses the development split; the test split is run after
freezing; a test rejects benchmark names appearing in retrieval or extraction code.

### 2.3 Extraction comparison

| Method | What it is |
|---|---|
| `regex` | the Phase 1 pattern pack, frozen |
| `pattern` | Semantica's unmodified method (expected to find few or no domain facts) |
| `llm` | Semantica `extract_entities_llm` / `extract_relations_llm` per chunk, temporal bounds on, through a metered, cached, rate-limited provider |

* Errors propagate; no silent fallback.
* Location: the sentence containing both endpoints, otherwise the span covering them,
  recorded with its precision. Entities or relations absent from the text are rejected.
* Scored separately: **entities** (document level and span level, per type),
  **relations** (per type, micro and macro), plus location accuracy, validity-date
  accuracy, alias-grouping quality, contradiction detection, tokens and time.
* Same documents and labels for all three; the original corpus and the paraphrase probe
  are run too.
* Duplicates and conflicts: blocking and a cap on review pairs; homonym guard;
  single-valued rules extended (lead, employer, owner, primary database, launch year)
  with supersession over time.

### 2.4 Graph retrieval v2

Phase 1 planners stay available as `graph_v1` / `hybrid_v1`, so improvement is measured.

| Topic | Proposal |
|---|---|
| Seeds | name and alias linking plus search over an entity index, with type hints |
| Traversal | relation-aware beam search over paths, depth chosen up to 4 |
| Ranking | path and fact similarity (precomputed embeddings), hub penalty |
| Temporal | as-of, before / after, "current", superseded facts |
| Filtering | de-duplication; conflicting claims travel together |
| Diversity | per-document cap with redundancy penalty |
| Budget | pack whole paths into a token budget; stop early when scores drop |
| Refusal | calibrated on the development split only |

Each component gets an on/off ablation. Retrieval stays model-free; LLM question
decomposition would be a separate, labelled variant if wanted.

### 2.5 Answer generation

* Same evidence envelope for every mode: `[E1] (file, location) text`.
* Strict JSON schema: `answerable`, refusal reason, answer, claims marked **stated** or
  **inferred** with citations (inferred claims list their premises), conflicts with the
  sources on each side.
* Called through the Groq SDK, because Semantica's wrapper exposes neither schemas nor
  usage.
* A deterministic verifier checks that citations exist and support the claim's entities
  and relation; unsupported claims are flagged, and an answer with none left is a
  refusal.

### 2.6 Evaluation

* Modes: vector, graph v2, hybrid v2, native `ContextRetriever`; v1 planners as
  reference for retrieval metrics only.
* Two graph conditions: **extracted** graph (end to end) and **gold** graph (isolates
  retrieval from extraction errors).
* Retrieval at 500 / 1,000 / 2,000 token budgets; generation at 1,000. Native results
  are truncated in their own order.
* Generation: 3 runs, temperature 0; mean, range and paired bootstrap intervals.

| Metric | Definition | Label in reports |
|---|---|---|
| Retrieval recall / precision | gold supporting statements covered / evidence overlapping them | measured |
| Answer correctness | structured answer against gold answers | measured |
| Faithfulness | verifier support rate; second-model judgement | measured; **model judgement** |
| Citation accuracy | citations supporting their claim; agreement with gold evidence | measured |
| Multi-hop accuracy | correctness by hop count | measured |
| Refusal accuracy | precision and recall of refusals | measured |
| Latency | retrieval and generation, median and p95, rate-limit waits excluded | measured |
| Tokens | API-reported usage; pre-call budgeting uses a local tokenizer | measured; budget is **estimate** |
| Extraction P / R / F1 | entities and relations | measured |

**Decided in advance:** v2 is called better than v1 or vector search only if the paired
interval on the test split excludes zero. Otherwise the report says it is not.

### 2.7 Dashboard and reproducibility

One command-line entry point (`generate`, `extract`, `build`, `retrieve`, `answer`,
`judge`, `report`), resumable by run id, with a response cache and a manifest (dataset
hash, models, settings, package versions, code revision). The Evaluation page browses
runs: extraction, retrieval curves, answers with citations and verifier flags, cost,
each value labelled.

## 3. Files

**Modified**

| File | Change |
|---|---|
| `backend/config.py`, `.env.example`, `.env` (model line only) | model default, workspaces, generation settings |
| `backend/extraction/extractor.py` | fail loudly; chunk-level LLM path; location precision |
| `backend/schemas.py` | answer, usage and location fields |
| `backend/graph/resolution.py` | blocking, review cap, homonym guard |
| `backend/graph/conflicts.py`, `builder.py`, `backend/ontology/schema.py` | wider single-valued rules, supersession |
| `backend/services/lab.py` | batch ingestion with one rebuild |
| `backend/retrieval/engine.py`, `vector.py`, `answer.py` | register new modes, fact and entity indexes, call new generator |
| `backend/evaluation/runner.py`, `metrics.py` | reuse only; Phase 1 behaviour unchanged |
| `app/pages/evaluation.py`, `query_playground.py`, `documents.py` | runs, cited answers, method choice |
| `docker-compose.yml` | `neo4j-bench` profile |
| `README.md`, `docs/architecture.md`, `database_design.md`, `semantica_integration.md` | documentation |

**New**

| Path | Content |
|---|---|
| `backend/llm/` | metered provider, cache, rate limiter |
| `backend/generation/` | schema, prompt, generator, verifier |
| `backend/retrieval/v2/` | seeds, paths, ranking, temporal, budget |
| `backend/evaluation/benchmark/` | extraction, retrieval, answer, judge, report |
| `benchmarks/v2/` | world model, renderer, questions, validator |
| `data/benchmarks/v2/` | documents, gold files, questions, review sheet |
| `data/sample_documents/gold_entities.json` | new entity labels |
| `scripts/benchmark.py` | command line |
| `docs/extraction_comparison.md`, `docs/benchmark_results.md` | reports |
| `tests/…` | regression snapshot and checksums, LLM failure handling, generator, v2 retrieval, verifier, metrics, bench integration |

## 4. Milestones and checkpoints

| | Work | Needs API | Stop for you |
|---|---|---|---|
| M0 | git baseline, regression tests, fix D1 to D4 | no | |
| M1 | dataset, gold, questions, review sheet | no | review sample |
| M2 | bench workspace, extraction harness, `regex` and `pattern` runs, 10-document LLM pilot | pilot | confirm token budget from measured pilot |
| M3 | retrieval v2 on development split, retrieval-only evaluation | no | |
| M4 | generation, verifier, judge; development pilot; frozen test run | yes | |
| M5 | dashboard, documentation, full test run, final reports | no | |

## 5. Cost and rate limits (estimates)

| Stage | Tokens |
|---|---|
| LLM extraction, whole benchmark | 1 to 2 million |
| Generation, 120 questions × 4 modes × 3 runs | about 3.5 million |
| Judging one run | about 1.2 million |
| Pilots and development | 0.5 to 1.5 million |
| **Total** | **about 6 to 8 million** |

At the [listed prices](https://console.groq.com/docs/models) (`gpt-oss-120b` $0.15 in /
$0.60 out per million; `qwen3.8-27b` as judge $0.80 / $4.00) this is roughly
**US$3 to 6**.

**Rate limits decide feasibility.** The [rate-limit
page](https://console.groq.com/docs/rate-limits) shows 8,000 tokens per minute and
200,000 per day for `gpt-oss-120b`, labelled developer base limits, while the models
page shows 250,000 per minute. The pages disagree; your account's Limits page is
authoritative. At 200,000 a day the full scope takes over a month.

| Option | Scope | Duration |
|---|---|---|
| A. Full (needs high limits) | all of the above | set by the account limits |
| B. Reduced (200k/day) | LLM extraction on about 25 documents; 40 test questions × 4 modes × 1 run; sampled judging; `regex`, `pattern` and all retrieval metrics still on the full set | about 4 to 5 days |

Option B gives wider uncertainty, no repeat-run variance, and an LLM graph covering
only the subset.

## 6. Risks

* Synthetic data and one author: results describe this benchmark, not real corpora.
* A model judge can be wrong in correlated ways; it is never the only measure.
* LLM extraction may not reach good recall, which would cap every graph mode.
* Groq can retire models again; the manifest records what was used.
* Preview model as judge may change; the fallback is `gpt-oss-20b`, less independent.

## 7. Decisions needed

1. **Approve the plan** and choose option A or B.
2. Will you review a sample of about 40 questions? (recommended)
3. May I run `git init` and commit the current state as the baseline? (recommended)
4. Models: `openai/gpt-oss-120b` for extraction and answers, `qwen/qwen3.8-27b` as
   judge, unless you prefer otherwise.
