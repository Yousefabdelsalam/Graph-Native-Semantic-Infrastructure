# Semantica integration

How this lab uses Semantica **0.7.0**, where the line between library and application
code runs, and what did not behave as expected. Every statement under "Findings" was
observed on the installed package; the ones marked 🧪 are locked in by a test, so an
upgrade that changes them will show up in the test run.

## How the APIs were verified

* PyPI's newest release was 0.7.0. The repository's `main` branch was 283 commits
  ahead (commit `b3efeb5f`, 2026-11-25), so signatures were read from the `v0.7.0`
  tag and then exercised against the installed package.
* Python support is 3.10 – 3.13 (`requires-python = ">=3.10,<3.14"`). This project
  was built and tested on 3.12.
* The HTML-collapse, FAISS-scoring and `ContextRetriever`-on-Neo4j code was also
  checked on `main`; all three were still present at that commit.

## Who does what

| Stage | Semantica (native) | Application (this repository) |
|---|---|---|
| Intake | `ingest.FileIngestor` | upload validation, content hashing, SQLite registry |
| Parsing | `parse.PDFParser`, `DOCXParser`, `HTMLParser`, `DocumentParser` | page / section character offsets; keeping HTML line breaks |
| Normalisation | `normalize.TextNormalizer` | applying it line by line so structure survives |
| Chunking | `split.TextSplitter(method="recursive")` | per-page splitting, stable chunk ids |
| Entities | `semantic_extract.NERExtractor` (`regex`, `pattern`, `llm`) | the domain pattern pack, overlap resolution, rejecting fallback output |
| Relations | `semantic_extract.RelationExtractor` | per-sentence patterns, ontology domain/range check, validity dates |
| Triplets, events | `TripletExtractor`, `EventDetector` | display only |
| Name handling | `normalize.EntityNormalizer` + alias map | honorific / legal-suffix stripping, unique-first-name rule |
| Deduplication | `deduplication.DuplicateDetector`, `EntityMerger` | thresholds, merge log, review list |
| Graph construction | `kg.GraphBuilder` | one edge per fact with all supporting evidence |
| Conflicts | `conflicts.ConflictDetector`, `SourceTracker`, `ConflictResolver` | claim records, temporal-overlap filter, tie detection, never auto-applying |
| Graph storage | `graph_store.GraphStore` (Neo4j) | upsert / delete diffing, safe reads |
| Traversal | `GraphStore.get_neighbors`, `shortest_path` | hop distances, path formatting |
| Provenance | `provenance.ProvenanceManager` (SQLite, PROV-O) | fact ids, document → fact chain, invalidation on change |
| Embeddings, vectors | `vector_store.VectorStore` + FAISS, FastEmbed | guard against fallback embeddings, cosine conversion |
| Native GraphRAG | `context.ContextRetriever` | supplying the graph as a dictionary |
| Graph / hybrid modes | (uses the modules above) | **the query planner and ranking are application code** |
| Answer generation | `llms.Groq` | grounded prompt, refusal when evidence is insufficient |
| Ontology | `ontology.OntologyEngine` (OWL, SHACL, validation, inference), `export.RDFExporter` | vocabulary definition, union-range correction |
| Reasoning, time | `reasoning.Reasoner`, `kg.TemporalGraphQuery` | rules, date detection in questions |
| Decisions | `context.AgentContext` | trace from decision to facts and sources |
| Analytics | `kg.CentralityCalculator`, `ConnectivityAnalyzer` | display (PageRank returns tuples where the others return dicts) |
| Visualisation | `visualization.KGVisualizer` | focus / filter views, Streamlit embedding |

Nothing here is a custom graph engine or a custom extractor. The two largest pieces of
application logic are **entity-resolution policy** and **graph retrieval planning**;
read the evaluation with that in mind, because "graph" and "hybrid" results measure
this planner as much as they measure Semantica.

## Module notes

### Ingestion and parsing
`FileIngestor.ingest_file` returns a `FileObject` with type and size. `PDFParser.parse`
returns `{"pages": [...], "full_text", "metadata"}`; `DOCXParser.parse` returns
`paragraphs` and `sections`; `HTMLParser.parse` returns an `HTMLData` object, not a
dictionary like the others.

### Extraction
Configuration used:

```python
NERExtractor(method="regex").extract_entities(sentence, patterns=ENTITY_PATTERNS)
RelationExtractor(method="regex").extract_relations(sentence, entities, patterns=...)
```

The built-in `pattern` method knows `PERSON`, `ORG` (needs "Inc"/"Corp"/…), `GPE` and
`DATE`, plus the relations `founded_by`, `located_in`, `works_for`, `born_in`,
`acquired_by`. On this corpus it produces **zero** facts in the domain vocabulary and
labels teams and projects as people. The `regex` method is the supported extension
point; the patterns live in [patterns.py](../backend/extraction/patterns.py).

Confidence values from `pattern`/`regex` are constants (0.7 / 0.75). They are stored
because the library reports them, but they carry no information.

LLM extraction (`method="llm"`, `provider="groq"`) is wired in and selectable when a
key is present. **It has not been run against the live service in this project.**

### Graph construction
`GraphBuilder(merge_entities=False, resolve_conflicts=False)`. Both are handled
explicitly beforehand so each merge and conflict can be explained.

### Neo4j storage model (Semantica's, not ours)
`create_node(labels, properties)` → `CREATE (n:Label …)`; relationships are typed
Neo4j relationships; lookups use Neo4j's internal `id()`. The app adds two things on
top of that model: a stable `id` property per node and relationship (needed to update
instead of duplicate) and a `managed_by` property so decision nodes written by
`AgentContext` are never touched by a re-sync. See [database_design.md](database_design.md).

### Provenance
```python
manager.track_entity(fact_id, source=filename, entity_type="relation",
    parent_entity_id=f"document:{id}", source_location=..., source_quote=...,
    confidence=..., activity_id="extraction:regex")
```
Tracking the same id again archives the previous version and links to it, entries are
hash-chained (`verify_chain()`), and `invalidate()` tombstones instead of deleting.
This module worked as documented and is one of the library's strongest parts.

### Vector store
`VectorStore(backend="faiss", dimension=384)`, `store_vectors(..., ids=...)`,
`search_vectors`, `delete_vectors`, `save` / `load` (writes `index.bin`,
`index.bin.meta.json`, `store_data.json`).

### Retrieval
`ContextRetriever(knowledge_graph={...}, vector_store=...).retrieve(question)` returns
merged vector and graph contexts. Graph contexts are generated prose per entity,
carrying one hop of relationships and **no source location**.

### Reasoning and temporal
`Reasoner` takes rules as `IF A(?x, ?y) AND B(?y, ?z) THEN C(?x, ?z)` over string
facts and handles multi-word names. `TemporalGraphQuery.query_at_time` filters
relationships on `valid_from` / `valid_until`; relationships without bounds are always
included.

### Decisions
`AgentContext(vector_store, knowledge_graph=graph_store, decision_tracking=True)`
writes a `Decision` node and `ABOUT` links. It matches entities on their `id` property.

## Findings: surprises and limitations

| # | Behaviour in 0.7.0 | Effect | What the lab does | |
|---|---|---|---|---|
| 1 | `GraphStore.add_nodes` / `add_edges` always `CREATE` | re-ingestion duplicates the graph | diffs by `id`, then `create_node` / `update_node` / `delete_*` | 🧪 |
| 2 | `Neo4jStore.execute_query` turns a returned node into a list of its property **names** | silent data loss on `RETURN n` | reads through `get_nodes`, or returns `properties(n)` | 🧪 |
| 3 | `ContextRetriever` given a `GraphStore` passes the question to it as Cypher; the syntax error is logged and swallowed | graph half of GraphRAG silently returns nothing | passes the graph as a dictionary loaded from Neo4j | 🧪 |
| 4 | `find_precedents` / `query_decisions` raise `KeyError: 'decision_id'` on Neo4j (follows from #2) | no precedent search on the persistent backend | reported in the UI; tracing is done through `ABOUT` links | 🧪 |
| 5 | FAISS `metric="inner_product"` scores every hit `1/(1+d)` | **ranking inverted** wherever `score` is used | keeps L2; cosine = `1 - d/2` | 🧪 |
| 6 | `GraphBuilder(merge_entities=True)` default fuzzy strategy | merges "Project Atlas" with "Project Cobalt" | merging disabled; explicit resolution | 🧪 |
| 7 | When an extraction method finds nothing, another method runs instead, down to "capitalised word" entities and adjacent `related_to` pairs | noise enters the graph unnoticed | filtered on `metadata["extraction_method"]`, counted under "Discarded" | 🧪 |
| 8 | SHACL generator keeps only the first range of a multi-range property | 11 false violations on this graph | rewrites those constraints to `sh:or`; shows both reports | 🧪 |
| 9 | OWL writer resolves plain domain/range names in a different namespace from declared class IRIs | domains point at undeclared classes | passes absolute IRIs | 🧪 |
| 10 | `HTMLParser` (`clean_text=True`) joins all lines with spaces | headings glue onto sentences, inventing entities | `clean_text=False` plus explicit block breaks | 🧪 |
| 11 | If the embedding model fails to load, hash or **random** vectors are used with only a warning | plausible-looking, meaningless retrieval | refuses to index unless the model is active | |
| 12 | `ContextRetriever` with a graph and a vector store needs `gensim` | `ImportError` despite not asking for node embeddings | installs the `graph-embeddings` extra | |
| 13 | `torch` / `transformers` are imported at start-up if installed | `embeddings-local` makes start-up much slower even when using FastEmbed | installs `fastembed` + `onnxruntime` only | |
| 14 | `ProgressTracker` is force-enabled by each component and prints to the console | noisy logs | `SEMANTICA_DISABLE_PROGRESS=1` | |
| 15 | `get_nodes` / `get_relationships` default to `limit=100` | silent truncation | passes an explicit large limit | |
| 16 | `ConflictResolver` returns a value even for an exact tie | a coin flip presented as a resolution | detects ties and labels them | |
| 17 | `ContextRetriever` offers no "insufficient evidence" signal | unanswerable questions get confident-looking context | measured as such in the evaluation | |
| 18 | `EventDetector(method="pattern")` participant lists contain duplicates and stray tokens | not usable as graph facts | shown as exploratory output only | |

## Not explored

Ontology versioning and alignment, community detection and global / drift retrieval,
node embeddings and link prediction, the triplet stores (RDF databases), the REST /
MCP servers and the Explorer UI, LLM-based ontology generation, and the non-file
ingestors (databases, web, streams).

## Optional dependencies and credentials

| Need | Package (extra) | Required |
|---|---|---|
| Neo4j | `neo4j` (`graph-neo4j`) | yes, plus a running Neo4j 5 |
| FAISS | `faiss-cpu` (`vectorstore-faiss`) | yes |
| PDF | `pdfplumber` (`parse-pdf`) | yes |
| DOCX / HTML | `python-docx`, `beautifulsoup4` (`documents`) | yes |
| SHACL validation | `pyshacl` (`shacl`) | yes |
| `ContextRetriever` | `gensim` (`graph-embeddings`) | yes |
| Embeddings | `fastembed`, `onnxruntime` | yes; model downloaded once |
| Visualisation | `plotly`, `matplotlib` | yes |
| LLM extraction / answers | `groq` (`llm-groq`) + `GROQ_API_KEY` | optional |

Messages such as "openai library not installed" or "litellm library not installed" at
start-up come from Semantica probing optional providers and can be ignored.
