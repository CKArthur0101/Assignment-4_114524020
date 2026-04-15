# Assignment 4: KG-based QA for NCU Regulations

## Due Date
April 9 - April 23, 2026

## TA Names
簡資烜, 葛亭妤

---

## What This Assignment Really Focuses On
This assignment is **not** about writing a long chatbot pipeline. It is mainly about whether your system can:

1. Build a usable Knowledge Graph from regulation PDFs.
2. Retrieve the correct regulation evidence (article/rule) for a question.
3. Generate an answer that is grounded in retrieved evidence.

In short: **Retrieval quality and grounding quality are the core.**

---

## Grading Priority

| Item           | Weight |
|----------------|--------|
| Query Accuracy | 40%    |
| Report          | 60%    |

To improve Query Accuracy, prioritize:
- Better KG coverage (fewer uncovered articles).
- Better retrieval precision/recall (typed + broad query strategy).
- Better grounded answer style (direct answer + cite source article).

---

## Current Code Reality (Important)
This section reflects the **actual project files** in this folder.

### Active scripts
- `setup_data.py`
- `build_kg.py`
- `query_system.py`
- `auto_test.py`

### Model setup used in this project
- The provided code path uses **local Hugging Face model loading** via `llm_loader.py`.
- Default model is `Qwen/Qwen2.5-3B-Instruct` (local cache supported).
- You may only switch to a model that is **smaller than the current default** (equal or larger models are not allowed).
- Using external API-key based model services is not allowed for this assignment.



---
## Important Note

You are free to deviate from the provided code templates and implement your own design. However, **auto_test.py must work with the provided `test_data.json` structure** to perform automatic grading. If `auto_test.py` fails to run correctly, the assignment will be considered incomplete and receive no points.

---

## Required KG / Query Contract (Do Not Break)
To keep `build_kg.py` ↔ `query_system.py` ↔ `auto_test.py` compatible, keep these names consistent.

### Graph schema
- `(:Regulation)-[:HAS_ARTICLE]->(:Article)-[:CONTAINS_RULE]->(:Rule)`

### Article properties
- `number`, `content`, `reg_name`, `category`

### Rule properties
- `rule_id`, `type`, `action`, `result`, `art_ref`, `reg_name`

### Fulltext indexes
- `article_content_idx`
- `rule_idx`

### SQLite filename
- `ncu_regulations.db`

If you rename these fields/indexes, retrieval and auto-testing may fail even if your logic is good.

---

## Functional Requirements

### 1) Data setup
- Parse PDFs in `source/` and store structured rows into SQLite.

### 2) KG construction
- Create Regulation and Article nodes from SQLite.
- Extract rule-level facts and write Rule nodes.
- Link Article → Rule using `:CONTAINS_RULE`.

### 3) Query system
- Convert question into structured query intent.
- Retrieve relevant Rule candidates from Neo4j.
- Return an evidence-grounded final answer.

#### Design note: why include DB Article snippets?
- Rule-level KG retrieval is the primary channel (fast and structured), but some Rule nodes are short and may miss full legal context.
- Therefore, it is acceptable to add a second evidence channel: **DB Article snippets routed by KG results** (KG first, DB second).
- This design helps reduce hallucination by giving the model sentence-level article text while keeping retrieval grounded in KG evidence.
- If KG-routed article fetch is sparse, a limited DB keyword fallback is allowed to improve recall.

### 4) Auto evaluation
- `auto_test.py` reads `test_data.json` and evaluates answer quality with an LLM-as-a-judge prompt.

---

## Prohibited / Allowed

### Prohibited
- Feeding all regulations directly to LLM and answering without KG retrieval.
- Switching to a model larger than `Qwen/Qwen2.5-3B-Instruct`.
- Using API-key based online model services.
- Dumping a large amount of regulation text to the model at once and asking it to pick the correct answer by itself.

### Allowed
- Local inference with the provided pipeline, and optional switch to smaller local models only.
- Prompt and query optimization (keyword expansion, typed/broad strategy, fallback retrieval).

---

## How to Run (Correct Order)

### 1. Prerequisites
- Python 3.11
- Docker Desktop

### 2. Start Neo4j
```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
```

### 3. Install dependencies
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Execute pipeline
```bash
# If ncu_regulations.db is already included, TA can skip setup_data.py.
# python setup_data.py
python build_kg.py
# Optional manual check
python query_system.py
python auto_test.py
```

Run all commands in the same working directory (repository root) to avoid relative-path issues.

---

## What to Include in Report
1. KG construction logic and design choices.
2. KG schema/diagram.
3. Key Cypher query design and retrieval strategy.
4. Failure analysis + improvements made.

---

## Submission
- Submit a **GitHub repository link** for this assignment.

Your repository must include at least the following files:
1. `README.md`
	- Describe your KG schema design.
	- Include screenshots showing key nodes and relationships in your graph.
2. `auto_test.py`
3. `build_kg.py`
4. `llm_loader.py`
5. `query_system.py`
6. `requirements.txt`
7. `.gitignore`

Deadline: **April 23, 2026**

