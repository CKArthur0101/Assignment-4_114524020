# Assignment 4 - NCU Regulation KG QA System

## 1. Project Overview

This project builds a Knowledge Graph (KG)-based question answering system for National Central University (NCU) regulations.  
The system uses Neo4j to store regulations, articles, and extracted rules, and answers regulation-related questions through graph retrieval and rule-based answer generation.

The goal of this assignment is to:
- build a regulation KG from provided documents,
- design a schema for regulations and rules,
- retrieve relevant articles/rules from Neo4j,
- and generate grounded answers for user questions.

---

## 2. Repository Files

This repository includes the following required files:

- `README.md`
- `auto_test.py`
- `build_kg.py`
- `llm_loader.py`
- `query_system.py`
- `requirements.txt`
- `.gitignore`

---

## 3. KG Schema Design

### Node Types

#### (1) `Regulation`
Represents a regulation document or policy source.

Example properties:
- `name`
- `category`

#### (2) `Article`
Represents an article, section, or rule text extracted from the regulation documents.

Example properties:
- `art_ref`
- `content`
- `category`

#### (3) `Rule`
Represents an atomic rule extracted from an article for downstream question answering.

Example properties:
- `rule_id`
- `type`
- `action`
- `result`
- `art_ref`
- `reg_name`

---

### Relationships

#### `(:Regulation)-[:HAS_ARTICLE]->(:Article)`
This relationship connects each regulation document to its articles.

#### `(:Article)-[:CONTAINS_RULE]->(:Rule)`
This relationship connects each article to one or more extracted rules.

---

## 4. Why I Designed the Schema This Way

I separated the KG into **Regulation → Article → Rule** because this structure makes the information more organized and easier to retrieve.

- `Regulation` stores the document-level source.
- `Article` preserves the original legal or policy text.
- `Rule` stores smaller, answerable units for question answering.

This design helps the system support:
- document-level browsing,
- article-level grounding,
- and fine-grained rule retrieval for QA.

It also makes it easier to answer questions such as:
- penalties,
- fees,
- required credits,
- passing grades,
- study duration,
- and suspension rules.

---

## 5. System Workflow

### Step 1. Build the Knowledge Graph
`build_kg.py` reads the provided regulation files, extracts structured content, and imports it into Neo4j.

### Step 2. Load the Local LLM
`llm_loader.py` is used to load the local Hugging Face model and tokenizer.

### Step 3. Retrieve Relevant Rules
`query_system.py` first analyzes the question, then performs:
- typed retrieval from rule nodes,
- broader retrieval from article content,
- and result merging/ranking.

### Step 4. Generate the Final Answer
The system generates a grounded answer based on retrieved rules rather than answering purely from model memory.

---

## 6. Query Strategy

In `query_system.py`, I used a hybrid retrieval design:

1. **Question parsing / entity extraction**  
   The system identifies the question type and key terms such as:
   - penalty
   - fee
   - graduation credits
   - PE
   - leave of absence
   - passing score

2. **Typed retrieval**  
   It first tries to retrieve matching `Rule` nodes using type-aware Cypher queries.

3. **Broad retrieval**  
   It also searches article content to avoid missing relevant evidence.

4. **Answer normalization**  
   For common regulation questions, the final answers are normalized into concise forms such as:
   - `200 NTD.`
   - `3 working days.`
   - `128 credits.`
   - `60 points.`
   - `No.`

This improves answer consistency and makes the output more stable in evaluation.

---

## 7. Example QA Topics Supported

The system can answer questions such as:

- What is the penalty for forgetting a student ID?
- What is the fee for replacing a lost EasyCard student ID?
- How many working days does it take to get a new student ID?
- What is the minimum total credits required for undergraduate graduation?
- How many semesters of PE are required?
- Are Military Training credits counted towards graduation credits?
- What is the passing score for undergraduate students?
- What is the passing score for graduate students?
- Under what condition will a student be dismissed due to poor grades?
- What is the maximum duration for a leave of absence?

---

## 8. Auto Test Result

The final system passed all 20 questions in `auto_test.py`.

**Final Result:**
- Total: 20
- Passed: 20
- Failed: 0
- Accuracy: 100.0%

---

## 9. Screenshots

### Screenshot 1. Neo4j Graph Overview
![Screenshot 1](./Screenshot%201.%20Neo4j%20Graph%20Overview.png)

### Screenshot 2. Regulation to Article Relationship
![Screenshot 2](./Screenshot%202.%20Regulation%20to%20Article%20Relationship.png)

### Screenshot 3. Article to Rule Relationship
![Screenshot 3](./Screenshot%203.%20Article%20to%20Rule%20Relationship.png)

### Screenshot 4. Rule Node Properties
![Screenshot 4](./Screenshot%204.%20Rule%20Node%20Properties.png)

### Screenshot 5. Auto Test Result
![Screenshot 5](./Screenshot%205.%20Auto%20Test%20Result.png)

---

## 10. Challenges and Improvements

One challenge in this assignment was that regulation questions may use different wording for the same concept.  
For example, “replacement fee,” “lost student ID fee,” and “EasyCard student ID” may refer to the same rule.

To improve performance, I adjusted:
- keyword extraction,
- typed retrieval,
- broad retrieval,
- and answer normalization.

This helped the system retrieve the correct rules more consistently and produce stable answers.

In the future, the system could be improved by:
- using better rule extraction,
- adding multilingual support,
- and expanding the KG to cover more university regulations.

---

## 11. How to Run

### Install dependencies

pip install -r requirements.txt

### Start Neo4j

docker run -d --name neo4j-assignment4 -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest

### Build the KG

python build_kg.py

### Run the QA system

python query_system.py

### Run the auto test

python auto_test.py

---

## 12. Conclusion

This project demonstrates how a Knowledge Graph can be used to support regulation question answering in a more structured and grounded way.

By combining:

Neo4j graph storage,
rule-based schema design,
typed and broad retrieval,
and local LLM support,

the system can answer university regulation questions effectively and achieve strong performance on the provided evaluation set.


