"""Minimal KG builder template for Assignment 4.

Keep this contract unchanged:
- Graph: (Regulation)-[:HAS_ARTICLE]->(Article)-[:CONTAINS_RULE]->(Rule)
- Article: number, content, reg_name, category
- Rule: rule_id, type, action, result, art_ref, reg_name
- Fulltext indexes: article_content_idx, rule_idx
- SQLite file: ncu_regulations.db
"""

import json
import os
import re
import sqlite3
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

from llm_loader import load_local_llm, get_tokenizer, get_raw_pipeline


# ========== 0) Initialization ==========
load_dotenv()

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (
    os.getenv("NEO4J_USER", "neo4j"),
    os.getenv("NEO4J_PASSWORD", "password"),
)

# 預設先走快速穩定版：規則式抽取
# 若你之後想真的加上本地 LLM 抽取，可把環境變數設成 1
USE_LLM_EXTRACTION = os.getenv("USE_LLM_EXTRACTION", "0") == "1"


def normalize_text(text: str) -> str:
    """Normalize whitespace and punctuation spacing."""
    text = (text or "").replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def truncate_text(text: str, max_len: int = 220) -> str:
    text = normalize_text(text)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def split_sentences(text: str) -> list[str]:
    """
    Split Chinese regulation text into sentence-like segments.
    Keeps enough granularity for rule creation without exploding node count.
    """
    text = normalize_text(text)
    if not text:
        return []

    parts = re.split(r"[。；;]\s*", text)
    segments: list[str] = []

    for part in parts:
        part = normalize_text(part)
        if not part:
            continue

        # 再拆一次條列
        sub_parts = re.split(r"(?:^|[，,])\s*(?=[一二三四五六七八九十]\s*[、.])", part)
        for sp in sub_parts:
            sp = normalize_text(sp)
            if len(sp) >= 6:
                segments.append(sp)

    # 去重保序
    seen: set[str] = set()
    deduped: list[str] = []
    for seg in segments:
        key = seg.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(seg)

    return deduped


def infer_rule_type(action: str, result: str, full_text: str) -> str:
    """Infer a coarse rule type for retrieval convenience."""
    combo = f"{action} {result} {full_text}"

    if any(k in combo for k in ["扣", "記過", "零分", "處分", "懲處", "退學", "作弊", "威脅", "disciplinary", "barred"]):
        return "penalty"
    if any(k in combo for k in ["費", "工本費", "元", "NTD", "補發", "申請費"]):
        return "fee"
    if any(k in combo for k in ["學分", "畢業"]):
        return "graduation"
    if any(k in combo for k in ["及格", "成績", "分數", "points", "score"]):
        return "score"
    if any(k in combo for k in ["分鐘", "工作天", "日", "天", "年", "學年", "學期", "期限", "期間"]):
        return "time_limit"
    if any(k in combo for k in ["應", "須", "需", "得", "不得", "可", "can", "must", "shall"]):
        return "requirement"

    return "general"


def split_action_result(segment: str) -> tuple[str, str]:
    """
    Try to split a regulation segment into action / result.
    If no clear split exists, keep the whole sentence as both fields
    so retrieval still has usable text.
    """
    seg = normalize_text(segment)
    if not seg:
        return "", ""

    consequence_markers = [
        "處",
        "扣",
        "記",
        "零分",
        "不予",
        "喪失",
        "退學",
        "懲處",
        "disciplinary",
        "barred",
    ]

    # 情況1：前半描述條件，後半描述結果
    m = re.search(
        r"^(.*?)(?:，|,)\s*((?:應|得|不得|須|需|可|應予|予以|扣|處|記|不予|零分).*)$",
        seg,
    )
    if m:
        action = normalize_text(m.group(1))
        result = normalize_text(m.group(2))
        if action and result:
            return action, result

    # 情況2：出現「者」通常後面是處置或結果
    m = re.search(r"^(.*?者)\s*(.*)$", seg)
    if m:
        left = normalize_text(m.group(1))
        right = normalize_text(m.group(2))
        if right and any(k in right for k in consequence_markers):
            return left, right

    # 情況3：條文本身就是完整規定
    return seg, seg


def sanitize_rule(rule: dict[str, Any], article_number: str, reg_name: str) -> dict[str, str] | None:
    """Normalize and validate a rule dict."""
    action = truncate_text(str(rule.get("action", "")).strip(), 220)
    result = truncate_text(str(rule.get("result", "")).strip(), 220)

    if not action or not result:
        return None

    rule_type = normalize_text(str(rule.get("type", "")).strip())
    if not rule_type:
        rule_type = infer_rule_type(action, result, f"{action} {result}")

    return {
        "type": rule_type,
        "action": action,
        "result": result,
        "art_ref": article_number,
        "reg_name": reg_name,
    }


def try_llm_extract(article_number: str, reg_name: str, content: str) -> list[dict[str, str]]:
    """
    Optional local-LLM extraction.
    Disabled by default because CPU-only runs can be slow.
    """
    if not USE_LLM_EXTRACTION:
        return []

    try:
        if get_tokenizer() is None or get_raw_pipeline() is None:
            load_local_llm()

        tok = get_tokenizer()
        pipe = get_raw_pipeline()
        if tok is None or pipe is None:
            return []

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You extract regulation rules into strict JSON only.\n"
                    'Return exactly one JSON object: {"rules":[...]}.\n'
                    'Each rule must contain keys: "type", "action", "result".\n'
                    "Do not output markdown. Do not explain.\n"
                    "Keep each action/result concise and grounded in the article."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Regulation: {reg_name}\n"
                    f"Article: {article_number}\n"
                    f"Content: {content}\n\n"
                    "Extract up to 3 important rules."
                ),
            },
        ]

        prompt = tok.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        raw = pipe(prompt, max_new_tokens=220)[0]["generated_text"].strip()

        # 找出 JSON 區塊
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return []

        obj = json.loads(match.group(0))
        raw_rules = obj.get("rules", [])
        if not isinstance(raw_rules, list):
            return []

        cleaned: list[dict[str, str]] = []
        for rule in raw_rules:
            if not isinstance(rule, dict):
                continue
            safe_rule = sanitize_rule(rule, article_number, reg_name)
            if safe_rule is not None:
                cleaned.append(safe_rule)

        return cleaned

    except Exception as e:
        print(f"[LLM extract skipped] Article {article_number}: {e}")
        return []


def build_fallback_rules(article_number: str, content: str) -> list[dict[str, str]]:
    """Deterministic fallback rules to guarantee coverage."""
    text = normalize_text(content)
    if not text:
        return []

    keyword_priority = [
        "不得",
        "應",
        "須",
        "需",
        "得",
        "可",
        "扣",
        "零分",
        "懲處",
        "處分",
        "補發",
        "工本費",
        "元",
        "分鐘",
        "工作天",
        "學分",
        "畢業",
        "及格",
        "休學",
        "延長",
    ]

    segments = split_sentences(text)

    # 優先保留比較像規則的句子
    prioritized = [s for s in segments if any(k in s for k in keyword_priority)]

    # 若一篇條文都沒抓到明顯規則，至少保留一條摘要規則以確保 coverage
    candidates = prioritized if prioritized else [truncate_text(text, 220)]

    rules: list[dict[str, str]] = []
    seen: set[str] = set()

    for seg in candidates[:3]:
        action, result = split_action_result(seg)
        rule_type = infer_rule_type(action, result, seg)

        action = truncate_text(action, 220)
        result = truncate_text(result, 220)

        if not action or not result:
            continue

        sig = f"{rule_type}|{action.lower()}|{result.lower()}"
        if sig in seen:
            continue
        seen.add(sig)

        rules.append(
            {
                "type": rule_type,
                "action": action,
                "result": result,
            }
        )

    # 最少保留一條
    if not rules:
        summary = truncate_text(text, 220)
        rules.append(
            {
                "type": infer_rule_type(summary, summary, text),
                "action": summary,
                "result": summary,
            }
        )

    return rules


def extract_entities(article_number: str, reg_name: str, content: str) -> dict[str, Any]:
    """
    Extract rule-like facts from one article and return:
    {"rules": [{"type": ..., "action": ..., "result": ...}, ...]}
    """
    # 先試 LLM（若有開）
    llm_rules = try_llm_extract(article_number, reg_name, content)

    # LLM 沒抽到就走穩定 fallback
    base_rules = llm_rules if llm_rules else build_fallback_rules(article_number, content)

    cleaned: list[dict[str, str]] = []
    seen: set[str] = set()

    for rule in base_rules:
        safe_rule = sanitize_rule(rule, article_number, reg_name)
        if safe_rule is None:
            continue

        sig = f"{safe_rule['type']}|{safe_rule['action'].lower()}|{safe_rule['result'].lower()}"
        if sig in seen:
            continue
        seen.add(sig)

        cleaned.append(
            {
                "type": safe_rule["type"],
                "action": safe_rule["action"],
                "result": safe_rule["result"],
            }
        )

    return {"rules": cleaned}


# SQLite tables used:
# - regulations(reg_id, name, category)
# - articles(reg_id, article_number, content)


def build_graph() -> None:
    """Build KG from SQLite into Neo4j using the fixed assignment schema."""
    sql_conn = sqlite3.connect("ncu_regulations.db")
    cursor = sql_conn.cursor()
    driver = GraphDatabase.driver(URI, auth=AUTH)

    # 只有真的要用 LLM 抽取時才 warm up，避免 CPU 跑太慢
    if USE_LLM_EXTRACTION:
        load_local_llm()

    with driver.session() as session:
        # Fixed strategy: clear existing graph data before rebuilding.
        session.run("MATCH (n) DETACH DELETE n")

        # 1) Read regulations and create Regulation nodes.
        cursor.execute("SELECT reg_id, name, category FROM regulations")
        regulations = cursor.fetchall()
        reg_map: dict[int, tuple[str, str]] = {}

        for reg_id, name, category in regulations:
            reg_map[reg_id] = (name, category)
            session.run(
                "MERGE (r:Regulation {id:$rid}) SET r.name=$name, r.category=$cat",
                rid=reg_id,
                name=name,
                cat=category,
            )

        # 2) Read articles and create Article + HAS_ARTICLE.
        cursor.execute("SELECT reg_id, article_number, content FROM articles")
        articles = cursor.fetchall()

        for reg_id, article_number, content in articles:
            reg_name, reg_category = reg_map.get(reg_id, ("Unknown", "Unknown"))
            session.run(
                """
                MATCH (r:Regulation {id: $rid})
                CREATE (a:Article {
                    number:   $num,
                    content:  $content,
                    reg_name: $reg_name,
                    category: $reg_category
                })
                MERGE (r)-[:HAS_ARTICLE]->(a)
                """,
                rid=reg_id,
                num=article_number,
                content=content,
                reg_name=reg_name,
                reg_category=reg_category,
            )

        # 3) Create full-text index on Article content.
        session.run(
            """
            CREATE FULLTEXT INDEX article_content_idx IF NOT EXISTS
            FOR (a:Article) ON EACH [a.content]
            """
        )

        rule_counter = 0
        logical_seen: set[str] = set()

        # 4) Extract rules and create Rule nodes + CONTAINS_RULE.
        for idx, (reg_id, article_number, content) in enumerate(articles, start=1):
            reg_name, _ = reg_map.get(reg_id, ("Unknown", "Unknown"))

            extracted = extract_entities(article_number, reg_name, content)
            rules = extracted.get("rules", [])

            # 保底，避免任何文章沒規則
            if not rules:
                rules = build_fallback_rules(article_number, content)

            for rule in rules:
                safe_rule = sanitize_rule(rule, article_number, reg_name)
                if safe_rule is None:
                    continue

                logic_key = (
                    f"{safe_rule['reg_name'].lower()}|"
                    f"{safe_rule['art_ref'].lower()}|"
                    f"{safe_rule['type'].lower()}|"
                    f"{safe_rule['action'].lower()}|"
                    f"{safe_rule['result'].lower()}"
                )
                if logic_key in logical_seen:
                    continue
                logical_seen.add(logic_key)

                rule_counter += 1
                rule_id = f"R{rule_counter:06d}"

                session.run(
                    """
                    MATCH (a:Article {number: $art_ref, reg_name: $reg_name})
                    CREATE (r:Rule {
                        rule_id:  $rule_id,
                        type:     $type,
                        action:   $action,
                        result:   $result,
                        art_ref:  $art_ref,
                        reg_name: $reg_name
                    })
                    MERGE (a)-[:CONTAINS_RULE]->(r)
                    """,
                    rule_id=rule_id,
                    type=safe_rule["type"],
                    action=safe_rule["action"],
                    result=safe_rule["result"],
                    art_ref=safe_rule["art_ref"],
                    reg_name=safe_rule["reg_name"],
                )

            if idx % 25 == 0 or idx == len(articles):
                print(f"[Progress] processed {idx}/{len(articles)} articles, rules={rule_counter}")

        # 5) Create full-text index on Rule fields.
        session.run(
            """
            CREATE FULLTEXT INDEX rule_idx IF NOT EXISTS
            FOR (r:Rule) ON EACH [r.action, r.result]
            """
        )

        # 6) Coverage audit.
        coverage = session.run(
            """
            MATCH (a:Article)
            OPTIONAL MATCH (a)-[:CONTAINS_RULE]->(r:Rule)
            WITH a, count(r) AS rule_count
            RETURN count(a) AS total_articles,
                   sum(CASE WHEN rule_count > 0 THEN 1 ELSE 0 END) AS covered_articles,
                   sum(CASE WHEN rule_count = 0 THEN 1 ELSE 0 END) AS uncovered_articles
            """
        ).single()

        total_articles = int((coverage or {}).get("total_articles", 0) or 0)
        covered_articles = int((coverage or {}).get("covered_articles", 0) or 0)
        uncovered_articles = int((coverage or {}).get("uncovered_articles", 0) or 0)

        print(
            f"[Coverage] covered={covered_articles}/{total_articles}, "
            f"uncovered={uncovered_articles}"
        )
        print(f"[Done] total rules created = {rule_counter}")

    driver.close()
    sql_conn.close()


if __name__ == "__main__":
    build_graph()