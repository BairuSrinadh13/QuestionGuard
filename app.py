from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import streamlit as st

# Optional integrations. QuestionGuard still runs without either package.
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None

APP_NAME = "QuestionGuard"
APP_VERSION = "2.0"
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("QUESTIONGUARD_DB", str(BASE_DIR / "questions.db")))
SEMANTIC_THRESHOLD = 0.80
REVIEW_THRESHOLD = 0.64

SEED_QUESTIONS = [
    {"question": "What is reflection of light?", "tags": {"topic": "reflection of light", "id": "Q1"}, "source": "seed"},
    {"question": "Define the reflection of light.", "tags": {"topic": "reflection of light", "id": "Q2"}, "source": "seed"},
    {"question": "What do you understand by the reflection of light?", "tags": {"topic": "reflection of light", "id": "Q3"}, "source": "seed"},
    {"question": "Explain the meaning of light reflection.", "tags": {"topic": "reflection of light", "id": "Q4"}, "source": "seed"},
    {"question": "State the laws of reflection of light.", "tags": {"topic": "reflection of light", "id": "Q5"}, "source": "seed"},
    {"question": "What is the angle of reflection?", "tags": {"topic": "reflection of light", "id": "Q6"}, "source": "seed"},
    {"question": "The angle of incidence is 30°. What is the angle of reflection?", "tags": {"topic": "reflection of light", "id": "Q7"}, "source": "seed"},
    {"question": "What is reflection?", "tags": {"topic": "reflection of light", "id": "Q8", "note": "broad/ambiguous wording"}, "source": "seed"},
    {"question": "What is the difference between regular and diffuse reflection?", "tags": {"topic": "reflection of light", "id": "Q9"}, "source": "seed"},
    {"question": "Define the reflection of light.", "tags": {"topic": "reflection of light", "id": "Q10"}, "source": "seed"},
]

# ----------------------------- Text analysis -----------------------------

def normalize(text: str) -> str:
    text = (text or "").lower().replace("°", " degree ")
    text = text.replace("’", "'").replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def content_tokens(text: str) -> set[str]:
    stop = {"what", "is", "the", "a", "an", "of", "by", "to", "do", "you", "understand", "explain", "define", "state", "how", "does", "are", "and", "in", "for", "between", "with", "from", "it", "this", "that"}
    return {w for w in normalize(text).split() if w not in stop and len(w) > 1}


def token_jaccard(a: str, b: str) -> float:
    aa, bb = set(normalize(a).split()), set(normalize(b).split())
    return len(aa & bb) / len(aa | bb) if aa | bb else 1.0


def char_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def intent(text: str) -> str:
    """Conservative intent labels used as a guard, not as a semantic model."""
    t = normalize(text)
    words = set(t.split())

    if any(p in t for p in ("difference between", "differentiate", "compare", "different from", "versus")):
        if "reflection" in words and ("regular" in words or "diffuse" in words):
            return "reflection_types_comparison"
        return "compare"
    if any(p in t for p in ("law of reflection", "laws of reflection", "state the laws", "two laws that govern")):
        return "reflection_laws"
    if "angle of incidence" in t and "angle of reflection" in t:
        return "angle_calculation"
    if "angle of reflection" in t:
        return "angle_of_reflection"
    if "angle of incidence" in t:
        return "angle_of_incidence"
    if any(p in t for p in ("calculate", "compute", "find the angle", "solve")):
        return "calculation"
    if "reflection" in words:
        if "light" in words:
            return "reflection_of_light"
        return "ambiguous_reflection"
    if "photosynthesis" in words or ("plants" in words and any(p in t for p in ("make food", "making food", "produce food", "using sunlight", "sunlight"))):
        return "photosynthesis"
    if "cpu" in words or "processor" in words:
        return "cpu"
    if "ram" in words or "storage" in words or "memory" in words:
        return "memory_or_storage"
    if any(p in t for p in ("what is", "define", "meaning", "explain", "describe", "how does", "how do", "what does", "purpose of", "function of")):
        return "definition_or_explanation"
    return "other"


def scope_conflict(a: str, b: str) -> bool:
    ia, ib = intent(a), intent(b)
    return {ia, ib} == {"ambiguous_reflection", "reflection_of_light"}


def is_photosynthesis_paraphrase(a: str, b: str) -> bool:
    aa, bb = normalize(a), normalize(b)
    wa, wb = set(aa.split()), set(bb.split())
    if intent(a) != "photosynthesis" or intent(b) != "photosynthesis":
        return False
    process_phrases = ("make food", "making food", "produce food", "produces food", "using sunlight", "use sunlight", "sunlight")
    a_process = "plants" in wa and any(p in aa for p in process_phrases)
    b_process = "plants" in wb and any(p in bb for p in process_phrases)
    return ("photosynthesis" in wa and b_process) or ("photosynthesis" in wb and a_process)


@st.cache_resource(show_spinner="Loading semantic language model…")
def load_embedder():
    if SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


def similarity_scores(candidate: str, existing_texts: list[str]) -> tuple[list[float], str]:
    if not existing_texts:
        return [], "no comparison corpus"
    model = load_embedder()
    if model is not None:
        try:
            vectors = model.encode([candidate] + existing_texts, normalize_embeddings=True)
            scores = (vectors[1:] @ vectors[0]).tolist()
            return [max(0.0, min(1.0, float(x))) for x in scores], "Sentence Transformers (all-MiniLM-L6-v2)"
        except Exception:
            pass
    if TfidfVectorizer is not None and cosine_similarity is not None:
        try:
            matrix = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode").fit_transform([candidate] + existing_texts)
            vals = cosine_similarity(matrix[0:1], matrix[1:]).ravel().tolist()
            return [float(x) for x in vals], "TF-IDF fallback"
        except Exception:
            pass
    return [SequenceMatcher(None, normalize(candidate), normalize(x)).ratio() for x in existing_texts], "character-similarity fallback"


def classify_pair(candidate: str, existing: str, semantic: float | None = None, method: str = "pair comparison") -> dict[str, Any]:
    """Classify one candidate against one known question."""
    a, b = normalize(candidate), normalize(existing)
    if not a:
        return {"status": "Invalid", "score": 0.0, "method": "validation", "reason": "Enter a question."}
    if a == b:
        return {"status": "Exact duplicate", "score": 1.0, "method": "normalized exact match", "reason": "Same words after case, punctuation, and whitespace normalization."}

    if semantic is None:
        scores, method = similarity_scores(candidate, [existing])
        semantic = scores[0] if scores else 0.0
    jac = token_jaccard(candidate, existing)
    chars = char_similarity(candidate, existing)
    ia, ib = intent(candidate), intent(existing)

    if is_photosynthesis_paraphrase(candidate, existing):
        status, reason = "Semantic duplicate", "Equivalent photosynthesis terminology and description of the process."
    elif scope_conflict(candidate, existing):
        status, reason = "Review", "One question refers to reflection broadly, while the other specifically refers to reflection of light. Their scopes may differ."
    elif ia != ib and ia not in ("other", "definition_or_explanation") and ib not in ("other", "definition_or_explanation"):
        if semantic >= 0.70 or jac >= 0.35:
            status, reason = "Review", "The wording is related, but the questions appear to ask about different subtopics or tasks."
        else:
            status, reason = "New question", "No sufficiently strong duplicate signal; the question intent differs."
    elif semantic >= SEMANTIC_THRESHOLD and (ia == ib or ia in ("other", "definition_or_explanation") or ib in ("other", "definition_or_explanation")):
        status, reason = "Semantic duplicate", "High semantic similarity with compatible question intent."
    elif semantic >= REVIEW_THRESHOLD or (jac >= 0.45 and (ia == ib or ia == "other" or ib == "other")) or chars >= 0.82:
        status, reason = "Review", "Related wording or concept, but equivalence is uncertain."
    else:
        status, reason = "New question", "No sufficiently strong duplicate signal."

    return {"status": status, "score": float(semantic), "method": method, "reason": reason, "jaccard": jac, "character_similarity": chars, "candidate_intent": ia, "existing_intent": ib}


def classify_candidate(candidate: str, corpus: list[dict[str, Any]], semantic_threshold: float = SEMANTIC_THRESHOLD, review_threshold: float = REVIEW_THRESHOLD) -> dict[str, Any]:
    """Compare with the whole corpus; return the strongest relevant match."""
    if not (candidate or "").strip():
        return {"status": "Invalid", "match": None, "score": 0.0, "method": "validation", "reason": "Enter a question."}
    normalized = normalize(candidate)
    if not normalized:
        return {"status": "Invalid", "match": None, "score": 0.0, "method": "validation", "reason": "No searchable text."}

    # Exact matches take precedence regardless of other corpus items.
    for item in corpus:
        q = item.get("question", "")
        if normalized == normalize(q):
            return {"status": "Exact duplicate", "match": item, "score": 1.0, "method": "normalized exact match", "reason": "Same words after case, punctuation, and whitespace normalization.", "jaccard": 1.0, "character_similarity": 1.0, "intent": intent(candidate)}

    texts = [x.get("question", "") for x in corpus]
    scores, method = similarity_scores(candidate, texts)
    ranked: list[tuple[float, float, int, dict[str, Any]]] = []
    for index, (item, sem) in enumerate(zip(corpus, scores)):
        pair = classify_pair(candidate, item.get("question", ""), semantic=sem, method=method)
        # The ranking rewards semantic closeness but preserves explicit scope-conflict review decisions.
        jac = pair.get("jaccard", token_jaccard(candidate, item.get("question", "")))
        chars = pair.get("character_similarity", char_similarity(candidate, item.get("question", "")))
        effective = max(float(sem), 0.92 * jac, 0.88 * chars)
        ranked.append((effective, float(sem), index, {**pair, "match": item, "method": method, "jaccard": jac, "character_similarity": chars}))

    if not ranked:
        return {"status": "New question", "match": None, "score": 0.0, "method": method, "reason": "Corpus is empty; no existing question was available for comparison.", "intent": intent(candidate)}

    # Choose the strongest pair; if scores tie, prefer review over a risky duplicate decision.
    ranked.sort(key=lambda x: (x[0], x[1], x[3]["status"] == "Review"), reverse=True)
    best = ranked[0][3]
    best["intent"] = intent(candidate)
    return best

# ------------------------------- Database ---------------------------------

def connect_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=20, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with connect_db() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            normalized TEXT,
            tags TEXT NOT NULL DEFAULT '{}',
            source TEXT NOT NULL DEFAULT 'manual',
            created_at TEXT NOT NULL DEFAULT ''
        )""")
        cols = {r[1] for r in conn.execute("PRAGMA table_info(questions)").fetchall()}
        if "normalized" not in cols:
            conn.execute("ALTER TABLE questions ADD COLUMN normalized TEXT")
        if "tags" not in cols:
            conn.execute("ALTER TABLE questions ADD COLUMN tags TEXT NOT NULL DEFAULT '{}' ")
        if "source" not in cols:
            conn.execute("ALTER TABLE questions ADD COLUMN source TEXT NOT NULL DEFAULT 'manual'")
        if "created_at" not in cols:
            conn.execute("ALTER TABLE questions ADD COLUMN created_at TEXT NOT NULL DEFAULT ''")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_questions_normalized ON questions(normalized)")
        conn.execute("""CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate TEXT NOT NULL,
            matched_question TEXT,
            decision TEXT NOT NULL,
            reason TEXT,
            created_at TEXT NOT NULL
        )""")
        # Backfill normalized values for databases created by earlier versions.
        rows = conn.execute("SELECT id, question FROM questions WHERE normalized IS NULL OR normalized = ''").fetchall()
        for row in rows:
            conn.execute("UPDATE questions SET normalized=? WHERE id=?", (normalize(row["question"]), row["id"]))
        count = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
        if count == 0:
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            for item in SEED_QUESTIONS:
                conn.execute("INSERT INTO questions(question, normalized, tags, source, created_at) VALUES(?,?,?,?,?)", (item["question"], normalize(item["question"]), json.dumps(item["tags"], ensure_ascii=False), item["source"], now))


def load_questions() -> list[dict[str, Any]]:
    with connect_db() as conn:
        rows = conn.execute("SELECT id, question, tags, source, created_at FROM questions ORDER BY id").fetchall()
    result = []
    for row in rows:
        try:
            tags = json.loads(row["tags"] or "{}")
            if not isinstance(tags, dict):
                tags = {}
        except (json.JSONDecodeError, TypeError):
            tags = {}
        result.append({"id": row["id"], "question": row["question"], "tags": tags, "source": row["source"], "created_at": row["created_at"]})
    return result


def save_question(question: str, tags: dict[str, Any] | None = None, source: str = "manual") -> tuple[bool, str]:
    question = (question or "").strip()
    if not question:
        return False, "Question cannot be empty."
    normalized = normalize(question)
    if not normalized:
        return False, "Question must contain searchable text."
    with connect_db() as conn:
        row = conn.execute("SELECT id, question FROM questions WHERE normalized=? LIMIT 1", (normalized,)).fetchone()
        if row:
            return False, f"This exact question is already stored (ID {row['id']})."
        # Also guard against legacy rows whose normalized field was not backfilled.
        for old in conn.execute("SELECT id, question FROM questions").fetchall():
            if normalize(old["question"]) == normalized:
                return False, f"This exact question is already stored (ID {old['id']})."
        conn.execute("INSERT INTO questions(question, normalized, tags, source, created_at) VALUES(?,?,?,?,?)", (question, normalized, json.dumps(tags or {}, ensure_ascii=False), source, datetime.now(timezone.utc).isoformat(timespec="seconds")))
    return True, "Question saved to the corpus."


def delete_question(question_id: int) -> None:
    with connect_db() as conn:
        conn.execute("DELETE FROM questions WHERE id=?", (int(question_id),))


def log_decision(candidate: str, match: str | None, decision: str, reason: str) -> None:
    with connect_db() as conn:
        conn.execute("INSERT INTO decisions(candidate, matched_question, decision, reason, created_at) VALUES(?,?,?,?,?)", (candidate, match, decision, reason, datetime.now(timezone.utc).isoformat(timespec="seconds")))


def load_decisions(limit: int = 200) -> list[dict[str, Any]]:
    with connect_db() as conn:
        rows = conn.execute("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]

# -------------------------- Generation and eval --------------------------
def generate_with_gemini(topic: str, count: int, existing: list[dict[str, Any]], api_key: str) -> list[str]:
    try:
        from google import genai
    except Exception as exc:
        raise RuntimeError("Gemini support is not installed. Install google-genai or run without an API key.") from exc
    client = genai.Client(api_key=api_key)
    topic_tokens = content_tokens(topic)
    ranked_existing = sorted(existing, key=lambda q: len(topic_tokens & content_tokens(q.get("question", ""))), reverse=True)
    examples = [q["question"] for q in ranked_existing[:40]]
    prompt = f"""You create high-quality educational assessment questions.
Topic: {topic}
Generate up to {min(max(count * 3, count), 24)} candidate questions.
Rules:
- Produce genuinely distinct questions, not paraphrases of existing questions.
- Explore different subtopics, applications, calculations, misconceptions, reasoning, and practical examples where suitable.
- Do not provide answers, numbering, commentary, or markdown.
- Keep every question clear and self-contained.
Existing question bank (avoid both exact and conceptual repeats):
{json.dumps(examples, ensure_ascii=False)}
Return ONLY a valid JSON array of strings."""
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    raw = (response.text or "").strip()
    raw = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw, flags=re.IGNORECASE).strip()
    start, end = raw.find("["), raw.rfind("]")
    if start < 0 or end < start:
        raise ValueError("Gemini did not return a JSON array. Please retry.")
    data = json.loads(raw[start:end + 1])
    if not isinstance(data, list):
        raise ValueError("Gemini response must be a JSON array.")
    accepted, seen = [], list(existing)
    seen_normalized: set[str] = set()
    for item in data:
        if not isinstance(item, str) or not item.strip():
            continue
        candidate = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", item.strip())
        key = normalize(candidate)
        if not key or key in seen_normalized:
            continue
        seen_normalized.add(key)
        result = classify_candidate(candidate, seen)
        # Keep Review candidates visible for human review; suppress only confirmed duplicates.
        if result["status"] in ("Exact duplicate", "Semantic duplicate"):
            continue
        accepted.append(candidate)
        seen.append({"question": candidate, "tags": {"topic": topic}, "source": "generated"})
        if len(accepted) >= count:
            break
    return accepted


def fallback_generate(topic: str, count: int, existing: list[dict[str, Any]]) -> list[str]:
    topic = topic.strip()
    if not topic:
        return []
    templates = [
        f"What are the main principles of {topic}?",
        f"How can {topic} be demonstrated with a practical example?",
        f"What factors affect {topic}?",
        f"How does {topic} differ from related concepts?",
        f"Why is {topic} important in real-world applications?",
        f"What is a common misconception about {topic}?",
        f"How would you solve a problem involving {topic}?",
        f"What experiment could be used to investigate {topic}?",
        f"What are the limitations of applying {topic}?",
        f"How can a result involving {topic} be verified?",
    ]
    accepted, seen = [], list(existing)
    for candidate in templates:
        result = classify_candidate(candidate, seen)
        if result["status"] in ("Exact duplicate", "Semantic duplicate"):
            continue
        accepted.append(candidate)
        seen.append({"question": candidate, "tags": {"topic": topic}, "source": "fallback"})
        if len(accepted) >= count:
            break
    return accepted


def run_demo_evaluation() -> list[dict[str, Any]]:
    pairs = [
        ("What is reflection of light?", "Define the reflection of light.", "Semantic duplicate"),
        ("What is reflection of light?", "What do you understand by the reflection of light?", "Semantic duplicate"),
        ("What is reflection of light?", "Explain the meaning of light reflection.", "Semantic duplicate"),
        ("What is reflection of light?", "State the laws of reflection of light.", "Different question"),
        ("What is reflection of light?", "What is the angle of reflection?", "Different question"),
        ("What is reflection of light?", "The angle of incidence is 30°. What is the angle of reflection?", "Different question"),
        ("What is reflection of light?", "What is reflection?", "Review"),
        ("What is reflection of light?", "What is the difference between regular and diffuse reflection?", "Different question"),
        ("Define the reflection of light.", "Define the reflection of light.", "Exact duplicate"),
        ("What is photosynthesis?", "How do plants make food using sunlight?", "Semantic duplicate"),
        ("What is photosynthesis?", "What is cellular respiration?", "Different question"),
    ]
    results = []
    for a, b, expected in pairs:
        got = classify_pair(a, b)
        status = got["status"]
        if expected == "Different question":
            passed = status == "New question"
        else:
            passed = status == expected
        results.append({"Question A": a, "Question B": b, "Expected": expected, "Actual": status, "Similarity": round(got["score"], 3), "Result": "PASS" if passed else "CHECK", "Reason": got["reason"]})
    return results

def _rows_to_csv(rows: list[dict[str, Any]]) -> str:
    import csv
    import io
    buffer = io.StringIO()
    if not rows:
        return ""
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()



# ---------------------------------- UI ------------------------------------
st.set_page_config(page_title=APP_NAME, page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
init_db()

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1450px;}
.qg-hero {padding: 1.25rem 1.4rem; border-radius: 16px; border: 1px solid rgba(128,128,128,.28); margin-bottom: 1rem;}
.qg-muted {opacity: .75; font-size: .92rem;}
[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); padding: .75rem; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

questions = load_questions()
with st.sidebar:
    st.markdown(f"# 🛡️ {APP_NAME}")
    st.caption(f"Question quality workspace · v{APP_VERSION}")
    st.divider()
    api_key = st.text_input("Gemini API key (optional)", type="password", help="Used only when you generate questions with Gemini. Leave blank to use local templates.")
    st.caption("Your API key is used for this session and is not written to the QuestionGuard database.")
    st.divider()
    st.caption(f"Database: `{DB_PATH.name}`")
    model_ready = load_embedder() is not None
    st.caption("Semantic engine: Sentence Transformers" if model_ready else "Semantic engine: TF-IDF/character fallback")

st.markdown(f"<div class='qg-hero'><h1 style='margin:0'>🛡️ {APP_NAME}</h1><p style='margin:.35rem 0 0 0'>Generate, compare, review, and maintain a higher-quality question bank.</p><p class='qg-muted' style='margin:.35rem 0 0 0'>Exact matching · semantic similarity · intent and scope safeguards · human review · SQLite persistence</p></div>", unsafe_allow_html=True)

questions = load_questions()
count_all = len(questions)
count_generated = sum(1 for q in questions if q.get("source") in ("generated", "fallback"))
count_manual = sum(1 for q in questions if q.get("source") in ("manual", "manual_review"))

# Keep the existing six-tab workflow so this can replace the previous app.py directly.
tabs = st.tabs(["Dashboard", "Generate questions", "Check duplicates", "Question corpus", "Evaluation", "Architecture"])

with tabs[0]:
    st.subheader("Corpus overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stored questions", count_all)
    c2.metric("Generated", count_generated)
    c3.metric("Manual additions", count_manual)
    c4.metric("Review decisions", len(load_decisions()))
    st.markdown("### Recent questions")
    if questions:
        for q in questions[-5:][::-1]:
            with st.container(border=True):
                st.write(f"**#{q['id']}** · {q['question']}")
                st.caption(f"Source: {q.get('source', 'unknown')} · Tags: {json.dumps(q.get('tags', {}), ensure_ascii=False)}")
    else:
        st.info("Your question bank is empty.")
    st.markdown("### How decisions work")
    a, b, c = st.columns(3)
    a.markdown("**Exact duplicate**\n\nSame normalized wording.")
    b.markdown("**Semantic duplicate**\n\nLikely asks the same thing using different wording.")
    c.markdown("**Review**\n\nRelated concepts or different scope; human judgment is recommended.")

with tabs[1]:
    st.subheader("Generate new questions")
    topic = st.text_input("Topic or category", value="reflection of light", key="generation_topic")
    gen_count = st.slider("Number of candidates", min_value=1, max_value=10, value=5, key="generation_count")
    if st.button("Generate and independently check", type="primary", key="generate_questions"):
        current = load_questions()
        try:
            if api_key.strip():
                generated = generate_with_gemini(topic, gen_count, current, api_key.strip())
                generation_mode = "Gemini API"
            else:
                generated = fallback_generate(topic, gen_count, current)
                generation_mode = "Local template fallback (not LLM-generated)"
            st.session_state["qg_generated_results"] = {"topic": topic, "mode": generation_mode, "items": generated}
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
            st.caption("Check the API key and internet connection, or leave the key blank to use local templates.")
    generated_state = st.session_state.get("qg_generated_results")
    if generated_state:
        st.write(f"Generation mode: **{generated_state['mode']}**")
        if not generated_state["items"]:
            st.info("No new candidates were returned. The generated candidates may overlap with your current corpus.")
        for idx, candidate in enumerate(generated_state["items"]):
            result = classify_candidate(candidate, load_questions())
            with st.container(border=True):
                st.write(candidate)
                st.write(f"**Decision:** {result['status']} · **Method:** {result['method']}")
                if result.get("match"):
                    st.caption(f"Closest match: {result['match']['question']}")
                st.caption(f"Similarity: {result.get('score', 0.0):.3f} · {result['reason']}")
                if result["status"] == "New question":
                    tags_text = st.text_input("Tags as JSON (optional)", value=json.dumps({"topic": generated_state["topic"]}, ensure_ascii=False), key=f"gen_tags_{hashlib.md5(candidate.encode()).hexdigest()}")
                    if st.button("Save accepted question", key=f"gen_save_{hashlib.md5(candidate.encode()).hexdigest()}"):
                        try:
                            tags = json.loads(tags_text or "{}")
                            if not isinstance(tags, dict):
                                raise ValueError("Tags must be a JSON key-value object.")
                            ok, message = save_question(candidate, tags, source="generated" if "Gemini" in generated_state["mode"] else "fallback")
                            (st.success if ok else st.warning)(message)
                            if ok:
                                st.rerun()
                        except (json.JSONDecodeError, ValueError) as exc:
                            st.error(f"Tags are invalid: {exc}")
                elif result["status"] == "Review":
                    st.warning("Review before adding. This candidate is not auto-saved.")
                    review_key = hashlib.md5((candidate + "review").encode()).hexdigest()
                    if st.button("Keep as new question", key=f"gen_keep_{review_key}"):
                        ok, message = save_question(candidate, {"topic": generated_state["topic"], "reviewed": True}, source="manual_review")
                        log_decision(candidate, result.get("match", {}).get("question") if result.get("match") else None, "Kept as new", result["reason"])
                        (st.success if ok else st.warning)(message)
                        if ok:
                            st.rerun()
                else:
                    st.warning("Confirmed duplicate: not auto-saved.")

with tabs[2]:
    st.subheader("Check a question")
    candidate = st.text_area("Enter a question to compare with the saved corpus", height=110, placeholder="e.g., Explain what reflection means.", key="manual_candidate")
    if st.button("Check for duplicates", type="primary", key="check_duplicate"):
        corpus = load_questions()
        result = classify_candidate(candidate, corpus)
        st.session_state["qg_last_check"] = {"candidate": candidate, "result": result}
        log_decision(candidate, result.get("match", {}).get("question") if result.get("match") else None, result["status"], result["reason"])
    last_check = st.session_state.get("qg_last_check")
    if last_check:
        result = last_check["result"]
        status = result["status"]
        if status == "Exact duplicate":
            st.error(f"{status}")
        elif status == "Semantic duplicate":
            st.warning(f"{status}")
        elif status == "Review":
            st.info(f"{status}")
        elif status == "Invalid":
            st.error(status)
        else:
            st.success(status)
        st.write(f"**Method:** {result.get('method', 'n/a')}")
        st.write(f"**Reason:** {result.get('reason', '')}")
        st.metric("Semantic similarity", f"{result.get('score', 0.0):.3f}")
        match = result.get("match")
        if match:
            st.markdown("**Closest match**")
            with st.container(border=True):
                st.write(match.get("question", ""))
                st.caption(f"Question ID: {match.get('id', 'n/a')} · Intent: {result.get('intent', 'n/a')}")
        if status in ("New question", "Review"):
            with st.expander("Save decision", expanded=(status == "New question")):
                tags_raw = st.text_input("Tags JSON", value='{"topic": "general"}', key="manual_tags")
                if status == "Review":
                    st.caption("Saving a reviewed candidate records your decision; it does not mean the detector judged it to be new automatically.")
                if st.button("Save to corpus", key="manual_save_candidate"):
                    try:
                        tags = json.loads(tags_raw or "{}")
                        if not isinstance(tags, dict):
                            raise ValueError("Tags must be a JSON object.")
                        ok, message = save_question(last_check["candidate"], tags, source="manual_review" if status == "Review" else "manual")
                        log_decision(last_check["candidate"], match.get("question") if match else None, "Saved by user" if ok else "Save blocked", message)
                        (st.success if ok else st.warning)(message)
                        if ok:
                            st.rerun()
                    except (json.JSONDecodeError, ValueError) as exc:
                        st.error(f"Tags are invalid: {exc}")

with tabs[3]:
    st.subheader("Question corpus")
    current = load_questions()
    f1, f2 = st.columns([2, 1])
    search_text = f1.text_input("Search questions or tags", key="corpus_search")
    source_filter = f2.selectbox("Source", ["All", "seed", "manual", "manual_review", "generated", "fallback"], key="corpus_source")
    filtered = []
    needle = normalize(search_text)
    for q in current:
        blob = normalize(q["question"] + " " + json.dumps(q.get("tags", {}), ensure_ascii=False))
        if needle and needle not in blob:
            continue
        if source_filter != "All" and q.get("source") != source_filter:
            continue
        filtered.append(q)
    st.caption(f"Showing {len(filtered)} of {len(current)} questions")
    for q in filtered:
        with st.container(border=True):
            left, right = st.columns([6, 1])
            left.markdown(f"**#{q['id']} · {q['question']}**")
            left.caption(f"Source: {q.get('source', 'unknown')} · Added: {q.get('created_at', '')}")
            left.caption(f"Tags: {json.dumps(q.get('tags', {}), ensure_ascii=False)}")
            if right.button("Delete", key=f"delete_q_{q['id']}"):
                delete_question(q["id"])
                st.rerun()
    st.divider()
    st.markdown("### Add a question manually")
    with st.form("add_question_form", clear_on_submit=True):
        new_question = st.text_area("Question", placeholder="Enter one clear question")
        new_tags = st.text_input("Tags JSON", value='{"topic": "general"}')
        submitted = st.form_submit_button("Add to corpus", type="primary")
    if submitted:
        try:
            tags = json.loads(new_tags or "{}")
            if not isinstance(tags, dict):
                raise ValueError("Tags must be a JSON object.")
            ok, message = save_question(new_question, tags, source="manual")
            (st.success if ok else st.warning)(message)
            if ok:
                st.rerun()
        except (json.JSONDecodeError, ValueError) as exc:
            st.error(f"Could not save question: {exc}")
    st.divider()
    export_payload = [{"id": q["id"], "question": q["question"], "tags": q["tags"], "source": q["source"], "created_at": q["created_at"]} for q in current]
    st.download_button("Export corpus as JSON", data=json.dumps(export_payload, ensure_ascii=False, indent=2), file_name="questionguard_corpus.json", mime="application/json")

with tabs[4]:
    st.subheader("Evaluation and decision history")
    st.write("Pairwise evaluation checks a specific pair of questions. Corpus checking instead compares a candidate against every saved question; these are different tasks.")
    if st.button("Run built-in evaluation", type="primary", key="run_eval"):
        st.session_state["qg_eval_results"] = run_demo_evaluation()
    eval_results = st.session_state.get("qg_eval_results")
    if eval_results:
        passed = sum(1 for r in eval_results if r["Result"] == "PASS")
        e1, e2, e3 = st.columns(3)
        e1.metric("Passed", passed)
        e2.metric("Needs review", len(eval_results) - passed)
        e3.metric("Pass rate", f"{passed / len(eval_results) * 100:.1f}%" if eval_results else "—")
        st.dataframe(eval_results, use_container_width=True, hide_index=True)
        st.download_button("Export evaluation CSV", data=_rows_to_csv(eval_results), file_name="questionguard_evaluation.csv", mime="text/csv")
    st.markdown("### Recent decisions")
    decisions = load_decisions()
    if decisions:
        st.dataframe(decisions, use_container_width=True, hide_index=True)
    else:
        st.caption("No decisions have been logged yet.")

with tabs[5]:
    st.subheader("System architecture")
    st.markdown("""
    ```text
    User input
       ↓
    Text normalization ──→ Exact-match check
       ↓                         ↓
    Semantic similarity ← Existing question corpus (SQLite)
       ↓
    Intent + scope safeguards
       ↓
    Decision: Exact duplicate / Semantic duplicate / Review / New question
       ↓
    Human decision + decision log
       ↓
    Optional Gemini generation → same independent duplicate pipeline
    ```
    """)
    st.markdown("### Components")
    st.markdown("- **Streamlit:** interactive interface.\n- **SQLite:** persistent question bank and decision history.\n- **Sentence Transformers:** semantic embeddings when available.\n- **TF-IDF / character similarity:** fallback when the embedding model cannot load.\n- **Gemini API:** optional question generation; the generated candidates still pass through QuestionGuard.\n- **Intent and scope safeguards:** help avoid treating related-but-different questions as duplicates.")
    st.warning("Similarity is evidence, not proof of equivalence. Review decisions are intentionally available for ambiguous wording and questions with different scope.")


def _rows_to_csv(rows: list[dict[str, Any]]) -> str:
    import csv
    import io
    buffer = io.StringIO()
    if not rows:
        return ""
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()
