import os
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pypdf import PdfReader

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

import tiktoken

from dotenv import load_dotenv
load_dotenv()

# Optional OpenAI generation
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# =========================
# Configuration
# =========================
DATA_DIR = "./data"
DEFAULT_PDF = os.path.join(DATA_DIR, "document.pdf")

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "thesis_chunks"

EMBED_MODEL = "all-MiniLM-L6-v2"

# Chunking
CHUNK_TOKENS = 450
CHUNK_OVERLAP_TOKENS = 80

# Retrieval
TOP_K_DENSE = 10
TOP_K_SPARSE = 10
FINAL_TOP_K = 6

DENSE_WEIGHT = 0.6
SPARSE_WEIGHT = 0.4

MIN_FINAL_SCORE = 0.10  # filter weak hits

# Generation
DEFAULT_LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.2

# Tokenizer for chunk sizing
enc = tiktoken.get_encoding("cl100k_base")


# =========================
# Helpers
# =========================
def tok_len(text: str) -> int:
    return len(enc.encode(text))


def clean_text(s: str) -> str:
    s = s.replace("\x00", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_heading(line: str) -> bool:
    """
    Heuristic heading detector for theses/manuals:
    - "CHAPTER X", "INTRODUCTION", "RELATED LITERATURE" (mostly uppercase)
    - Numbered headings like "3.5 Subjective Logic Principle"
    """
    line_stripped = line.strip()
    if not line_stripped:
        return False

    # Numbered headings: 1.2, 3.5.2 etc + title
    if re.match(r"^\d+(\.\d+){0,3}\s+\S+", line_stripped):
        return True

    # CHAPTER keyword
    if re.match(r"^(CHAPTER|APPENDIX)\b", line_stripped.upper()):
        return True

    # Mostly uppercase and reasonable length (avoid shouting sentences)
    letters = re.sub(r"[^A-Za-z]+", "", line_stripped)
    if len(letters) >= 6:
        upper_ratio = sum(c.isupper() for c in letters) / len(letters)
        if upper_ratio > 0.85 and 4 <= len(line_stripped.split()) <= 10:
            return True

    return False


def split_into_blocks_by_headings(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert PDF pages into blocks with (chapter/section-ish) headings.
    Output blocks have:
      - "title_path": list[str] representing hierarchy
      - "page_start", "page_end"
      - "text": block content (non-heading text)
    """
    title_path: List[str] = []
    blocks: List[Dict[str, Any]] = []

    cur_lines: List[str] = []
    cur_page_start = 1
    cur_page_end = 1

    def flush():
        nonlocal cur_lines, cur_page_start, cur_page_end
        text = clean_text("\n".join(cur_lines))
        if text and tok_len(text) > 30:
            blocks.append({
                "title_path": title_path.copy(),
                "page_start": cur_page_start,
                "page_end": cur_page_end,
                "text": text
            })
        cur_lines = []

    for p in pages:
        page_no = p["page"]
        raw = p["text"] or ""
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

        if page_no == 1:
            cur_page_start = page_no
        cur_page_end = page_no

        for ln in lines:
            if is_heading(ln):
                flush()

                if re.match(r"^\d+(\.\d+){0,3}\s+", ln):
                    depth = ln.split()[0].count(".") + 1
                    if depth == 1:
                        title_path = [ln]
                    else:
                        if len(title_path) >= depth:
                            title_path = title_path[:depth - 1]
                        title_path.append(ln)
                else:
                    title_path = [ln]

                cur_page_start = page_no
                cur_page_end = page_no
            else:
                cur_lines.append(ln)

    flush()
    return blocks


def paragraph_chunk(text: str, chunk_tokens: int, overlap_tokens: int) -> List[str]:
    """
    Paragraph-aware chunking with overlap.
    """
    paras = re.split(r"(?:\n\s*\n)+", text)
    if len(paras) == 1:
        paras = re.split(r"(?<=[.!?])\s+", text)

    chunks: List[str] = []
    cur: List[str] = []
    cur_tokens = 0

    def flush_with_overlap():
        nonlocal cur, cur_tokens
        if not cur:
            return
        chunk = clean_text(" ".join(cur))
        if chunk:
            chunks.append(chunk)

        toks = enc.encode(chunk)
        tail = toks[-overlap_tokens:] if len(toks) > overlap_tokens else toks
        tail_text = enc.decode(tail).strip()
        cur = [tail_text] if tail_text else []
        cur_tokens = tok_len(tail_text) if tail_text else 0

    for para in paras:
        para = clean_text(para)
        if not para:
            continue
        p_tokens = tok_len(para)

        if p_tokens > chunk_tokens:
            sents = re.split(r"(?<=[.!?])\s+", para)
            for s in sents:
                s = clean_text(s)
                if not s:
                    continue
                s_tokens = tok_len(s)
                if cur_tokens + s_tokens > chunk_tokens and cur:
                    flush_with_overlap()
                cur.append(s)
                cur_tokens += s_tokens
            continue

        if cur_tokens + p_tokens > chunk_tokens and cur:
            flush_with_overlap()

        cur.append(para)
        cur_tokens += p_tokens

    if cur:
        chunk = clean_text(" ".join(cur))
        if chunk:
            chunks.append(chunk)

    return chunks


def tokenize_for_bm25(text: str) -> List[str]:
    """
    Thesis-friendly tokenizer:
    - keeps decimals like 7.2, 4.2.3
    - keeps hyphenated words like e-sers
    - keeps words and numbers
    """
    text = (text or "").lower()
    return re.findall(r"[a-z]+(?:-[a-z]+)*|\d+(?:\.\d+)+|\d+", text)


def is_list_like(text: str) -> bool:
    """
    Detect 'List of Tables/Figures' style chunks that mention many Table/Figure IDs.
    We down-rank these because they often outrank the actual content.
    """
    return len(re.findall(r"\b(table|figure)\s+\d+(?:\.\d+)+\b", (text or "").lower())) >= 3


def section_boost(meta: dict) -> float:
    title = (meta or {}).get("title_path", "") or ""
    t = title.lower()

    boost = 0.0

    # Very strong positives
    if "4.1.3 limitations" in t:
        boost += 0.20
    if "4.2 e-sers" in t:
        boost += 0.18

    # General positives
    if "limitations" in t:
        boost += 0.08
    if "e-sers" in t:
        boost += 0.06
    if "sers" in t:
        boost += 0.04
    if "architecture" in t:
        boost += 0.02

    # Negatives
    if "system design and evaluation" in t:
        boost -= 0.10
    if re.search(r"\b7\.\d\b", t):     # Chapter 7.* sections
        boost -= 0.06
    if "appendix" in t:
        boost -= 0.12
    if "survey" in t:
        boost -= 0.08

    return boost



def weighted_overlap_rerank(query: str, texts: Dict[str, str]) -> Dict[str, float]:
    """
    Better than plain Jaccard for answer quality:
    rewards coverage of the query terms (like a tiny 'soft reranker').
    """
    q = tokenize_for_bm25(query)
    q_counts = Counter(q)
    if not q_counts:
        return {k: 0.0 for k in texts}

    den = float(sum(q_counts.values()))
    out: Dict[str, float] = {}
    for _id, txt in texts.items():
        d = tokenize_for_bm25(txt or "")
        d_counts = Counter(d)
        if not d_counts:
            out[_id] = 0.0
            continue
        num = 0.0
        for tok, qc in q_counts.items():
            num += min(float(qc), float(d_counts.get(tok, 0)))
        out[_id] = float(num / (den + 1e-9))
    return out

import html

def html_escape(s: str) -> str:
    return html.escape(s or "", quote=True)

def highlight_terms(text: str, query: str, max_terms: int = 12) -> str:
    safe = html_escape(text)
    q_tokens = tokenize_for_bm25(query)

    seen = set()
    terms = []
    for t in q_tokens:
        if len(t) <= 2 or t in seen:
            continue
        seen.add(t)
        terms.append(t)
        if len(terms) >= max_terms:
            break

    terms.sort(key=len, reverse=True)

    for term in terms:
        pattern = re.compile(rf"(?i)\b({re.escape(term)})\b")
        safe = pattern.sub(r"<mark>\1</mark>", safe)

    return safe

def build_citation_label(meta: Dict[str, Any]) -> str:
    title = meta.get("title_path", "") or "Untitled section"
    ps = meta.get("page_start", "?")
    pe = meta.get("page_end", "?")
    return f"{title} (pp. {ps}-{pe})"

def make_ask_html_page(query: str, answer: str, retrieved: List[Dict[str, Any]], explain: Dict[str, Any]) -> str:
    q = html_escape(query)
    ans = html_escape(answer).replace("\n", "<br/>")

    cards = []
    for i, r in enumerate(retrieved, 1):
        meta = r["meta"]
        label = html_escape(build_citation_label(meta))
        snippet = highlight_terms(r["text"][:900], query)

        cards.append(f"""
        <details class="card">
          <summary><b>{i}.</b> {label}</summary>
          <div class="snippet">{snippet}</div>
        </details>
        """)

    cards_html = "\n".join(cards)

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>Thesis RAG Answer</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 900px; margin: auto; }}
.answer {{ padding: 14px; border: 1px solid #ddd; border-radius: 8px; }}
.card {{ margin-top: 10px; border: 1px solid #ddd; border-radius: 8px; padding: 10px; }}
mark {{ background: #ffeb3b; }}
</style>
</head>
<body>
<h2>Query</h2>
<p>{q}</p>

<h2>Answer</h2>
<div class="answer">{ans}</div>

<h2>Citations</h2>
{cards_html}

</body>
</html>"""


# =========================
# Storage / Index Globals
# =========================
os.makedirs(CHROMA_DIR, exist_ok=True)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=st_ef,
    metadata={"description": "Thesis/manual chunks with structure metadata"}
)

# BM25 in-memory (rebuilt from Chroma)
BM25_MODEL: Optional[BM25Okapi] = None
BM25_DOC_IDS: List[str] = []
BM25_TOKENS: List[List[str]] = []

# Optional OpenAI client
OPENAI_CLIENT = OpenAI() if (OpenAI is not None and os.getenv("OPENAI_API_KEY")) else None


# =========================
# Ingestion
# =========================
def read_pdf_pages(pdf_path: str) -> List[Dict[str, Any]]:
    reader = PdfReader(pdf_path)
    pages = []
    for i, pg in enumerate(reader.pages):
        pages.append({"page": i + 1, "text": pg.extract_text() or ""})
    return pages


def ingest(pdf_path: str) -> Dict[str, Any]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = read_pdf_pages(pdf_path)
    blocks = split_into_blocks_by_headings(pages)

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    doc_count = 0
    for b_idx, b in enumerate(blocks):
        title_path = b["title_path"]
        block_text = b["text"]

        chunks = paragraph_chunk(block_text, CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS)
        for c_idx, ch in enumerate(chunks):
            doc_id = f"{os.path.basename(pdf_path)}::b{b_idx}::c{c_idx}"
            meta = {
                "source": os.path.basename(pdf_path),
                "title_path": " > ".join(title_path) if title_path else "",
                "page_start": b["page_start"],
                "page_end": b["page_end"],
                "block_index": b_idx,
                "chunk_index_in_block": c_idx,
            }
            ids.append(doc_id)
            docs.append(ch)
            metas.append(meta)
            doc_count += 1

    # Demo reset: delete & recreate entire collection
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    global collection
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=st_ef,
        metadata={"description": "Thesis/manual chunks with structure metadata"}
    )
    collection.upsert(ids=ids, documents=docs, metadatas=metas)

    return {
        "pdf": os.path.basename(pdf_path),
        "pages": len(pages),
        "blocks": len(blocks),
        "chunks": doc_count,
        "chunk_tokens": CHUNK_TOKENS,
        "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
        "embedding_model": EMBED_MODEL,
        "bm25_enabled": True
    }


def rebuild_bm25_from_chroma() -> Dict[str, Any]:
    global BM25_MODEL, BM25_DOC_IDS, BM25_TOKENS

    data = collection.get(include=["documents"])
    ids = data.get("ids", [])
    docs = data.get("documents", [])

    if not ids or not docs:
        BM25_MODEL = None
        BM25_DOC_IDS = []
        BM25_TOKENS = []
        return {"bm25_ready": False, "docs": 0}

    BM25_DOC_IDS = ids
    BM25_TOKENS = [tokenize_for_bm25(t or "") for t in docs]
    BM25_MODEL = BM25Okapi(BM25_TOKENS)

    return {"bm25_ready": True, "docs": len(BM25_DOC_IDS)}


# =========================
# Retrieval
# =========================
def dense_retrieve(query: str, k: int) -> List[Dict[str, Any]]:
    res = collection.query(
        query_texts=[query],
        n_results=k,
        include=["distances", "metadatas", "documents"]
    )

    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]

    out = []
    for _id, doc, meta, dist in zip(ids, docs, metas, dists):
        sim = 1.0 / (1.0 + float(dist))
        out.append({
            "id": _id,
            "text": doc,
            "meta": meta,
            "dense_score": sim
        })
    return out


def sparse_retrieve(query: str, k: int) -> Dict[str, float]:
    if BM25_MODEL is None:
        return {}

    q_tokens = tokenize_for_bm25(query)
    scores_arr = BM25_MODEL.get_scores(q_tokens)

    top_idx = np.argsort(scores_arr)[::-1][:k]
    top_scores = scores_arr[top_idx]
    max_s = float(np.max(top_scores)) if len(top_scores) else 1.0

    out: Dict[str, float] = {}
    for idx in top_idx:
        _id = BM25_DOC_IDS[int(idx)]
        out[_id] = float(scores_arr[int(idx)] / (max_s + 1e-9))
    return out

def dedupe_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in results:
        m = r.get("meta", {}) or {}
        key = (m.get("title_path", ""), m.get("page_start"), m.get("page_end"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

def hybrid_retrieve(query: str) -> Dict[str, Any]:
    """
    Best-answer-quality hybrid retrieval:
      - Thesis-friendly BM25 tokenization
      - Dynamic weights
      - Smarter "conceptual vs reference vs keyword-like" detection
      - Stronger rerank (weighted overlap)
      - List-of-tables/figures penalty
      - Section-aware boost (limitations / E-SERS / SERS)
      - Conceptual dense threshold
    """

    # ---- 1) Dense retrieval ----
    dense_hits = dense_retrieve(query, TOP_K_DENSE)
    dense_map = {h["id"]: h for h in dense_hits}
    dense_scores = {h["id"]: float(h.get("dense_score", 0.0)) for h in dense_hits}

    # ---- 2) Sparse retrieval ----
    sparse_scores = sparse_retrieve(query, TOP_K_SPARSE)
    debug_sparse_top5 = sorted(sparse_scores.items(), key=lambda x: x[1], reverse=True)[:5]

    # ---- 3) Query type detection (smarter) ----
    ql = query.lower()
    reference_like = bool(re.search(r"\b(table|figure|eq|equation|algorithm)\b", ql))
    keyword_like = bool(re.search(r"\b(e-sers|sers|apk|permission|risk matrix|4×4|3×3)\b", ql))

    # Conceptual = neither explicit reference (table/eq) nor keyword-heavy technical
    conceptual = (not reference_like) and (not keyword_like)

    # ---- 4) Dynamic weights ----
    if conceptual:
        dense_w, sparse_w = 0.85, 0.15
    else:
        dense_w, sparse_w = DENSE_WEIGHT, SPARSE_WEIGHT

    # ---- 5) Candidate union + combine ----
    candidates = set(dense_scores.keys()) | set(sparse_scores.keys())

    combined: List[Tuple[str, float, float, float]] = []
    for _id in candidates:
        d = dense_scores.get(_id, 0.0)
        s = sparse_scores.get(_id, 0.0)
        score = (dense_w * d) + (sparse_w * s)
        combined.append((_id, score, d, s))
    combined.sort(key=lambda x: x[1], reverse=True)

    # ---- 6) Conceptual filter ----
    conceptual_dense_min = 0.55
    keyword_like_dense_min = 0.40
   

    # ---- 7) Rerank on wider pool (fetch texts as needed) ----
    top_ids = [x[0] for x in combined[: max(FINAL_TOP_K * 4, 20)]]

    rerank_texts: Dict[str, str] = {}
    for _id in top_ids:
        if _id in dense_map:
            rerank_texts[_id] = dense_map[_id]["text"]
        else:
            fetched = collection.get(ids=[_id], include=["documents"])
            if fetched and fetched.get("documents"):
                rerank_texts[_id] = fetched["documents"][0]

    rerank_map = weighted_overlap_rerank(query, rerank_texts)

    # ---- 8) Final scoring + penalties + boosts ----
    final: List[Tuple[str, float, float, float, float]] = []
    for _id, score, d, s in combined:
        if conceptual and d < conceptual_dense_min:
            continue
        if keyword_like and d < keyword_like_dense_min:
            continue

        rr = float(rerank_map.get(_id, 0.0))
        base = (0.85 * score) + (0.15 * rr)

        # Get text for list-like penalty
        if _id in rerank_texts:
            text_for_penalty = rerank_texts[_id]
        elif _id in dense_map:
            text_for_penalty = dense_map[_id]["text"]
        else:
            fetched = collection.get(ids=[_id], include=["documents"])
            text_for_penalty = fetched["documents"][0] if fetched and fetched.get("documents") else ""

        penalty = 0.15 if (text_for_penalty and is_list_like(text_for_penalty)) else 0.0

        # Get meta for section boost
        if _id in dense_map:
            meta_for_boost = dense_map[_id].get("meta", {})
        else:
            fetched_meta = collection.get(ids=[_id], include=["metadatas"])
            meta_for_boost = fetched_meta["metadatas"][0] if fetched_meta and fetched_meta.get("metadatas") else {}

        boost = section_boost(meta_for_boost)

        final_score = base - penalty + boost

        if final_score >= MIN_FINAL_SCORE:
            final.append((_id, final_score, d, s, rr))

    is_diff_q = bool(re.search(r"\b(difference|different|compare|vs)\b", query.lower()))
    final_k = 4 if is_diff_q else FINAL_TOP_K

    final.sort(key=lambda x: x[1], reverse=True)
    final = final[:final_k]

    # ---- 9) Materialize results ----
    results: List[Dict[str, Any]] = []
    for _id, f, d, s, rr in final:
        if _id in dense_map:
            text = dense_map[_id]["text"]
            meta = dense_map[_id]["meta"]
        else:
            fetched = collection.get(ids=[_id], include=["documents", "metadatas"])
            text = fetched["documents"][0] if fetched and fetched.get("documents") else ""
            meta = fetched["metadatas"][0] if fetched and fetched.get("metadatas") else {}

        results.append({
            "id": _id,
            "final_score": float(f),
            "dense_score": float(dense_scores.get(_id, d)),
            "sparse_score": float(sparse_scores.get(_id, 0.0)),
            "rerank_score": float(rr),
            "meta": meta,
            "text": text
        })

    # ---- 10) Explainability ----
    explain = {
        "chosen_strategies": {
            "chunking": {
                "type": "heading-aware blocks + paragraph chunking",
                "chunk_tokens": CHUNK_TOKENS,
                "overlap_tokens": CHUNK_OVERLAP_TOKENS,
                "why_best_for_thesis": (
                    "Academic content is section-structured and context dependent; "
                    "chunking by detected headings and paragraphs reduces splitting of definitions/formulas "
                    "and improves citation accuracy."
                )
            },
            "embedding": {
                "model": EMBED_MODEL,
                "type": "single-vector-per-chunk",
                "why_best_for_thesis": (
                    "Semantic embeddings capture conceptual similarity across sections, "
                    "helping match conceptual questions to the correct explanations."
                )
            },
            "retrieval": {
                "type": "hybrid dense + BM25 + weighted-overlap rerank + list-of-tables penalty + section boost",
                "weights": {
                    "dense": dense_w,
                    "sparse": sparse_w,
                    "rerank_mix": "0.85/0.15",
                    "list_like_penalty": 0.15,
                    "conceptual_dense_min": conceptual_dense_min
                },
                "why_best_for_thesis": (
                    "Theses include exact references (tables/equations) and conceptual questions. "
                    "Dense embeddings capture semantics; BM25 catches exact identifiers; a lightweight rerank "
                    "improves precision; list/TOC penalties and section-aware boosts reduce noise."
                )
            }
        },
        "debug": {
            "dense_top_k": TOP_K_DENSE,
            "sparse_top_k": TOP_K_SPARSE,
            "final_top_k": FINAL_TOP_K,
            "min_final_score": MIN_FINAL_SCORE,
            "conceptual_query": conceptual,
            "reference_like": reference_like,
            "keyword_like": keyword_like,
            "dense_w": dense_w,
            "sparse_w": sparse_w,
            "sparse_top5": debug_sparse_top5
        }
    }
    
    results = dedupe_results(results)
    results = results[:FINAL_TOP_K]
    return {"results": results, "explain": explain}


# =========================
# Answering
# =========================
def grounded_extract_answer(query: str, retrieved: List[Dict[str, Any]]) -> str:
    """
    If no LLM key is available, return an extractive, grounded response:
    - show top evidence snippets and where they came from.
    """
    lines = [f"I can’t generate with an LLM (OPENAI_API_KEY not set). Here are the most relevant excerpts:\n"]
    for i, r in enumerate(retrieved, 1):
        meta = r["meta"]
        cite = f"(pages {meta['page_start']}-{meta['page_end']}, {meta.get('title_path','')})"
        snippet = (r["text"] or "")[:600].strip()
        lines.append(f"{i}) {cite}\n{snippet}\n")
    lines.append("Tip: set OPENAI_API_KEY to enable a fully-written answer grounded in these excerpts.")
    return "\n".join(lines)

def enforce_numbers_from_context(answer: str, context: str) -> str:
    ctx_nums = set(re.findall(r"\d+(?:\.\d+)?", context))
    ans_nums = set(re.findall(r"\d+(?:\.\d+)?", answer))

    illegal = ans_nums - ctx_nums
    if not illegal:
        return answer

    # Remove sentences containing illegal numbers
    sents = re.split(r"(?<=[.!?])\s+", answer)
    kept = []
    for s in sents:
        if any(n in s for n in illegal):
            continue
        kept.append(s)
    return " ".join(kept).strip()


def llm_answer(query: str, retrieved: List[Dict[str, Any]], model: str) -> str:
    if OPENAI_CLIENT is None:
        return grounded_extract_answer(query, retrieved)

    context_blocks = []
    for r in retrieved:
        meta = r["meta"]
        tag = f"[pages {meta['page_start']}-{meta['page_end']}]"
        title = meta.get("title_path", "")
        context_blocks.append(f"{tag} {title}\n{r['text']}")

    context = "\n\n---\n\n".join(context_blocks)

    system = (
        "You are an academic assistant. Use ONLY the provided excerpts as evidence. "
        "Do NOT add any facts, numbers, dataset sizes, or claims that are not explicitly stated in the excerpts. "
        "If a numeric value is not present in the excerpts, do not guess it—omit it. "
        "Write 1 short paragraph answer first, then bullet points. "
        "Each bullet MUST end with citations like [pages X-Y]. "
        "Do not mention dataset sizes or experimental counts unless explicitly stated. "
        "If the excerpts do not support a detail, say: 'Not enough evidence in the provided text.'"
    )

    resp = OPENAI_CLIENT.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Question:\n{query}\n\nContext:\n{context}"}
        ],
        temperature=TEMPERATURE
    )

    answer = resp.choices[0].message.content or ""
    # Number-guard happens here, where `context` exists:
    answer = enforce_numbers_from_context(answer, context)
    return answer



# =========================
# FastAPI
# =========================
app = FastAPI(title="Explainable Thesis RAG Assistant")


class IngestRequest(BaseModel):
    pdf_path: Optional[str] = None


class AskRequest(BaseModel):
    query: str
    llm_model: Optional[str] = None

class AskHtmlRequest(BaseModel):
    query: str
    llm_model: Optional[str] = None


@app.post("/ingest")
def ingest_route(req: IngestRequest):
    pdf_path = req.pdf_path or DEFAULT_PDF
    t0 = time.time()
    stats = ingest(pdf_path)

    bm25_stats = rebuild_bm25_from_chroma()
    return {"ok": True, "stats": stats, "bm25": bm25_stats, "seconds": round(time.time() - t0, 2)}


@app.post("/ask")
def ask_route(req: AskRequest):
    query = req.query.strip()
    if not query:
        return {"ok": False, "error": "query is required"}

    t0 = time.time()
    pack = hybrid_retrieve(query)
    retrieved = pack["results"]
    explain = pack["explain"]

    model = req.llm_model or DEFAULT_LLM_MODEL
    answer = llm_answer(query, retrieved, model)    

    sources = []
    for r in retrieved:
        m = r["meta"]
        sources.append({
            "title_path": m.get("title_path", ""),
            "page_start": m.get("page_start"),
            "page_end": m.get("page_end"),
            "scores": {
                "final": r["final_score"],
                "dense": r["dense_score"],
                "sparse": r["sparse_score"],
                "rerank": r["rerank_score"],
            }
        })

    return {
        "ok": True,
        "answer": answer,
        "sources": sources,
        "explain": explain,
        "seconds": round(time.time() - t0, 2)
    }

@app.post("/ask_html", response_class=HTMLResponse)
def ask_html_route(req: AskHtmlRequest):
    query = req.query.strip()
    if not query:
        return HTMLResponse("<h3>Error: query is required</h3>", status_code=400)

    pack = hybrid_retrieve(query)
    retrieved = pack["results"]
    explain = pack["explain"]

    #  ignore Swagger placeholder value
    model = req.llm_model
    if not model or model.strip().lower() == "string":
        model = DEFAULT_LLM_MODEL

    try:
        answer = llm_answer(query, retrieved, model)
    except Exception as e:
        # Return readable error in HTML instead of generic 500
        return HTMLResponse(f"<h3>LLM Error</h3><pre>{html_escape(str(e))}</pre>", status_code=500)

    html_page = make_ask_html_page(query, answer, retrieved, explain)
    return HTMLResponse(html_page, status_code=200)


@app.get("/ui", response_class=HTMLResponse)
def ui_page():
    return HTMLResponse("""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Thesis RAG UI</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 980px; margin: 24px auto; padding: 0 12px; }
    textarea { width: 100%; min-height: 90px; padding: 10px; }
    button { padding: 10px 14px; cursor: pointer; border-radius: 10px; border: 1px solid #ddd; background: #fff; }
    .row { display:flex; gap:10px; align-items:center; margin-top:10px; flex-wrap: wrap; }
    .card { border: 1px solid #e2e2e2; border-radius: 12px; padding: 12px; margin-top: 12px; background: #fff; }
    .muted { color: #555; font-size: 12px; }
    .answer { white-space: pre-wrap; line-height: 1.5; }
    mark { background: #ffeb3b; padding: 0 2px; border-radius: 4px; }
    details { margin-top: 10px; }
    summary { cursor: pointer; }
    code { background: #f4f4f4; padding: 2px 6px; border-radius: 6px; border: 1px solid #e2e2e2; }
    .pill { font-size: 12px; border: 1px solid #e2e2e2; padding: 4px 8px; border-radius: 999px; background: #fafafa; }
    .grid { display:grid; grid-template-columns: 1fr; gap: 10px; }
    @media (min-width: 820px) { .grid { grid-template-columns: 1fr 1fr; } }
  </style>
</head>
<body>
  <h2>Explainable Thesis RAG Assistant</h2>

  <div class="card">
    <div class="muted">Ask a question</div>
    <textarea id="q" placeholder="e.g., What problem does this document address?"></textarea>

    <div class="row">
      <button onclick="ask()">Ask</button>
      <span id="status" class="muted"></span>
      <span id="timing" class="pill" style="display:none;"></span>
    </div>

    <div class="row">
      <label class="muted"><input type="checkbox" id="showDev" /> Developer details</label>
    </div>
  </div>

  <div id="out"></div>

<script>
function escapeHtml(s){
  return (s||"")
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#039;");
}

function highlight(text, query){
  let safe = escapeHtml(text || "");
  const terms = (query||"").toLowerCase().match(/[a-z0-9_]+/g) || [];
  const uniq = [...new Set(terms)]
    .filter(t => t.length > 2)
    .slice(0, 12)
    .sort((a,b)=>b.length-a.length);

  for (const t of uniq){
    const re = new RegExp("\\\\b(" + t.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&') + ")\\\\b", "gi");
    safe = safe.replace(re, "<mark>$1</mark>");
  }
  return safe;
}

function fmt3(x){
  if (x === null || x === undefined) return "—";
  const n = Number(x);
  if (Number.isNaN(n)) return String(x);
  return n.toFixed(3);
}

async function ask(){
  const q = document.getElementById("q").value.trim();
  if(!q) return;

  document.getElementById("status").textContent = "Thinking...";
  document.getElementById("timing").style.display = "none";
  document.getElementById("out").innerHTML = "";

  let res, data;
  try {
    res = await fetch("/ask", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({query: q})
    });
    data = await res.json();
  } catch (e) {
    document.getElementById("status").textContent = "Network error";
    return;
  }

  if(!data.ok){
    document.getElementById("status").textContent = "Error: " + (data.error || "unknown");
    return;
  }

  document.getElementById("status").textContent = "";
  if (data.seconds !== undefined) {
    const t = document.getElementById("timing");
    t.textContent = `Response: ${data.seconds}s`;
    t.style.display = "inline-block";
  }

  // Answer
  const answerHtml = `
    <div class="card">
      <div class="muted"><b>Answer</b></div>
      <div class="answer" style="margin-top:8px;">${escapeHtml(data.answer)}</div>
    </div>
  `;

  // Why this answer? (collapsed)
  const whyThis = `
    <details class="card">
      <summary><b>Why this answer?</b></summary>
      <p style="margin-top:10px; line-height:1.5;">
        This answer was produced by retrieving the most relevant thesis sections using
        heading-aware chunking and semantic similarity search. The system prioritizes
        passages that best match your question and cites the original page ranges so you
        can verify the response in the document.
      </p>
      <p class="muted" style="margin-top:8px;">
        Tip: Expand “Sources & Evidence” to see the supporting excerpts.
      </p>
    </details>
  `;

  // Citations (collapsed) — uses /ask sources list only.
  // If you want highlighted excerpts here too, we can add snippets to /ask or call /ask_html instead.
  const sources = (data.sources || []);
  const citationsCards = sources.map((s, idx) => {
    const title = escapeHtml(s.title_path || "Untitled section");
    const pages = `pp. ${s.page_start}-${s.page_end}`;
    const scoreLine = `final=${fmt3(s.scores?.final)}, dense=${fmt3(s.scores?.dense)}, sparse=${fmt3(s.scores?.sparse)}, rerank=${fmt3(s.scores?.rerank)}`;

    return `
      <details class="card">
        <summary><b>${idx+1}.</b> ${title} (${pages})</summary>
        <div class="muted" style="margin-top:10px;">
          <b>Scores:</b> <code>${escapeHtml(scoreLine)}</code>
        </div>
        <div class="muted" style="margin-top:8px;">
          For highlighted excerpts, use the <code>/ask_html</code> endpoint or we can add snippet text into <code>/ask</code>.
        </div>
      </details>
    `;
  }).join("");

  const citationsSection = `
    <details class="card">
      <summary><b>Sources & Evidence</b> <span class="muted">(click to expand)</span></summary>
      <div style="margin-top:10px;">
        ${citationsCards || "<div class='muted'>No sources returned.</div>"}
      </div>
    </details>
  `;

  // Developer details (optional)
  const devOn = document.getElementById("showDev").checked;
  const devSection = devOn ? `
    <details class="card">
      <summary><b>Developer details</b></summary>
      <div class="grid" style="margin-top:10px;">
        <div>
          <div class="muted"><b>Explain</b></div>
          <pre style="white-space:pre-wrap;">${escapeHtml(JSON.stringify(data.explain || {}, null, 2))}</pre>
        </div>
      </div>
    </details>
  ` : "";

  document.getElementById("out").innerHTML = answerHtml + whyThis + citationsSection + devSection;
}
</script>
</body>
</html>
""")


@app.get("/")
def root():
    return {
        "service": "Explainable Thesis RAG Assistant",
        "endpoints": ["/ingest (POST)", "/ask (POST)"],
        "note": "Place your PDF at ./data/thesis.pdf or pass pdf_path to /ingest."
    }


@app.get("/debug_bm25")
def debug_bm25():
    return {
        "bm25_ready": BM25_MODEL is not None,
        "bm25_docs": len(BM25_DOC_IDS),
        "bm25_tokens": len(BM25_TOKENS),
        "sample_tokens": BM25_TOKENS[0][:25] if BM25_TOKENS else []
    }


# Rebuild BM25 on startup (if Chroma already has data)
try:
    rebuild_bm25_from_chroma()
except Exception:
    pass
