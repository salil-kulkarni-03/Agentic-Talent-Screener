"""
skills.py — Resume-to-JD Skills Matching Engine
================================================
Takes parsed output from main.py (UniversalResumeParser) and ranks
candidates against a job description using a 3-component hybrid score:

  • 60 % — Semantic similarity    (sentence-transformers + FAISS cosine)
  • 25 % — Keyword Jaccard        (skill token overlap via CountVectorizer)
  • 15 % — Experience TF-IDF      (sklearn TfidfVectorizer cosine)

Storage strategy (minimum footprint):
  • Embeddings cached on disk as float16 .npz  (keyed by 8-char SHA-256)
  • FAISS IndexFlatIP kept in RAM only — rebuilt cheaply from cache
  • Optional save_index / load_index for full binary persistence
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# ── Soft-import heavy libs with helpful errors ────────────────────────────────
def _require(pkg: str, pip: str | None = None):
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError:
        install = pip or pkg
        raise ImportError(
            f"Package '{pkg}' is required. Install it with:\n"
            f"  pip install {install}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-zA-Z0-9\+\#\-\.]{2,}")

def _tokenize(text: str) -> set[str]:
    """Lowercase tokens from text; used for Jaccard keyword overlap."""
    return {t.lower() for t in _TOKEN_RE.findall(text)}


def _doc_from_parsed(parsed: dict) -> str:
    """
    Build a single rich text document from a parsed resume dict.
    Priority order: skills > experience > education > raw_text (first 2000 chars).
    Returns '' for completely empty parsed dicts.
    """
    parts: list[str] = []

    # 1. Skills
    skills: dict = parsed.get("skills") or {}
    if skills.get("summary"):
        parts.append(skills["summary"])
    if skills.get("flat"):
        parts.append(" ".join(skills["flat"]))

    # 2. Experience
    for exp in parsed.get("experience") or []:
        s = exp.get("summary") or ""
        if s:
            parts.append(s)

    # 3. Education
    for edu in parsed.get("education") or []:
        s = edu.get("summary") or ""
        if s:
            parts.append(s)

    # 4. Raw text — truncated to 2000 chars (enough context, low overhead)
    raw = (parsed.get("raw_text") or "")[:2000]
    if raw:
        parts.append(raw)

    return " ".join(parts).strip()


def _content_hash(text: str) -> str:
    """8-char prefix of SHA-256 of the text — used as cache key."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:8]


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """In-place L2 normalisation; zeros are left as-is (avoids NaN)."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# SkillsMatcher
# ─────────────────────────────────────────────────────────────────────────────

_GLOBAL_MODEL_CACHE = {}


class SkillsMatcher:
    """
    Parameters
    ----------
    model_name : str
        Sentence-transformers model. Default 'all-MiniLM-L6-v2' (22 MB).
    cache_dir  : str | Path
        Directory for float16 .npz embedding cache.  Created automatically.
    weights    : tuple[float, float, float]
        (semantic, keyword, experience) weights — must sum to 1.0.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: str | Path = ".skill_cache",
        weights: tuple[float, float, float] = (0.60, 0.25, 0.15),
    ):
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {weights}")

        self.model_name = model_name
        self.cache_dir   = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.w_sem, self.w_kw, self.w_exp = weights

        # Lazy-loaded heavy objects
        self._model      = None   # SentenceTransformer
        self._faiss      = None   # faiss module
        self._index      = None   # faiss.IndexFlatIP
        self._tfidf      = None   # fitted TfidfVectorizer

        # Candidate store
        self._candidates: list[dict[str, Any]] = []  # raw parsed dicts
        self._docs:       list[str]            = []  # rich text per candidate
        self._embs:       np.ndarray | None    = None  # (N, D) float32 normalised
        self._exp_texts:  list[str]            = []  # experience blob per candidate

    # ── Private: lazy loaders ────────────────────────────────────────────────

    def _load_model(self):
        global _GLOBAL_MODEL_CACHE
        if self._model is None:
            if self.model_name not in _GLOBAL_MODEL_CACHE:
                SentenceTransformer = _require(
                    "sentence_transformers", "sentence-transformers"
                ).SentenceTransformer
                warnings.filterwarnings("ignore", category=FutureWarning)
                
                # Limit PyTorch CPU threads to 1 to reduce RAM overhead on low-memory servers!
                try:
                    import torch
                    torch.set_num_threads(1)
                    torch.set_num_interop_threads(1)
                except Exception:
                    pass

                _GLOBAL_MODEL_CACHE[self.model_name] = SentenceTransformer(self.model_name)
            self._model = _GLOBAL_MODEL_CACHE[self.model_name]
        return self._model

    def _load_faiss(self):
        if self._faiss is None:
            self._faiss = _require("faiss", "faiss-cpu")
        return self._faiss

    # ── Private: embedding cache ─────────────────────────────────────────────

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.npz"

    def _load_cached(self, key: str) -> np.ndarray | None:
        p = self._cache_path(key)
        if p.exists():
            try:
                data = np.load(p)
                return data["emb"].astype(np.float32)
            except Exception:
                p.unlink(missing_ok=True)
        return None

    def _save_cached(self, key: str, emb: np.ndarray):
        p = self._cache_path(key)
        np.savez_compressed(p, emb=emb.astype(np.float16))

    # ── Private: encode docs (with per-doc cache) ────────────────────────────

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Return (N, D) float32 normalised embeddings for texts."""
        model = self._load_model()
        results: list[np.ndarray] = []
        to_encode_idx: list[int]  = []
        to_encode_txt: list[str]  = []

        # Check cache per-document
        for i, text in enumerate(texts):
            key = _content_hash(text) if text.strip() else "__blank__"
            cached = self._load_cached(key)
            if cached is not None:
                results.append((i, cached))
            else:
                to_encode_idx.append(i)
                to_encode_txt.append(text if text.strip() else " ")

        # Batch encode cache misses
        if to_encode_txt:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                batch_embs = model.encode(
                    to_encode_txt,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                ).astype(np.float32)

            for local_i, (global_i, text) in enumerate(
                zip(to_encode_idx, to_encode_txt)
            ):
                emb = batch_embs[local_i]
                key = _content_hash(texts[global_i]) if texts[global_i].strip() else "__blank__"
                self._save_cached(key, emb)
                results.append((global_i, emb))

        # Re-sort by original index
        results.sort(key=lambda x: x[0])
        matrix = np.stack([r[1] for r in results], axis=0)
        return _l2_normalize(matrix)

    # ── Private: build/rebuild FAISS index ──────────────────────────────────

    def _build_index(self, embs: np.ndarray):
        faiss = self._load_faiss()
        dim   = embs.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product == cosine after L2-norm
        index.add(embs)
        self._index = index

    # ── Public API ───────────────────────────────────────────────────────────

    def add_candidates(self, parsed_list: list[dict]) -> None:
        """
        Ingest parsed resume dicts from UniversalResumeParser.
        Deduplicates by (name, content hash).  Can be called multiple times
        to incrementally add candidates.

        Parameters
        ----------
        parsed_list : list[dict]
            Each dict must be the direct output of UniversalResumeParser.parse().
        """
        if not parsed_list:
            return

        seen_keys: set[str] = set()
        # Build dedup set from existing candidates
        for cand in self._candidates:
            doc = _doc_from_parsed(cand)
            seen_keys.add(f"{cand.get('name','')}::{_content_hash(doc)}")

        new_parsed: list[dict] = []
        new_docs:   list[str]  = []
        new_exp:    list[str]  = []

        for p in parsed_list:
            if not isinstance(p, dict):
                continue
            doc = _doc_from_parsed(p)
            key = f"{p.get('name', '')}::{_content_hash(doc)}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            exp_blob = " ".join(
                e.get("summary", "") for e in (p.get("experience") or [])
            )
            new_parsed.append(p)
            new_docs.append(doc)
            new_exp.append(exp_blob)

        if not new_parsed:
            return  # all duplicates

        # Encode new candidates
        new_embs = self._encode(new_docs)

        # Merge with existing
        self._candidates.extend(new_parsed)
        self._docs.extend(new_docs)
        self._exp_texts.extend(new_exp)

        if self._embs is None:
            self._embs = new_embs
        else:
            self._embs = np.vstack([self._embs, new_embs])

        # Rebuild FAISS index (cheap — IndexFlatIP just wraps the matrix)
        self._build_index(self._embs)

        # Invalidate TF-IDF (will be refit on next rank() call)
        self._tfidf = None

    def rank(
        self,
        job_description: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Rank candidates against a job description.

        Parameters
        ----------
        job_description : str  — Free-form JD text.
        top_k           : int  — Number of top candidates to return (default 10).
                                 If larger than candidate count, all are returned.

        Returns
        -------
        list of dicts, sorted by composite_score descending:
        [
          {
            "rank":            1,
            "name":            "Jane Doe",
            "composite_score": 0.812,   # 0–1
            "semantic_score":  0.891,
            "keyword_score":   0.654,
            "experience_score":0.421,
            "skills_matched":  ["Python", "PyTorch", ...],
            "skills_missing":  ["Kubernetes", ...],
          },
          ...
        ]
        """
        # ── Validation ────────────────────────────────────────────────────────
        if not self._candidates:
            raise ValueError(
                "No candidates loaded. Call add_candidates() first."
            )
        jd = (job_description or "").strip()
        if not jd:
            raise ValueError("job_description must not be empty.")

        n_cands = len(self._candidates)
        k = min(top_k, n_cands)

        # ── 1. Semantic score (FAISS) ─────────────────────────────────────────
        jd_emb = self._encode([jd])                  # (1, D) normalised
        faiss  = self._load_faiss()
        # Search all candidates (top-k from FAISS)
        sims, indices = self._index.search(jd_emb, k)  # (1, k)
        sem_scores_topk = np.clip(sims[0], 0.0, 1.0)   # inner product ≡ cosine
        topk_indices    = indices[0]

        # ── 2. Keyword Jaccard score ──────────────────────────────────────────
        jd_tokens = _tokenize(jd)
        kw_scores = np.zeros(n_cands, dtype=np.float32)
        for i, parsed in enumerate(self._candidates):
            skills_flat = (parsed.get("skills") or {}).get("flat") or []
            cand_tokens = _tokenize(" ".join(skills_flat)) | _tokenize(self._docs[i])
            if jd_tokens and (jd_tokens | cand_tokens):
                kw_scores[i] = len(jd_tokens & cand_tokens) / len(jd_tokens | cand_tokens)

        # ── 3. Experience TF-IDF score ────────────────────────────────────────
        exp_scores = np.zeros(n_cands, dtype=np.float32)
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            # Fit TF-IDF on experience texts + JD
            corpus = self._exp_texts + [jd]
            # Lazy refit when candidates change
            if self._tfidf is None:
                self._tfidf = TfidfVectorizer(
                    min_df=1,
                    max_features=8000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                )
                self._tfidf.fit(corpus)
            exp_matrix = self._tfidf.transform(self._exp_texts)
            jd_vec     = self._tfidf.transform([jd])
            sims_exp   = cosine_similarity(exp_matrix, jd_vec).flatten()
            exp_scores = np.clip(sims_exp.astype(np.float32), 0.0, 1.0)
        except ImportError:
            pass  # scikit-learn unavailable — experience component is 0

        # ── 4. Composite scores & JD skill gap ────────────────────────────────
        jd_skill_tokens = _tokenize(jd)

        results: list[dict[str, Any]] = []
        seen_topk = set(topk_indices.tolist())

        # Build result for FAISS top-k
        for local_rank, cand_idx in enumerate(topk_indices):
            _append_result(
                results, cand_idx, local_rank,
                sem_scores_topk[local_rank],
                kw_scores[cand_idx],
                exp_scores[cand_idx],
                self._candidates[cand_idx],
                jd_skill_tokens,
                self.w_sem, self.w_kw, self.w_exp,
            )

        # Sort by composite score
        results.sort(key=lambda x: x["composite_score"], reverse=True)

        # Re-assign ranks after final sort
        for r_i, r in enumerate(results):
            r["rank"] = r_i + 1

        return results

    # ── Optional persistence ─────────────────────────────────────────────────

    def save_index(self, path: str | Path) -> None:
        """
        Serialize the FAISS index + candidate metadata to a single .npz file.
        Useful for instant startup without re-encoding.
        """
        if self._index is None or self._embs is None:
            raise RuntimeError("Nothing to save — add candidates first.")
        faiss = self._load_faiss()
        path  = Path(path)
        index_bytes = faiss.serialize_index(self._index)
        meta = [
            {
                "name":       c.get("name", ""),
                "skills_flat": (c.get("skills") or {}).get("flat") or [],
                "doc":        self._docs[i],
                "exp":        self._exp_texts[i],
                "parsed":     c,
            }
            for i, c in enumerate(self._candidates)
        ]
        np.savez_compressed(
            path,
            embs        = self._embs.astype(np.float16),
            index_bytes = np.frombuffer(index_bytes, dtype=np.uint8),
            meta_json   = np.array([json.dumps(meta, default=str)]),
        )

    def load_index(self, path: str | Path) -> None:
        """Load a previously saved index. Replaces current state."""
        faiss = self._load_faiss()
        path  = Path(path)
        data  = np.load(path, allow_pickle=False)
        meta  = json.loads(str(data["meta_json"][0]))
        index_bytes = data["index_bytes"].tobytes()

        self._embs      = data["embs"].astype(np.float32)
        self._index     = faiss.deserialize_index(index_bytes)
        self._candidates= [m["parsed"] for m in meta]
        self._docs      = [m["doc"]    for m in meta]
        self._exp_texts = [m["exp"]    for m in meta]
        self._tfidf     = None  # will be refit on next rank()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers (outside class — no self needed)
# ─────────────────────────────────────────────────────────────────────────────

def _append_result(
    results:         list,
    cand_idx:        int,
    local_rank:      int,
    sem_score:       float,
    kw_score:        float,
    exp_score:       float,
    parsed:          dict,
    jd_skill_tokens: set,
    w_sem:           float,
    w_kw:            float,
    w_exp:           float,
):
    composite = float(
        w_sem * sem_score +
        w_kw  * kw_score  +
        w_exp * exp_score
    )

    skills_flat = (parsed.get("skills") or {}).get("flat") or []
    cand_tokens = {s.lower() for s in skills_flat}
    matched = sorted(
        s for s in skills_flat if s.lower() in jd_skill_tokens
    )
    missing = sorted(
        t for t in jd_skill_tokens
        if t not in cand_tokens and len(t) > 3
    )[:10]  # cap at 10 missing skills shown

    res = {
        "rank":             local_rank + 1,       # will be re-assigned after sort
        "name":             parsed.get("name", "Unknown"),
        "composite_score":  round(composite,  4),
        "semantic_score":   round(float(sem_score), 4),
        "keyword_score":    round(float(kw_score),  4),
        "experience_score": round(float(exp_score), 4),
        "skills_matched":   matched,
        "skills_missing":   missing,
    }
    if "file_hash" in parsed:
        res["file_hash"] = parsed["file_hash"]
    results.append(res)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def rank_candidates(
    parsed_list:     list[dict],
    job_description: str,
    top_k:           int = 10,
    model_name:      str = "all-MiniLM-L6-v2",
    cache_dir:       str = ".skill_cache",
    weights:         tuple[float, float, float] = (0.60, 0.25, 0.15),
) -> list[dict]:
    """
    One-shot convenience function.

    Example
    -------
    from main import UniversalResumeParser
    from skills import rank_candidates

    parser = UniversalResumeParser()
    parsed = [parser.parse(f) for f in resume_files]
    results = rank_candidates(parsed, job_description="We need a Python ML engineer...")
    for r in results:
        print(r["rank"], r["name"], r["composite_score"])
    """
    matcher = SkillsMatcher(
        model_name=model_name,
        cache_dir=cache_dir,
        weights=weights,
    )
    matcher.add_candidates(parsed_list)
    return matcher.rank(job_description, top_k=top_k)


# ─────────────────────────────────────────────────────────────────────────────
# __main__ — Demo / edge-case test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # ── Try to parse real resumes (if they exist) ─────────────────────────────
    try:
        from main import UniversalResumeParser
        parser = UniversalResumeParser()
        _base  = Path(__file__).parent

        resume_files = [
            _base / "Unstructured_Example.pdf",
            _base / "Salil Kulkarni Resume.pdf",
            _base / "Salil Resume.png",
            _base / "Tesseract_Example.png",
        ]
        parsed_list = []
        for rf in resume_files:
            if rf.exists():
                try:
                    result = parser.parse(str(rf))
                    parsed_list.append(result)
                    print(f"✅ Parsed: {rf.name}  →  {result['name']}")
                except Exception as e:
                    print(f"⚠️  Skipped {rf.name}: {e}")

    except ImportError as e:
        print(f"Could not import UniversalResumeParser: {e}")
        parsed_list = []

    # ── Inject synthetic candidates so demo always works ─────────────────────
    synthetic = [
        {
            "name": "Alice ML Engineer",
            "skills": {
                "flat": ["Python", "PyTorch", "TensorFlow", "scikit-learn",
                         "Docker", "AWS", "FastAPI", "pandas", "numpy"],
                "structured": {"Languages": ["Python"], "Frameworks": ["PyTorch", "TensorFlow"]},
                "summary": "Languages: Python | Frameworks: PyTorch, TensorFlow | Cloud: AWS",
            },
            "experience": [
                {"summary": "DeepMind | ML Research Engineer | 2022–Present"},
                {"summary": "Google Brain | Software Engineer Intern | 2021"},
            ],
            "education": [{"summary": "IIT Bombay | 2022 | M.Tech Computer Science | CGPA: 9.2"}],
            "raw_text":  "Experienced ML engineer with 3+ years in deep learning, NLP, and MLOps.",
        },
        {
            "name": "Bob Full Stack Dev",
            "skills": {
                "flat": ["JavaScript", "React", "Node.js", "MongoDB", "Docker", "CSS", "HTML"],
                "structured": {"Languages": ["JavaScript"], "Frontend": ["React"]},
                "summary": "Languages: JavaScript | Frontend: React | Backend: Node.js",
            },
            "experience": [
                {"summary": "Startup XYZ | Full Stack Developer | 2021–2023"},
            ],
            "education": [{"summary": "BITS Pilani | 2021 | B.Tech CSE | CGPA: 8.1"}],
            "raw_text":  "Full stack developer specialising in React and Node.js web applications.",
        },
        {
            "name": "Carol Data Scientist",
            "skills": {
                "flat": ["Python", "R", "pandas", "scikit-learn", "Tableau",
                         "SQL", "PostgreSQL", "Spark", "Airflow"],
                "structured": {"Languages": ["Python", "R"], "BI Tools": ["Tableau"]},
                "summary": "Languages: Python, R | BI: Tableau | Databases: PostgreSQL",
            },
            "experience": [
                {"summary": "Analytics Corp | Data Scientist | 2020–2023"},
                {"summary": "University Research | RA | 2019–2020"},
            ],
            "education": [{"summary": "NIT Trichy | 2020 | B.Tech EE | CGPA: 8.8"}],
            "raw_text":  "Data scientist with expertise in statistical modelling and business analytics.",
        },
        # Edge-case: blank resume
        {
            "name": "Ghost Blank",
            "skills": {"flat": [], "structured": {}, "summary": ""},
            "experience": [],
            "education":  [],
            "raw_text":   "",
        },
    ]
    parsed_list.extend(synthetic)

    JD = """
    We are looking for a Web Developer with hands-on experience in HTML, CSS, JavaScript, and React.js. The ideal candidate should be comfortable working with backend technologies like Node.js or Django, databases such as MySQL or MongoDB, and version control using Git and GitHub. Familiarity with REST APIs, responsive design using Figma or Bootstrap, and deployment on platforms like Vercel or AWS is expected. A Bachelor's degree in Computer Science or a related field is preferred.
    """

    print("\n" + "=" * 65)
    print("🔍  Matching candidates against JD...")
    print("=" * 65)

    # ── Test edge case: empty candidate list ──────────────────────────────────
    try:
        SkillsMatcher().rank(JD)
    except ValueError as e:
        print(f"✅ Edge case (no candidates): {e}")

    # ── Test edge case: empty JD ──────────────────────────────────────────────
    try:
        m = SkillsMatcher()
        m.add_candidates(parsed_list)
        m.rank("")
    except ValueError as e:
        print(f"✅ Edge case (empty JD): {e}")

    # ── Main ranking ──────────────────────────────────────────────────────────
    matcher = SkillsMatcher(cache_dir=".skill_cache")
    matcher.add_candidates(parsed_list)

    # top_k > candidate count — should not crash
    results = matcher.rank(JD, top_k=100)

    # ── Pretty print ──────────────────────────────────────────────────────────
    print(f"\n{'Rank':<5} {'Name':<28} {'Score':>7}  {'Semantic':>9}  {'Keyword':>8}  {'Exp TF-IDF':>10}")
    print("-" * 75)
    for r in results:
        print(
            f"{r['rank']:<5} {r['name']:<28} {r['composite_score']:>7.4f}"
            f"  {r['semantic_score']:>9.4f}  {r['keyword_score']:>8.4f}"
            f"  {r['experience_score']:>10.4f}"
        )

    print()
    top = results[0]
    print(f"🏆 Top candidate: {top['name']}")
    print(f"   Skills matched : {top['skills_matched'] or '(none)'}")
    print(f"   Skills missing  : {top['skills_missing'] or '(none)'}")

    # ── Duplicate add — should not double-count ───────────────────────────────
    before = len(matcher._candidates)
    matcher.add_candidates(synthetic)
    after  = len(matcher._candidates)
    assert after == before, f"Duplicate dedup failed: {before} → {after}"
    print(f"\n✅ Duplicate detection: {before} candidates before re-add, {after} after (correct)")

    print("\n✅ All checks passed.")
