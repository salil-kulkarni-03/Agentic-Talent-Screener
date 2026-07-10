"""
ml.py — Explainable Hiring Prediction Engine
=============================================
Stage 3 of the AI Recruiter pipeline:
  main.py  → parse resumes (PDF, PNG, JPG, HTML, DOCX)
  skills.py → rank candidates (semantic + keyword + TF-IDF)
  ml.py    → predict hire probability (LightGBM + SMOTE + SHAP)

Feature set
-----------
  7-feature base mode  :  skills.py scores only (backward-compat, cold-start)
  15-feature rich mode :  +8 features from main.py parsed dicts
      total_exp_years, recent_exp_years, job_hops,
      edu_gpa, tech_keyword_count, certifications_count,
      projects_count, skills_count

Three-tier output (document best practice)
-------------------------------------------
  prob >= 0.70  →  [INTERVIEW]
  prob 0.40-0.69→  [PHONE SCREEN]
  prob <  0.40  →  [AUTO-REJECT]
"""

from __future__ import annotations

import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


# ── Soft-import helpers ───────────────────────────────────────────────────────

def _try_import(pkg: str):
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError:
        return None


def _require(pkg: str, pip: str | None = None):
    m = _try_import(pkg)
    if m is None:
        raise ImportError(
            f"Package '{pkg}' is required.  Install with:\n"
            f"  pip install {pip or pkg}"
        )
    return m


# ── Feature schemas ───────────────────────────────────────────────────────────

_FEATURE_COLS_BASE = [
    "composite_score",
    "semantic_score",
    "keyword_score",
    "experience_score",
    "skills_matched_count",
    "skills_missing_count",
    "skills_match_ratio",
]

_FEATURE_COLS_RICH_EXTRA = [
    "total_exp_years",
    "recent_exp_years",
    "job_hops",
    "edu_gpa",
    "tech_keyword_count",
    "certifications_count",
    "projects_count",
    "skills_count",
]

_FEATURE_COLS_RICH = _FEATURE_COLS_BASE + _FEATURE_COLS_RICH_EXTRA

_MIN_TRAIN_SAMPLES  = 10
_MIN_POSITIVE_RATIO = 0.05

# ── Decision thresholds ───────────────────────────────────────────────────────

_TIER_INTERVIEW    = 0.54
_TIER_PHONE_SCREEN = 0.40


def _make_decision(prob: float) -> str:
    if prob >= _TIER_INTERVIEW:
        return "[INTERVIEW]"
    if prob >= _TIER_PHONE_SCREEN:
        return "[PHONE SCREEN]"
    return "[AUTO-REJECT]"


# ─────────────────────────────────────────────────────────────────────────────
# Rich feature extraction helpers (from main.py parsed dicts)
# ─────────────────────────────────────────────────────────────────────────────

_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

_CERT_KW = {
    "aws", "azure", "gcp", "google cloud", "kubernetes", "docker certified",
    "terraform", "cisco", "comptia",
}

_TECH_KW = {
    "python", "java", "javascript", "typescript", "c++", "c#", "go",
    "rust", "sql", "nosql", "react", "angular", "vue", "node", "django",
    "flask", "fastapi", "docker", "kubernetes", "aws", "gcp", "azure",
    "spark", "kafka", "airflow", "tensorflow", "pytorch", "scikit",
    "pandas", "numpy", "git", "linux", "mongodb", "postgresql", "redis",
}


def _parse_exp_years(parsed: dict) -> tuple[float, float]:
    """
    Return (total_exp_years, recent_exp_years).
    Uses pre-extracted LLM values if available, otherwise falls back to regex.
    """
    # Try reading LLM-extracted values first
    llm_total = parsed.get("total_experience_years")
    llm_recent = parsed.get("recent_experience_years")
    
    if llm_total is not None and llm_recent is not None:
        try:
            return round(min(float(llm_total), 40.0), 2), round(min(float(llm_recent), 10.0), 2)
        except (ValueError, TypeError):
            pass

    # Fallback to local regex-based parsing
    now_year = datetime.now().year
    total  = 0.0
    recent = 0.0
    for exp in (parsed.get("experience") or []):
        summary = str(exp.get("summary") or "")
        years   = _YEAR_RE.findall(summary)
        if len(years) >= 2:
            try:
                start, end = int(years[0]), int(years[-1])
                if start > end:
                    start, end = end, start
                end = min(end, now_year)
                total  += max(0, end - start)
                recent += max(0, min(end, now_year) - max(start, now_year - 3))
            except (ValueError, OverflowError):
                pass
        elif len(years) == 1:
            total += 1.0   # single year → assume ~1 yr
    return round(min(total, 40.0), 2), round(min(recent, 10.0), 2)


def _parse_edu_gpa(parsed: dict) -> float:
    """
    Extract highest GPA/CGPA/percentage normalised to 0-10.
    Uses pre-extracted LLM values if available, otherwise falls back to regex.
    """
    # Try reading LLM-extracted normalized GPA first
    llm_gpa = parsed.get("normalized_gpa_scaled_to_10")
    if llm_gpa is not None:
        try:
            return round(min(float(llm_gpa), 10.0), 2)
        except (ValueError, TypeError):
            pass

    # Fallback to local regex-based parsing
    best    = 0.0
    gpa_re  = re.compile(r"(?:cgpa|gpa)[^\d]*([0-9]+\.?[0-9]*)", re.IGNORECASE)
    pct_re  = re.compile(r"([0-9]+\.?[0-9]*)\s*%")
    for edu in (parsed.get("education") or []):
        summary = str(edu.get("summary") or "")
        m = gpa_re.search(summary)
        if m:
            val = float(m.group(1))
            # If val looks like it's on a 4.0 scale, convert
            best = max(best, min(val * 2.5 if val <= 4.0 else val, 10.0))
            continue
        m = pct_re.search(summary)
        if m:
            best = max(best, min(float(m.group(1)) / 10.0, 10.0))
    return round(best, 2)


def _count_tech_keywords(parsed: dict) -> int:
    """Count how many standard tech keywords appear in skills + raw text."""
    skills_flat = (parsed.get("skills") or {}).get("flat") or []
    raw  = str(parsed.get("raw_text") or "")
    text = " ".join(skills_flat).lower() + " " + raw.lower()
    return sum(1 for kw in _TECH_KW if kw in text)


def _count_certifications(parsed: dict) -> int:
    """Count cloud/tech certification mentions."""
    skills_flat = (parsed.get("skills") or {}).get("flat") or []
    raw  = str(parsed.get("raw_text") or "")
    text = " ".join(skills_flat).lower() + " " + raw.lower()
    return sum(1 for kw in _CERT_KW if kw in text)


def _count_projects(parsed: dict) -> int:
    """Count explicit projects or 'project' keyword occurrences (capped at 10)."""
    projs = parsed.get("projects") or []
    if projs:
        return min(len(projs), 10)
    raw = str(parsed.get("raw_text") or "").lower()
    return min(raw.count("project"), 10)


def _count_skills(parsed: dict) -> int:
    """Count items in the flat skills list."""
    return len((parsed.get("skills") or {}).get("flat") or [])


def _extract_rich_row(parsed: dict) -> list[float]:
    """Return list of 8 rich feature values for one parsed resume dict."""
    total_exp, recent_exp = _parse_exp_years(parsed)
    job_hops = len(parsed.get("experience") or [])
    return [
        total_exp,
        recent_exp,
        float(job_hops),
        _parse_edu_gpa(parsed),
        float(_count_tech_keywords(parsed)),
        float(_count_certifications(parsed)),
        float(_count_projects(parsed)),
        float(_count_skills(parsed)),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Feature matrix builder
# ─────────────────────────────────────────────────────────────────────────────

def build_features(
    ranked_results: list[dict],
    parsed_list:    list[dict] | None = None,
) -> np.ndarray:
    """
    Build feature matrix from ranked candidates.

    Parameters
    ----------
    ranked_results : list[dict]
        Output rows from SkillsMatcher.rank().
    parsed_list    : list[dict] | None
        Original parsed dicts from UniversalResumeParser, aligned 1-to-1
        with ranked_results. Enables 15-feature mode; otherwise 7.

    Returns
    -------
    np.ndarray of shape (N, 7) or (N, 15), dtype float32.
    """
    rows = []
    for i, r in enumerate(ranked_results):
        matched = len(r.get("skills_matched") or [])
        missing = len(r.get("skills_missing") or [])
        total   = matched + missing
        base = [
            r.get("composite_score",  np.nan),
            r.get("semantic_score",   np.nan),
            r.get("keyword_score",    np.nan),
            r.get("experience_score", np.nan),
            float(matched),
            float(missing),
            float(matched / total) if total > 0 else 0.0,
        ]
        if parsed_list is not None and i < len(parsed_list):
            try:
                base += _extract_rich_row(parsed_list[i])
            except Exception:
                base += [0.0] * len(_FEATURE_COLS_RICH_EXTRA)
        rows.append(base)
    return np.array(rows, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# HiringPredictor
# ─────────────────────────────────────────────────────────────────────────────

class HiringPredictor:
    """
    Explainable hire-probability predictor.

    Parameters
    ----------
    threshold   : float  — Minimum probability for PHONE SCREEN (default 0.40).
    smote_ratio : float  — SMOTE target minority/majority ratio (default 0.30).
    n_estimators: int    — LightGBM trees (default 300).
    model_path  : path   — If given, auto-save after fit() and load from here.
    """

    def __init__(
        self,
        threshold:    float = 0.40,
        smote_ratio:  float = 0.30,
        n_estimators: int   = 300,
        model_path:   str | Path | None = None,
    ):
        self.threshold    = threshold
        self.smote_ratio  = smote_ratio
        self.n_estimators = n_estimators
        self.model_path   = Path(model_path) if model_path else None

        self._pipeline       = None
        self._shap_explainer = None
        self._trained        = False
        self._rich_mode      = False       # True when 15-feature mode
        self._X_train        = None        # kept for SHAP background
        self._feature_cols   = _FEATURE_COLS_BASE

    # ── Internal pipeline builder ─────────────────────────────────────────────

    def _build_pipeline(self, X: np.ndarray, y: np.ndarray):
        """Build and fit: Imputer -> Scaler -> (SMOTE) -> LightGBM."""
        from sklearn.pipeline      import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute        import SimpleImputer

        LGBMClassifier = _require("lightgbm", "lightgbm").LGBMClassifier

        n_samples   = len(y)
        n_positives = int(y.sum())
        n_negatives = n_samples - n_positives
        pos_ratio   = n_positives / n_samples if n_samples > 0 else 0.0

        if pos_ratio < _MIN_POSITIVE_RATIO:
            warnings.warn(
                f"Only {pos_ratio:.1%} positive labels in training data. "
                "Collect more 'hired' examples for reliable predictions.",
                UserWarning, stacklevel=3,
            )

        # SMOTE (gracefully skipped if imblearn not installed or data too small)
        use_smote  = False
        smote_step = None
        if n_samples >= _MIN_TRAIN_SAMPLES and n_positives >= 2 and n_negatives >= 2:
            imblearn = _try_import("imblearn")
            if imblearn:
                from imblearn.over_sampling import SMOTE
                k = min(3, n_positives - 1, n_negatives - 1)
                if k >= 1:
                    smote_step = SMOTE(
                        sampling_strategy=min(self.smote_ratio, 1.0),
                        k_neighbors=k,
                        random_state=42,
                    )
                    use_smote = True

        scale_pw = (n_negatives / n_positives) if (not use_smote and n_positives > 0) else 1.0

        lgbm = LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=20,
            max_depth=-1,
            scale_pos_weight=scale_pw,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        )

        preprocess = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]

        if use_smote:
            from imblearn.pipeline import Pipeline as ImbPipeline
            pipeline = ImbPipeline(preprocess + [("smote", smote_step), ("model", lgbm)])
        else:
            pipeline = Pipeline(preprocess + [("model", lgbm)])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X, y)
        return pipeline

    # ── SHAP ─────────────────────────────────────────────────────────────────

    def _get_explainer(self, X_background: np.ndarray):
        if self._shap_explainer is not None:
            return self._shap_explainer
        shap_mod = _try_import("shap")
        if shap_mod is None:
            return None
        try:
            lgbm_model = self._pipeline.named_steps["model"]
            X_trans    = _transform_until_model(self._pipeline, X_background)
            self._shap_explainer = shap_mod.TreeExplainer(
                lgbm_model,
                data=shap_mod.sample(X_trans, min(50, len(X_trans))),
                feature_names=self._feature_cols,
            )
        except Exception as e:
            warnings.warn(f"SHAP explainer could not be built: {e}", stacklevel=2)
            return None
        return self._shap_explainer

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        ranked_results: list[dict],
        labels:         list[int],
        parsed_list:    list[dict] | None = None,
    ) -> "HiringPredictor":
        """
        Train on historical ranked candidates.

        Parameters
        ----------
        ranked_results : Output of SkillsMatcher.rank().
        labels         : Binary hire label per candidate (1 = hired, 0 = not).
        parsed_list    : Optional main.py parsed dicts (enables 15-feature mode).
        """
        if len(ranked_results) != len(labels):
            raise ValueError(
                f"ranked_results and labels must have same length "
                f"(got {len(ranked_results)} vs {len(labels)})."
            )
        if not ranked_results:
            raise ValueError("ranked_results is empty — nothing to train on.")

        self._rich_mode    = bool(parsed_list and len(parsed_list) > 0)
        self._feature_cols = _FEATURE_COLS_RICH if self._rich_mode else _FEATURE_COLS_BASE

        X = build_features(ranked_results, parsed_list if self._rich_mode else None)
        y = np.array(labels, dtype=np.int32)

        if len(np.unique(y)) < 2:
            warnings.warn(
                "Labels contain only one class. "
                "Provide both hired (1) and rejected (0) examples.",
                UserWarning, stacklevel=2,
            )

        self._pipeline       = self._build_pipeline(X, y)
        self._shap_explainer = None
        self._X_train        = X
        self._trained        = True

        if self.model_path:
            self.save(self.model_path)

        return self

    def predict_proba(
        self,
        ranked_results: list[dict],
        parsed_list:    list[dict] | None = None,
    ) -> np.ndarray:
        """Return hire probabilities (shape N,). Requires fit() first."""
        if not self._trained:
            raise RuntimeError("Call fit() before predict_proba().")

        use_parsed = parsed_list if (self._rich_mode and parsed_list) else None
        X = build_features(ranked_results, use_parsed)

        # If rich mode but no parsed_list provided, pad extra cols with zeros
        if self._rich_mode and use_parsed is None and X.shape[1] == len(_FEATURE_COLS_BASE):
            extra = np.zeros((X.shape[0], len(_FEATURE_COLS_RICH_EXTRA)), dtype=np.float32)
            X = np.hstack([X, extra])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._pipeline.predict_proba(X)[:, 1].astype(np.float32)

    def explain(
        self,
        ranked_results: list[dict],
        candidate_idx:  int = 0,
        parsed_list:    list[dict] | None = None,
    ) -> dict[str, float] | None:
        """Return SHAP feature contributions for one candidate, or None if unavailable."""
        if not self._trained:
            return None
        use_parsed = parsed_list if (self._rich_mode and parsed_list) else None
        X = build_features(ranked_results, use_parsed)
        if self._rich_mode and use_parsed is None and X.shape[1] == len(_FEATURE_COLS_BASE):
            extra = np.zeros((X.shape[0], len(_FEATURE_COLS_RICH_EXTRA)), dtype=np.float32)
            X = np.hstack([X, extra])
        explainer = self._get_explainer(self._X_train)
        if explainer is None:
            return None
        try:
            X_trans = _transform_until_model(self._pipeline, X[candidate_idx:candidate_idx + 1])
            sv = explainer.shap_values(X_trans)
            if isinstance(sv, list):
                sv = sv[1]
            return dict(zip(self._feature_cols, sv[0].tolist()))
        except Exception as e:
            warnings.warn(f"SHAP explanation failed: {e}", stacklevel=2)
            return None

    def report(
        self,
        ranked_results: list[dict],
        top_k:          int = 10,
        parsed_list:    list[dict] | None = None,
        labels:         list[int] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generate full per-candidate hiring report.

        Returns list of dicts sorted by hire_probability (descending):
            rank, name, hire_probability, decision,
            composite_score, semantic_score, keyword_score, experience_score,
            skills_matched, skills_missing, shap_explanation, true_label
        """
        if not ranked_results:
            return []

        results  = ranked_results[:top_k] if top_k > 0 else ranked_results
        p_slice  = (parsed_list[:top_k] if parsed_list and top_k > 0 else parsed_list)

        if self._trained:
            probs = self.predict_proba(results, p_slice)
        else:
            probs = np.array(
                [r.get("composite_score", 0.0) for r in results], dtype=np.float32
            )

        order  = np.argsort(probs)[::-1]
        report = []
        for new_rank, idx in enumerate(order):
            r    = results[int(idx)]
            prob = float(probs[int(idx)])

            shap_exp = None
            if self._trained:
                shap_exp = self.explain(results, int(idx), p_slice)

            label = None
            if labels is not None and int(idx) < len(labels):
                label = int(labels[int(idx)])

            report.append({
                "rank":              new_rank + 1,
                "name":              r.get("name", "Unknown"),
                "hire_probability":  round(prob, 4),
                "decision":          _make_decision(prob),
                "composite_score":   r.get("composite_score",  0.0),
                "semantic_score":    r.get("semantic_score",   0.0),
                "keyword_score":     r.get("keyword_score",    0.0),
                "experience_score":  r.get("experience_score", 0.0),
                "skills_matched":    r.get("skills_matched",   []),
                "skills_missing":    r.get("skills_missing",   []),
                "shap_explanation":  shap_exp,
                "true_label":        label,
            })

        return report

    def print_report(self, report: list[dict], verbose: bool = True) -> None:
        """Pretty-print the hiring report."""
        if not report:
            print("No candidates to report.")
            return

        interview   = [r for r in report if r["hire_probability"] >= _TIER_INTERVIEW]
        phone       = [r for r in report if _TIER_PHONE_SCREEN <= r["hire_probability"] < _TIER_INTERVIEW]
        auto_reject = [r for r in report if r["hire_probability"] < _TIER_PHONE_SCREEN]

        mode_str = (
            f"{'15-feature rich' if self._rich_mode else '7-feature base'} mode"
            if self._trained else "cold-start (composite score — no ML model)"
        )

        print("\n" + "=" * 76)
        print("  HIRING PREDICTION REPORT")
        print(f"  Mode         : {mode_str}")
        print(f"  Interview    : {len(interview)}  |  "
              f"Phone Screen : {len(phone)}  |  Auto-Reject : {len(auto_reject)}")
        print("=" * 76)

        header = (
            f"{'Rk':<4} {'Name':<26} {'Prob':>6}  {'Decision':<14}"
            f"  {'Composite':>9}  {'Semantic':>8}  {'Keyword':>7}  {'Exp':>6}"
        )
        print(header)
        print("-" * 76)

        for r in report:
            tl = " (H)" if r["true_label"] == 1 else (" (R)" if r["true_label"] == 0 else "")
            print(
                f"{r['rank']:<4} {(r['name'] + tl):<26}"
                f" {r['hire_probability']:>6.1%}  {r['decision']:<14}"
                f"  {r['composite_score']:>9.4f}"
                f"  {r['semantic_score']:>8.4f}"
                f"  {r['keyword_score']:>7.4f}"
                f"  {r['experience_score']:>6.4f}"
            )

        if verbose:
            print()
            for r in report:
                if r["shap_explanation"]:
                    print(f"\n  SHAP -- {r['name']} ({r['hire_probability']:.1%})")
                    for feat, val in sorted(
                        r["shap_explanation"].items(),
                        key=lambda x: abs(x[1]), reverse=True,
                    ):
                        bar = "+" if val >= 0 else "-"
                        print(f"     {bar}  {feat:<29} {val:+.4f}")
                if r["skills_matched"]:
                    print(f"  Matched : {', '.join(r['skills_matched'][:8])}"
                          + (" ..." if len(r["skills_matched"]) > 8 else ""))
                if r["skills_missing"]:
                    print(f"  Missing : {', '.join(r['skills_missing'][:8])}"
                          + (" ..." if len(r["skills_missing"]) > 8 else ""))

        print("\n" + "=" * 76)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save trained pipeline to .joblib (compress=3, small footprint)."""
        joblib = _require("joblib")
        joblib.dump({
            "pipeline":     self._pipeline,
            "threshold":    self.threshold,
            "smote_ratio":  self.smote_ratio,
            "rich_mode":    self._rich_mode,
            "feature_cols": self._feature_cols,
        }, str(path), compress=3)
        print(f"Model saved -> {path}")

    def load(self, path: str | Path) -> "HiringPredictor":
        """Load a previously saved model. Returns self."""
        joblib = _require("joblib")
        p = joblib.load(str(path))
        self._pipeline     = p["pipeline"]
        self.threshold     = p.get("threshold",    self.threshold)
        self._rich_mode    = p.get("rich_mode",    False)
        self._feature_cols = p.get("feature_cols", _FEATURE_COLS_BASE)
        self._trained      = True
        self._shap_explainer = None
        print(f"Model loaded <- {path}")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Internal utility
# ─────────────────────────────────────────────────────────────────────────────

def _transform_until_model(pipeline, X: np.ndarray) -> np.ndarray:
    """Apply all pipeline steps except the final estimator ('model')."""
    Xt = X
    for name, step in pipeline.steps[:-1]:
        if name == "smote":
            continue
        Xt = step.transform(Xt)
    return Xt


# ─────────────────────────────────────────────────────────────────────────────
# Convenience one-shot function
# ─────────────────────────────────────────────────────────────────────────────

def predict_hires(
    ranked_results:    list[dict],
    parsed_list:       list[dict] | None = None,
    historical_ranked: list[dict] | None = None,
    historical_labels: list[int]  | None = None,
    historical_parsed: list[dict] | None = None,
    threshold:         float              = 0.40,
    top_k:             int                = 10,
) -> list[dict]:
    """
    One-shot convenience wrapper.

    Cold-start (no training):
        ranked  = SkillsMatcher().rank(jd)
        results = predict_hires(ranked, parsed_list=parsed_list)

    Trained (with real historical data):
        results = predict_hires(
            ranked, parsed_list,
            historical_ranked=hist_r,
            historical_labels=hist_l,
            historical_parsed=hist_p,
        )
    """
    predictor = HiringPredictor(threshold=threshold)
    if historical_ranked and historical_labels:
        predictor.fit(historical_ranked, historical_labels, historical_parsed)
    report = predictor.report(ranked_results, top_k=top_k, parsed_list=parsed_list)
    predictor.print_report(report)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic history generator (demo / bootstrapping only)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_synthetic_history(n: int = 2000, seed: int = 42):
    """
    Generate n synthetic historical candidates with ALL 15 features.
    Uses AND-based rule labels (document Method 2) for honest labelling.

    Label rule:
      hired = skills_match > 0.75 AND total_exp > 2 AND skills_count > 6 AND edu_gpa > 7.0

    Returns (ranked_list, parsed_list, labels).
    Replace with real historical data for production use.
    """
    rng = np.random.default_rng(seed)
    ranked, parsed, labels = [], [], []

    for i in range(n):
        base = float(rng.beta(2, 3))   # shared talent, mean ~0.40

        def j(off=0.0, std=0.12):
            return float(np.clip(base + off + rng.normal(0, std), 0.0, 1.0))

        comp  = j(0.00, 0.12)
        sem   = j(0.05, 0.10)
        kw    = j(-0.05, 0.15)
        exp   = j(-0.10, 0.18)
        n_m   = int(np.clip(rng.poisson(max(base * 8, 0)), 0, 12))
        n_mis = int(np.clip(rng.poisson(max((1 - base) * 6, 0)), 0, 10))

        # Rich feature values (correlated with base talent)
        total_exp  = float(np.clip(rng.exponential(max(base * 6, 0.1)), 0, 15))
        recent_exp = float(np.clip(rng.exponential(max(base * 2, 0.1)), 0, 5))
        job_hops   = int(np.clip(rng.poisson(max(total_exp / 2, 0.5)), 0, 10))
        edu_gpa    = float(np.clip(rng.normal(base * 10, 1.5), 0, 10))
        tech_kw    = int(np.clip(rng.poisson(max(base * 15, 1)), 0, 20))
        certs      = int(np.clip(rng.poisson(max(base * 3, 0.2)), 0, 8))
        projects   = int(np.clip(rng.poisson(max(base * 4, 0.5)), 0, 12))
        skills_cnt = int(np.clip(rng.poisson(max(base * 10, 2)), 0, 20))

        ranked.append({
            "name":             f"Candidate_{i}",
            "composite_score":  round(comp, 4),
            "semantic_score":   round(sem,  4),
            "keyword_score":    round(kw,   4),
            "experience_score": round(exp,  4),
            "skills_matched":   [f"skill_{j}" for j in range(n_m)],
            "skills_missing":   [f"gap_{j}"   for j in range(n_mis)],
        })

        # Synthetic parsed dict (mirrors main.py output structure)
        start_yr = max(2000, 2026 - int(total_exp) - 1)
        parsed.append({
            "experience": [
                {"summary": f"Corp_{k} | Dev | {start_yr + k * 2}-{start_yr + k * 2 + 2}"}
                for k in range(job_hops)
            ],
            "education": [{"summary": f"University | 2022 | B.Tech | CGPA: {edu_gpa:.1f}"}],
            "skills":    {"flat": [f"tech_{k}" for k in range(skills_cnt)]},
            "raw_text":  " ".join(
                ["project"] * projects + ["certified"] * certs +
                list(_TECH_KW)[:tech_kw]
            ),
        })

        # AND-based label (honest rule — not arbitrary thresholds)
        hired = (
            comp       > 0.75 and
            total_exp  > 2.0  and
            skills_cnt > 6    and
            edu_gpa    > 7.0
        )
        labels.append(1 if hired else 0)

    # Shuffle together
    perm = rng.permutation(n).tolist()
    return (
        [ranked[i] for i in perm],
        [parsed[i] for i in perm],
        [labels[i] for i in perm],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Two-stage model helpers
# ─────────────────────────────────────────────────────────────────────────────

_STAGE2_FEATURE_COLS = [
    "total_exp_years", "recent_exp_years", "job_hops",
    "edu_gpa", "tech_keyword_count", "certifications_count",
    "projects_count", "skills_count",
]


def _load_stage2_model(path: str | Path | None = None):
    """
    Load the JD-independent LightGBM model trained by data_loader.py.
    Returns the sklearn pipeline, or None if the file is not found.
    """
    joblib = _try_import("joblib")
    if joblib is None:
        return None
    candidates = [
        path,
        Path(__file__).parent / "hiring_model.joblib",
        Path("hiring_model.joblib"),
    ]
    for p in candidates:
        if p and Path(p).exists():
            try:
                payload = joblib.load(str(p))
                print(f"Stage-2 model loaded <- {Path(p).resolve()}")
                return payload["pipeline"]
            except Exception as e:
                print(f"Could not load model from {p}: {e}")
    return None


def _two_stage_predict(
    ranked_results: list[dict],
    parsed_list:    list[dict],
    stage2_model,
    w_match: float = 0.60,
    w_strength: float = 0.40,
) -> np.ndarray:
    """
    Compute two-stage hire probability.

    Stage 1 (JD-dependent)  : composite_score from skills.py
    Stage 2 (JD-independent): candidate_strength from hiring_model.joblib
    Final                   : Weighted average (Stage 1 x w_match, Stage 2 x w_strength)

    Adjust the weights below to prioritize JD match vs general strength.
    """
    # Stage 1: composite scores from skills.py (already computed)
    composites = np.array(
        [r.get("composite_score", 0.0) for r in ranked_results], dtype=np.float32
    )

    # Stage 2: build 8-feature matrix from parsed dicts
    rows = []
    for i, _ in enumerate(ranked_results):
        p = parsed_list[i] if (parsed_list and i < len(parsed_list)) else {}
        rows.append(_extract_rich_row(p))   # reuse existing helper from ml.py
    X2 = np.array(rows, dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        strengths = stage2_model.predict_proba(X2)[:, 1].astype(np.float32)

    # Weighted Average
    hire_probs = (w_match * composites) + (w_strength * strengths)
    return hire_probs.astype(np.float32)


def _two_stage_report(
    ranked_results: list[dict],
    parsed_list:    list[dict],
    stage2_model,
    top_k: int = 10,
) -> list[dict]:
    """
    Full two-stage report with SHAP from stage-1 HiringPredictor.
    Returns sorted report dicts.
    """
    if not ranked_results:
        return []
    results  = ranked_results[:top_k] if top_k > 0 else ranked_results
    p_slice  = parsed_list[:top_k]    if (parsed_list and top_k > 0) else parsed_list

    probs  = _two_stage_predict(results, p_slice, stage2_model)
    order  = np.argsort(probs)[::-1]
    report = []
    for new_rank, idx in enumerate(order):
        r     = results[int(idx)]
        prob  = float(probs[int(idx)])
        p     = p_slice[int(idx)] if (p_slice and int(idx) < len(p_slice)) else {}

        # Stage 1 strength (for display)
        comp     = r.get("composite_score", 0.0)
        
        # Recover strength: if prob = 0.6*comp + 0.4*str -> str = (prob - 0.6*comp) / 0.4
        WEIGHT_COMPOSITE = 0.60
        WEIGHT_STRENGTH  = 0.40
        strength_val = (prob - (WEIGHT_COMPOSITE * comp)) / WEIGHT_STRENGTH
        strength = max(0.0, min(strength_val, 1.0))

        report.append({
            "rank":              new_rank + 1,
            "name":              r.get("name", "Unknown"),
            "hire_probability":  round(prob, 4),
            "decision":          _make_decision(prob),
            "composite_score":   comp,
            "semantic_score":    r.get("semantic_score",   0.0),
            "keyword_score":     r.get("keyword_score",    0.0),
            "experience_score":  r.get("experience_score", 0.0),
            "candidate_strength": round(min(strength, 1.0), 4),
            "skills_matched":    r.get("skills_matched",   []),
            "skills_missing":    r.get("skills_missing",   []),
        })
    return report


def _print_two_stage_report(report: list[dict]) -> None:
    """Pretty-print the two-stage report."""
    if not report:
        print("No candidates to report.")
        return

    interview   = [r for r in report if r["hire_probability"] >= _TIER_INTERVIEW]
    phone       = [r for r in report if _TIER_PHONE_SCREEN <= r["hire_probability"] < _TIER_INTERVIEW]
    auto_reject = [r for r in report if r["hire_probability"] < _TIER_PHONE_SCREEN]

    print("\n" + "=" * 80)
    print("  TWO-STAGE HIRING REPORT")
    print("  Formula : hire_prob = (0.60 × composite_score) + (0.40 × candidate_strength)")
    print(f"  Interview : {len(interview)}  |  Phone Screen : {len(phone)}  |  Auto-Reject : {len(auto_reject)}")
    print("=" * 80)
    print(f"{'Rk':<4} {'Name':<26} {'Prob':>6}  {'Decision':<14}  {'Composite':>9}  {'Strength':>8}")
    print("-" * 80)
    for r in report:
        print(
            f"{r['rank']:<4} {r['name']:<26}"
            f" {r['hire_probability']:>6.1%}  {r['decision']:<14}"
            f"  {r['composite_score']:>9.4f}  {r['candidate_strength']:>8.4f}"
        )
        if r["skills_matched"]:
            print(f"       Matched : {', '.join(r['skills_matched'][:6])}"
                  + (" ..." if len(r["skills_matched"]) > 6 else ""))
        if r["skills_missing"]:
            print(f"       Missing : {', '.join(r['skills_missing'][:6])}"
                  + (" ..." if len(r["skills_missing"]) > 6 else ""))
    print("=" * 80)


# ─────────────────────────────────────────────────────────────────────────────
# __main__ — real resumes only, no synthetic candidates
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":

    # ── 1. Discover and parse ALL resume files in the project folder ─────────
    ranked_real: list[dict] = []
    parsed_real: list[dict] = []

    try:
        from main   import UniversalResumeParser
        from skills import SkillsMatcher

        parser = UniversalResumeParser()
        _base  = Path(__file__).parent

        # Auto-discover every supported format
        resume_files = (
            sorted(_base.glob("*.pdf")) +
            sorted(_base.glob("*.png")) +
            sorted(_base.glob("*.jpg")) +
            sorted(_base.glob("*.jpeg")) +
            sorted(_base.glob("*.docx")) +
            sorted(_base.glob("*.html"))
        )
        # Exclude non-resume files (scripts, outputs)
        resume_files = [
            f for f in resume_files
            if f.suffix.lower() != ".py"
            and not f.name.startswith("_")
            and f.name != "ml_output.txt"
        ]

        parsed_full: list[dict] = []
        import time
        for rf in resume_files:
            try:
                p = parser.parse(str(rf))
                parsed_full.append(p)
                print(f"Parsed : {rf.name}  ->  {p.get('name', 'Unknown')}")
                # Groq allows 30 RPM — 3 second pause is very safe
                time.sleep(3)
            except Exception as e:
                print(f"Skipped: {rf.name}: {e}")

        if not parsed_full:
            print("\nNo resume files found in the project folder.")
            print("Add PDF / PNG / JPG / DOCX / HTML resumes to:")
            print(f"  {_base}")
        else:
            # ── Paste or update your JD here ──────────────────────────────
            JD = """
            Junior Full-Stack Developer.
            Required: Python, Javascript, HTML, CSS, MySQL, MongoDB, Git.
            Preferred: Java, C++, VS Code, Jupyter, Vercel, PyTorch, TensorFlow.
            Experience in web development, data analysis, REST APIs, frontend development.
            """

            matcher = SkillsMatcher()
            matcher.add_candidates(parsed_full)
            ranked_real = matcher.rank(JD, top_k=50)

            # Align parsed dicts to the ranked order (match by name)
            name_to_parsed = {p.get("name", f"__idx_{i}"): p
                              for i, p in enumerate(parsed_full)}
            parsed_real = [
                name_to_parsed.get(r.get("name", ""), {})
                for r in ranked_real
            ]
            print(f"\nRanked {len(ranked_real)} real candidates.\n")

    except Exception as e:
        print(f"Could not load real resumes: {e}\n")

    # ── 2. Cold-start report (100% real, no ML model needed) ──────────────
    if ranked_real:
        print("-" * 76)
        print("COLD-START REPORT  (semantic ranking — no training data needed)")
        print("Most unbiased predictor. Use this when no trained model is available.")
        print("-" * 76)
        cold = HiringPredictor()
        cold_report = cold.report(ranked_real, top_k=10, parsed_list=parsed_real)
        cold.print_report(cold_report, verbose=False)

    # ── 3. Two-Stage Report: Stage1 (skills.py) x Stage2 (hiring_model.joblib) ─
    print("\n" + "-" * 76)
    print("TWO-STAGE REPORT  (composite_score + candidate_strength weighted avg)")
    print("Stage 1: JD-dependent semantic match (skills.py)         [60% weight]")
    print("Stage 2: JD-independent quality model (hiring_model.joblib) [40% weight]")
    print("-" * 76)

    stage2_model = _load_stage2_model()

    if stage2_model is None:
        print("hiring_model.joblib not found.")
        print("To generate it:")
        print("  1. Upload data_loader.py + resume dataset to Google Colab")
        print("  2. Run: !python data_loader.py")
        print("  3. Download hiring_model.joblib -> place in this folder")
        print("  4. Re-run: python ml.py")
        print("\nUsing cold-start (composite score) as fallback.")
    elif ranked_real:
        ts_report = _two_stage_report(ranked_real, parsed_real, stage2_model, top_k=10)
        _print_two_stage_report(ts_report)
    else:
        print("No real resumes to score. Add resume files to the project folder.")

    # ── 4. Edge-case checks ─────────────────────────────────────────
    print("\n-- Edge-case checks --")
    assert HiringPredictor().report([]) == [], "Empty input should return empty report."
    print("OK: Empty candidates -> empty report (correct)")

    try:
        HiringPredictor().predict_proba([{"composite_score": 0.5}])
        print("FAIL: Should have raised RuntimeError")
    except RuntimeError:
        print("OK: predict_proba before fit() raises RuntimeError (correct)")

    print("\nml.py demo complete.")
