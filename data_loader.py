"""
data_loader.py — JD-Independent LightGBM Trainer
==================================================
Run this on Google Colab (not on your local PC).

Steps:
  1. Upload this file + zipped "resume dataset" folder to Colab
  2. Run:  !python data_loader.py
  3. Download hiring_model.joblib → place in your AI Recruiter folder

What it does:
  - Scans ALL 24 category folders for PDFs (2484 total)
  - Parses each PDF through UniversalResumeParser (main.py)
  - Extracts 8 JD-INDEPENDENT features per resume
  - Labels based on resume QUALITY — not category (fully generalized)
  - Trains LightGBM + SMOTE
  - Saves hiring_model.joblib (~200KB)

Label rule (quality-based, generalized for any JD):
  hired = (exp >= 1yr OR skills >= 5)
       AND edu_gpa >= 6.0
       AND (tech_keywords >= 3 OR skills >= 4)
"""

from __future__ import annotations

import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
def _find_dataset_dir() -> Path:
    """Auto-discovers the dataset directory regardless of how it was unzipped."""
    base = Path(".")
    # Look for the 'ENGINEERING' folder which we know is one of the category folders
    for p in base.rglob("ENGINEERING"):
        if p.is_dir():
            return p.parent  # The parent folder contains all 24 categories
    return Path("resume dataset/data/data")

DATASET_DIR  = _find_dataset_dir()
MODEL_OUTPUT = Path("hiring_model.joblib")

# ── Feature vocab (same as ml.py) ────────────────────────────────────────────
_TECH_KW = {
    "python", "java", "javascript", "typescript", "c++", "c#", "go",
    "rust", "sql", "nosql", "react", "angular", "vue", "node", "django",
    "flask", "fastapi", "docker", "kubernetes", "aws", "gcp", "azure",
    "spark", "kafka", "airflow", "tensorflow", "pytorch", "scikit",
    "pandas", "numpy", "git", "linux", "mongodb", "postgresql", "redis",
}
_CERT_KW = {
    "aws", "azure", "gcp", "google cloud", "kubernetes",
    "terraform", "cisco", "comptia",
}
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

FEATURE_COLS = [
    "total_exp_years", "recent_exp_years", "job_hops",
    "edu_gpa", "tech_keyword_count", "certifications_count",
    "projects_count", "skills_count",
]


# ── Feature extraction (mirrors ml.py helpers exactly) ───────────────────────

def _parse_exp_years(parsed: dict) -> tuple[float, float]:
    now_year = datetime.now().year
    total = recent = 0.0
    for exp in (parsed.get("experience") or []):
        yrs = _YEAR_RE.findall(str(exp.get("summary") or ""))
        if len(yrs) >= 2:
            try:
                s, e = int(yrs[0]), int(yrs[-1])
                if s > e:
                    s, e = e, s
                e = min(e, now_year)
                total  += max(0, e - s)
                recent += max(0, min(e, now_year) - max(s, now_year - 3))
            except (ValueError, OverflowError):
                pass
        elif len(yrs) == 1:
            total += 1.0
    return round(min(total, 40.0), 2), round(min(recent, 10.0), 2)


def _parse_edu_gpa(parsed: dict) -> float:
    best   = 0.0
    gpa_re = re.compile(r"(?:cgpa|gpa)[^\d]*([0-9]+\.?[0-9]*)", re.IGNORECASE)
    pct_re = re.compile(r"([0-9]+\.?[0-9]*)\s*%")
    for edu in (parsed.get("education") or []):
        s = str(edu.get("summary") or "")
        m = gpa_re.search(s)
        if m:
            v = float(m.group(1))
            best = max(best, min(v * 2.5 if v <= 4.0 else v, 10.0))
            continue
        m = pct_re.search(s)
        if m:
            best = max(best, min(float(m.group(1)) / 10.0, 10.0))
    return round(best, 2)


def _count_tech_kw(parsed: dict) -> int:
    flat = (parsed.get("skills") or {}).get("flat") or []
    text = " ".join(flat).lower() + " " + str(parsed.get("raw_text") or "").lower()
    return sum(1 for kw in _TECH_KW if kw in text)


def _count_certs(parsed: dict) -> int:
    flat = (parsed.get("skills") or {}).get("flat") or []
    text = " ".join(flat).lower() + " " + str(parsed.get("raw_text") or "").lower()
    return sum(1 for kw in _CERT_KW if kw in text)


def _count_projects(parsed: dict) -> int:
    projs = parsed.get("projects") or []
    if projs:
        return min(len(projs), 10)
    return min(str(parsed.get("raw_text") or "").lower().count("project"), 10)


def _count_skills(parsed: dict) -> int:
    return len((parsed.get("skills") or {}).get("flat") or [])


def extract_features(parsed: dict) -> List[float]:
    """Extract all 8 JD-independent features from one parsed resume."""
    total_exp, recent_exp = _parse_exp_years(parsed)
    job_hops = len(parsed.get("experience") or [])
    return [
        total_exp,
        recent_exp,
        float(job_hops),
        _parse_edu_gpa(parsed),
        float(_count_tech_kw(parsed)),
        float(_count_certs(parsed)),
        float(_count_projects(parsed)),
        float(_count_skills(parsed)),
    ]


def quality_label(feats: List[float]) -> int:
    """
    Generalized quality label with a PROBABILISTIC approach.
    Instead of a hard cliff (exactly 1 or exactly 0), we calculate a 
    continuous strength score and sample the label. This forces the 
    LightGBM model to output smooth probabilities (e.g., 0.82 or 0.45) 
    instead of just 1.000 or 0.000.
    """
    total_exp, recent_exp, job_hops, edu_gpa, tech_kw, certs, projs, skills = feats
    
    # Calculate a continuous quality score (0.0 to 1.0)
    score = 0.0
    
    # Experience (up to 0.35)
    score += min(total_exp / 8.0, 1.0) * 0.35
    
    # Education (up to 0.20)
    # Senior candidates (>4 years) get auto-pass for GPA since industry experience matters more
    if total_exp >= 4.0:
        score += 0.20
    elif edu_gpa >= 5.0:
        score += min((edu_gpa - 5.0) / 5.0, 1.0) * 0.20
        
    # Technical depth (up to 0.30)
    score += min(tech_kw / 10.0, 1.0) * 0.15
    score += min(skills / 15.0, 1.0) * 0.15
    
    # Extras (up to 0.15)
    score += min(certs / 3.0, 1.0) * 0.10
    score += min(projs / 5.0, 1.0) * 0.05
    
    # Cap between 5% and 95% so nothing is ever exactly 1 or 0
    prob_of_hire = max(0.05, min(score, 0.95))
    
    # Sample the actual 1/0 label based on that probability
    return 1 if np.random.rand() < prob_of_hire else 0


# ── Main training pipeline ────────────────────────────────────────────────────

def main(dry_run: bool = False):
    # ── 0. Check and Unzip Dataset if needed ─────────────────────────────────
    zip_path = Path("resume dataset.zip")
    if zip_path.exists():
        import zipfile
        print("Found 'resume dataset.zip'. Extracting now (this might take a moment)...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Extraction complete!")
        
        # Now that we've unzipped, we need to update our dynamic path
        global DATASET_DIR
        DATASET_DIR = _find_dataset_dir()

    if not DATASET_DIR.exists():
        print(f"ERROR: Dataset folder not found at: {DATASET_DIR.resolve()}")
        print("\nWhat Colab currently sees in this folder:")
        for p in Path(".").iterdir():
            print(f"  - {p.name}")
        print("\nPlease make sure you uploaded the dataset and unzipped it correctly.")
        sys.exit(1)

    # ── 1. Import parser ─────────────────────────────────────────────────────
    try:
        from main import UniversalResumeParser
    except ImportError:
        print("ERROR: main.py not found. Upload it alongside data_loader.py.")
        sys.exit(1)

    parser = UniversalResumeParser()

    # ── 2. Scan all PDFs ──────────────────────────────────────────────────────
    pdf_files = sorted(DATASET_DIR.rglob("*.pdf"))
    if dry_run:
        # Use only 5 PDFs per category for a quick sanity check
        from itertools import islice
        by_cat: dict[str, list] = {}
        for p in pdf_files:
            cat = p.parent.name
            by_cat.setdefault(cat, []).append(p)
        pdf_files = [p for cat_files in by_cat.values() for p in list(cat_files)[:5]]
        print(f"[DRY RUN] Testing on {len(pdf_files)} PDFs (5 per category)")

    total   = len(pdf_files)
    rows    = []
    failed  = 0

    print(f"\nParsing {total} resume PDFs...")
    for i, pdf_path in enumerate(pdf_files):
        if i % 100 == 0:
            pct = i / total * 100
            print(f"  {i}/{total}  ({pct:.0f}%)  failures so far: {failed}")
        try:
            parsed = parser.parse(str(pdf_path))
            feats  = extract_features(parsed)
            label  = quality_label(feats)
            rows.append(feats + [label])
        except Exception as e:
            failed += 1
            if failed == 1:
                print(f"\n[!] First failure details: {type(e).__name__} - {e}")

    n_parsed = len(rows)
    print(f"\nParsed: {n_parsed}  |  Failed/skipped: {failed}")

    if n_parsed < 50:
        print("ERROR: Too few successful parses. Check dataset structure.")
        sys.exit(1)

    # ── 3. Build feature matrix ───────────────────────────────────────────────
    import pandas as pd
    df = pd.DataFrame(rows, columns=FEATURE_COLS + ["hired"])

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df["hired"].values.astype(np.int32)

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    print(f"Label distribution: Hired={n_pos} ({n_pos/len(y):.1%})  |  Rejected={n_neg}")

    if n_pos < 10 or n_neg < 10:
        print("WARNING: Very few positive or negative labels. Check label thresholds.")

    # ── 4. Train/test split ───────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ── 5. Build pipeline ─────────────────────────────────────────────────────
    import lightgbm as lgb
    from sklearn.impute        import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline      import Pipeline

    k_n = max(1, min(3, n_pos - 1, n_neg - 1))

    lgbm = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=15,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=max(1.0, n_neg / max(1, n_pos)),  # Safe division
        random_state=42, verbose=-1, n_jobs=-1,
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("lgbm",    lgbm),
    ])

    print("\nTraining LightGBM...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(X_train, y_train)

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    from sklearn.metrics import accuracy_score, roc_auc_score

    proba = pipeline.predict_proba(X_test)[:, 1]
    pred  = (proba >= 0.50).astype(int)
    acc   = accuracy_score(y_test, pred)
    try:
        auc = roc_auc_score(y_test, proba)
    except Exception:
        auc = float("nan")

    print(f"Held-out eval  |  Accuracy: {acc:.3f}  |  AUC-ROC: {auc:.3f}")
    print(f"Prediction range: {proba.min():.3f} – {proba.max():.3f}")

    # ── 7. Save model ─────────────────────────────────────────────────────────
    import joblib
    joblib.dump({
        "pipeline":     pipeline,
        "feature_cols": FEATURE_COLS,
        "mode":         "jd_independent",
        "n_train":      len(y_train),
        "accuracy":     round(acc, 4),
        "auc":          round(auc, 4),
    }, str(MODEL_OUTPUT), compress=3)

    print(f"\nModel saved  ->  {MODEL_OUTPUT.resolve()}")
    print("Download hiring_model.joblib and place it in your AI Recruiter folder.")
    print("Then run: python ml.py")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    main(dry_run=dry)
