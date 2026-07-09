from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
import hashlib
from pathlib import Path
import json

# Add the parent directory (project root) to sys.path so we can import our existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import UniversalResumeParser
from skills import SkillsMatcher
from ml import _TIER_INTERVIEW, _TIER_PHONE_SCREEN, _two_stage_predict, _extract_rich_row
import joblib
import numpy as np
from backend.agent import chat_agent
from backend.auditor import audit_resume
from backend.github import extract_github_username, fetch_github_stats

app = FastAPI(title="AI Recruiter API")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-Memory Database for Prototype ──────────────────────────────────────────
# In a real production app, this would be SQLite/PostgreSQL.
db_candidates = []
db_jd = ""

# Load the trained LightGBM model once at startup
try:
    model_dict = joblib.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hiring_model.joblib"))
    stage2_model = model_dict.get("pipeline")
except Exception as e:
    print(f"Warning: Could not load hiring_model.joblib: {e}")
    stage2_model = None

parser = UniversalResumeParser()


def _make_decision(prob: float) -> str:
    if prob >= _TIER_INTERVIEW:
        return "[INTERVIEW]"
    if prob >= _TIER_PHONE_SCREEN:
        return "[PHONE SCREEN]"
    return "[AUTO-REJECT]"


@app.get("/")
def read_root():
    return {"status": "ok", "message": "AI Recruiter Backend is running!"}


@app.post("/api/jobs")
def set_job_description(payload: dict):
    """Sets the global Job Description used for ranking."""
    global db_jd
    jd = payload.get("jd", "").strip()
    if not jd:
        raise HTTPException(status_code=400, detail="Job description cannot be empty")
    db_jd = jd
    return {"message": "Job description updated successfully"}


@app.get("/api/jobs")
def get_job_description():
    return {"jd": db_jd}


@app.post("/api/resumes/upload")
async def upload_resume(file: UploadFile = File(...)):
    """Uploads a resume file, parses it via Groq LLM, and saves it in memory."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
        
    temp_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "temp"
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / file.filename

    # Save uploaded file temporarily and compute its SHA-256 hash
    try:
        content = await file.read()
        file_hash = hashlib.sha256(content).hexdigest()
        
        # Check if the exact same file content has already been uploaded
        for candidate in db_candidates:
            if candidate.get("file_hash") == file_hash:
                return {
                    "message": f"Resume {file.filename} is already uploaded.",
                    "data": candidate
                }
                
        with open(temp_path, "wb") as buffer:
            buffer.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Parse with our Groq-powered UniversalResumeParser
    try:
        parsed_data = parser.parse(str(temp_path))
        parsed_data["file_hash"] = file_hash
        parsed_data["github_username"] = extract_github_username(parsed_data.get("raw_text", ""))
        
        # Add to our in-memory "database"
        db_candidates.append(parsed_data)
        
        return {"message": f"Successfully parsed {file.filename}", "data": parsed_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing failed: {e}")
    finally:
        # Cleanup temp file
        if temp_path.exists():
            temp_path.unlink()


@app.get("/api/candidates")
def get_ranked_candidates(
    w_sem: float = 0.60,
    w_kw: float = 0.25,
    w_exp: float = 0.15,
    w_match: float = 0.60,
    w_strength: float = 0.40
):
    """Ranks all parsed candidates against the current JD and returns the Two-Stage report."""
    if not db_candidates:
        return {"candidates": []}
        
    if not db_jd:
        raise HTTPException(status_code=400, detail="No Job Description set. Please set a JD first.")

    # Normalize Group 1 weights (Semantic, Keyword, Experience)
    sum_g1 = w_sem + w_kw + w_exp
    if sum_g1 > 0:
        w_sem = w_sem / sum_g1
        w_kw = w_kw / sum_g1
        w_exp = w_exp / sum_g1
    else:
        w_sem, w_kw, w_exp = 0.60, 0.25, 0.15

    # Normalize Group 2 weights (JD Match vs. Profile Strength)
    sum_g2 = w_match + w_strength
    if sum_g2 > 0:
        w_match = w_match / sum_g2
        w_strength = w_strength / sum_g2
    else:
        w_match, w_strength = 0.60, 0.40

    try:
        # Stage 1: Semantic + Keyword ranking with custom weights
        matcher = SkillsMatcher(weights=(w_sem, w_kw, w_exp))
        matcher.add_candidates(db_candidates)
        ranked_real = matcher.rank(db_jd, top_k=50)

        # Re-align parsed dicts to the ranked order
        p_slice = []
        for r in ranked_real:
            matched = False
            # First try aligning using the unique file_hash
            if "file_hash" in r:
                for p in db_candidates:
                    if p.get("file_hash") == r["file_hash"]:
                        p_slice.append(p)
                        matched = True
                        break
            # Fallback to name matching if file_hash is not present
            if not matched:
                for p in db_candidates:
                    if p.get("name", "Unknown") == r.get("name", "Unknown"):
                        p_slice.append(p)
                        break

        # Stage 2: LightGBM Prediction with custom weights
        if not stage2_model:
            raise HTTPException(status_code=500, detail="ML model not loaded.")
            
        probs = _two_stage_predict(ranked_real, p_slice, stage2_model, w_match=w_match, w_strength=w_strength)
        order = np.argsort(probs)[::-1]
        
        report = []
        for new_rank, idx in enumerate(order):
            r = ranked_real[int(idx)]
            prob = float(probs[int(idx)])
            
            # Recover candidate strength dynamically
            comp = float(r.get("composite_score", 0.0))
            strength_val = (prob - (w_match * comp)) / w_strength if w_strength > 0 else 0.0
            strength = max(0.0, min(strength_val, 1.0))

            # Run the Resume Optimization Auditor dynamically
            p = p_slice[int(idx)] if (p_slice and int(idx) < len(p_slice)) else {}
            resume_raw = p.get("raw_text", "")
            audit_status = audit_resume(resume_raw, db_jd)

            # Extract GitHub username dynamically if missing
            github_username = p.get("github_username")
            if not github_username and resume_raw:
                github_username = extract_github_username(resume_raw)
                p["github_username"] = github_username

            report.append({
                "rank": int(new_rank + 1),
                "name": str(r.get("name", "Unknown")),
                "file_hash": str(r.get("file_hash", "") or p.get("file_hash", "")),
                "github_username": github_username,
                "hire_probability": float(round(prob, 4)),
                "decision": str(_make_decision(prob)),
                "composite_score": float(round(comp, 4)),
                "semantic_score": float(round(r.get("semantic_score", 0.0), 4)),
                "keyword_score": float(round(r.get("keyword_score", 0.0), 4)),
                "experience_score": float(round(r.get("experience_score", 0.0), 4)),
                "candidate_strength": float(round(min(strength, 1.0), 4)),
                "skills_matched": list(r.get("skills_matched", [])),
                "skills_missing": list(r.get("skills_missing", [])),
                "audit_status": audit_status,
            })

        return {"candidates": report}
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Crash:\n{error_msg}")


@app.post("/api/chat")
def chat_with_agent(payload: dict):
    """Chat endpoint using the LangGraph agent state machine."""
    message = payload.get("message", "").strip()
    history = payload.get("history", [])
    
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
        
    try:
        # Invoke the LangGraph agent
        # We pass db_candidates as the candidates list context, the current query, and history.
        initial_state = {
            "messages": history,
            "query": message,
            "candidates": db_candidates,
            "is_in_scope": False,
            "reply": ""
        }
        
        result = chat_agent.invoke(initial_state)
        return {"reply": result.get("reply", "No response generated.")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent Error: {e}")


@app.get("/api/candidates/{file_hash}/github")
def get_candidate_github(file_hash: str):
    """Fetches GitHub developer metrics for a candidate profile."""
    candidate = None
    for p in db_candidates:
        if p.get("file_hash") == file_hash:
            candidate = p
            break
            
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
        
    username = candidate.get("github_username")
    if not username:
        return {"username": None, "stats": None}
        
    stats = fetch_github_stats(username)
    if stats is None:
        raise HTTPException(status_code=400, detail=f"Failed to fetch GitHub stats for username: {username}")
        
    return {"username": username, "stats": stats}


@app.post("/api/candidates/{file_hash}/github")
def update_candidate_github(file_hash: str, payload: dict):
    """Allows manual linking/modifying of a candidate's GitHub handle."""
    username = payload.get("github_username", "").strip()
    
    candidate = None
    for p in db_candidates:
        if p.get("file_hash") == file_hash:
            candidate = p
            break
            
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
        
    candidate["github_username"] = username if username else None
    return {"message": "GitHub handle updated successfully.", "github_username": candidate["github_username"]}


@app.delete("/api/candidates")
def clear_candidates():
    """Clears the in-memory database of candidates."""
    global db_candidates
    db_candidates = []
    return {"message": "All candidates cleared."}
