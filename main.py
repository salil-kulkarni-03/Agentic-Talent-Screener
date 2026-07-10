"""
main.py — Groq-Powered Universal Resume Parser
================================================
Uses Groq's blazing-fast Llama-3 API (free, no credit card required).

Supports any format with zero layout bias:
  PDF, PNG, JPG, JPEG, DOCX, HTML, TXT

Stage 1 of the AI Recruiter pipeline:
  main.py  → parse resumes (this file)
  skills.py → rank candidates (semantic + keyword + TF-IDF)
  ml.py    → predict hire probability (two-stage LightGBM)

Setup:
  pip install groq PyMuPDF python-dotenv
  Set GROQ_API_KEY in your .env file.

Output dict (same shape as before — fully backward-compatible):
  {
    "name":       "John Smith",
    "raw_text":   "Full resume text ...",
    "skills":     {"flat": ["Python", "Docker"], "summary": "Python, Docker"},
    "experience": [{"summary": "Google | SWE | 2020–2023"}],
    "education":  [{"summary": "MIT | 2020 | B.Sc CS | GPA: 3.9"}],
  }
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()  # Loads variables from a .env file if it exists
except ImportError:
    pass

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_API_KEY_HERE")

# Model to use — Llama 3.3 70B is the most accurate free model on Groq
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Fallback: PII scrubber (runs locally before any API call) ─────────────────
_EMAIL_RE  = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE  = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")
_ADDR_RE   = re.compile(r"\d{1,5}\s+\w+\s+(Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Blvd)\b", re.I)

def _scrub_pii(text: str) -> str:
    """Strip PII from text before sending to any external API (privacy protection)."""
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    text = _ADDR_RE.sub("[ADDRESS]", text)
    return text


# ── Raw text extraction (local, no API) ───────────────────────────────────────
def _raw_text_from_file(file_path: str) -> str:
    """Extract raw text from any file type locally (used as input to the LLM)."""
    p = Path(file_path)
    suffix = p.suffix.lower()

    if suffix == ".pdf":
        text = ""
        # Attempt 1: Direct text extraction via PyMuPDF
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(p))
            text = "\n".join(page.get_text() for page in doc)
            
            # Extract structural hyperlinks (e.g. hyperlinked portfolio words like "GitHub")
            try:
                links_list = []
                for page in doc:
                    for link in page.get_links():
                        uri = link.get("uri")
                        if uri:
                            links_list.append(uri)
                if links_list:
                    text += "\n\n[Extracted Hyperlinks]\n" + "\n".join(links_list)
            except Exception:
                pass
        except Exception:
            pass

        # Attempt 2: pdfminer fallback
        if len(text.strip()) < 50:
            try:
                from pdfminer.high_level import extract_text
                text = extract_text(str(p))
            except Exception:
                pass

        # Attempt 3: OCR fallback for image-based PDFs
        #   Renders each page as an image, then runs Tesseract OCR
        if len(text.strip()) < 50:
            try:
                import fitz
                import pytesseract
                from PIL import Image
                import io
                doc = fitz.open(str(p))
                ocr_pages = []
                for page in doc:
                    pix = page.get_pixmap(dpi=300)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_pages.append(pytesseract.image_to_string(img))
                text = "\n".join(ocr_pages)
            except Exception:
                pass

        return text

    elif suffix in (".png", ".jpg", ".jpeg"):
        try:
            import pytesseract
            from PIL import Image
            return pytesseract.image_to_string(Image.open(str(p)))
        except Exception:
            return ""

    elif suffix == ".html":
        try:
            from bs4 import BeautifulSoup
            return BeautifulSoup(p.read_text(errors="ignore"), "html.parser").get_text(" ")
        except Exception:
            return p.read_text(errors="ignore")

    elif suffix == ".docx":
        try:
            import docx
            doc = docx.Document(str(p))
            return "\n".join(para.text for para in doc.paragraphs)
        except Exception:
            return ""

    elif suffix == ".txt":
        return p.read_text(errors="ignore")

    return ""


# ── Groq API call ─────────────────────────────────────────────────────────────
_PROMPT = """You are an expert technical recruiter. Analyze this resume text and extract the following information.
Return ONLY a valid JSON object with exactly these keys, no extra text, no markdown fences:

{
  "name": "Full candidate name",
  "skills": ["skill1", "skill2", ...],
  "experience": [
    {"company": "Company Name", "role": "Job Title", "period": "2020-2023"}
  ],
  "education": [
    {"institution": "University Name", "degree": "B.Tech Computer Engineering", "year": "2022", "gpa": "8.5 or empty string"}
  ],
  "total_experience_years": 4.5,
  "recent_experience_years": 2.0,
  "normalized_gpa_scaled_to_10": 8.8
}

Rules:
- name: Extract the full name. If truly not found, use "Unknown"
- skills: list ALL technical skills mentioned anywhere (languages, tools, frameworks, platforms, methodologies, soft skills)
- experience: list each job/internship/project role separately
  - period should be a year range like "2018-2022" or "2023-Present"
  - If dates are not mentioned, use "" (empty string), NEVER use "Unknown"
  - Include internships, freelance work, and significant project roles
- education: list each degree separately
  - gpa: use the numeric value as-is (e.g. "8.5" or "3.9" or "85%"). Use "" if not mentioned
  - If the score is a percentage, include the % sign (e.g. "95%")
- total_experience_years: A float (rounded to 1 decimal) representing total calendar years of experience.
  - Calculate this by checking all work experience entries.
  - IMPORTANT: If job periods overlap, merge them so you do not double-count overlapping years.
- recent_experience_years: A float (rounded to 1 decimal) representing experience years falling within the last 3 years (relative to today).
- normalized_gpa_scaled_to_10: A float (rounded to 2 decimals) representing their highest educational score scaled to 10.0.
  - If on a 4.0 scale (e.g. "3.8/4.0"), multiply by 2.5 (e.g., 9.5).
  - If a percentage (e.g. "85%"), divide by 10 (e.g., 8.5).
  - If already on a 10.0 scale, output as-is.
  - Use null if not mentioned.
- If a field is not found at all, use empty list []
- Do NOT include contact info (email, phone, address)
- Do NOT hallucinate or invent information that is not in the text
"""

def _call_groq(raw_text: str) -> dict:
    """
    Sends the locally-extracted resume text to Groq's Llama-3 API with exponential backoff retries.
    """
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    scrubbed = _scrub_pii(raw_text[:12000])

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert resume parser. Return ONLY valid JSON, no markdown fences."},
                    {"role": "user",   "content": f"{_PROMPT}\n\nResume text:\n{scrubbed}"},
                ],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )

            text = response.choices[0].message.content.strip()
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
            text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
            return json.loads(text)

        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "429" in error_str or "rate limit" in error_str
            
            if is_rate_limit and attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s backoff
                print(f"  [Groq] Rate limit hit (429). Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"  [Groq] API error on attempt {attempt+1}: {e}")
                if attempt == max_retries - 1:
                    return {}
    return {}


# ── Output normalizer ─────────────────────────────────────────────────────────
def _normalize_output(llm_dict: dict, raw_text: str) -> dict:
    """
    Converts the LLM's JSON response into the exact output dict shape that
    skills.py and ml.py expect. Backward-compatible with old parser output.
    """
    # Name
    name = str(llm_dict.get("name") or "Unknown").strip()
    if name.lower() in ("unknown", "", "null", "none"):
        name = "Unknown"

    # Skills
    raw_skills = llm_dict.get("skills") or []
    if isinstance(raw_skills, str):
        raw_skills = [s.strip() for s in raw_skills.split(",") if s.strip()]
    # Filter out garbage artifacts from bad text extraction
    flat_skills = [
        str(s).strip() for s in raw_skills
        if s
        and len(str(s).strip()) >= 2           # At least 2 chars
        and not str(s).strip().isdigit()        # Not purely numeric
        and re.search(r"[a-zA-Z]", str(s))     # Must contain at least one letter
        and not re.match(r"^[A-Z]\d+$", str(s).strip())  # Filter cell refs like G14, B2
    ]
    skills_summary = ", ".join(flat_skills)

    # Experience
    raw_exp = llm_dict.get("experience") or []
    experience = []
    for e in raw_exp:
        if isinstance(e, dict):
            company = e.get("company", "")
            role    = e.get("role", "")
            period  = e.get("period", "")
            summary = " | ".join(filter(None, [company, role, period]))
            if summary:
                experience.append({
                    "company": company,
                    "role":    role,
                    "period":  period,
                    "summary": summary,
                })

    # Education
    raw_edu = llm_dict.get("education") or []
    education = []
    for e in raw_edu:
        if isinstance(e, dict):
            institution = e.get("institution", "")
            degree      = e.get("degree", "")
            year        = e.get("year", "")
            gpa         = e.get("gpa", "")
            score_str   = f"GPA: {gpa}" if gpa else ""
            summary     = " | ".join(filter(None, [institution, year, degree, score_str]))
            if summary:
                education.append({
                    "institution":  institution,
                    "degree":       degree,
                    "year":         year,
                    "score_label":  "GPA" if gpa else "",
                    "score_value":  gpa,
                    "summary":      summary,
                })

    return {
        "name":       name,
        "raw_text":   raw_text,
        "skills":     {"flat": flat_skills, "summary": skills_summary},
        "experience": experience,
        "education":  education,
        "total_experience_years": llm_dict.get("total_experience_years"),
        "recent_experience_years": llm_dict.get("recent_experience_years"),
        "normalized_gpa_scaled_to_10": llm_dict.get("normalized_gpa_scaled_to_10")
    }


# ── Public API ────────────────────────────────────────────────────────────────
class UniversalResumeParser:
    """
    Groq-powered resume parser (Llama-3).
    Supports PDF, PNG, JPG, JPEG, HTML, DOCX, TXT.
    Zero layout bias — uses LLM for all formats.
    """

    def __init__(self, api_key: str | None = None):
        global GROQ_API_KEY
        if api_key:
            GROQ_API_KEY = api_key

        if GROQ_API_KEY == "YOUR_API_KEY_HERE":
            print(
                "\n[!] Groq API key not set!\n"
                "    Get a free key at: https://console.groq.com\n"
                "    Then add to your .env file: GROQ_API_KEY=gsk_your_key_here\n"
            )

    def parse(self, file_path: str) -> dict:
        """
        Parse a resume file and return a structured dict.

        Parameters
        ----------
        file_path : str
            Path to resume file (PDF, PNG, JPG, HTML, DOCX, TXT)

        Returns
        -------
        dict with keys: name, raw_text, skills, experience, education
        """
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Resume file not found: {file_path}")

        # Step 1: Extract raw text locally (Groq is text-only, not multimodal)
        raw_text = _raw_text_from_file(str(p))

        if not raw_text.strip():
            print(f"  [Warning] Could not extract text from: {p.name}")
            return _normalize_output({}, raw_text)

        # Step 2: Call Groq API with the extracted text
        llm_result = _call_groq(raw_text)

        # Step 3: Normalize to expected output shape
        parsed = _normalize_output(llm_result, raw_text)
        return parsed


# ── Standalone test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = UniversalResumeParser()
    _base  = Path(__file__).parent

    # Auto-discover all resume files in the project folder
    resume_files = (
        sorted(_base.glob("*.pdf")) +
        sorted(_base.glob("*.png")) +
        sorted(_base.glob("*.jpg")) +
        sorted(_base.glob("*.jpeg")) +
        sorted(_base.glob("*.docx")) +
        sorted(_base.glob("*.html"))
    )
    resume_files = [
        f for f in resume_files
        if not f.name.startswith("_")
        and f.name != "ml_output.txt"
    ]

    if not resume_files:
        print("No resume files found. Add PDF/PNG/JPG/DOCX/HTML to the project folder.")
        sys.exit(0)

    for rf in resume_files:
        print(f"\n{'='*60}")
        print(f"File : {rf.name}")
        print(f"{'='*60}")
        try:
            result = parser.parse(str(rf))
            print(f"Name       : {result['name']}")
            print(f"Skills     : {result['skills']['flat']}")
            print(f"Experience : {[e['summary'] for e in result['experience']]}")
            print(f"Education  : {[e['summary'] for e in result['education']]}")
            print(f"Raw text   : {len(result['raw_text'])} chars")
        except Exception as e:
            print(f"Error: {e}")

        # Groq allows 30 RPM, so 3 seconds between requests is very safe
        time.sleep(3)

