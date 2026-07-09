import re

STOPWORDS = {
    "the", "and", "to", "in", "of", "for", "with", "on", "a", "an", "is", "at", 
    "by", "from", "as", "our", "we", "you", "that", "this", "or", "are", "be", 
    "experience", "project", "work", "using", "with", "management", "development",
    "skills", "years", "candidate", "role", "team", "strong"
}

def audit_resume(resume_text: str, jd_text: str) -> dict:
    """
    Audits a resume for keyword stuffing and verbatim job description copy-pasting.
    Does not impact the semantic score itself, but provides warning flags.
    """
    if not resume_text or not jd_text:
        return {
            "flagged": False,
            "copy_paste_percentage": 0.0,
            "stuffed_keywords": [],
            "warnings": []
        }

    # Extract words
    resume_words = re.findall(r'\b\w+\b', resume_text.lower())
    jd_words = re.findall(r'\b\w+\b', jd_text.lower())
    
    # ── 1. Verbatim Copy-Pasting Detector (N-Gram Overlap) ───────────────────
    N = 6  # Sequence length of exact matching words
    jd_ngrams = set()
    for i in range(len(jd_words) - N + 1):
        jd_ngrams.add(" ".join(jd_words[i:i+N]))

    resume_ngrams = []
    for i in range(len(resume_words) - N + 1):
        resume_ngrams.append(" ".join(resume_words[i:i+N]))

    matched_ngrams = 0
    unique_matches = set()
    for rg in resume_ngrams:
        if rg in jd_ngrams:
            matched_ngrams += 1
            unique_matches.add(rg)

    total_ngrams = len(resume_ngrams)
    overlap_pct = (matched_ngrams / total_ngrams * 100) if total_ngrams > 0 else 0.0
    
    # ── 2. Keyword Stuffing Detector ──────────────────────────────────────────
    freq = {}
    total_filtered_words = 0
    for w in resume_words:
        if w not in STOPWORDS and len(w) > 2:
            freq[w] = freq.get(w, 0) + 1
            total_filtered_words += 1

    stuffed_keywords = []
    if total_filtered_words > 10:
        for word, count in freq.items():
            density = count / total_filtered_words
            # Flag if a single technical/non-stopword exceeds 4% density and appears at least 5 times
            if density >= 0.04 and count >= 5:
                stuffed_keywords.append(word)

    # ── 3. Synthesize Warnings ────────────────────────────────────────────────
    warnings_list = []
    flagged = False

    # Threshold for verbatim copy-pasting is set at 8% sequence overlap
    if overlap_pct >= 8.0:
        warnings_list.append(f"Verbatim copy-pasting from JD detected ({round(overlap_pct, 1)}% sequence overlap).")
        flagged = True

    if stuffed_keywords:
        kw_list = ", ".join(stuffed_keywords)
        warnings_list.append(f"Abnormally high keyword density for: {kw_list}.")
        flagged = True

    return {
        "flagged": flagged,
        "copy_paste_percentage": round(overlap_pct, 2),
        "stuffed_keywords": stuffed_keywords,
        "warnings": warnings_list
    }
