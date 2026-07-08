"""
Patch: fix get_institution (strip date noise words) inside _extract_education.
"""
import re

with open('main.py', 'r', encoding='utf-8') as f:
    src = f.read()

# ── Fix 1: Replace get_institution with a version that strips date-noise words ──
OLD = '''        def get_institution(line):
            """Strip year tokens; return remaining alphabetic words as name."""
            clean = yr_range.sub("", line)
            clean = yr_only.sub("", clean).strip(" |-\\u2013.")
            words = [w for w in clean.split() if re.match(r'^[A-Za-z&\\',.\\-]+$', w)]
            return " ".join(words).strip() if words else clean[:70].strip()'''

NEW = '''        # Words that appear near years but are NOT part of the institution name
        _date_noise = re.compile(
            r'\\b(expected|present|ongoing|current|till|since|from|in|of|the'
            r'|january|february|march|april|may|june|july|august'
            r'|september|october|november|december'
            r'|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\\b',
            re.IGNORECASE
        )

        def get_institution(line):
            """Strip year tokens and date-noise words; return the institution name."""
            clean = yr_range.sub("", line)
            clean = yr_only.sub("", clean)
            clean = _date_noise.sub("", clean).strip(" |-\\u2013.,")
            words = [w for w in clean.split() if re.match(r"^[A-Za-z&\\']+$", w) and len(w) > 1]
            return " ".join(words).strip() if words else clean[:70].strip()'''

if OLD in src:
    src = src.replace(OLD, NEW, 1)
    print("Fix 1 (get_institution): applied")
else:
    # Try to find and print the actual get_institution body for debugging
    m = re.search(r'def get_institution\(line\):.*?return.*?\n', src, re.DOTALL)
    if m:
        print("Fix 1 NOT found. Current get_institution:\n" + m.group()[:400])
    else:
        print("Fix 1 NOT found. get_institution not matched at all.")

with open('main.py', 'w', encoding='utf-8') as f:
    f.write(src)

print("Done.")
