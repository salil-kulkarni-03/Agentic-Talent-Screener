import os
import re
import requests
from typing import Dict, Any, Optional

# Regular expression to extract GitHub username from links in resume text
_GITHUB_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9_\-]+)", re.I)

# In-memory cache to save API rate limits
_github_cache: Dict[str, Dict[str, Any]] = {}

def extract_github_username(resume_text: str) -> Optional[str]:
    """
    Parses raw resume text for occurrences of github.com/username
    and returns the username if found.
    """
    if not resume_text:
        return None
    match = _GITHUB_RE.search(resume_text)
    if match:
        username = match.group(1).strip()
        # Filter out common false positives or layout elements
        if username and username.lower() not in {"com", "io", "repo", "project", "link"}:
            return username
    return None

def fetch_github_stats(username: str) -> Optional[Dict[str, Any]]:
    """
    Queries public GitHub REST API for candidate's repo counts,
    total stars, forks, and language usage distribution.
    Uses in-memory cache to avoid rate limit throttling.
    """
    username = username.strip()
    if not username:
        return None

    # Check cache first
    if username in _github_cache:
        return _github_cache[username]

    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Read GITHUB_TOKEN if recruiter added it to boost rate limits
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        # 1. Fetch user base profile details
        user_url = f"https://api.github.com/users/{username}"
        user_res = requests.get(user_url, headers=headers, timeout=5)
        
        if user_res.status_code == 404:
            return None
        elif user_res.status_code == 403:
            # Rate limit exceeded fallback
            return {
                "rate_limit_exceeded": True,
                "username": username,
                "repos_count": 0,
                "stars_count": 0,
                "forks_count": 0,
                "languages": {}
            }
        
        user_res.raise_for_status()
        user_data = user_res.json()

        # 2. Fetch public repositories details (max 100)
        repos_url = f"https://api.github.com/users/{username}/repos?per_page=100"
        repos_res = requests.get(repos_url, headers=headers, timeout=5)
        repos_res.raise_for_status()
        repos_data = repos_res.json()

        # 3. Calculate statistics
        total_repos = len(repos_data)
        total_stars = sum(repo.get("stargazers_count", 0) for repo in repos_data)
        total_forks = sum(repo.get("forks_count", 0) for repo in repos_data)

        # Aggregate language distribution by repo sizes
        language_sizes = {}
        total_size = 0
        for repo in repos_data:
            lang = repo.get("language")
            size = repo.get("size", 0)
            if lang and size > 0:
                language_sizes[lang] = language_sizes.get(lang, 0) + size
                total_size += size

        # Convert sizes to percentages
        language_pct = {}
        if total_size > 0:
            for lang, size in language_sizes.items():
                pct = round((size / total_size) * 100, 1)
                if pct >= 1.0: # Only show languages with at least 1% share
                    language_pct[lang] = pct

        # Sort languages by percentage descending
        sorted_languages = dict(sorted(language_pct.items(), key=lambda x: x[1], reverse=True))

        stats = {
            "rate_limit_exceeded": False,
            "username": username,
            "name": user_data.get("name"),
            "repos_count": total_repos,
            "stars_count": total_stars,
            "forks_count": total_forks,
            "languages": sorted_languages
        }

        # Cache stats
        _github_cache[username] = stats
        return stats

    except Exception as e:
        print(f"Error fetching GitHub details for {username}: {e}")
        return None
