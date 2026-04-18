import os
import json
import feedparser
import time
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types

base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, '../templates')
app = Flask(__name__, template_folder=template_dir)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Single-source feeds (used for most categories)
RSS_FEEDS = {
    "Pakistan": None,           # Multi-source — handled separately
    "World": "https://www.aljazeera.com/xml/rss/all.xml",
    "Politics": "https://tribune.com.pk/feed/pakistan",
    "Technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "Business": "https://www.dawn.com/feeds/business",
    "Sports": None,             # Multi-source — handled separately
}

# Pakistan aggregates multiple local sources
PAKISTAN_FEEDS = [
    "https://www.dawn.com/feeds/home",
    "https://arynews.tv/feed/",
    "https://tribune.com.pk/feed/home",
    "https://www.geo.tv/rss/1/7",
]

# Sports aggregates BBC Sport, Al Jazeera, ARY Sports
SPORTS_FEEDS = [
    "https://feeds.bbci.co.uk/sport/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://arynews.tv/category/sports/feed/",
]

# ---------------- NEWS ----------------

def fetch_feed_safe(url):
    try:
        feed = feedparser.parse(url)
        return feed.entries
    except Exception:
        return []


def get_live_headlines(category, query=None):
    if category == "Pakistan":
        all_entries = []
        for url in PAKISTAN_FEEDS:
            entries = fetch_feed_safe(url)
            source_name = (
                "Dawn" if "dawn" in url else
                "ARY News" if "arynews" in url else
                "Tribune" if "tribune" in url else
                "Geo News"
            )
            for e in entries:
                e["_source_label"] = source_name
            all_entries.extend(entries)

    elif category == "Sports":
        all_entries = []
        sports_keywords = [
            "sport", "cricket", "football", "soccer", "tennis",
            "psl", "fifa", "olympic", "match", "league", "cup",
            "hockey", "rugby", "golf", "f1", "racing", "champion",
            "wicket", "innings", "goal", "score", "player", "team"
        ]
        for url in SPORTS_FEEDS:
            entries = fetch_feed_safe(url)
            source_name = (
                "BBC Sport" if "bbc" in url else
                "Al Jazeera" if "aljazeera" in url else
                "ARY Sports"
            )
            for e in entries:
                title_lower = e.title.lower()
                summary_lower = getattr(e, 'summary', '').lower()
                if "aljazeera" in url:
                    if not any(k in title_lower or k in summary_lower for k in sports_keywords):
                        continue
                e["_source_label"] = source_name
                all_entries.append(e)

    else:
        url = RSS_FEEDS.get(category, RSS_FEEDS["World"])
        all_entries = fetch_feed_safe(url)
        for e in all_entries:
            e["_source_label"] = category

    # Sort by date descending
    all_entries = sorted(
        all_entries,
        key=lambda x: x.get('published_parsed', time.gmtime(0)),
        reverse=True
    )

    # Keyword filter (search)
    if query:
        q = query.lower()
        all_entries = [
            e for e in all_entries
            if q in e.title.lower() or q in getattr(e, 'summary', '').lower()
        ]

    # Deduplicate by title
    seen = set()
    unique = []
    for e in all_entries:
        key = e.title.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return [{
        "title": e.title,
        "link": e.link,
        "published": e.get("published", "Just Now"),
        "source_label": getattr(e, '_source_label', category)
    } for e in unique[:12]]


def summarize_with_ai(headlines, category):
    if not headlines:
        return []

    fallback = [{
        "title": h['title'],
        "summary": "Live coverage update.",
        "url": h['link'],
        "source": h.get('source_label', category),
        "category": category,
        "time": h['published']
    } for h in headlines]

    prompt = f"""
    Summarize each headline in 1 sentence.
    Return JSON only — no markdown, no backticks — with keys:
    title, summary, url, source, category, time

    Use the provided source_label value as the source for each headline.
    Headlines: {json.dumps(headlines)}
    """

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )

        text = response.text.strip()
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()

        return json.loads(text)
    except:
        return fallback


# ---------------- RISK ENGINE ----------------

COUNTRY_KEYWORDS = {
    "Pakistan": ["pakistan", "islamabad", "karachi"],
    "India": ["india", "delhi"],
    "United States of America": ["usa", "united states", "america"],
    "China": ["china", "beijing"],
    "Russia": ["russia", "moscow"],
    "United Kingdom": ["uk", "britain", "london"],
    "Ukraine": ["ukraine"],
    "Iran": ["iran", "tehran"],
    "Israel": ["israel"],
    "France": ["france", "paris"],
    "Germany": ["germany", "berlin"],
    "Brazil": ["brazil"],
    "Japan": ["japan", "tokyo"]
}

COUNTRY_COORDS = {
    "Pakistan": [30.3753, 69.3451],
    "India": [20.5937, 78.9629],
    "United States of America": [37.0902, -95.7129],
    "China": [35.8617, 104.1954],
    "Russia": [61.5240, 105.3188],
    "United Kingdom": [55.3781, -3.4360],
    "Ukraine": [48.3794, 31.1656],
    "Iran": [32.4279, 53.6880],
    "Israel": [31.0461, 34.8516],
    "France": [46.2276, 2.2137],
    "Germany": [51.1657, 10.4515],
    "Brazil": [-14.2350, -51.9253],
    "Japan": [36.2048, 138.2529]
}

NEGATIVE = [
    "war", "attack", "bomb", "conflict", "violence",
    "protest", "military", "strike", "killed", "explosion",
    "crisis", "tension", "terror"
]


def calculate_risk():
    all_news = []
    for cat in RSS_FEEDS:
        all_news.extend(get_live_headlines(cat))

    scores = {c: 0 for c in COUNTRY_KEYWORDS}

    for a in all_news:
        text = a["title"].lower()
        for country, keys in COUNTRY_KEYWORDS.items():
            if any(k in text for k in keys):
                scores[country] += 1
                if any(n in text for n in NEGATIVE):
                    scores[country] += 3

    result = {}
    for c, s in scores.items():
        if s >= 12:
            result[c] = "critical"
        elif s >= 6:
            result[c] = "high"
        elif s >= 2:
            result[c] = "low"
        else:
            result[c] = "none"

    return result


# ---------------- ROUTES ----------------

@app.route("/")
def home():
    return render_template("index.html", categories=list(RSS_FEEDS.keys()))


@app.route("/api/news")
def news():
    cat = request.args.get("category", "Pakistan")
    query = request.args.get("q", "")
    live = get_live_headlines(cat, query)
    return jsonify({"success": True, "articles": summarize_with_ai(live, cat)})


@app.route("/api/summary")
def summary():
    cat = request.args.get("category", "Pakistan")
    live = get_live_headlines(cat)

    if not live:
        return jsonify({"success": False, "summary": "No data"})

    try:
        prompt = f"Summarize in 15 words: {live[0]['title']}"
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=1.0)
        )
        return jsonify({"success": True, "summary": response.text.strip()})
    except:
        return jsonify({"success": False, "summary": live[0]["title"]})


@app.route("/api/risk")
def risk():
    try:
        risk_data = calculate_risk()
        enriched = {}
        for country, level in risk_data.items():
            enriched[country] = {
                "level": level,
                "coords": COUNTRY_COORDS.get(country, [0, 0])
            }
        return jsonify({"success": True, "risk": enriched})
    except Exception as e:
        return jsonify({"success": False, "risk": {}})


if __name__ == "__main__":
    app.run(debug=True)
