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

RSS_FEEDS = {
    "Pakistan": "https://www.dawn.com/feeds/home",
    "ARY News": "https://arynews.tv/feed/",
    "World": "https://www.aljazeera.com/xml/rss/all.xml",
    "Politics": "https://tribune.com.pk/feed/pakistan",
    "Technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "Business": "https://www.dawn.com/feeds/business"
}

# ---------------- NEWS ----------------

def get_live_headlines(category, query=None):
    url = RSS_FEEDS.get(category, RSS_FEEDS["World"])
    feed = feedparser.parse(url)

    entries = sorted(
        feed.entries,
        key=lambda x: x.get('published_parsed', time.gmtime(0)),
        reverse=True
    )

    if query:
        query = query.lower()
        entries = [
            e for e in entries
            if query in e.title.lower() or query in getattr(e, 'summary', '').lower()
        ]

    return [{
        "title": e.title,
        "link": e.link,
        "published": e.get("published", "Just Now")
    } for e in entries[:12]]


def summarize_with_ai(headlines, category):
    if not headlines:
        return []

    fallback = [{
        "title": h['title'],
        "summary": "Live coverage update.",
        "url": h['link'],
        "source": category,
        "category": category,
        "time": h['published']
    } for h in headlines]

    prompt = f"""
    Summarize each headline in 1 sentence.
    Return JSON only with keys:
    title, summary, url, source, category, time

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
        text = (a["title"]).lower()

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
        return jsonify({"success": True, "risk": calculate_risk()})
    except Exception as e:
        return jsonify({"success": False, "risk": {}})


if __name__ == "__main__":
    app.run(debug=True)
