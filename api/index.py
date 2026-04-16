import os
import json
import feedparser
from flask import Flask, request, jsonify, render_template
from google import genai

base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, '../templates')

app = Flask(__name__, template_folder=template_dir)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

RSS_FEEDS = {
    "Pakistan": "https://www.dawn.com/feeds/home",
    "ARY News": "https://arynews.tv/feed/",
    "World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "Politics": "https://www.dawn.com/feeds/pakistan",
    "Technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "Business": "https://www.dawn.com/feeds/business"
}


def get_live_headlines(category):
    url = RSS_FEEDS.get(category, RSS_FEEDS["World"])
    feed = feedparser.parse(url)

    return [
        {
            "title": entry.title,
            "link": entry.link,
            "published": entry.get("published", "Recently")
        }
        for entry in feed.entries[:10]
    ]


def summarize(headlines, category, query=None):
    if not headlines:
        return []

    # simple search filter (server-side search)
    if query:
        headlines = [
            h for h in headlines
            if query.lower() in h["title"].lower()
        ]

    fallback = [
        {
            "title": h["title"],
            "summary": "Live update: Tap to read full story.",
            "url": h["link"],
            "source": "Live Feed",
            "category": category,
            "time": h.get("published", "Now")
        }
        for h in headlines[:6]
    ]

    prompt = f"""
Summarize these news headlines for category {category}.
Return ONLY valid JSON list:
{json.dumps(headlines[:6])}
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


@app.route("/")
def home():
    return render_template("index.html", categories=list(RSS_FEEDS.keys()))


@app.route("/api/news")
def news():
    cat = request.args.get("category", "Pakistan")
    query = request.args.get("q", "")

    data = get_live_headlines(cat)
    articles = summarize(data, cat, query)

    return jsonify({"success": True, "articles": articles})


@app.route("/api/summary")
def summary():
    cat = request.args.get("category", "Pakistan")
    data = get_live_headlines(cat)

    try:
        prompt = f"Give short news briefing: {data[0]['title'] if data else 'Latest News'}"
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )
        return jsonify({"success": True, "summary": response.text})
    except:
        return jsonify({"success": False, "summary": "Loading updates..."})


app = app
