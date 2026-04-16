import os
import json
import feedparser
import time
from flask import Flask, request, jsonify, render_template
from google import genai

base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, '../templates')
app = Flask(__name__, template_folder=template_dir)

# Initialize Gemini 3 Client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Pakistani and Global Sources
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
    headlines = []
    for entry in feed.entries[:8]:
        headlines.append({
            "title": entry.title,
            "link": entry.link,
            "published": entry.get("published", "Recently")
        })
    return headlines

def summarize_with_ai(headlines, category):
    if not headlines: return []

    # Fallback Logic
    source_name = "ARY News" if category == "ARY News" else "Dawn News" if category in ["Pakistan", "Politics", "Business"] else "BBC News"
    fallback_news = []
    for h in headlines[:6]:
        fallback_news.append({
            "title": h['title'],
            "summary": "Live update: Tap to view full coverage on " + source_name,
            "url": h['link'],
            "source": source_name,
            "category": category,
            "time": "Just Now"
        })

    prompt = f"Summarize these {category} headlines into a clean JSON list for a Pakistani audience: {json.dumps(headlines)}"
    
    try:
        response = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
        text = response.text.strip()
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        return json.loads(text)
    except:
        return fallback_news

@app.route("/")
def home():
    return render_template("index.html", categories=list(RSS_FEEDS.keys()))

@app.route("/api/news")
def news():
    cat = request.args.get("category", "Pakistan")
    live_data = get_live_headlines(cat)
    processed_news = summarize_with_ai(live_data, cat)
    return jsonify({"success": True, "articles": processed_news})

@app.route("/api/summary")
def summary():
    cat = request.args.get("category", "Pakistan")
    live_data = get_live_headlines(cat)
    top_story = live_data[0]['title'] if live_data else "Latest Updates"
    
    try:
        prompt = f"Provide a high-impact news briefing on: {top_story}. No markdown."
        response = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
        return jsonify({"success": True, "summary": response.text})
    except:
        return jsonify({"success": False, "summary": "Syncing with local and global news bureaus..."})

app = app
