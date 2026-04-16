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

def get_live_headlines(category, query=None):
    url = RSS_FEEDS.get(category, RSS_FEEDS["World"])
    feed = feedparser.parse(url)
    headlines = []
    
    entries = feed.entries
    # --- SEARCH LOGIC INTEGRITY ---
    if query:
        query = query.lower()
        # Filters locally first to save API tokens and increase speed
        entries = [e for e in entries if query in e.title.lower() or query in getattr(e, 'summary', '').lower()]

    for entry in entries[:10]:
        headlines.append({
            "title": entry.title,
            "link": entry.link,
            "published": entry.get("published", "Recently")
        })
    return headlines

def summarize_with_ai(headlines, category):
    if not headlines: return []
    
    source_name = "ARY News" if category == "ARY News" else "Dawn News" if category in ["Pakistan", "Politics", "Business"] else "BBC News"
    
    # Robust Fallback in case of Rate Limits (429)
    fallback_news = [{
        "title": h['title'],
        "summary": f"Live update from {source_name}. Tap to read full coverage on the official bureau website.",
        "url": h['link'],
        "source": source_name,
        "category": category,
        "time": "Just Now"
    } for h in headlines[:6]]

    prompt = f"Summarize these headlines for a news dashboard in JSON format. Only output JSON: {json.dumps(headlines)}"
    
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
    query = request.args.get("q", "")
    live_data = get_live_headlines(cat, query)
    processed_news = summarize_with_ai(live_data, cat)
    return jsonify({"success": True, "articles": processed_news})

@app.route("/api/summary")
def summary():
    cat = request.args.get("category", "Pakistan")
    live_data = get_live_headlines(cat)
    top_story = live_data[0]['title'] if live_data else "Latest Updates"
    
    try:
        prompt = f"Provide a one-sentence high-impact news briefing on: {top_story}. No markdown."
        response = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
        return jsonify({"success": True, "summary": response.text})
    except:
        return jsonify({"success": False, "summary": "Intelligence feed synchronized with local bureaus."})

app = app
