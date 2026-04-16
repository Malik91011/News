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

RSS_FEEDS = {
    "World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "Politics": "https://feeds.bbci.co.uk/news/politics/rss.xml",
    "Technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "Science": "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "Health": "https://feeds.bbci.co.uk/news/health/rss.xml"
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
    if not headlines:
        return []

    # FALLBACK: Create default news items in case AI is busy
    fallback_news = []
    for h in headlines[:6]:
        fallback_news.append({
            "title": h['title'],
            "summary": "Live update: Tap to read the full report from the source.",
            "url": h['link'],
            "source": "BBC News",
            "category": category,
            "time": "Just Now"
        })

    prompt = f"Summarize these headlines into a JSON list: {json.dumps(headlines)}"
    
    try:
        # Try once. If it fails, we fall back immediately to avoid 429 loops
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )
        text = response.text.strip()
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"Gemini Busy. Serving Fallback News.")
        return fallback_news

@app.route("/")
def home():
    return render_template("index.html", categories=list(RSS_FEEDS.keys()))

@app.route("/api/news")
def news():
    cat = request.args.get("category", "World")
    live_data = get_live_headlines(cat)
    processed_news = summarize_with_ai(live_data, cat)
    return jsonify({"success": True, "articles": processed_news})

@app.route("/api/summary")
def summary():
    cat = request.args.get("category", "World")
    live_data = get_live_headlines(cat)
    top_story = live_data[0]['title'] if live_data else "Global Events"
    
    try:
        # We use a very short prompt for the summary to save tokens/quota
        prompt = f"1-sentence news flash: {top_story}. No markdown."
        response = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
        return jsonify({"success": True, "summary": response.text})
    except:
        return jsonify({"success": False, "summary": "Intelligence systems busy. Fetching real-time feed..."})

app = app
