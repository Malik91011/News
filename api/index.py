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

    prompt = f"Summarize these real headlines for {category} into a JSON list: {json.dumps(headlines)}"
    
    # Retry logic for 429 Too Many Requests
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt
            )
            text = response.text.strip()
            if "```" in text:
                text = text.split("```")[1].replace("json", "").strip()
            return json.loads(text)
        except Exception as e:
            if "429" in str(e):
                time.sleep(2) # Wait and try again
                continue
            return []
    return []

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
    top_story = live_data[0]['title'] if live_data else "Global News"
    
    prompt = f"Give me a one-sentence dramatic briefing about: {top_story}. No markdown."
    try:
        response = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
        return jsonify({"success": True, "summary": response.text})
    except:
        return jsonify({"success": True, "summary": "Intelligence systems are currently busy. Refreshing..."})

app = app
