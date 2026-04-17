import os
import json
import feedparser
import time
from flask import Flask, request, jsonify, render_template
from google import genai

base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, '../templates')
app = Flask(__name__, template_folder=template_dir)

# Initialize Gemini Client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Updated Reliable RSS Feeds for 2026
RSS_FEEDS = {
    "Pakistan": "https://www.dawn.com/feeds/home",
    "ARY News": "https://arynews.tv/feed/",
    "World": "https://www.aljazeera.com/xml/rss/all.xml",
    "Politics": "https://tribune.com.pk/feed/pakistan",
    "Technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "Business": "https://www.dawn.com/feeds/business"
}

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
        entries = [e for e in entries if query in e.title.lower() or query in getattr(e, 'summary', '').lower()]

    headlines = []
    for entry in entries[:12]:
        headlines.append({
            "title": entry.title,
            "link": entry.link,
            "published": entry.get("published", "Just Now")
        })
    return headlines

def summarize_with_ai(headlines, category):
    if not headlines: return []
    
    source_name = "ARY News" if category == "ARY News" else "Dawn News" if category in ["Pakistan", "Politics", "Business"] else "Global Bureau"
    
    fallback_news = [{
        "title": h['title'],
        "summary": f"Latest breaking coverage from {source_name}. Reported on {h['published']}.",
        "url": h['link'],
        "source": source_name,
        "category": category,
        "time": h['published']
    } for h in headlines[:8]]

    prompt = (
        f"Analyze these news headlines from {source_name}: {json.dumps(headlines)}. "
        "Create a short, engaging 1-sentence summary for each. "
        "Return ONLY a JSON list of objects with these keys: title, summary, url, source, category, time. "
        "Do not include any markdown formatting or backticks."
    )
    
    try:
        # UPDATED TO GEMINI 3 FLASH
        response = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
        text = response.text.strip()
        
        # Robust JSON cleaning
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        
        return json.loads(text)
    except Exception as e:
        print(f"AI Processing Error: {e}")
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
    if not live_data:
        return jsonify({"success": False, "summary": "Intelligence feed synchronized..."})
    
    top_story = live_data[0]['title']
    try:
        prompt = f"Write a hard-hitting, one-sentence news flash about: {top_story}. No hashtags."
        # UPDATED TO GEMINI 3 FLASH
        response = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
        return jsonify({"success": True, "summary": response.text.strip()})
    except Exception as e:
        print(f"Summary Error: {e}")
        return jsonify({"success": False, "summary": f"LIVE: {top_story}"})

if __name__ == "__main__":
    app.run(debug=True)
