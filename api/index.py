import os
import json
import feedparser
import time
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types

# Initialize Flask
# Assuming your folder structure has a 'templates' folder at the root
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, 'templates')
app = Flask(__name__, template_folder=template_dir)

# Initialize Gemini Client
# Ensure you have 'GEMINI_API_KEY' set in your environment variables
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Your Intelligence Sources (RSS Feeds)
RSS_FEEDS = {
    "Pakistan": "https://www.dawn.com/feeds/home",
    "ARY News": "https://arynews.tv/feed/",
    "World": "https://www.aljazeera.com/xml/rss/all.xml",
    "Politics": "https://tribune.com.pk/feed/pakistan",
    "Technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "Business": "https://www.dawn.com/feeds/business"
}

# --- 1. RISK INTELLIGENCE ROUTE ---
@app.route("/api/risk-data")
def risk_data():
    """
    MANUAL INTELLIGENCE FEED
    These keys (PK, US, etc.) match the Leaflet map GeoJSON properties.
    """
    intelligence_feed = {
        "PK": {"level": "CRITICAL", "color": "#ff4d4d", "info": "High volatility; monitoring active."},
        "US": {"level": "STABLE", "color": "#4dff88", "info": "Regional stability nominal."},
        "GB": {"level": "STABLE", "color": "#4dff88", "info": "Standard protocol."},
        "AE": {"level": "STABLE", "color": "#4dff88", "info": "Trade corridors open."},
        "UA": {"level": "HIGH", "color": "#e8c547", "info": "Geopolitical tension detected."},
        "JP": {"level": "STABLE", "color": "#4dff88", "info": "System synchronized."}
    }
    return jsonify(intelligence_feed)

# --- 2. NEWS PROCESSING UTILITIES ---
def get_live_headlines(category, query=None):
    url = RSS_FEEDS.get(category, RSS_FEEDS["World"])
    feed = feedparser.parse(url)
    entries = sorted(feed.entries, key=lambda x: x.get('published_parsed', time.gmtime(0)), reverse=True)

    if query:
        query = query.lower()
        entries = [e for e in entries if query in e.title.lower()]

    return [{"title": entry.title, "link": entry.link} for entry in entries[:12]]

def summarize_with_ai(headlines, category):
    if not headlines:
        return []
    
    prompt = (
        f"Analyze these news headlines from {category}: {json.dumps(headlines)}. "
        "Create a short, engaging 1-sentence summary for each. "
        "Return ONLY a JSON list of objects with these keys: title, summary, url, source. "
        "Strictly no markdown tags or code blocks."
    )
    
    try:
        response = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
        text = response.text.strip()
        # Clean potential markdown formatting
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return None  # Return None so we can trigger the fallback

# --- 3. WEB ROUTES ---
@app.route("/")
def home():
    return render_template("index.html", categories=list(RSS_FEEDS.keys()))

@app.route("/api/news")
def news():
    cat = request.args.get("category", "Pakistan")
    query = request.args.get("q", "")
    
    # Get raw data
    live_data = get_live_headlines(cat, query)
    
    # Attempt AI Summarization
    processed_news = summarize_with_ai(live_data, cat)
    
    # CRITICAL FALLBACK: If AI fails/limits, show raw headlines so grid isn't empty
    if processed_news is None:
        processed_news = [
            {
                "title": h['title'], 
                "summary
