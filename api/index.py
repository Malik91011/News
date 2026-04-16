import os
import json
import feedparser
from flask import Flask, request, jsonify, render_template
from google import genai

# Path setup for Vercel
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, '../templates')
app = Flask(__name__, template_folder=template_dir)

# Initialize Gemini 3 Client
# Make sure GEMINI_API_KEY is in Vercel Settings -> Environment Variables
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Mapping buttons to LIVE RSS Feeds for 100% accuracy
RSS_FEEDS = {
    "World": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "Politics": "https://feeds.bbci.co.uk/news/politics/rss.xml",
    "Technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "Science": "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "Health": "https://feeds.bbci.co.uk/news/health/rss.xml"
}

def get_live_headlines(category):
    """Fetches real headlines from RSS feeds."""
    url = RSS_FEEDS.get(category, RSS_FEEDS["World"])
    feed = feedparser.parse(url)
    headlines = []
    # Grab the top 8 stories
    for entry in feed.entries[:8]:
        headlines.append({
            "title": entry.title,
            "link": entry.link,
            "published": entry.get("published", "Recently")
        })
    return headlines

def summarize_with_ai(headlines, category):
    """Uses Gemini 3 to create clean summaries of real news."""
    if not headlines:
        return []

    prompt = f"""
    I have these real-time {category} headlines: {json.dumps(headlines)}
    
    Task: Convert these into a clean JSON list of 6 items. 
    Keep the original title and link. Write a 1-sentence 'summary' for each.
    
    JSON Format:
    [
      {{
        "title": "Title Here",
        "summary": "AI Summary Here",
        "url": "Original Link",
        "source": "BBC News",
        "category": "{category}",
        "time": "Recently"
      }}
    ]
    """
    
    try:
        # Using Gemini 3 Flash (April 2026 Stable)
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )
        text = response.text.strip()
        
        # Clean JSON markdown blocks
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        
        return json.loads(text)
    except Exception as e:
        print(f"AI ERROR: {e}")
        return []

@app.route("/")
def home():
    # Pass category keys to the template
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
    
    # Modern 2026 AI Briefing
    prompt = f"Give me a one-sentence dramatic news briefing about: {top_story}. No markdown."
    try:
        response = client.models.generate_content(model="gemini-3-flash-preview", contents=prompt)
        return jsonify({"success": True, "summary": response.text})
    except:
        return jsonify({"success": True, "summary": "Keeping you updated on the latest shifts."})

# Export for Vercel
app = app
