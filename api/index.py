import os
import json
from flask import Flask, request, jsonify, render_template
from google import genai
from datetime import datetime

# Path setup for Vercel
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, '../templates')

app = Flask(__name__, template_folder=template_dir)

# Initialize Gemini Client (Ensure GEMINI_API_KEY is in Vercel Settings)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

NEWS_CATEGORIES = ["World", "Politics", "Technology", "Business", "Science", "Health"]

def fetch_news(category="World", query="", is_summary=False):
    # Anchoring the AI to today's date in 2026
    current_date = "Thursday, April 16, 2026"

    if is_summary:
        prompt = f"Today is {current_date}. Provide a one-sentence high-level briefing of the most recent {category} news. Use present tense. No markdown."
    else:
        prompt = f"""
        Today is {current_date}. 
        Return exactly 6 HIGHLY RECENT news items for {category} {query} in STRICT JSON format.
        Focus on events from the last 24 hours.
        JSON Format:
        [
          {{
            "title": "Headline",
            "summary": "1-2 sentence description",
            "url": "https://news.google.com",
            "source": "Source Name",
            "category": "{category}",
            "time": "2 hours ago"
          }}
        ]
        """

    try:
        # Using Gemini 3 Flash (the 2026 Free Tier workhorse)
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt
        )
        
        text = response.text.strip()
        if is_summary:
            return text

        # Clean markdown if Gemini includes it
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        
        return json.loads(text)

    except Exception as e:
        print(f"FETCH ERROR: {e}")
        return "Briefing currently unavailable." if is_summary else []

@app.route("/")
def home():
    return render_template("index.html", categories=NEWS_CATEGORIES)

@app.route("/api/news")
def news():
    category = request.args.get("category", "World")
    query = request.args.get("q", "")
    articles = fetch_news(category, query)
    return jsonify({"success": True, "articles": articles})

@app.route("/api/summary")
def summary():
    category = request.args.get("category", "World")
    summary_text = fetch_news(category, is_summary=True)
    return jsonify({"success": True, "summary": summary_text})

app = app
