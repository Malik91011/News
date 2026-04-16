import os
import json
from flask import Flask, request, jsonify, render_template

# THE CRITICAL CHANGE: Use this specific import path
from google import genai

base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, '../templates')

app = Flask(__name__, template_folder=template_dir)

# Initialize Client
# Make sure GEMINI_API_KEY is in Vercel Settings -> Environment Variables
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

NEWS_CATEGORIES = ["World", "Politics", "Technology", "Business", "Science", "Health"]

def fetch_news(category="World", query="", is_summary=False):
    if is_summary:
        prompt = f"Provide a one-sentence high-level briefing of the current state of {category} news. No markdown."
    else:
        prompt = f"Return exactly 6 news items for {category} {query} in STRICT JSON format: [{{'title': 'string', 'summary': 'string', 'url': 'string', 'source': 'string', 'category': '{category}', 'time': 'Recently'}}]"

    try:
        # Generate content using Gemini 1.5 Flash
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        
        text = response.text.strip()

        if is_summary:
            return text

        # Strip markdown if present
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        
        return json.loads(text.strip())

    except Exception as e:
        print(f"FETCH ERROR: {e}")
        return "Briefing unavailable." if is_summary else []

@app.route("/")
def home():
    # Passing categories list is required by your HTML
    return render_template("index.html", categories=NEWS_CATEGORIES)

@app.route("/api/news")
def news():
    category = request.args.get("category", "World")
    query = request.args.get("q", "")
    return jsonify({"success": True, "articles": fetch_news(category, query)})

@app.route("/api/summary")
def summary():
    category = request.args.get("category", "World")
    return jsonify({"success": True, "summary": fetch_news(category, is_summary=True)})

app = app
