import os
import json
from flask import Flask, request, jsonify, render_template
from anthropic import Anthropic

# Finds the path of this file and goes one level up to find /templates
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, '../templates')

app = Flask(__name__, template_folder=template_dir)

# Vercel reads this from your Project Settings -> Environment Variables
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# CRITICAL: This list fixes the "Undefined is not JSON serializable" error
NEWS_CATEGORIES = ["World", "Politics", "Technology", "Business", "Science", "Health"]

def fetch_news(category="World", query="", is_summary=False):
    if is_summary:
        prompt = f"Provide a one-sentence high-level briefing of the current state of {category} news. Do not use markdown, just plain text."
    else:
        prompt = f"""
        Return exactly 6 news items for {category} {query} in STRICT JSON format.
        Format:
        [
          {{
            "title": "string",
            "summary": "string",
            "url": "string",
            "source": "string",
            "category": "{category}",
            "time": "Recently"
          }}
        ]
        """

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-latest", 
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text.strip()

        if is_summary:
            return text

        # Clean Claude's markdown wrappers if they exist
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        
        return json.loads(text.strip())

    except Exception as e:
        print(f"Error: {e}")
        return "Briefing unavailable." if is_summary else []

@app.route("/")
def home():
    # We MUST pass categories here so index.html doesn't crash
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

# Essential for Vercel's WSGI detection
app = app
