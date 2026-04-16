import os
import json
from flask import Flask, request, jsonify, render_template
from anthropic import Anthropic

base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, '../templates')

app = Flask(__name__, template_folder=template_dir)

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

# Define categories here so they match your UI
NEWS_CATEGORIES = ["World", "Politics", "Technology", "Business", "Science", "Health"]

def fetch_news(category="World", query="", is_summary=False):
    if is_summary:
        prompt = f"Provide a one-sentence high-level briefing of the current state of {category} news. No markdown, just text."
    else:
        prompt = f"Return exactly 6 news items for {category} {query} in STRICT JSON format: [{{'title': 'string', 'summary': 'string', 'url': 'string', 'source': 'string'}}]"

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text.strip()
        
        if is_summary:
            return text
            
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"): text = text[4:]
        return json.loads(text.strip())
    except Exception as e:
        return "Briefing unavailable." if is_summary else []

@app.route("/")
def home():
    # CRITICAL: We pass the categories list here to fix the TypeError
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
