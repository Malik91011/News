import os
import json
import google.generativeai as genai
from flask import Flask, request, jsonify, render_template

base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, '../templates')

app = Flask(__name__, template_folder=template_dir)

# 1. Setup Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

NEWS_CATEGORIES = ["World", "Politics", "Technology", "Business", "Science", "Health"]

def fetch_news(category="World", query="", is_summary=False):
    if is_summary:
        prompt = f"Provide a one-sentence high-level briefing of the current state of {category} news. No markdown."
    else:
        prompt = f"""
        Return exactly 6 news items for {category} {query} in STRICT JSON format.
        Format must be exactly like this:
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
        response = model.generate_content(prompt)
        text = response.text.strip()

        if is_summary:
            return text

        # Clean Gemini's markdown code blocks
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        
        return json.loads(text.strip())

    except Exception as e:
        print(f"GEMINI ERROR: {e}")
        return "Briefing unavailable." if is_summary else []

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
