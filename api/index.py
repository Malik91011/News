import os
import json
from flask import Flask, request, jsonify, render_template
from anthropic import Anthropic

# Using absolute paths to ensure Vercel finds the templates folder correctly
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, '../templates')

app = Flask(__name__, template_folder=template_dir)

# Initialize Anthropic client - Make sure ANTHROPIC_API_KEY is in Vercel Settings
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

def fetch_news(category="World", query=""):
    prompt = f"""
Return exactly 6 news items in STRICT JSON format only.

Category: {category}
Query: {query}

Format:
[
  {{
    "title": "string",
    "summary": "string",
    "url": "string"
  }}
]
"""
    try:
        # Updated to the latest stable model as of April 2026
        response = client.messages.create(
            model="claude-3-5-sonnet-latest", 
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text.strip()

        # Clean any markdown formatting Claude might return
        if text.startswith("```"):
            text = text.split("```json")[-1].split("```")[0].strip()

        return json.loads(text)

    except Exception as e:
        # Prevent the app from crashing; return a helpful error object
        return [
            {
                "title": "Error fetching news",
                "summary": f"Could not retrieve news: {str(e)}",
                "url": "#"
            }
        ]

@app.route("/")
def home():
    """Serves the main website page"""
    return render_template("index.html")

@app.route("/api/news")
def news():
    """API endpoint for fetching news via AJAX/Fetch"""
    category = request.args.get("category", "World")
    query = request.args.get("q", "")
    
    articles = fetch_news(category, query)
    return jsonify({
        "success": True,
        "articles": articles
    })

# Essential for Vercel's WSGI detection
app = app
