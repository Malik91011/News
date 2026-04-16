import os
import json
from flask import Flask, request, jsonify, render_template
from anthropic import Anthropic

# We point template_folder to '../templates' because this file is inside the /api folder
app = Flask(__name__, template_folder='../templates')

# Safe API key loading from Vercel Environment Variables
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
        # Using the Sonnet 3.5 model (or your specified version)
        response = client.messages.create(
            model="claude-3-5-sonnet-latest" 
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text.strip()

        # Clean markdown wrappers if Claude includes them
        text = text.replace("```json", "").replace("```", "").strip()

        return json.loads(text)

    except Exception as e:
        # Graceful error handling so the site doesn't 500
        return [
            {
                "title": "Error fetching news",
                "summary": str(e),
                "url": ""
            }
        ]

@app.route("/")
def home():
    """Serves the main HTML page from /templates/index.html"""
    return render_template("index.html")

@app.route("/api/news")
def news():
    """API endpoint for your frontend to fetch news data"""
    category = request.args.get("category", "World")
    query = request.args.get("q", "")

    return jsonify({
        "success": True,
        "articles": fetch_news(category, query)
    })

# This is required for Vercel to pick up the app instance
app = app
