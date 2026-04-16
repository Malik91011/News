import os
import json
from flask import Flask, request, jsonify
from anthropic import Anthropic

app = Flask(__name__)

# Safe API key loading
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
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text.strip()

        # Clean markdown safely
        text = text.replace("```json", "").replace("```", "").strip()

        return json.loads(text)

    except Exception as e:
        # Never crash API on Vercel
        return [
            {
                "title": "Error fetching news",
                "summary": str(e),
                "url": ""
            }
        ]


@app.route("/")
def home():
    return "API running"


@app.route("/api/news")
def news():
    category = request.args.get("category", "World")
    query = request.args.get("q", "")

    return jsonify({
        "success": True,
        "articles": fetch_news(category, query)
    })


# REQUIRED for Vercel
app = app
