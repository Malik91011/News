import os
import json
from flask import Flask, request, jsonify
from anthropic import Anthropic

app = Flask(__name__)

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

CATEGORIES = ["World", "Technology", "Business", "Science", "Health", "Sports", "Politics"]


def fetch_news(category="World", query=""):
    topic = query if query else f"latest {category} news"

    prompt = f"""
Return 6 news items as JSON ONLY.

Each item must have:
title, summary, source, category, time, url

Topic: {topic}
"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text.strip()

        if text.startswith("```"):
            text = text.split("```")[1]

        return json.loads(text)[:6]

    except Exception as e:
        print("ERROR:", e)
        return []


@app.route("/api/news")
def news():
    category = request.args.get("category", "World")
    query = request.args.get("q", "")

    return jsonify({
        "success": True,
        "articles": fetch_news(category, query)
    })


@app.route("/")
def home():
    return "News API is running"


# IMPORTANT for Vercel
app = app
