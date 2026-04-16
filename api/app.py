import os
from flask import Flask, render_template, request, jsonify
import anthropic
import json

app = Flask(__name__, template_folder="../templates")

client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

CATEGORIES = ["World", "Technology", "Business", "Science", "Health", "Sports", "Politics"]


def fetch_news_with_claude(category="World", query=""):
    search_topic = query if query else f"latest {category} news today"

    prompt = f"""
Search for: "{search_topic}"

Return ONLY a valid JSON array with exactly 6 items.

Each item must have:
title, summary, source, category, time, url

No explanation. No markdown. Only JSON.
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    text = ""
    for block in response.content:
        if hasattr(block, "text"):
            text += block.text

    try:
        text = text.strip()

        # remove ``` if present
        if text.startswith("```"):
            text = text.split("```")[1]

        articles = json.loads(text)
        return articles[:6]

    except Exception as e:
        print("ERROR:", e)
        print("RAW:", text)
        return []


@app.route("/")
def home():
    return render_template("index.html", categories=CATEGORIES)


@app.route("/api/news")
def news():
    category = request.args.get("category", "World")
    query = request.args.get("q", "")

    articles = fetch_news_with_claude(category, query)

    return jsonify({
        "success": True,
        "articles": articles
    })


@app.route("/api/summary")
def summary():
    category = request.args.get("category", "World")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": f"Give a 3 sentence summary of today's {category} news."
        }]
    )

    text = ""
    for block in response.content:
        if hasattr(block, "text"):
            text += block.text

    return jsonify({
        "success": True,
        "summary": text.strip()
    })


# IMPORTANT: this line is REQUIRED for Vercel
app = app
