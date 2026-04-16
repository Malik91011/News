import os
import json
from anthropic import Anthropic

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

CATEGORIES = ["World", "Technology", "Business", "Science", "Health", "Sports", "Politics"]


def fetch_news(category="World", query=""):
    topic = query if query else f"latest {category} news"

    prompt = f"""
Return 6 news items as JSON ONLY.

Each item:
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


# 🔥 Vercel entry point (IMPORTANT)
def handler(request):
    from urllib.parse import parse_qs, urlparse

    path = request.get("path", "/")

    if path == "/api/news":
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "success": True,
                "articles": fetch_news()
            })
        }

    return {
        "statusCode": 404,
        "body": "Not found"
    }
