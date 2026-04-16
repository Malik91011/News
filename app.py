import os
from flask import Flask, render_template, request, jsonify
import anthropic
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

CATEGORIES = ["World", "Technology", "Business", "Science", "Health", "Sports", "Politics"]

def fetch_news_with_claude(category: str = "World", query: str = "") -> list[dict]:
    """Use Claude with web_search to fetch and summarize real news."""
    search_topic = query if query else f"latest {category} news today"
    
    prompt = f"""Search the web for: "{search_topic}"

Return ONLY a valid JSON array (no markdown, no explanation) with exactly 6 news items.
Each item must have these exact keys:
- "title": compelling headline (string)
- "summary": 2-3 sentence summary (string)
- "source": news outlet name (string)
- "category": one of {CATEGORIES} (string)
- "time": relative time like "2 hours ago" (string)
- "url": article URL if available, else "#" (string)

Example format:
[{{"title": "...", "summary": "...", "source": "...", "category": "...", "time": "...", "url": "..."}}]

Return only the JSON array, nothing else."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": prompt}]
    )

    # Extract text from response blocks
    full_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            full_text += block.text

    # Parse JSON from response
    import json, re
    try:
        # Strip markdown fences if present
        clean = re.sub(r"```json|```", "", full_text).strip()
        # Find first JSON array
        match = re.search(r"\[.*\]", clean, re.DOTALL)
        if match:
            articles = json.loads(match.group())
            return articles[:6]
    except (json.JSONDecodeError, AttributeError):
        pass

    return []


@app.route("/")
def index():
    return render_template("index.html", categories=CATEGORIES)


@app.route("/api/news")
def api_news():
    category = request.args.get("category", "World")
    query = request.args.get("q", "")
    try:
        articles = fetch_news_with_claude(category, query)
        return jsonify({"success": True, "articles": articles, "category": category})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "articles": []}), 500


@app.route("/api/summary")
def api_summary():
    """Get a brief AI-written briefing for a category."""
    category = request.args.get("category", "World")
    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{
                "role": "user",
                "content": f"Search for today's top {category} news and write a 3-sentence briefing summary of the most important developments. Be concise and factual."
            }]
        )
        summary = ""
        for block in response.content:
            if hasattr(block, "text"):
                summary += block.text
        return jsonify({"success": True, "summary": summary.strip()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
