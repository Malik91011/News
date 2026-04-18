import os
import json
import feedparser
import time
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types

base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, '../templates')
app = Flask(__name__, template_folder=template_dir)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Single-source feeds (used for most categories)
RSS_FEEDS = {
    "Pakistan": None,           # Multi-source — handled separately
    "World": "https://www.aljazeera.com/xml/rss/all.xml",
    "Politics": "https://tribune.com.pk/feed/pakistan",
    "Technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "Business": "https://www.dawn.com/feeds/business",
    "Sports": None,             # Multi-source — handled separately
}

# Pakistan aggregates multiple local sources
PAKISTAN_FEEDS = [
    "https://www.dawn.com/feeds/home",
    "https://arynews.tv/feed/",
    "https://tribune.com.pk/feed/home",
    "https://www.geo.tv/rss/1/7",
]

# Sports aggregates BBC Sport, Al Jazeera, ARY Sports
SPORTS_FEEDS = [
    "https://feeds.bbci.co.uk/sport/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://arynews.tv/category/sports/feed/",
]

# ---------------- NEWS ----------------

def fetch_feed_safe(url):
    try:
        feed = feedparser.parse(url)
        return feed.entries
    except Exception:
        return []


def get_live_headlines(category, query=None):
    if category == "Pakistan":
        all_entries = []
        for url in PAKISTAN_FEEDS:
            entries = fetch_feed_safe(url)
            source_name = (
                "Dawn" if "dawn" in url else
                "ARY News" if "arynews" in url else
                "Tribune" if "tribune" in url else
                "Geo News"
            )
            for e in entries:
                e["_source_label"] = source_name
            all_entries.extend(entries)

    elif category == "Sports":
        all_entries = []
        sports_keywords = [
            "sport", "cricket", "football", "soccer", "tennis",
            "psl", "fifa", "olympic", "match", "league", "cup",
            "hockey", "rugby", "golf", "f1", "racing", "champion",
            "wicket", "innings", "goal", "score", "player", "team"
        ]
        for url in SPORTS_FEEDS:
            entries = fetch_feed_safe(url)
            source_name = (
                "BBC Sport" if "bbc" in url else
                "Al Jazeera" if "aljazeera" in url else
                "ARY Sports"
            )
            for e in entries:
                title_lower = e.title.lower()
                summary_lower = getattr(e, 'summary', '').lower()
                if "aljazeera" in url:
                    if not any(k in title_lower or k in summary_lower for k in sports_keywords):
                        continue
                e["_source_label"] = source_name
                all_entries.append(e)

    else:
        url = RSS_FEEDS.get(category, RSS_FEEDS["World"])
        all_entries = fetch_feed_safe(url)
        for e in all_entries:
            e["_source_label"] = category

    # Sort by date descending
    all_entries = sorted(
        all_entries,
        key=lambda x: x.get('published_parsed', time.gmtime(0)),
        reverse=True
    )

    # Keyword filter (search)
    if query:
        q = query.lower()
        all_entries = [
            e for e in all_entries
            if q in e.title.lower() or q in getattr(e, 'summary', '').lower()
        ]

    # Deduplicate by title
    seen = set()
    unique = []
    for e in all_entries:
        key = e.title.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return [{
        "title": e.title,
        "link": e.link,
        "published": e.get("published", "Just Now"),
        "source_label": getattr(e, '_source_label', category)
    } for e in unique[:12]]


def summarize_with_ai(headlines, category):
    if not headlines:
        return []

    fallback = [{
        "title": h["title"],
        "summary": h["title"],
        "assessment": "Monitoring situation as details emerge.",
        "precaution": "Stay informed through official sources.",
        "url": h["link"],
        "source": h.get("source_label", category),
        "category": category,
        "time": h["published"]
    } for h in headlines]

    items = [{"title": h["title"], "url": h["link"],
              "source_label": h.get("source_label", category),
              "time": h["published"]} for h in headlines]

    prompt = (
        "You are a professional news intelligence analyst. "
        "For each headline return a JSON array where each object has exactly these keys:\n"
        "title (copy exactly), summary (2 sentences: what happened and why it matters), "
        "assessment (1 sentence: broader significance or implication), "
        "precaution (1 sentence: practical advice or what to watch), "
        "url (copy exactly), source (copy source_label exactly), "
        "category (always \"" + category + "\"), time (copy exactly).\n"
        "Return ONLY a valid JSON array starting with [ and ending with ]. "
        "No markdown, no backticks.\n\nHeadlines:\n" + json.dumps(items)
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        text = response.text.strip()

        if "```" in text:
            for part in text.split("```"):
                part = part.strip().lstrip("json").strip()
                if part.startswith("["):
                    text = part
                    break

        start = text.find("[")
        end = text.rfind("]") + 1
        if start != -1 and end > start:
            text = text[start:end]

        parsed = json.loads(text)
        for i, item in enumerate(parsed):
            item.setdefault("summary", headlines[i]["title"] if i < len(headlines) else "")
            item.setdefault("assessment", "Situation under review.")
            item.setdefault("precaution", "Follow credible news sources for updates.")
        return parsed

    except Exception:
        return fallback


# ---------------- RISK ENGINE ----------------

# Dedicated global feeds used ONLY for risk scoring (not shown as news categories)
RISK_FEEDS = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",           # BBC World
    "https://www.aljazeera.com/xml/rss/all.xml",             # Al Jazeera
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml", # NYT World
    "https://feeds.reuters.com/reuters/worldNews",            # Reuters World
    "https://feeds.skynews.com/feeds/rss/world.xml",         # Sky News World
    "https://www.dawn.com/feeds/home",                       # Dawn (South Asia)
    "https://arynews.tv/feed/",                              # ARY (Pakistan/Global)
    "https://tribune.com.pk/feed/home",                      # Tribune
]

COUNTRY_KEYWORDS = {
    # Asia
    "Pakistan":              ["pakistan", "islamabad", "karachi", "lahore", "peshawar"],
    "India":                 ["india", "indian", "delhi", "mumbai", "modi", "new delhi"],
    "China":                 ["china", "chinese", "beijing", "shanghai", "xi jinping"],
    "Japan":                 ["japan", "japanese", "tokyo", "osaka"],
    "South Korea":           ["south korea", "korean", "seoul"],
    "North Korea":           ["north korea", "pyongyang", "kim jong"],
    "Afghanistan":           ["afghanistan", "afghan", "kabul", "taliban"],
    "Bangladesh":            ["bangladesh", "dhaka"],
    "Myanmar":               ["myanmar", "burma", "yangon"],
    # Middle East
    "Iran":                  ["iran", "iranian", "tehran", "khamenei"],
    "Israel":                ["israel", "israeli", "tel aviv", "netanyahu", "gaza", "west bank"],
    "Saudi Arabia":          ["saudi", "saudi arabia", "riyadh", "mbs"],
    "Turkey":                ["turkey", "turkish", "ankara", "erdogan", "türkiye"],
    "Iraq":                  ["iraq", "iraqi", "baghdad"],
    "Syria":                 ["syria", "syrian", "damascus"],
    "Yemen":                 ["yemen", "yemeni", "houthi", "sanaa"],
    "Lebanon":               ["lebanon", "lebanese", "beirut", "hezbollah"],
    # Europe
    "Russia":                ["russia", "russian", "moscow", "putin", "kremlin"],
    "Ukraine":               ["ukraine", "ukrainian", "kyiv", "zelensky", "zelenskyy"],
    "United Kingdom":        ["uk", "britain", "british", "london", "england", "scotland"],
    "France":                ["france", "french", "paris", "macron"],
    "Germany":               ["germany", "german", "berlin", "scholz"],
    "Poland":                ["poland", "polish", "warsaw"],
    "Spain":                 ["spain", "spanish", "madrid"],
    "Italy":                 ["italy", "italian", "rome"],
    "Serbia":                ["serbia", "serbian", "belgrade"],
    # Americas
    "United States":         ["united states", "usa", "american", "washington", "trump", "biden", "white house", "congress", "pentagon"],
    "Canada":                ["canada", "canadian", "ottawa", "toronto", "trudeau"],
    "Mexico":                ["mexico", "mexican", "mexico city"],
    "Brazil":                ["brazil", "brazilian", "brasilia", "lula"],
    "Argentina":             ["argentina", "argentine", "buenos aires"],
    "Venezuela":             ["venezuela", "venezuelan", "caracas", "maduro"],
    "Colombia":              ["colombia", "colombian", "bogota"],
    "Cuba":                  ["cuba", "cuban", "havana"],
    # Africa
    "Nigeria":               ["nigeria", "nigerian", "abuja", "lagos"],
    "South Africa":          ["south africa", "south african", "pretoria", "johannesburg"],
    "Sudan":                 ["sudan", "sudanese", "khartoum"],
    "Ethiopia":              ["ethiopia", "ethiopian", "addis ababa"],
    "Somalia":               ["somalia", "somali", "mogadishu", "al-shabaab"],
    "Libya":                 ["libya", "libyan", "tripoli"],
    "Egypt":                 ["egypt", "egyptian", "cairo"],
    "Kenya":                 ["kenya", "kenyan", "nairobi"],
    "Congo":                 ["congo", "congolese", "kinshasa", "drc"],
    # Oceania
    "Australia":             ["australia", "australian", "canberra", "sydney"],
}

COUNTRY_COORDS = {
    "Pakistan":              [30.3753, 69.3451],
    "India":                 [20.5937, 78.9629],
    "China":                 [35.8617, 104.1954],
    "Japan":                 [36.2048, 138.2529],
    "South Korea":           [35.9078, 127.7669],
    "North Korea":           [40.3399, 127.5101],
    "Afghanistan":           [33.9391, 67.7100],
    "Bangladesh":            [23.6850, 90.3563],
    "Myanmar":               [21.9162, 95.9560],
    "Iran":                  [32.4279, 53.6880],
    "Israel":                [31.0461, 34.8516],
    "Saudi Arabia":          [23.8859, 45.0792],
    "Turkey":                [38.9637, 35.2433],
    "Iraq":                  [33.2232, 43.6793],
    "Syria":                 [34.8021, 38.9968],
    "Yemen":                 [15.5527, 48.5164],
    "Lebanon":               [33.8547, 35.8623],
    "Russia":                [61.5240, 105.3188],
    "Ukraine":               [48.3794, 31.1656],
    "United Kingdom":        [55.3781, -3.4360],
    "France":                [46.2276, 2.2137],
    "Germany":               [51.1657, 10.4515],
    "Poland":                [51.9194, 19.1451],
    "Spain":                 [40.4637, -3.7492],
    "Italy":                 [41.8719, 12.5674],
    "Serbia":                [44.0165, 21.0059],
    "United States":         [37.0902, -95.7129],
    "Canada":                [56.1304, -106.3468],
    "Mexico":                [23.6345, -102.5528],
    "Brazil":                [-14.2350, -51.9253],
    "Argentina":             [-38.4161, -63.6167],
    "Venezuela":             [6.4238, -66.5897],
    "Colombia":              [4.5709, -74.2973],
    "Cuba":                  [21.5218, -77.7812],
    "Nigeria":               [9.0820, 8.6753],
    "South Africa":          [-30.5595, 22.9375],
    "Sudan":                 [12.8628, 30.2176],
    "Ethiopia":              [9.1450, 40.4897],
    "Somalia":               [5.1521, 46.1996],
    "Libya":                 [26.3351, 17.2283],
    "Egypt":                 [26.8206, 30.8025],
    "Kenya":                 [-0.0236, 37.9062],
    "Congo":                 [-4.0383, 21.7587],
    "Australia":             [-25.2744, 133.7751],
}

NEGATIVE = [
    "war", "attack", "bomb", "conflict", "violence", "protest", "military",
    "strike", "killed", "explosion", "crisis", "tension", "terror", "troops",
    "missile", "shooting", "coup", "sanction", "arrested", "riot", "casualt",
    "invasion", "airstrike", "refugee", "hostage", "nuclear", "threat", "unrest"
]


def fetch_risk_headlines():
    """Pull headlines from all dedicated global risk feeds."""
    all_headlines = []
    for url in RISK_FEEDS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries:
                all_headlines.append(e.title.lower())
        except Exception:
            continue
    return all_headlines


def calculate_risk():
    # Use dedicated global risk feeds — much broader coverage
    headlines = fetch_risk_headlines()

    scores = {c: 0 for c in COUNTRY_KEYWORDS}

    for text in headlines:
        for country, keys in COUNTRY_KEYWORDS.items():
            if any(k in text for k in keys):
                scores[country] += 1
                if any(n in text for n in NEGATIVE):
                    scores[country] += 2   # negative event bonus

    result = {}
    for c, s in scores.items():
        if s >= 8:
            result[c] = "critical"
        elif s >= 4:
            result[c] = "high"
        elif s >= 1:
            result[c] = "low"
        else:
            result[c] = "none"

    return result


# ---------------- ROUTES ----------------

@app.route("/")
def home():
    return render_template("index.html", categories=list(RSS_FEEDS.keys()))


@app.route("/api/news")
def news():
    cat = request.args.get("category", "Pakistan")
    query = request.args.get("q", "")
    live = get_live_headlines(cat, query)
    return jsonify({"success": True, "articles": summarize_with_ai(live, cat)})


@app.route("/api/summary")
def summary():
    # Collect top headlines from multiple global feeds for a world briefing
    try:
        world_headlines = []
        for url in [
            "https://feeds.bbci.co.uk/news/world/rss.xml",
            "https://www.aljazeera.com/xml/rss/all.xml",
            "https://feeds.reuters.com/reuters/worldNews",
        ]:
            try:
                feed = feedparser.parse(url)
                world_headlines.extend([e.title for e in feed.entries[:4]])
            except Exception:
                continue

        if not world_headlines:
            return jsonify({"success": False, "summary": "Intelligence feed synchronized."})

        prompt = (
            "You are a world news anchor. Based on these global headlines, write ONE "
            "concise 25-word briefing sentence summarizing the state of the world right now. "
            "Be specific, factual and impactful. No filler phrases.\n\nHeadlines:\n"
            + "\n".join(world_headlines[:12])
        )
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.4)
        )
        return jsonify({"success": True, "summary": response.text.strip()})
    except Exception:
        return jsonify({"success": False, "summary": "Global intelligence feed active."})


@app.route("/api/risk")
def risk():
    try:
        risk_data = calculate_risk()
        enriched = {}
        for country, level in risk_data.items():
            coords = COUNTRY_COORDS.get(country, [0, 0])
            if coords == [0, 0]:
                continue  # skip countries with no coordinates
            enriched[country] = {
                "level": level,
                "coords": coords
            }
        return jsonify({"success": True, "risk": enriched})
    except Exception as e:
        return jsonify({"success": False, "risk": {}})


if __name__ == "__main__":
    app.run(debug=True)
