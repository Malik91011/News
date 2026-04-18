import os
import json
import feedparser
import time
import requests
import logging
from flask import Flask, request, jsonify, render_template

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── SIMPLE IN-MEMORY CACHE ──────────────────────────────────
_cache = {}   # key -> (timestamp, value)
CACHE_TTL = 300  # 5 minutes

def cache_get(key):
    entry = _cache.get(key)
    if entry and (time.time() - entry[0]) < CACHE_TTL:
        return entry[1]
    return None

def cache_set(key, value):
    _cache[key] = (time.time(), value)

base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, '../templates')
app = Flask(__name__, template_folder=template_dir)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# Models tried in order — first success wins
GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
]
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

def gemini(prompt, temperature=0.3):
    """Call Gemini REST API — one attempt per model, fail fast, no waits."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set")
    last_err = None
    for model in GEMINI_MODELS:
        url = f"{GEMINI_BASE}/{model}:generateContent?key={GEMINI_API_KEY}"
        try:
            resp = requests.post(
                url,
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": temperature, "maxOutputTokens": 2048}
                },
                timeout=20
            )
            if resp.status_code in (404, 429):
                logger.warning(f"Model {model} returned {resp.status_code}, trying next")
                last_err = resp.status_code
                continue
            resp.raise_for_status()
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            logger.info(f"Gemini success: {model}")
            return text.strip()
        except Exception as e:
            last_err = e
            logger.warning(f"Model {model} failed: {e}")
            continue
    raise Exception(f"All Gemini models failed. Last: {last_err}")

# ─── RSS FEEDS ────────────────────────────────────────────────

RSS_FEEDS = {
    "Pakistan": None,
    "World":    "https://www.aljazeera.com/xml/rss/all.xml",
    "Politics": "https://tribune.com.pk/feed/pakistan",
    "Technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "Business": "https://www.dawn.com/feeds/business",
    "Sports":   None,
}

PAKISTAN_FEEDS = [
    "https://www.dawn.com/feeds/home",
    "https://arynews.tv/feed/",
    "https://tribune.com.pk/feed/home",
    "https://www.geo.tv/rss/1/7",
]

SPORTS_FEEDS = [
    "https://feeds.bbci.co.uk/sport/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://arynews.tv/category/sports/feed/",
]

RISK_FEEDS = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://feeds.skynews.com/feeds/rss/world.xml",
    "https://www.dawn.com/feeds/home",
    "https://arynews.tv/feed/",
    "https://tribune.com.pk/feed/home",
]

# ─── NEWS ─────────────────────────────────────────────────────

def fetch_feed_safe(url):
    try:
        return feedparser.parse(url).entries
    except Exception:
        return []

def get_live_headlines(category, query=None):
    if category == "Pakistan":
        all_entries = []
        for url in PAKISTAN_FEEDS:
            entries = fetch_feed_safe(url)
            src = ("Dawn" if "dawn" in url else
                   "ARY News" if "arynews" in url else
                   "Tribune" if "tribune" in url else "Geo News")
            for e in entries:
                e["_src"] = src
            all_entries.extend(entries)

    elif category == "Sports":
        sports_kw = ["sport","cricket","football","soccer","tennis","psl","fifa",
                     "olympic","match","league","cup","hockey","rugby","golf",
                     "f1","racing","champion","wicket","innings","goal","score"]
        all_entries = []
        for url in SPORTS_FEEDS:
            entries = fetch_feed_safe(url)
            src = ("BBC Sport" if "bbc" in url else
                   "Al Jazeera" if "aljazeera" in url else "ARY Sports")
            for e in entries:
                tl = e.title.lower()
                if "aljazeera" in url and not any(k in tl for k in sports_kw):
                    continue
                e["_src"] = src
                all_entries.append(e)
    else:
        url = RSS_FEEDS.get(category, RSS_FEEDS["World"])
        all_entries = fetch_feed_safe(url)
        for e in all_entries:
            e["_src"] = category

    all_entries.sort(key=lambda x: x.get("published_parsed", time.gmtime(0)), reverse=True)

    if query:
        q = query.lower()
        all_entries = [e for e in all_entries
                       if q in e.title.lower() or q in getattr(e, "summary", "").lower()]

    seen, unique = set(), []
    for e in all_entries:
        k = e.title.lower().strip()
        if k not in seen:
            seen.add(k)
            unique.append(e)

    return [{"title": e.title, "link": e.link,
             "published": e.get("published", "Just Now"),
             "source_label": getattr(e, "_src", category)} for e in unique[:12]]


# ─── KEYWORD-BASED LOCAL ANALYSIS (no API needed) ────────────

RISK_WORDS   = {"war","attack","bomb","conflict","killed","explosion","crisis",
                "terror","missile","troops","invasion","airstrike","coup",
                "violence","shooting","nuclear","sanctions","arrested"}
ECON_WORDS   = {"economy","inflation","stock","market","trade","gdp","deficit",
                "recession","currency","oil","gas","price","rate","debt","tax"}
HEALTH_WORDS = {"disease","outbreak","virus","pandemic","hospital","death",
                "health","medical","vaccine","drug","infection","casualties"}
TECH_WORDS   = {"ai","tech","cyber","hack","data","software","startup","robot",
                "space","satellite","launch","digital","electric","energy"}

def local_analyze(title):
    """Generate assessment + precaution from headline keywords — zero API calls."""
    t = title.lower()

    if any(w in t for w in RISK_WORDS):
        assessment = "This event carries potential for regional or international escalation and warrants close monitoring."
        precaution = "Avoid the affected region if possible and follow official government travel advisories."
    elif any(w in t for w in ECON_WORDS):
        assessment = "Economic developments of this nature can have downstream effects on markets, trade, and consumer prices."
        precaution = "Review any exposure to affected markets or currencies and consult a financial advisor if needed."
    elif any(w in t for w in HEALTH_WORDS):
        assessment = "Health-related developments require timely public awareness and coordinated institutional response."
        precaution = "Follow guidance from local health authorities and avoid crowded areas if an outbreak is indicated."
    elif any(w in t for w in TECH_WORDS):
        assessment = "Technological shifts of this kind can rapidly reshape industries, privacy norms, and national security postures."
        precaution = "Stay informed about implications for data privacy and infrastructure security in your region."
    else:
        assessment = "The situation is developing and its full impact remains to be assessed by relevant authorities."
        precaution = "Monitor reputable news sources for updates and avoid sharing unverified information."

    return assessment, precaution


def summarize_with_ai(headlines, category):
    """Try Gemini once; if rate-limited fall back to local keyword analysis instantly."""
    if not headlines:
        return []

    cache_key = f"news_{category}"
    cached = cache_get(cache_key)
    if cached:
        logger.info(f"Cache HIT: {cache_key}")
        return cached

    def build_result(h, summary=None, assessment=None, precaution=None):
        a, p = local_analyze(h["title"])
        return {
            "title":      h["title"],
            "summary":    summary or h["title"],
            "assessment": assessment or a,
            "precaution": precaution or p,
            "url":        h["link"],
            "source":     h.get("source_label", category),
            "category":   category,
            "time":       h["published"],
        }

    # Try Gemini with a compact prompt
    items = [{"i": i, "t": h["title"]} for i, h in enumerate(headlines)]
    prompt = (
        "News analyst. Return a JSON array — one object per item. "
        "Keys: i (copy), s (1-sentence summary), a (1-sentence assessment), p (1-sentence precaution). "
        "No markdown. Start with [\n\n"
        + json.dumps(items)
    )

    try:
        raw = gemini(prompt, temperature=0.2)
        logger.info(f"Gemini OK: {raw[:120]}")

        text = raw.strip()
        if "```" in text:
            for part in text.split("```"):
                part = part.strip().lstrip("json").strip()
                if part.startswith("["):
                    text = part
                    break
        s, e = text.find("["), text.rfind("]") + 1
        if s != -1 and e > s:
            text = text[s:e]

        parsed = json.loads(text)
        idx_map = {obj.get("i", ix): obj for ix, obj in enumerate(parsed)}

        results = []
        for i, h in enumerate(headlines):
            obj = idx_map.get(i, {})
            results.append(build_result(
                h,
                summary    = obj.get("s"),
                assessment = obj.get("a"),
                precaution = obj.get("p"),
            ))
        cache_set(cache_key, results)
        logger.info(f"Gemini results cached for {cache_key}")
        return results

    except Exception as ex:
        logger.warning(f"Gemini unavailable ({ex}), using local analysis")
        # Local analysis: no API, instant, always works
        results = [build_result(h) for h in headlines]
        cache_set(cache_key, results)   # cache local results too
        return results



# ─── RISK ENGINE ─────────────────────────────────────────────

COUNTRY_KEYWORDS = {
    "Pakistan":     ["pakistan","islamabad","karachi","lahore","peshawar"],
    "India":        ["india","indian","delhi","mumbai","modi","new delhi"],
    "China":        ["china","chinese","beijing","shanghai","xi jinping"],
    "Japan":        ["japan","japanese","tokyo","osaka"],
    "South Korea":  ["south korea","korean","seoul"],
    "North Korea":  ["north korea","pyongyang","kim jong"],
    "Afghanistan":  ["afghanistan","afghan","kabul","taliban"],
    "Iran":         ["iran","iranian","tehran","khamenei"],
    "Israel":       ["israel","israeli","tel aviv","netanyahu","gaza","west bank"],
    "Saudi Arabia": ["saudi","saudi arabia","riyadh"],
    "Turkey":       ["turkey","turkish","ankara","erdogan"],
    "Iraq":         ["iraq","iraqi","baghdad"],
    "Syria":        ["syria","syrian","damascus"],
    "Yemen":        ["yemen","yemeni","houthi"],
    "Lebanon":      ["lebanon","lebanese","beirut","hezbollah"],
    "Russia":       ["russia","russian","moscow","putin","kremlin"],
    "Ukraine":      ["ukraine","ukrainian","kyiv","zelensky"],
    "United Kingdom":["uk","britain","british","london","england"],
    "France":       ["france","french","paris","macron"],
    "Germany":      ["germany","german","berlin"],
    "Poland":       ["poland","polish","warsaw"],
    "Spain":        ["spain","spanish","madrid"],
    "Italy":        ["italy","italian","rome"],
    "United States":["united states","usa","american","washington","trump","white house","congress","pentagon"],
    "Canada":       ["canada","canadian","ottawa","toronto"],
    "Mexico":       ["mexico","mexican","mexico city"],
    "Brazil":       ["brazil","brazilian","brasilia","lula"],
    "Argentina":    ["argentina","buenos aires"],
    "Venezuela":    ["venezuela","venezuelan","caracas","maduro"],
    "Colombia":     ["colombia","colombian","bogota"],
    "Nigeria":      ["nigeria","nigerian","abuja","lagos"],
    "South Africa": ["south africa","pretoria","johannesburg"],
    "Sudan":        ["sudan","sudanese","khartoum"],
    "Ethiopia":     ["ethiopia","addis ababa"],
    "Somalia":      ["somalia","mogadishu","al-shabaab"],
    "Libya":        ["libya","libyan","tripoli"],
    "Egypt":        ["egypt","egyptian","cairo"],
    "Congo":        ["congo","congolese","kinshasa","drc"],
    "Australia":    ["australia","australian","canberra","sydney"],
}

COUNTRY_COORDS = {
    "Pakistan":       [30.3753,  69.3451],
    "India":          [20.5937,  78.9629],
    "China":          [35.8617, 104.1954],
    "Japan":          [36.2048, 138.2529],
    "South Korea":    [35.9078, 127.7669],
    "North Korea":    [40.3399, 127.5101],
    "Afghanistan":    [33.9391,  67.7100],
    "Iran":           [32.4279,  53.6880],
    "Israel":         [31.0461,  34.8516],
    "Saudi Arabia":   [23.8859,  45.0792],
    "Turkey":         [38.9637,  35.2433],
    "Iraq":           [33.2232,  43.6793],
    "Syria":          [34.8021,  38.9968],
    "Yemen":          [15.5527,  48.5164],
    "Lebanon":        [33.8547,  35.8623],
    "Russia":         [61.5240, 105.3188],
    "Ukraine":        [48.3794,  31.1656],
    "United Kingdom": [55.3781,  -3.4360],
    "France":         [46.2276,   2.2137],
    "Germany":        [51.1657,  10.4515],
    "Poland":         [51.9194,  19.1451],
    "Spain":          [40.4637,  -3.7492],
    "Italy":          [41.8719,  12.5674],
    "United States":  [37.0902, -95.7129],
    "Canada":         [56.1304,-106.3468],
    "Mexico":         [23.6345,-102.5528],
    "Brazil":         [-14.2350,-51.9253],
    "Argentina":      [-38.4161,-63.6167],
    "Venezuela":      [  6.4238,-66.5897],
    "Colombia":       [  4.5709,-74.2973],
    "Nigeria":        [  9.0820,  8.6753],
    "South Africa":   [-30.5595, 22.9375],
    "Sudan":          [ 12.8628, 30.2176],
    "Ethiopia":       [  9.1450, 40.4897],
    "Somalia":        [  5.1521, 46.1996],
    "Libya":          [ 26.3351, 17.2283],
    "Egypt":          [ 26.8206, 30.8025],
    "Congo":          [ -4.0383, 21.7587],
    "Australia":      [-25.2744,133.7751],
}

NEGATIVE = ["war","attack","bomb","conflict","violence","protest","military",
            "strike","killed","explosion","crisis","tension","terror","troops",
            "missile","shooting","coup","sanction","riot","casualt","invasion",
            "airstrike","refugee","hostage","nuclear","threat","unrest","dead"]

def calculate_risk():
    headlines = []
    for url in RISK_FEEDS:
        try:
            for e in feedparser.parse(url).entries:
                headlines.append(e.title.lower())
        except Exception:
            continue

    scores = {c: 0 for c in COUNTRY_KEYWORDS}
    for text in headlines:
        for country, keys in COUNTRY_KEYWORDS.items():
            if any(k in text for k in keys):
                scores[country] += 1
                if any(n in text for n in NEGATIVE):
                    scores[country] += 2

    result = {}
    for c, s in scores.items():
        if s >= 8:   result[c] = "critical"
        elif s >= 4: result[c] = "high"
        elif s >= 1: result[c] = "low"
        else:        result[c] = "none"
    return result


# ─── ROUTES ──────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html", categories=list(RSS_FEEDS.keys()))


@app.route("/api/news")
def news():
    cat   = request.args.get("category", "Pakistan")
    query = request.args.get("q", "")
    live  = get_live_headlines(cat, query)
    return jsonify({"success": True, "articles": summarize_with_ai(live, cat)})


@app.route("/api/summary")
def summary():
    """World briefing for the status bar — cached 5 min to avoid rate limits."""
    cached = cache_get("world_summary")
    if cached:
        logger.info("Cache HIT: world_summary")
        return jsonify({"success": True, "summary": cached})

    headlines = []
    for url in ["https://feeds.bbci.co.uk/news/world/rss.xml",
                "https://www.aljazeera.com/xml/rss/all.xml",
                "https://feeds.skynews.com/feeds/rss/world.xml"]:
        try:
            for e in feedparser.parse(url).entries[:4]:
                headlines.append(e.title)
        except Exception:
            continue

    if not headlines:
        return jsonify({"success": False, "summary": "Intelligence feed synchronized."})

    prompt = (
        "You are a world news anchor. Write ONE sharp, factual 25-word sentence "
        "summarizing the biggest story in the world right now based on these headlines. "
        "Be specific. No filler phrases like 'in a world of'. No intro.\n\n"
        "Headlines:\n" + "\n".join(headlines[:12])
    )
    try:
        text = gemini(prompt, temperature=0.3)
        text = text.strip().strip('"').strip("'")
        cache_set("world_summary", text)
        return jsonify({"success": True, "summary": text})
    except Exception as ex:
        logger.error(f"summary FAILED: {ex}")
        fallback = headlines[0] if headlines else "Intelligence feed synchronized."
        return jsonify({"success": False, "summary": fallback})


@app.route("/api/risk")
def risk():
    try:
        enriched = {}
        for country, level in calculate_risk().items():
            coords = COUNTRY_COORDS.get(country)
            if coords:
                enriched[country] = {"level": level, "coords": coords}
        return jsonify({"success": True, "risk": enriched})
    except Exception as ex:
        logger.error(f"risk FAILED: {ex}")
        return jsonify({"success": False, "risk": {}})


if __name__ == "__main__":
    app.run(debug=True)
