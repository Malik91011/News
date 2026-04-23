import os
import json
import feedparser
import time
import requests
import logging
from flask import Flask, request, jsonify, render_template

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── IN-MEMORY CACHE ─────────────────────────────────────────
_cache = {}
CACHE_TTL = 300  # 5 minutes

def cache_get(key):
    entry = _cache.get(key)
    if entry and (time.time() - entry[0]) < CACHE_TTL:
        logger.info(f"Cache HIT: {key}")
        return entry[1]
    return None

def cache_set(key, value):
    _cache[key] = (time.time(), value)

# ─── APP ─────────────────────────────────────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, '../templates')
app = Flask(__name__, template_folder=template_dir)

# ─── API KEYS ────────────────────────────────────────────────
GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# ═══════════════════════════════════════════════════════════════
#  API LAYER
# ═══════════════════════════════════════════════════════════════

def call_groq(prompt, temperature=0.3, max_tokens=1024):
    """GROQ — ultra-fast, low cost."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set")
    models = ["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"]
    last_err = None
    for model in models:
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}],
                      "temperature": temperature, "max_tokens": max_tokens},
                timeout=15
            )
            if resp.status_code == 429:
                logger.warning(f"GROQ {model} 429, trying next")
                last_err = 429
                continue
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            logger.info(f"GROQ success: {model}")
            return text
        except Exception as e:
            last_err = e
            logger.warning(f"GROQ {model} failed: {e}")
            continue
    raise Exception(f"GROQ all models failed. Last: {last_err}")

def call_deepseek(prompt, temperature=0.3, max_tokens=1024):
    """DEEPSEEK — strong reasoning."""
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY not set")
    try:
        resp = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": "deepseek-chat",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": temperature, "max_tokens": max_tokens},
            timeout=20
        )
        if resp.status_code == 429:
            raise Exception("DeepSeek 429 rate limited")
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        logger.info("DeepSeek success")
        return text
    except Exception as e:
        raise Exception(f"DeepSeek failed: {e}")

def call_gemini(prompt, temperature=0.3, max_tokens=2048):
    """GEMINI — best reasoning."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set")
    base = "https://generativelanguage.googleapis.com/v1beta/models"
    models = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash"]
    last_err = None
    for model in models:
        try:
            resp = requests.post(
                f"{base}/{model}:generateContent?key={GEMINI_API_KEY}",
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"temperature": temperature,
                                           "maxOutputTokens": max_tokens}},
                timeout=20
            )
            if resp.status_code in (404, 429):
                logger.warning(f"Gemini {model} {resp.status_code}, trying next")
                last_err = resp.status_code
                continue
            resp.raise_for_status()
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            logger.info(f"Gemini success: {model}")
            return text.strip()
        except Exception as e:
            last_err = e
            logger.warning(f"Gemini {model} failed: {e}")
            continue
    raise Exception(f"Gemini all models failed. Last: {last_err}")

def call_openrouter(prompt, temperature=0.3, max_tokens=1024):
    """OPENROUTER — routes to multiple free models."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set")
    models = [
        "meta-llama/llama-4-scout:free",
        "meta-llama/llama-4-maverick:free",
        "google/gemma-3-27b-it:free",
        "mistralai/mistral-small-3.1-24b-instruct:free",
    ]
    last_err = None
    for model in models:
        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://pulseai.vercel.app",
                    "X-Title": "PulseAI News"
                },
                json={"model": model,
                      "messages": [{"role": "user", "content": prompt}],
                      "temperature": temperature, "max_tokens": max_tokens},
                timeout=20
            )
            if resp.status_code == 429:
                logger.warning(f"OpenRouter {model} 429, trying next")
                last_err = 429
                continue
            if resp.status_code in (400, 404):
                logger.warning(f"OpenRouter {model} {resp.status_code}, trying next")
                continue
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            logger.info(f"OpenRouter success: {model}")
            return text
        except Exception as e:
            last_err = e
            logger.warning(f"OpenRouter {model} failed: {e}")
            continue
    raise Exception(f"OpenRouter all models failed. Last: {last_err}")

# ═══════════════════════════════════════════════════════════════
#  ORCHESTRATION ENGINE
# ═══════════════════════════════════════════════════════════════

def extract_json_array(text):
    text = text.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip().lstrip("json").strip()
            if part.startswith("["):
                text = part
                break
    s, e = text.find("["), text.rfind("]") + 1
    if s != -1 and e > s:
        return json.loads(text[s:e])
    raise ValueError("No JSON array found")

def extract_json_object(text):
    text = text.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip().lstrip("json").strip()
            if part.startswith("{"):
                text = part
                break
    s, e = text.find("{"), text.rfind("}") + 1
    if s != -1 and e > s:
        return json.loads(text[s:e])
    raise ValueError("No JSON object found")

# ── LOCAL FALLBACK ───────────────────────────────────────────
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
    t = title.lower()
    if any(w in t for w in RISK_WORDS):
        return (
            "This event carries potential for regional or international escalation.",
            "high",
            "Follow official government travel advisories and monitor situation closely."
        )
    elif any(w in t for w in ECON_WORDS):
        return (
            "Economic developments of this nature can affect markets and consumer prices.",
            "medium",
            "Review market exposure and consult a financial advisor if needed."
        )
    elif any(w in t for w in HEALTH_WORDS):
        return (
            "Health developments require timely public awareness and institutional response.",
            "medium",
            "Follow local health authority guidance and avoid crowded areas if indicated."
        )
    elif any(w in t for w in TECH_WORDS):
        return (
            "Technological shifts can rapidly reshape industries and security postures.",
            "low",
            "Stay informed about data privacy and infrastructure security implications."
        )
    else:
        return (
            "Situation is developing; full impact remains to be assessed by authorities.",
            "low",
            "Monitor reputable news sources and avoid sharing unverified information."
        )

# ── TASKS ────────────────────────────────────────────────────

def task_summarize_articles(headlines):
    """GROQ → DeepSeek → Gemini → OpenRouter for bulk summarization."""
    items = [{"i": i, "t": h["title"]} for i, h in enumerate(headlines)]
    prompt = (
        "You are a factual news summarizer. For each item write a 1-sentence factual summary. "
        "Return a JSON array. Each object: i (copy number), s (1-sentence summary). "
        "No markdown. Start with [\n\n" + json.dumps(items)
    )
    for fn, name in [(call_groq, "GROQ"), (call_deepseek, "DeepSeek"), (call_gemini, "Gemini"), (call_openrouter, "OpenRouter")]:
        try:
            raw = fn(prompt, temperature=0.2, max_tokens=800)
            parsed = extract_json_array(raw)
            idx_map = {obj.get("i", ix): obj.get("s", "") for ix, obj in enumerate(parsed)}
            logger.info(f"Summarize via {name}")
            return {i: idx_map.get(i, "") for i in range(len(headlines))}
        except Exception as e:
            logger.warning(f"{name} summarize failed: {e}")
    return {}

def task_assess_articles(headlines):
    """GROQ → DeepSeek → Gemini → OpenRouter for structured assessment."""
    items = [{"i": i, "t": h["title"]} for i, h in enumerate(headlines)]
    prompt = (
        "You are a news intelligence analyst. For each headline provide a structured assessment. "
        "Return a JSON array. Each object must have: "
        "i (copy number), "
        "importance (low|medium|high), "
        "impact (1 sentence on potential consequences), "
        "bias (neutral|slightly_left|slightly_right|unknown). "
        "No markdown. Return only the JSON array starting with [\n\n"
        + json.dumps(items)
    )
    for fn, name in [(call_groq, "GROQ"), (call_deepseek, "DeepSeek"), (call_gemini, "Gemini"), (call_openrouter, "OpenRouter")]:
        try:
            raw = fn(prompt, temperature=0.2, max_tokens=1000)
            parsed = extract_json_array(raw)
            logger.info(f"Assessment via {name}")
            return {obj.get("i", ix): obj for ix, obj in enumerate(parsed)}
        except Exception as e:
            logger.warning(f"{name} assess failed: {e}")
    return {}

def task_advisory_articles(headlines):
    """GROQ → Gemini → DeepSeek → OpenRouter for per-article advisories."""
    items = [{"i": i, "t": h["title"]} for i, h in enumerate(headlines)]
    prompt = (
        "You are a responsible news advisor. "
        "For each headline write a 1-sentence practical advisory for the reader. "
        "Be neutral, factual, helpful. No sensationalism. "
        "Return a JSON array. Each object: i (copy number), p (1-sentence advisory). "
        "No markdown. Start with [\n\n" + json.dumps(items)
    )
    for fn, name in [(call_groq, "GROQ"), (call_gemini, "Gemini"), (call_deepseek, "DeepSeek"), (call_openrouter, "OpenRouter")]:
        try:
            raw = fn(prompt, temperature=0.3, max_tokens=800)
            parsed = extract_json_array(raw)
            logger.info(f"Advisory via {name}")
            return {obj.get("i", ix): obj.get("p", "") for ix, obj in enumerate(parsed)}
        except Exception as e:
            logger.warning(f"{name} advisory failed: {e}")
    return {}

def task_overall_summary(headlines):
    """GROQ → Gemini → DeepSeek → OpenRouter for status bar."""
    titles = [h["title"] for h in headlines[:12]]
    if not titles:
        return "Intelligence feed synchronized."
    prompt = (
        "You are a world news anchor. Based on these headlines, write ONE sharp factual "
        "25-word sentence summarizing the most important global story right now. "
        "Be specific. No filler phrases. Return only the sentence, nothing else.\n\n"
        "Headlines:\n" + "\n".join(titles)
    )
    for fn, name in [(call_groq, "GROQ"), (call_gemini, "Gemini"), (call_deepseek, "DeepSeek"), (call_openrouter, "OpenRouter")]:
        try:
            text = fn(prompt, temperature=0.3, max_tokens=120)
            logger.info(f"Overall summary via {name}")
            return text.strip().strip('"').strip("'")
        except Exception as e:
            logger.warning(f"{name} overall summary failed: {e}")
    return titles[0] if titles else "Intelligence feed synchronized."

# ── MAIN ORCHESTRATOR ────────────────────────────────────────
def orchestrate(headlines, category):
    """Dispatch tasks to APIs and cache results."""
    if not headlines:
        return []

    cache_key = f"news_{category}"
    cached = cache_get(cache_key)
    if cached:
        logger.info(f"Cache HIT: {cache_key}")
        return cached

    n = len(headlines)
    logger.info(f"Orchestrating {n} articles for category: {category}")

    summaries   = task_summarize_articles(headlines)
    assessments = task_assess_articles(headlines)
    advisories  = task_advisory_articles(headlines)

    results = []
    for i, h in enumerate(headlines):
        local_assess, local_importance, local_advisory = local_analyze(h["title"])

        assessment_obj = assessments.get(i, {})
        importance = assessment_obj.get("importance", local_importance)
        impact     = assessment_obj.get("impact", local_assess)
        bias       = assessment_obj.get("bias", "unknown")

        assessment_text = impact
        if bias not in ("unknown", "neutral", ""):
            assessment_text += f" (Framing: {bias.replace('_', ' ')})"

        article = {
            "title":      h["title"],
            "summary":    summaries.get(i) or h["title"],
            "assessment": assessment_text or local_assess,
            "precaution": advisories.get(i) or local_advisory,
            "importance": importance,
            "url":        h["link"],
            "source":     h.get("source_label", category),
            "category":   category,
            "time":       h["published"],
        }
        results.append(article)
        

    cache_set(cache_key, results)
    logger.info(f"Orchestration complete for {category}, cached.")
    return results

# ─── RSS FEEDS ────────────────────────────────────────────────

RSS_FEEDS = {
    "Pakistan":   None,
    "World":      "https://www.aljazeera.com/xml/rss/all.xml",
    "Politics":   "https://tribune.com.pk/feed/pakistan",
    "Technology": "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "Business":   "https://www.dawn.com/feeds/business",
    "Sports":     None,
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
             "source_label": getattr(e, "_src", category)} for e in unique[:8]]

# ─── RISK ENGINE ─────────────────────────────────────────────

COUNTRY_KEYWORDS = {
    "Pakistan":      ["pakistan","islamabad","karachi","lahore","peshawar"],
    "India":         ["india","indian","delhi","mumbai","modi","new delhi"],
    "China":         ["china","chinese","beijing","shanghai","xi jinping"],
    "Japan":         ["japan","japanese","tokyo","osaka"],
    "South Korea":   ["south korea","korean","seoul"],
    "North Korea":   ["north korea","pyongyang","kim jong"],
    "Afghanistan":   ["afghanistan","afghan","kabul","taliban"],
    "Iran":          ["iran","iranian","tehran","khamenei"],
    "Israel":        ["israel","israeli","tel aviv","netanyahu","gaza","west bank"],
    "Saudi Arabia":  ["saudi","saudi arabia","riyadh"],
    "Turkey":        ["turkey","turkish","ankara","erdogan"],
    "Iraq":          ["iraq","iraqi","baghdad"],
    "Syria":         ["syria","syrian","damascus"],
    "Yemen":         ["yemen","yemeni","houthi"],
    "Lebanon":       ["lebanon","lebanese","beirut","hezbollah"],
    "Russia":        ["russia","russian","moscow","putin","kremlin"],
    "Ukraine":       ["ukraine","ukrainian","kyiv","zelensky"],
    "United Kingdom":["uk","britain","british","london","england"],
    "France":        ["france","french","paris","macron"],
    "Germany":       ["germany","german","berlin"],
    "Poland":        ["poland","polish","warsaw"],
    "Spain":         ["spain","spanish","madrid"],
    "Italy":         ["italy","italian","rome"],
    "United States": ["united states","usa","american","washington","trump","white house","congress","pentagon"],
    "Canada":        ["canada","canadian","ottawa","toronto"],
    "Mexico":        ["mexico","mexican","mexico city"],
    "Brazil":        ["brazil","brazilian","brasilia","lula"],
    "Argentina":     ["argentina","buenos aires"],
    "Venezuela":     ["venezuela","venezuelan","caracas","maduro"],
    "Colombia":      ["colombia","colombian","bogota"],
    "Nigeria":       ["nigeria","nigerian","abuja","lagos"],
    "South Africa":  ["south africa","pretoria","johannesburg"],
    "Sudan":         ["sudan","sudanese","khartoum"],
    "Ethiopia":      ["ethiopia","addis ababa"],
    "Somalia":       ["somalia","mogadishu","al-shabaab"],
    "Libya":         ["libya","libyan","tripoli"],
    "Egypt":         ["egypt","egyptian","cairo"],
    "Congo":         ["congo","congolese","kinshasa","drc"],
    "Australia":     ["australia","australian","canberra","sydney"],
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
    query = request.args.get("q", "").strip()
    live  = get_live_headlines(cat, query)
    
    if query:
        # Search: bypass cache, filter results, skip AI (too slow for search)
        results = [{
            "title":      h["title"],
            "summary":    h["title"],
            "assessment": "",
            "precaution": "",
            "importance": "",
            "url":        h["link"],
            "source":     h.get("source_label", cat),
            "category":   cat,
            "time":       h["published"],
        } for h in live]
        logger.info(f"Search '{query}' returned {len(results)} results")
        return jsonify({"success": True, "articles": results})
    
    return jsonify({"success": True, "articles": orchestrate(live, cat)})

@app.route("/api/summary")
def summary():
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
                headlines.append({"title": e.title})
        except Exception:
            continue

    text = task_overall_summary(headlines)
    cache_set("world_summary", text)
    return jsonify({"success": True, "summary": text})

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
