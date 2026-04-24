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

# ── COUNTRY/REGION DETECTION ─────────────────────────────────
REGION_MAP = {
    "pakistan": "Pakistan", "india": "India", "china": "China", "russia": "Russia",
    "ukraine": "Ukraine", "israel": "Israel", "iran": "Iran", "gaza": "Gaza",
    "usa": "United States", "america": "United States", "europe": "Europe",
    "middle east": "Middle East", "africa": "Africa", "nato": "NATO countries",
    "turkey": "Turkey", "saudi": "Saudi Arabia", "yemen": "Yemen",
    "korea": "Korean Peninsula", "taiwan": "Taiwan Strait",
}
INDUSTRY_MAP = {
    "oil": "Energy", "gas": "Energy", "opec": "Energy", "petroleum": "Energy",
    "bank": "Finance", "stock": "Finance", "market": "Finance", "crypto": "Finance",
    "bitcoin": "Finance", "trade": "Trade & Commerce", "tariff": "Trade & Commerce",
    "tech": "Technology", "ai": "Technology", "cyber": "Technology",
    "pharma": "Healthcare", "vaccine": "Healthcare", "hospital": "Healthcare",
    "military": "Defense", "army": "Defense", "weapon": "Defense", "nuclear": "Defense",
    "food": "Agriculture", "grain": "Agriculture", "wheat": "Agriculture",
    "shipping": "Logistics", "port": "Logistics", "supply chain": "Logistics",
}

def local_analyze(title):
    """Extended local analysis — returns full decision intelligence dict."""
    t = title.lower()

    # Detect affected countries
    affected_countries = list({v for k, v in REGION_MAP.items() if k in t})[:4]
    # Detect affected industries
    affected_industries = list({v for k, v in INDUSTRY_MAP.items() if k in t})[:4]

    if any(w in t for w in RISK_WORDS):
        return {
            "assessment": "Signals active escalation. Regional stability is directly threatened.",
            "importance": "high",
            "advisory": "Avoid affected region. Follow government travel advisories. Verify information before acting.",
            "impact": "Increases risk to regional security, civilian movement, and international supply lines.",
            "affected_countries": affected_countries or ["Regional"],
            "affected_industries": affected_industries or ["Defense", "Humanitarian"],
            "next_triggers": "Track ceasefire status, troop deployments, and UN Security Council responses within 24h.",
            "confidence_score": "medium",
            "why_this_matters": "Unchecked escalation triggers broader conflict and international intervention."
        }
    elif any(w in t for w in ECON_WORDS):
        return {
            "assessment": "Applies direct pressure on markets, trade flows, and consumer prices.",
            "importance": "medium",
            "advisory": "Reassess market exposure. Monitor currency and commodity movements before committing capital.",
            "impact": "Drives volatility in equity markets, exchange rates, and import-dependent economies.",
            "affected_countries": affected_countries or ["Global"],
            "affected_industries": affected_industries or ["Finance", "Trade & Commerce"],
            "next_triggers": "Watch central bank statements, inflation data, and commodity indices within 48h.",
            "confidence_score": "medium",
            "why_this_matters": "Economic shocks propagate rapidly across interconnected global markets."
        }
    elif any(w in t for w in HEALTH_WORDS):
        return {
            "assessment": "Indicates active public health pressure requiring institutional response.",
            "importance": "medium",
            "advisory": "Follow WHO and local health authority directives. Limit non-essential exposure.",
            "impact": "Strains healthcare capacity, disrupts logistics, and elevates mortality risk in vulnerable populations.",
            "affected_countries": affected_countries or ["Regional"],
            "affected_industries": affected_industries or ["Healthcare", "Logistics"],
            "next_triggers": "Track WHO declarations, case trajectory, and border control measures within 72h.",
            "confidence_score": "medium",
            "why_this_matters": "Health crises cross borders fast and overwhelm underprepared systems."
        }
    elif any(w in t for w in TECH_WORDS):
        return {
            "assessment": "Reshapes competitive dynamics and exposes new regulatory and security vectors.",
            "importance": "low",
            "advisory": "Audit data exposure and infrastructure dependencies in affected technology sectors.",
            "impact": "Disrupts incumbents, shifts investment flows, and introduces new compliance requirements.",
            "affected_countries": affected_countries or ["Global"],
            "affected_industries": affected_industries or ["Technology", "Finance"],
            "next_triggers": "Monitor regulatory filings, competitor responses, and enterprise adoption signals.",
            "confidence_score": "low",
            "why_this_matters": "Technology shifts restructure industries faster than policy can respond."
        }
    else:
        return {
            "assessment": "Situation is evolving. Intelligence confidence remains limited at this stage.",
            "importance": "low",
            "advisory": "Track verified sources only. Withhold judgment until confirmed reporting emerges.",
            "impact": "Scope and downstream effects remain unquantified pending further reporting.",
            "affected_countries": affected_countries or [],
            "affected_industries": affected_industries or [],
            "next_triggers": "Await official statements and second-source confirmation within 24h.",
            "confidence_score": "low",
            "why_this_matters": "Early signals often precede larger developments. Situational awareness is critical."
        }

# ── TASKS ────────────────────────────────────────────────────

def task_summarize_articles(headlines):
    """GROQ → DeepSeek → Gemini → OpenRouter for bulk summarization."""
    items = [{"i": i, "t": h["title"]} for i, h in enumerate(headlines)]
    prompt = (
        "You are a senior intelligence analyst. For each item write a direct, declarative 1-sentence summary. "
        "Use strong, active language. Never use: may, could, might, suggests, appears. "
        "Use instead: confirms, signals, triggers, indicates, drives, forces, increases risk. "
        "Max 20 words per summary. Return a JSON array. Each object: i (copy number), s (summary). "
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
    """GROQ → DeepSeek → Gemini → OpenRouter for full decision intelligence."""
    items = [{"i": i, "t": h["title"]} for i, h in enumerate(headlines)]
    prompt = (
        "You are a senior geopolitical intelligence analyst. For each headline return a JSON array. "
        "Use authoritative, declarative language. Never use: may, could, might, suggests, appears, seems. "
        "Use instead: signals, confirms, indicates, drives, forces, triggers, increases, threatens. "
        "Each object must have exactly these keys: "
        "i (copy number), "
        "importance (low|medium|high), "
        "impact (1 direct sentence — active voice, max 15 words), "
        "affected_countries (array of up to 3 country/region names), "
        "affected_industries (array of up to 3 sector names), "
        "next_triggers (1 sentence: specific indicators to watch in next 24-72h), "
        "confidence_score (low|medium|high), "
        "why_this_matters (max 10 words, punchy and specific), "
        "bias (neutral|slightly_left|slightly_right|unknown). "
        "No markdown. Return only the JSON array starting with [\n\n"
        + json.dumps(items)
    )
    for fn, name in [(call_groq, "GROQ"), (call_deepseek, "DeepSeek"), (call_gemini, "Gemini"), (call_openrouter, "OpenRouter")]:
        try:
            raw = fn(prompt, temperature=0.2, max_tokens=1500)
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
        "You are an intelligence briefing officer. "
        "For each headline write a direct, actionable 1-sentence advisory. "
        "Be specific and decisive. Never hedge with: may, could, might, consider, perhaps. "
        "Use: monitor, avoid, reassess, verify, track, act, prepare. Max 18 words. "
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

# ── DEDUPLICATION + IMPORTANCE SCORING ──────────────────────
SEVERITY_KEYWORDS = {
    "war": 10, "nuclear": 10, "invasion": 9, "airstrike": 9, "coup": 9,
    "collapse": 8, "explosion": 8, "killed": 8, "crisis": 7, "sanctions": 7,
    "missile": 7, "terror": 7, "attack": 6, "conflict": 6, "protest": 5,
    "election": 5, "summit": 5, "deal": 4, "talks": 3, "report": 2,
}

def score_headline(title):
    """Score a headline by severity keyword weight."""
    t = title.lower()
    return sum(v for k, v in SEVERITY_KEYWORDS.items() if k in t)

def deduplicate_headlines(headlines):
    """Remove near-duplicate headlines using word overlap."""
    unique = []
    seen_words = []
    for h in headlines:
        words = set(h["title"].lower().split())
        # Remove common stop words
        words -= {"the","a","an","in","on","at","to","of","and","or","for","is","are","was","were","as","by"}
        is_dup = any(len(words & prev) / max(len(words | prev), 1) > 0.55 for prev in seen_words)
        if not is_dup:
            unique.append(h)
            seen_words.append(words)
    return unique

# ── MAIN ORCHESTRATOR ────────────────────────────────────────
def orchestrate(headlines, category):
    """Dispatch tasks to APIs, deduplicate, score, and cache results."""
    if not headlines:
        return []

    cache_key = f"news_{category}"
    cached = cache_get(cache_key)
    if cached:
        logger.info(f"Cache HIT: {cache_key}")
        return cached

    # Deduplicate first
    headlines = deduplicate_headlines(headlines)
    # Sort by importance score
    headlines = sorted(headlines, key=lambda h: score_headline(h["title"]), reverse=True)

    logger.info(f"Orchestrating {len(headlines)} articles for category: {category}")

    summaries   = task_summarize_articles(headlines)
    assessments = task_assess_articles(headlines)
    advisories  = task_advisory_articles(headlines)

    results = []
    for i, h in enumerate(headlines):
        local = local_analyze(h["title"])

        assessment_obj = assessments.get(i, {})
        importance  = assessment_obj.get("importance", local["importance"])
        impact      = assessment_obj.get("impact", local["assessment"])
        bias        = assessment_obj.get("bias", "unknown")
        assessment_text = impact
        if bias not in ("unknown", "neutral", ""):
            assessment_text += f" (Framing: {bias.replace('_', ' ')})"

        article = {
            "title":              h["title"],
            "summary":            summaries.get(i) or h["title"],
            "assessment":         assessment_text or local["assessment"],
            "precaution":         advisories.get(i) or local["advisory"],
            "importance":         importance,
            "impact":             assessment_obj.get("impact", local["impact"]),
            "affected_countries": assessment_obj.get("affected_countries", local["affected_countries"]),
            "affected_industries":assessment_obj.get("affected_industries", local["affected_industries"]),
            "next_triggers":      assessment_obj.get("next_triggers", local["next_triggers"]),
            "confidence_score":   assessment_obj.get("confidence_score", local["confidence_score"]),
            "why_this_matters":   assessment_obj.get("why_this_matters", local["why_this_matters"]),
            "severity_score":     score_headline(h["title"]),
            "source_count":       h.get("source_count", 1),
            "credibility":        h.get("credibility", "Low"),
            "url":                h["link"],
            "source":             h.get("source_label", category),
            "category":           category,
            "time":               h["published"],
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

    # Count how many distinct sources covered each story (multi-source credibility)
    title_sources = {}
    for e in all_entries:
        key = e.title.lower().strip()
        src = getattr(e, "_src", category)
        if key not in title_sources:
            title_sources[key] = set()
        title_sources[key].add(src)

    result = []
    for e in unique[:8]:
        key = e.title.lower().strip()
        source_count = len(title_sources.get(key, {1}))
        credibility = "High" if source_count >= 3 else "Medium" if source_count == 2 else "Low"
        result.append({
            "title":          e.title,
            "link":           e.link,
            "published":      e.get("published", "Just Now"),
            "source_label":   getattr(e, "_src", category),
            "source_count":   source_count,
            "credibility":    credibility,
        })
    return result

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

SEVERITY_WEIGHTS = {
    "war": 5, "nuclear": 5, "invasion": 5, "airstrike": 4, "coup": 4,
    "collapse": 4, "explosion": 4, "killed": 3, "crisis": 3, "sanctions": 3,
    "missile": 3, "terror": 3, "attack": 2, "conflict": 2, "protest": 2,
    "tension": 2, "troops": 2, "threat": 2, "riot": 2,
}

def calculate_risk():
    """Calculate risk with 0-100 numeric scores and trend data."""
    all_headlines = []
    for url in RISK_FEEDS:
        try:
            for e in feedparser.parse(url).entries:
                all_headlines.append(e.title.lower())
        except Exception:
            continue

    # Raw mention + severity scores
    raw_scores = {c: 0 for c in COUNTRY_KEYWORDS}
    country_headlines = {c: [] for c in COUNTRY_KEYWORDS}
    
    for text in all_headlines:
        for country, keys in COUNTRY_KEYWORDS.items():
            if any(k in text for k in keys):
                severity = sum(v for k, v in SEVERITY_WEIGHTS.items() if k in text)
                raw_scores[country] += 1 + severity
                country_headlines[country].append(text)

    # Normalize to 0-100
    max_score = max(raw_scores.values()) if raw_scores.values() else 1
    max_score = max(max_score, 1)

    result = {}
    for c, s in raw_scores.items():
        risk_score = min(int((s / max_score) * 100), 100)
        if s >= 8:   level = "critical"
        elif s >= 4: level = "high"
        elif s >= 1: level = "low"
        else:        level = "none"
        
        # Detect top threats from headlines
        threats = []
        for h in country_headlines[c][:10]:
            for kw in ["war", "attack", "sanctions", "nuclear", "coup", "crisis", "invasion", "missile", "explosion"]:
                if kw in h and kw not in threats:
                    threats.append(kw)
        
        hl = country_headlines[c]
        conflict_score  = min(int(sum(SEVERITY_WEIGHTS.get(k,0) for h in hl for k in ["war","invasion","airstrike","missile","troops","coup"] if k in h) / max(len(hl),1) * 25), 100)
        political_score = min(int(sum(SEVERITY_WEIGHTS.get(k,0) for h in hl for k in ["sanctions","coup","election","protest","government","diplomatic"] if k in h) / max(len(hl),1) * 25), 100)
        economic_score  = min(int(sum(SEVERITY_WEIGHTS.get(k,0) for h in hl for k in ["oil","currency","inflation","debt","recession","market","collapse"] if k in h) / max(len(hl),1) * 25), 100)
        news_intensity  = min(len(hl) * 8, 100)
        result[c] = {
            "level": level,
            "risk_score": risk_score,
            "threats": threats[:3],
            "recent_headlines": country_headlines[c][:5],
            "risk_trend": "increasing" if s > 8 else "stable" if s > 2 else "decreasing",
            "breakdown": {
                "conflict":   conflict_score,
                "political":  political_score,
                "economic":   economic_score,
                "intensity":  news_intensity,
            }
        }
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
    cached = cache_get("risk_data")
    if cached:
        return jsonify({"success": True, "risk": cached})
    try:
        enriched = {}
        risk_data = calculate_risk()
        for country, data in risk_data.items():
            coords = COUNTRY_COORDS.get(country)
            if coords:
                enriched[country] = {
                    "level":            data["level"],
                    "risk_score":       data["risk_score"],
                    "threats":          data["threats"],
                    "risk_trend":       data["risk_trend"],
                    "recent_headlines": data["recent_headlines"],
                    "breakdown":        data.get("breakdown", {}),
                    "coords":           coords
                }
        cache_set("risk_data", enriched)
        return jsonify({"success": True, "risk": enriched})
    except Exception as ex:
        logger.error(f"risk FAILED: {ex}")
        return jsonify({"success": False, "risk": {}})

@app.route("/api/global-brief")
def global_brief():
    """Top 5-7 most important global events for the Risk Brief section."""
    cached = cache_get("global_brief")
    if cached:
        return jsonify({"success": True, "events": cached})
    
    # Pull from risk feeds and score
    all_items = []
    for url in RISK_FEEDS:
        try:
            for e in feedparser.parse(url).entries[:8]:
                score = score_headline(e.title)
                if score > 0:
                    local = local_analyze(e.title)
                    all_items.append({
                        "title":           e.title,
                        "why_this_matters": local["why_this_matters"],
                        "impact":          local["impact"],
                        "importance":      local["importance"],
                        "affected_countries": local["affected_countries"],
                        "severity_score":  score,
                        "source":          url.split("/")[2].replace("www.", "").replace("feeds.", ""),
                    })
        except Exception:
            continue

    # Deduplicate and take top 7
    seen = set()
    unique = []
    for item in sorted(all_items, key=lambda x: x["severity_score"], reverse=True):
        words = frozenset(item["title"].lower().split()[:5])
        if words not in seen:
            seen.add(words)
            unique.append(item)
        if len(unique) >= 7:
            break

    cache_set("global_brief", unique)
    return jsonify({"success": True, "events": unique})

BREAKING_KEYWORDS = {"war","invasion","nuclear","coup","airstrike","explosion","escalation","collapse","offensive","ceasefire broken","emergency"}

@app.route("/api/top-risks")
def top_risks():
    """Top 5 global risks + breaking alert detection."""
    cached = cache_get("top_risks")
    if cached:
        return jsonify({"success": True, **cached})
    try:
        risk_data = calculate_risk()
        ranked = sorted(
            [(c, d) for c, d in risk_data.items() if d["risk_score"] > 0],
            key=lambda x: x[1]["risk_score"], reverse=True
        )[:5]
        result = [{"country": c, **d} for c, d in ranked]

        # Detect breaking alerts: top country with critical level + breaking keyword
        alerts = []
        for c, d in ranked[:3]:
            if d["level"] == "critical":
                for h in d.get("recent_headlines", []):
                    if any(kw in h for kw in BREAKING_KEYWORDS):
                        alerts.append({"country": c, "headline": h.title(), "score": d["risk_score"]})
                        break

        payload = {"risks": result, "alerts": alerts[:2]}
        cache_set("top_risks", payload)
        return jsonify({"success": True, **payload})
    except Exception as ex:
        logger.error(f"top-risks FAILED: {ex}")
        return jsonify({"success": False, "risks": [], "alerts": []})

if __name__ == "__main__":
    app.run(debug=True)
