"""
News Sentry – High‑Sensitivity News Monitor for D‑Wave Quantum Inc. (QBTS)
-----------------------------------------------------------------------
Pulls fresh headlines from **Google News RSS** (free, no API key), classifies
relevance using OpenAI, and emails you when any of the following topics are
mentioned in connection with QBTS / D‑Wave Quantum:

* Litigation  
* Earnings / financial results  
* Technology developments  
* Collaborations / partnerships  
* **Geo‑political events** (sanctions, export controls, trade policy, conflicts, government legislation, etc.)

### Logging & Console Output
* Structured `logging` output (INFO level) streams to **STDOUT**.
* Matching `print()` statements echo the key events (📰 / 🔍 / 🔔) so runtimes that
  only capture raw stdout still show what’s happening.

Prerequisites
=============
```bash
pip install feedparser apscheduler openai pydantic python-dotenv
```

Environment Variables (set in `.env` or your shell)
--------------------------------------------------
```
OPENAI_API_KEY=<your_openai_key>
SMTP_HOST=smtp.gmail.com
SMTP_PORT=465
SMTP_USER=<your_email>
SMTP_PASS=<app_password>
ALERT_TO=<destination_email>
```

Run
===
```bash
python news_sentry.py  # logs + prints appear in console
```
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import smtplib
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import List
from urllib.parse import quote_plus

import email.utils as eut
import feedparser
import openai
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from pydantic import BaseModel, HttpUrl

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger("news_sentry")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TICKER = "QBTS"
COMPANY = "D-Wave Quantum"
QUERY = f"{TICKER} OR \"{COMPANY}\""
RSS_URL = (
    "https://news.google.com/rss/search?q=" + quote_plus(QUERY) + "&hl=en-US&gl=US&ceid=US:en"
)

KEYWORDS = {
    "litigation": [
        "lawsuit",
        "court",
        "legal",
        "litigation",
        "settlement",
    ],
    "earnings": [
        "earnings",
        "results",
        "quarter",
        "guidance",
        "revenue",
        "profit",
    ],
    "tech": [
        "technology",
        "product",
        "update",
        "roadmap",
        "quantum",
        "release",
        "launch",
    ],
    "collaboration": [
        "partnership",
        "collaboration",
        "alliance",
        "agreement",
        "joint venture",
        "mou",
    ],
    "geo_political": [
        "sanction",
        "export control",
        "tariff",
        "trade war",
        "regulation",
        "legislation",
        "government",
        "policy",
        "geopolitical",
        "conflict",
        "war",
        "ukraine",
        "israel hamas",
        "china",
        "us-china",
        "eu",
    ],
}
CHECK_INTERVAL_SECONDS = 300  # 5‑minute poll
DB_PATH = "news_sentry.db"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class Article(BaseModel):
    id: str
    published_at: datetime
    headline: str
    summary: str | None = None
    url: HttpUrl

# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def init_db() -> None:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS articles (
                   id TEXT PRIMARY KEY,
                   published_at TEXT
               )"""
        )
        conn.commit()
    logger.info("SQLite store initialised at %s", DB_PATH)


def already_seen(article_id: str) -> bool:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        cur = conn.execute("SELECT 1 FROM articles WHERE id = ?", (article_id,))
        return cur.fetchone() is not None


def mark_seen(article: Article) -> None:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO articles (id, published_at) VALUES (?, ?)",
            (article.id, article.published_at.isoformat()),
        )
        conn.commit()

# ---------------------------------------------------------------------------
# Fetch – Google News RSS
# ---------------------------------------------------------------------------

def fetch_news(since: datetime) -> List[Article]:
    feed = feedparser.parse(RSS_URL)
    articles: List[Article] = []
    for entry in feed.entries:
        try:
            pub = eut.parsedate_to_datetime(entry.published)
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if pub < since:
            continue
        url = entry.link
        aid = hashlib.sha256(url.encode()).hexdigest()
        art = Article(
            id=aid,
            published_at=pub,
            headline=entry.title,
            summary=getattr(entry, "summary", None),
            url=url,
        )
        articles.append(art)
    return articles

# ---------------------------------------------------------------------------
# OpenAI classification
# ---------------------------------------------------------------------------

def classify(article: Article) -> tuple[bool, int]:
    logger.info("🔍 Classifying article with OpenAI: %s", article.headline)
    print("🔍 Classifying:", article.headline)
    system = (
        "You are a financial news assistant. Return JSON: {\\\"relevant\\\": bool, \\\"sentiment\\\": int} where relevant is true iff the text concerns "
        "litigation, earnings, tech developments, collaborations, or geo‑political events (sanctions, export controls, legislation, conflicts, etc.) "
        "for D‑Wave Quantum Inc. (QBTS). Sentiment scale: -5 very negative, 0 neutral, 5 very positive."
    )
    user = f"Headline: {article.headline}\nSummary: {article.summary or ''}"
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        max_tokens=50,
    )
    data = json.loads(resp.choices[0].message.content)
    return data.get("relevant", False), int(data.get("sentiment", 0))

# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

def send_email(article: Article, sentiment: int) -> None:
    msg = EmailMessage()
    msg["Subject"] = f"QBTS alert ({sentiment:+}) – {article.headline}"
    msg["From"] = os.environ["SMTP_USER"]
    msg["To"] = os.environ["ALERT_TO"]
    body = (
        f"Headline: {article.headline}\n"
        f"Sentiment: {sentiment:+}\n"
        f"Published at: {article.published_at}\n"
        f"URL: {article.url}\n"
    )
    msg.set_content(body)
    context = smtplib.ssl.create_default_context()
    with smtplib.SMTP_SSL(
        os.environ["SMTP_HOST"], int(os.environ.get("SMTP_PORT", 465)), context=context
    ) as server:
        server.login(os.environ["SMTP_USER"], os.environ["SMTP_PASS"])
        server.send_message(msg)
    logger.info("🔔 Email sent: %s", article.headline)
    print("🔔 Email sent:", article.headline)

# ---------------------------------------------------------------------------
# Quick keyword filter
# ---------------------------------------------------------------------------

def quick_filter(article: Article) -> bool:
    text = (article.headline + " " + (article.summary or "")).lower()
    if TICKER.lower() not in text and COMPANY.lower() not in text:
        return False
    return any(any(word in text for word in words) for words in KEYWORDS.values())

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def job() -> None:
    since = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(hours=6)
    alerts_sent = 0
    for art in fetch_news(since):
        if already_seen(art.id):
            continue
        if not quick_filter(art):
            continue
        logger.info("📰 New article detected: %s", art.headline)
        print("📰 New article:", art.headline)
        relevant, sentiment = classify(art)
        if relevant:
            send_email(art, sentiment)
            alerts_sent += 1
        mark_seen(art)
    logger.info("Cycle complete – %d alert(s) sent", alerts_sent)

# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    scheduler = BackgroundScheduler(timezone
