"""
News Sentry – High‑Sensitivity News Monitor for D‑Wave Quantum Inc. (QBTS)
-----------------------------------------------------------------------
Pulls fresh headlines from **Google News RSS** (free, no API key), classifies
relevance using OpenAI, and emails you when litigation, earnings, tech, or
collaboration news is detected.

### Logging & Console Output
* **INFO** level log lines stream to **STDOUT**.
* Additionally, the script now uses `print()` so Docker/Kubernetes log
  collectors that only capture raw stdout (and ignore Python loggers) still
  show the key events:
  * 📰  New article headline
  * 🔍  OpenAI classification start
  * 🔔  Email alert dispatched

Prerequisites
=============
$ pip install feedparser apscheduler openai pydantic python-dotenv

Environment Variables (set in .env or your shell)
------------------------------------------------
OPENAI_API_KEY=<your_openai_key>
SMTP_HOST=smtp.gmail.com
SMTP_PORT=465
SMTP_USER=<your_email>
SMTP_PASS=<app_password>
ALERT_TO=<destination_email>

Run
===
$ python news_sentry.py   # logs + prints appear in the console

Docker
------
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python","news_sentry.py"]
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import smtplib
import sqlite3
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
TICKER = "AMD"
COMPANY = "Advanced Micro Devices"
QUERY = f"{TICKER} OR \"{COMPANY}\""
RSS_URL = (
    "https://news.google.com/rss/search?q=" + quote_plus(QUERY) + "&hl=en-US&gl=US&ceid=US:en"
)
KEYWORDS = {
    "litigation": ["lawsuit", "court", "legal", "litigation", "settlement"],
    "earnings": ["earnings", "results", "quarter", "guidance"],
    "tech": ["technology", "product", "update", "roadmap", "quantum"],
    "collaboration": ["partnership", "collaboration", "alliance", "agreement"],
}
CHECK_INTERVAL_SECONDS = 300
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

def init_db():
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


def mark_seen(article: Article):
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
        "You are a financial news assistant. Return JSON: {\"relevant\": bool, \"sentiment\": int} "
        "where relevant is true iff the text concerns litigation, earnings, tech developments, or collaborations for D‑Wave Quantum Inc. (QBTS). "
        "Sentiment: -5 very negative, 0 neutral, 5 very positive."
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

def send_email(article: Article, sentiment: int):
    msg = EmailMessage()
    msg["Subject"] = f"QBTS alert ({sentiment:+}) – {article.headline}"
    msg["From"] = os.environ["SMTP_USER"]
    msg["To"] = os.environ["ALERT_TO"]
    body = (
        f"Headline: {article.headline}\nSentiment: {sentiment:+}\nPublished at: {article.published_at}\nURL: {article.url}\n"
    )
    msg.set_content(body)
    context = smtplib.ssl.create_default_context()
    with smtplib.SMTP_SSL(os.environ["SMTP_HOST"], int(os.environ.get("SMTP_PORT", 465)), context=context) as server:
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

def job():
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
    scheduler = BackgroundScheduler(timezone=timezone.utc)
    scheduler.add_job(job, IntervalTrigger(seconds=CHECK_INTERVAL_SECONDS), next_run_time=datetime.utcnow())
    scheduler.start()
    logger.info("Scheduler started – polling every %d s (CTRL+C to stop)", CHECK_INTERVAL_SECONDS)
    print("Scheduler started – polling every", CHECK_INTERVAL_SECONDS, "seconds (CTRL+C to stop)")
    try:
        import time
        while True:
            time.sleep
