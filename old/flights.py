"""Flight fare monitor → multi‑provider search, .env secrets, e‑mail alerts, and
optional APScheduler polling loop.

Command‑line example (poll every 30
min):
    python flight_search.py --origin TLV --dest ANY --month 2025-08 \
        --passengers 2 --weekend yes --price-limit 350 --every 30

Install deps:
    pip install python-dotenv requests apscheduler fast-flights

Required .env (or shell) keys:
    # Alert mail (SMTP_SSL only)
    ALERTS_SMTP_SERVER=smtp.gmail.com
    ALERTS_SMTP_PORT=465
    ALERTS_SMTP_USER=my@gmail.com
    ALERTS_SMTP_PASS=app‑password
    ALERTS_EMAIL_FROM="Flights Bot <bot@my.co>"   # optional
    ALERTS_EMAIL_TO=ronen@example.com,team@example.com
"""



from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import logging
import os
import smtplib
import sys
import time
from dataclasses import asdict, dataclass
from email.message import EmailMessage
from typing import Any, Optional

import requests
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from fast_flights import get_flights, FlightData, Passengers, Result

###############################################################################
# Logging
###############################################################################
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

###############################################################################
# Environment / secrets
###############################################################################
load_dotenv()  # pull variables from .env into os.environ

# Alert mail configuration
a_missing = []
for key in (
    "ALERTS_SMTP_SERVER",
    "ALERTS_SMTP_PORT",
    "ALERTS_SMTP_USER",
    "ALERTS_SMTP_PASS",
    "ALERTS_EMAIL_TO",
):
    if not os.getenv(key):
        a_missing.append(key)
if a_missing:
    logger.warning("Missing alert env keys: %s", ", ".join(a_missing))

SMTP_SERVER = os.getenv("ALERTS_SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("ALERTS_SMTP_PORT", "465"))
SMTP_USER = os.getenv("ALERTS_SMTP_USER", "")
SMTP_PASS = os.getenv("ALERTS_SMTP_PASS", "")
EMAIL_FROM = os.getenv("ALERTS_EMAIL_FROM", SMTP_USER)
EMAIL_TO = os.getenv("ALERTS_EMAIL_TO", EMAIL_FROM)

###############################################################################
# Data model
###############################################################################


@dataclass
class FlightOffer:
    origin: str
    destination: str
    departure: str  # YYYY
    return_date: str  # YYYY
    price: float    # total price for all passengers in provider currency
    currency: str
    deep_link: Optional[str]

###############################################################################
# Helper functions
###############################################################################


def first_day_and_last_day(year: int, month: int) -> tuple[dt.date, dt.date]:
    last_day = calendar.monthrange(year, month)[1]
    return dt.date(year, month, 1), dt.date(year, month, last_day)


def is_weekend(date_str: str) -> bool:
    return dt.date.fromisoformat(date_str).weekday() >= 4  # Fri


def within_price_limit(price: float, limit: float | None) -> bool:
    return limit is None or price <= limit

###############################################################################
# Google Flights provider
###############################################################################


def search_google_flights(origin: str, dest: str, dep_from: dt.date, dep_to: dt.date, passengers: int, trip_duration: int) -> list[FlightOffer]:
    """
    Scrapes Google Flights for the given parameters.
    Note: This uses an unofficial library and may be brittle.
    """
    offers: list[FlightOffer] = []
    day = dep_from
    while day <= dep_to:
        try:
            return_date = day + dt.timedelta(days=trip_duration)
            logger.info(f"Scraping flights from {origin} to {dest} from {day.isoformat()} to {return_date.isoformat()}...")
            
            result: Result = get_flights(
                flight_data=[
                    FlightData(date=day.isoformat(), from_airport=origin, to_airport=dest),
                    FlightData(date=return_date.isoformat(), from_airport=dest, to_airport=origin)
                ],
                trip="round-trip",
                seat="economy",
                passengers=Passengers(adults=passengers),
            )

            if result and result.flights:
                for flight in result.flights:
                    offers.append(
                        FlightOffer(
                            origin=origin,
                            destination=dest,
                            departure=flight.departure.strftime("%Y-%m-%d"),
                            return_date=(flight.departure + dt.timedelta(days=trip_duration)).strftime("%Y-%m-%d"),
                            price=float(flight.price.replace("$", "").replace(",", "")),
                            currency="USD",
                            deep_link=None,
                        )
                    )
            else:
                logger.info("No data returned from scraper.")
        except Exception as e:
            logger.exception(f"Failed to scrape Google Flights for {day.isoformat()}: {e}")
        
        day += dt.timedelta(days=1)
        time.sleep(2)

    logger.debug("Google Flights offers: %d", len(offers))
    return offers

###############################################################################
# Core gather
###############################################################################


def gather_offers(origin: str, dest: str, month: str, passengers: int, weekend_only: bool, price_limit: float | None, trip_duration: int) -> list[FlightOffer]:
    year, month_int = map(int, month.split("-"))
    dep_from, dep_to = first_day_and_last_day(year, month_int)

    offers = search_google_flights(origin, dest, dep_from, dep_to, passengers, trip_duration)

    # filters
    offers = [o for o in offers if (not weekend_only or is_weekend(o.departure)) and within_price_limit(o.price, price_limit)]

    # dedup unique key
    uniq: dict[tuple, FlightOffer] = {}
    for o in offers:
        uniq[(o.origin, o.destination, o.departure, o.return_date, o.price)] = o
    return sorted(uniq.values(), key=lambda x: x.price)


###############################################################################
# Alert mail
###############################################################################


def send_email(subject: str, body: str) -> None:
    if not (SMTP_USER and SMTP_PASS):
        logger.warning("Email alerts are not configured, skipping sending email")
        return
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg.set_content(body)
    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as s:
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        logger.info("\ud83d\udce8 Alert sent: %s", subject)
    except smtplib.SMTPException:
        logger.exception("SMTP failure when sending alert")


###############################################################################
# CLI + scheduler
###############################################################################


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser("Flight fare monitor")
    p.add_argument("--origin", required=True)
    p.add_argument("--dest", default="ANY")
    p.add_argument("--month", required=True, help="YYYY-MM")
    p.add_argument("--passengers", type=int, default=1)
    p.add_argument("--duration", type=int, default=7, help="Trip duration in days")
    p.add_argument("--weekend", choices=["yes", "no"], default="no")
    p.add_argument("--price-limit", type=float, default=None)
    p.add_argument("--json", action="store_true")
    p.add_argument("--every", type=int, default=0, metavar="MINUTES", help="Polling interval (0 = run once)")
    return p.parse_args(argv)


def job(args: argparse.Namespace) -> None:
    offers = gather_offers(
        origin=args.origin.upper(),
        dest=args.dest.upper(),
        month=args.month,
        passengers=args.passengers,
        weekend_only=(args.weekend == "yes"),
        price_limit=args.price_limit,
        trip_duration=args.duration,
    )

    if args.json:
        print(json.dumps([asdict(o) for o in offers], ensure_ascii=False, indent=2))
        return

    if not offers:
        logger.info("No flights found matching criteria")
        return

    logger.info("Found %d flights", len(offers))

    lines: list[str] = []
    for o in offers:
        txt = f"{o.origin}->{o.destination} {o.departure}\u2013{o.return_date} | {o.price:.0f} {o.currency}"
        if o.deep_link:
            txt += f" \u2192 {o.deep_link}"
        print(txt)
        lines.append(txt)

    send_email(
        subject=f"\u2708\ufe0f {len(offers)} flight(s) from {args.origin.upper()} in {args.month}",
        body="\n".join(lines),
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.every > 0:
        sched = BackgroundScheduler()
        sched.add_job(lambda: job(args), "interval", minutes=args.every)
        logger.info("Scheduler started \u2013 every %d min (Ctrl\u2011C to exit)", args.every)
        job(args)  # immediate first run
        sched.start()
        try:
            while True:
                time.sleep(3600)
        except (KeyboardInterrupt, SystemExit):
            sched.shutdown()
    else:
        job(args)


if __name__ == "__main__":
    main()
