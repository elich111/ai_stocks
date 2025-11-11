import datetime as dt
import sys
import os
from unittest.mock import patch, MagicMock
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flights import first_day_and_last_day, is_weekend, within_price_limit, search_google_flights, FlightOffer


def test_first_day_and_last_day():
    """Tests the first_day_and_last_day function."""
    first, last = first_day_and_last_day(2025, 8)
    assert first == dt.date(2025, 8, 1)
    assert last == dt.date(2025, 8, 31)

def test_is_weekend():
    """Tests the is_weekend function."""
    assert is_weekend("2025-07-11") is True # Friday
    assert is_weekend("2025-07-12") is True # Saturday
    assert is_weekend("2025-07-13") is True # Sunday
    assert is_weekend("2025-07-14") is False # Monday

def test_within_price_limit():
    """Tests the within_price_limit function."""
    assert within_price_limit(100, 200) is True
    assert within_price_limit(200, 200) is True
    assert within_price_limit(201, 200) is False
    assert within_price_limit(100, None) is True

@patch('flights.get_flights')
def test_search_google_flights(mock_get_flights):
    """Tests the search_google_flights function with a mocked scraper."""
    mock_flight = MagicMock()
    mock_flight.departure = dt.datetime(2025, 8, 1)
    mock_flight.price = "$1,000"

    mock_result = MagicMock()
    mock_result.flights = [mock_flight]
    mock_get_flights.return_value = mock_result

    dep_from = dt.date(2025, 8, 1)
    dep_to = dt.date(2025, 8, 1)

    offers = search_google_flights('TLV', 'JFK', dep_from, dep_to, 1, 7)

    assert len(offers) == 1
    assert offers[0].price == 1000
    mock_get_flights.assert_called_once()