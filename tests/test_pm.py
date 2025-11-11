import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import importlib
import pandas as pd
from datetime import datetime, time

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set dummy environment variables before importing the module to prevent sys.exit()
os.environ['ALERTS_EMAIL_FROM'] = 'dummy@example.com'
os.environ['ALERTS_EMAIL_TO'] = 'dummy@example.com'
os.environ['ALERTS_SMTP_SERVER'] = 'dummy'
os.environ['ALERTS_SMTP_PORT'] = '0'
os.environ['ALERTS_SMTP_USER'] = 'dummy'
os.environ['ALERTS_SMTP_PASS'] = 'dummy'

import pm

class TestPriceMonitor(unittest.TestCase):

    def setUp(self):
        # Reload module before each test to reset its state and clear sent alerts
        importlib.reload(pm)
        pm.ALERTS_SENT_TODAY.clear()

    @patch.dict(os.environ, {
        "ALERTS_TICKERS": "TEST1,TEST2",
        "ALERTS_THRESHOLD": "10",
    })
    def test_config_loading(self):
        importlib.reload(pm)  # Reload to pick up patched env vars
        self.assertEqual(pm.TICKERS, ["TEST1", "TEST2"])
        self.assertEqual(pm.THRESHOLD, 10.0)

    @patch('pm.dt')
    def test_market_session_now(self, mock_dt):
        # Mocking datetime to control the time
        with patch('pm.ET_TZ', pm.ZoneInfo("US/Eastern")):
            mock_dt.datetime.now.return_value = datetime(2025, 7, 14, 10, 0).replace(tzinfo=pm.ET_TZ)
            self.assertEqual(pm.market_session_now(), "open")

            mock_dt.datetime.now.return_value = datetime(2025, 7, 14, 8, 0).replace(tzinfo=pm.ET_TZ)
            self.assertEqual(pm.market_session_now(), "pre")

            mock_dt.datetime.now.return_value = datetime(2025, 7, 14, 17, 0).replace(tzinfo=pm.ET_TZ)
            self.assertEqual(pm.market_session_now(), "post")

    @patch('pm.yf.Ticker')
    @patch('pm.dt.datetime')
    def test_fetch_price_and_change_success(self, mock_datetime, mock_ticker):
        # Mock time to be within market hours
        mock_now = datetime(2025, 7, 14, 10, 30)
        with patch('pm.NY_TZ', pm.pytz.timezone("America/New_York")) as mock_tz:
            mock_now_ny = mock_now.replace(tzinfo=mock_tz)
            mock_datetime.now.return_value = mock_now_ny

            mock_instance = mock_ticker.return_value
            
            # Mock for daily history call
            daily_df = pd.DataFrame({'Open': [100.0]}, index=[pd.to_datetime(mock_now_ny.date())])
            
            # Mock for minute history call
            minute_df = pd.DataFrame({'Close': [105.0]}, index=[pd.to_datetime(mock_now_ny)])

            mock_instance.history.side_effect = [daily_df, minute_df]
            
            result = pm.fetch_price_and_change("TEST")
            
            self.assertIsNotNone(result)
            change, price = result
            self.assertAlmostEqual(change, 5.0)
            self.assertAlmostEqual(price, 105.0)

    @patch('pm.send_email')
    @patch('pm.fetch_price_and_change')
    def test_check_ticker_sends_alert_on_threshold(self, mock_fetch, mock_send_email):
        pm.THRESHOLD = 5.0
        mock_fetch.return_value = (6.0, 106.0)  # 6% change, above threshold
        pm.check_ticker("TEST")
        mock_send_email.assert_called_once()

    @patch('pm.send_email')
    @patch('pm.fetch_price_and_change')
    def test_check_ticker_does_not_send_if_below_threshold(self, mock_fetch, mock_send_email):
        pm.THRESHOLD = 5.0
        mock_fetch.return_value = (4.0, 104.0)  # 4% change, below threshold
        pm.check_ticker("TEST")
        mock_send_email.assert_not_called()

    @patch('pm.send_email')
    @patch('pm.fetch_price_and_change')
    def test_check_ticker_does_not_send_if_already_alerted(self, mock_fetch, mock_send_email):
        pm.THRESHOLD = 5.0
        mock_fetch.return_value = (6.0, 106.0)
        pm.ALERTS_SENT_TODAY.add("TEST")
        pm.check_ticker("TEST")
        mock_send_email.assert_not_called()

if __name__ == '__main__':
    unittest.main()