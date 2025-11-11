import unittest
from unittest.mock import patch, MagicMock, mock_open

import ts

class TestTickerSearch(unittest.TestCase):

    @patch('ts.client.chat.completions.create')
    @patch('ts.yf.Ticker')
    @patch('ts.send_email')
    @patch('builtins.open', new_callable=mock_open, read_data='TEST;TICKER')
    def test_find_opportunities_with_opportunities(self, mock_file, mock_send_email, mock_ticker, mock_openai_create):
        """Test find_opportunities when opportunities are found."""
        mock_ticker.return_value.info = {
            'longName': 'Test Company',
            'trailingPE': 10,
            'forwardEps': 12,
            'trailingEps': 10,
            'currentPrice': 100,
        }
        
        # Create a mock for the chat completion response
        mock_chat_completion = MagicMock()
        mock_chat_completion.choices = [MagicMock()]
        mock_chat_completion.choices[0].message = MagicMock()
        mock_chat_completion.choices[0].message.content = "This is a test analysis."
        mock_openai_create.return_value = mock_chat_completion

        ts.find_opportunities()

        mock_send_email.assert_called_once()
        self.assertIn('Stock Opportunities Found!', mock_send_email.call_args[0][0])
        self.assertIn('AI Analysis: This is a test analysis.', mock_send_email.call_args[0][1])

        # Check the prompt
        mock_openai_create.assert_called_once()
        messages = mock_openai_create.call_args[1]['messages']
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[1]['role'], 'user')
        expected_prompt = "Provide a brief analysis for investing in Test Company (TEST). The current price is $100.00, P/E ratio is 10.00, and EPS growth is 20.00%. Is this a good investment opportunity and why? Provide a short summary."
        self.assertEqual(messages[1]['content'], expected_prompt)

    @patch('ts.yf.Ticker')
    @patch('ts.send_email')
    @patch('builtins.open', new_callable=mock_open, read_data='TEST;TICKER')
    def test_find_opportunities_no_opportunities(self, mock_file, mock_send_email, mock_ticker):
        """Test find_opportunities when no opportunities are found."""
        mock_ticker.return_value.info = {
            'longName': 'Test Company',
            'trailingPE': 30,  # High P/E
            'forwardEps': 10.5,
            'trailingEps': 10,
            'currentPrice': 100,
        }

        ts.find_opportunities()

        # This should result in a potential opportunity, so email should be sent
        mock_send_email.assert_called_once()
        self.assertIn('Stock Opportunities Found!', mock_send_email.call_args[0][0])
        self.assertIn('potential opportunities', mock_send_email.call_args[0][1])


    @patch('ts.yf.Ticker')
    @patch('ts.send_email')
    @patch('builtins.open', new_callable=mock_open, read_data='TEST;TICKER')
    def test_find_opportunities_missing_data(self, mock_file, mock_send_email, mock_ticker):
        """Test find_opportunities when a ticker has missing data."""
        mock_ticker.return_value.info = {
            'longName': 'Test Company',
            'trailingPE': 10,
            'forwardEps': 12,
            # Missing trailingEps
            'currentPrice': 100,
        }

        ts.find_opportunities()

        # This should result in a potential opportunity with low P/E
        mock_send_email.assert_called_once()
        self.assertIn('Stock Opportunities Found!', mock_send_email.call_args[0][0])
        self.assertIn('potential opportunities', mock_send_email.call_args[0][1])
        self.assertIn('Reason: Low P/E', mock_send_email.call_args[0][1])

    @patch('ts.yf.Ticker', side_effect=Exception("Test Exception"))
    @patch('ts.send_email')
    @patch('builtins.open', new_callable=mock_open, read_data='TEST;TICKER')
    def test_find_opportunities_yfinance_exception(self, mock_file, mock_send_email, mock_ticker):
        """Test find_opportunities when yfinance raises an exception."""
        ts.find_opportunities()

        mock_send_email.assert_not_called()

if __name__ == '__main__':
    unittest.main()