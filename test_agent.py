"""Unit and mock tests for Weather Prediction Agent."""

import unittest
from unittest.mock import patch, MagicMock

from weather_service import (
    calculate_temperature_trend,
    search_city,
    fetch_weather_data,
)
from weather_agent import WeatherPredictionAgent


class TestWeatherService(unittest.TestCase):
    def test_calculate_trend_downward(self):
        # Temperatures decreasing steadily
        past_temps = [35.0, 34.0, 33.0, 32.0, 31.0, 30.0, 29.0]
        trend = calculate_temperature_trend(past_temps)

        self.assertEqual(trend["trend_direction"], "downward")
        self.assertAlmostEqual(trend["slope"], -1.0, places=1)
        self.assertEqual(trend["max_temp"], 35.0)
        self.assertEqual(trend["min_temp"], 29.0)
        self.assertAlmostEqual(trend["projected_temp"], 28.0, places=1)

    def test_calculate_trend_upward(self):
        # Temperatures increasing steadily
        past_temps = [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0]
        trend = calculate_temperature_trend(past_temps)

        self.assertEqual(trend["trend_direction"], "upward")
        self.assertAlmostEqual(trend["slope"], 1.0, places=1)
        self.assertEqual(trend["min_temp"], 20.0)
        self.assertEqual(trend["max_temp"], 26.0)
        self.assertAlmostEqual(trend["projected_temp"], 27.0, places=1)

    def test_calculate_trend_stable(self):
        # Temperatures virtually identical
        past_temps = [25.0, 25.1, 24.9, 25.0, 25.2, 25.0, 25.0]
        trend = calculate_temperature_trend(past_temps)

        self.assertEqual(trend["trend_direction"], "stable")
        self.assertEqual(trend["avg_temp"], 25.0)

    @patch("requests.get")
    def test_search_city_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {
                    "name": "Hyderabad",
                    "country": "India",
                    "latitude": 17.385,
                    "longitude": 78.4867,
                    "timezone": "Asia/Kolkata",
                }
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = search_city("Hyderabad")
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Hyderabad")
        self.assertEqual(result["country"], "India")
        self.assertEqual(result["latitude"], 17.385)

    @patch("requests.get")
    def test_search_city_not_found(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"results": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = search_city("NonExistentCityXYZ123")
        self.assertIsNone(result)

    @patch("requests.get")
    def test_fetch_weather_data(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "daily": {
                "time": [
                    "2026-11-09",
                    "2026-11-10",
                    "2026-11-11",
                    "2026-11-12",
                    "2026-11-13",
                    "2026-11-14",
                    "2026-11-15",
                    "2026-11-16",
                    "2026-11-17",
                ],
                "temperature_2m_max": [33.1, 32.5, 32.0, 31.8, 31.5, 31.3, 31.0, 31.1, 31.2],
                "temperature_2m_min": [22.0, 21.8, 21.5, 21.2, 21.0, 20.9, 20.8, 20.7, 20.8],
            }
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        data = fetch_weather_data(17.385, 78.4867)
        self.assertIsNotNone(data)
        self.assertEqual(len(data["past_7_days"]), 7)
        self.assertEqual(data["tomorrow_forecast"]["max_temp"], 31.2)
        self.assertEqual(data["tomorrow_forecast"]["min_temp"], 20.8)


class TestWeatherPredictionAgent(unittest.TestCase):
    @patch("weather_agent.search_city")
    @patch("weather_agent.fetch_weather_data")
    def test_agent_predict_analytical_fallback(self, mock_fetch, mock_search):
        mock_search.return_value = {
            "name": "Hyderabad",
            "country": "India",
            "latitude": 17.385,
            "longitude": 78.4867,
            "timezone": "Asia/Kolkata",
        }
        mock_fetch.return_value = {
            "past_7_days": [
                {"date": f"2026-11-0{i}", "max_temp": 33.0 - (i * 0.3), "min_temp": 21.0}
                for i in range(7)
            ],
            "tomorrow_forecast": {"date": "2026-11-17", "max_temp": 31.2, "min_temp": 20.8},
        }

        agent = WeatherPredictionAgent()
        # Force ollama unavailable to test fallback
        agent.is_ollama_available = MagicMock(return_value=False)

        result = agent.predict("Hyderabad", show_thinking=False)
        self.assertTrue(result["success"])
        self.assertIn("Prediction: Tomorrow's maximum will be around", result["prediction_text"])
        self.assertIn("Official forecast for tomorrow → max 31.2°C, min 20.8°C", result["forecast_line"])

    @patch("weather_agent.search_city")
    def test_agent_predict_city_not_found(self, mock_search):
        mock_search.return_value = None
        agent = WeatherPredictionAgent()

        result = agent.predict("UnknownCity", show_thinking=False)
        self.assertFalse(result["success"])
        self.assertIn("wasn't found", result["error"])


if __name__ == "__main__":
    unittest.main()
