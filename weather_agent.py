"""Weather Prediction Agent implementation.

Coordinates weather data lookup, trend analysis, AI prediction via Ollama,
and comparison against official forecasts.
"""

import os
import urllib.request
import urllib.error
from typing import Any, Dict, Optional
from dotenv import load_dotenv

from weather_service import (
    search_city,
    fetch_weather_data,
    calculate_temperature_trend,
)

load_dotenv()


class WeatherPredictionAgent:
    """Agent that analyzes 7-day temperature trends and predicts tomorrow's maximum."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        ollama_host: Optional[str] = None,
        temperature: float = 0.2,
    ):
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3.2")
        self.ollama_host = ollama_host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.temperature = float(os.getenv("MODEL_TEMPERATURE", str(temperature)))

    def is_ollama_available(self, timeout: float = 2.0) -> bool:
        """Check if the local Ollama server is running and reachable."""
        try:
            url = f"{self.ollama_host.rstrip('/')}/api/tags"
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _generate_analytical_prediction(self, city_name: str, trend: Dict[str, Any]) -> str:
        """Fallback analytical prediction if Ollama is not running."""
        rounded_proj = round(trend["projected_temp"])
        slope = trend["slope"]
        sign = "+" if slope > 0 else ""
        slope_str = f"{sign}{slope:.1f}°C per day"

        if trend["trend_direction"] == "downward":
            desc = (
                f"The past week shows a slight downward trend of {slope_str},\n"
                f"suggesting a mild cool-down from the recent high of {trend['max_temp']:.0f}°C."
            )
        elif trend["trend_direction"] == "upward":
            desc = (
                f"The past week shows a slight upward trend of {slope_str},\n"
                f"suggesting continued warming above the recent average of {trend['avg_temp']:.0f}°C."
            )
        else:
            desc = (
                f"The past week shows a steady pattern ({slope_str}),\n"
                f"holding consistent near the recent average of {trend['avg_temp']:.0f}°C."
            )

        return f"Prediction: Tomorrow's maximum will be around {rounded_proj}°C.\n{desc}"

    def _generate_ai_prediction(
        self,
        city_name: str,
        country: str,
        past_temps: list,
        trend: Dict[str, Any],
    ) -> str:
        """Invoke local Ollama model to generate reasoned temperature prediction."""
        try:
            from langchain_ollama import ChatOllama
            from langchain_core.messages import SystemMessage, HumanMessage

            llm = ChatOllama(
                model=self.model_name,
                base_url=self.ollama_host,
                temperature=self.temperature,
            )

            system_prompt = (
                "You are an expert meteorological analyst AI. "
                "Your role is to predict tomorrow's maximum temperature for a city "
                "based on recent 7-day temperature trends.\n\n"
                "Output strictly in this format (no extra markdown headings, greetings, or fluff):\n"
                "Prediction: Tomorrow's maximum will be around <number>°C.\n"
                "<1 to 2 sentences explaining the trend, citing the slope in °C per day and recent high/low.>"
            )

            sign = "+" if trend["slope"] > 0 else ""
            user_prompt = (
                f"City: {city_name}, {country}\n"
                f"Past 7 Days Max Temperatures (°C): {past_temps}\n"
                f"Calculated Daily Trend Slope: {sign}{trend['slope']:.1f}°C per day ({trend['trend_direction']} trend)\n"
                f"Past Week High: {trend['max_temp']}°C, Low: {trend['min_temp']}°C, Average: {trend['avg_temp']}°C\n"
                f"Most Recent Day Max: {trend['recent_temp']}°C\n"
                f"Statistical Trend Projection: around {round(trend['projected_temp'])}°C\n\n"
                "Provide tomorrow's maximum temperature prediction now."
            )

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            response = llm.invoke(messages)
            content = response.content.strip()

            if "Prediction:" in content:
                return content
            else:
                rounded_proj = round(trend["projected_temp"])
                return f"Prediction: Tomorrow's maximum will be around {rounded_proj}°C.\n{content}"

        except Exception:
            # Fall back to analytical engine if Ollama fails mid-request
            return self._generate_analytical_prediction(city_name, trend)

    def predict(self, city_name: str, show_thinking: bool = True) -> Dict[str, Any]:
        """Perform full prediction pipeline for a given city name."""
        clean_name = city_name.strip()
        if not clean_name:
            return {"success": False, "error": "Please enter a valid city name."}

        # Step 1: Thinking & City lookup
        if show_thinking:
            print(f"Thinking... looking up past temperatures for {clean_name}")

        city_info = search_city(clean_name)
        if not city_info:
            return {
                "success": False,
                "error": f"City '{clean_name}' wasn't found. Please check spelling.",
            }

        resolved_name = city_info["name"]
        country = city_info.get("country", "")

        # Step 2: Fetch 7-day historical + tomorrow forecast
        weather_data = fetch_weather_data(
            latitude=city_info["latitude"],
            longitude=city_info["longitude"],
            timezone=city_info.get("timezone", "auto"),
        )

        if not weather_data:
            return {
                "success": False,
                "error": "Could not fetch weather data. Please check your internet connection.",
            }

        past_records = weather_data.get("past_7_days", [])
        tomorrow = weather_data.get("tomorrow_forecast", {})

        past_max_temps = [r["max_temp"] for r in past_records if r.get("max_temp") is not None]
        if len(past_max_temps) < 3:
            return {
                "success": False,
                "error": "Insufficient historical temperature data available for this location.",
            }

        # Step 3: Thinking & Trend calculation
        if show_thinking:
            print("Thinking... calculating the temperature trend")

        trend = calculate_temperature_trend(past_max_temps)

        # Step 4: AI Prediction
        ollama_online = self.is_ollama_available()
        if ollama_online:
            prediction_text = self._generate_ai_prediction(
                city_name=resolved_name,
                country=country,
                past_temps=past_max_temps,
                trend=trend,
            )
            ai_source = f"Ollama ({self.model_name})"
        else:
            prediction_text = self._generate_analytical_prediction(
                city_name=resolved_name,
                trend=trend,
            )
            ai_source = "Statistical Trend Engine (Ollama server offline)"

        # Step 5: Official Forecast Comparison Line
        official_max = tomorrow.get("max_temp")
        official_min = tomorrow.get("min_temp")

        if official_max is not None and official_min is not None:
            forecast_line = f"Official forecast for tomorrow → max {official_max:.1f}°C, min {official_min:.1f}°C"
        elif official_max is not None:
            forecast_line = f"Official forecast for tomorrow → max {official_max:.1f}°C"
        else:
            forecast_line = "Official forecast for tomorrow → data unavailable"

        return {
            "success": True,
            "city": resolved_name,
            "country": country,
            "past_7_days": past_records,
            "trend": trend,
            "prediction_text": prediction_text,
            "official_forecast": tomorrow,
            "forecast_line": forecast_line,
            "ai_source": ai_source,
        }
