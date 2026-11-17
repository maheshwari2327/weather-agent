"""LangChain tools for the Weather Prediction Agent.

Exposes tools for geocoding, retrieving past temperature records,
computing trend statistics, and getting official forecasts.
"""

import json
from langchain_core.tools import tool
from weather_service import (
    search_city,
    fetch_weather_data,
    calculate_temperature_trend,
)


@tool
def lookup_city_coordinates(city_name: str) -> str:
    """Look up geographic latitude and longitude for any city name."""
    info = search_city(city_name)
    if not info:
        return json.dumps({"error": f"City '{city_name}' was not found."})
    return json.dumps(info)


@tool
def get_past_temperatures_and_forecast(latitude: float, longitude: float, timezone: str = "auto") -> str:
    """Retrieve the past 7 days of daily temperatures and tomorrow's forecast for coordinates."""
    data = fetch_weather_data(latitude, longitude, timezone)
    if not data:
        return json.dumps({"error": "Failed to fetch weather data from service."})
    return json.dumps(data)


@tool
def analyze_temperature_trend(past_max_temperatures: list[float]) -> str:
    """Calculate the trend slope, average, high, low, and statistical projection from a list of past temperatures."""
    trend = calculate_temperature_trend(past_max_temperatures)
    return json.dumps(trend)
