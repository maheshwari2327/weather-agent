"""Weather Service module for Weather Prediction Agent.

Fetches 7-day historical weather data and official forecasts using
the free Open-Meteo API (no API key required).
"""

from typing import Any, Dict, List, Optional
import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def search_city(city_name: str, timeout: int = 8) -> Optional[Dict[str, Any]]:
    """Search for a city by name and retrieve its geographic coordinates.

    Args:
        city_name: Name of the city (e.g. 'Hyderabad', 'London', 'Tokyo').
        timeout: Request timeout in seconds.

    Returns:
        Dict with city metadata (name, country, latitude, longitude, timezone)
        or None if city was not found.
    """
    try:
        response = requests.get(
            GEOCODING_URL,
            params={
                "name": city_name.strip(),
                "count": 1,
                "language": "en",
                "format": "json",
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results")
        if not results or len(results) == 0:
            return None

        city_info = results[0]
        return {
            "name": city_info.get("name", city_name),
            "country": city_info.get("country", ""),
            "admin1": city_info.get("admin1", ""),
            "latitude": city_info.get("latitude"),
            "longitude": city_info.get("longitude"),
            "timezone": city_info.get("timezone", "auto"),
        }
    except requests.RequestException:
        return None


def fetch_weather_data(
    latitude: float,
    longitude: float,
    timezone: str = "auto",
    timeout: int = 8,
) -> Optional[Dict[str, Any]]:
    """Fetch past 7 days temperatures and tomorrow's official forecast.

    Args:
        latitude: Latitude of the location.
        longitude: Longitude of the location.
        timezone: Local timezone string or 'auto'.
        timeout: Request timeout in seconds.

    Returns:
        Dict containing past 7 days records and tomorrow's forecast,
        or None if the request failed.
    """
    try:
        response = requests.get(
            FORECAST_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "daily": ["temperature_2m_max", "temperature_2m_min"],
                "past_days": 7,
                "forecast_days": 2,
                "timezone": timezone or "auto",
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        daily = data.get("daily", {})

        times: List[str] = daily.get("time", [])
        max_temps: List[float] = daily.get("temperature_2m_max", [])
        min_temps: List[float] = daily.get("temperature_2m_min", [])

        if not times or not max_temps or len(times) < 8:
            return None

        # When past_days=7 and forecast_days=2:
        # Indices 0..6: past 7 days
        # Index -2: today
        # Index -1: tomorrow (official forecast)
        past_count = min(7, len(times) - 2)
        past_records = []
        for i in range(past_count):
            past_records.append(
                {
                    "date": times[i],
                    "max_temp": max_temps[i],
                    "min_temp": min_temps[i] if i < len(min_temps) else None,
                }
            )

        tomorrow_forecast = {
            "date": times[-1],
            "max_temp": max_temps[-1],
            "min_temp": min_temps[-1] if len(min_temps) > 0 else None,
        }

        today_data = {
            "date": times[-2],
            "max_temp": max_temps[-2],
            "min_temp": min_temps[-2] if len(min_temps) > 1 else None,
        }

        return {
            "past_7_days": past_records,
            "today": today_data,
            "tomorrow_forecast": tomorrow_forecast,
        }
    except requests.RequestException:
        return None


def calculate_temperature_trend(past_max_temps: List[float]) -> Dict[str, Any]:
    """Calculate the trend slope, averages, highs/lows and statistical projection.

    Args:
        past_max_temps: List of maximum daily temperatures from the past 7 days.

    Returns:
        Dict with slope, trend_direction, avg_temp, min_temp, max_temp, projected_temp.
    """
    n = len(past_max_temps)
    if n == 0:
        return {
            "slope": 0.0,
            "trend_direction": "stable",
            "avg_temp": 0.0,
            "min_temp": 0.0,
            "max_temp": 0.0,
            "projected_temp": 0.0,
        }

    avg_temp = sum(past_max_temps) / n
    min_temp = min(past_max_temps)
    max_temp = max(past_max_temps)

    # Linear regression slope: m = sum((x - x_bar) * (y - y_bar)) / sum((x - x_bar)^2)
    x_vals = list(range(n))
    x_bar = sum(x_vals) / n
    denom = sum((x - x_bar) ** 2 for x in x_vals)
    if denom > 0:
        slope = sum((x - x_bar) * (y - avg_temp) for x, y in zip(x_vals, past_max_temps)) / denom
    else:
        slope = 0.0

    if slope >= 0.15:
        direction = "upward"
    elif slope <= -0.15:
        direction = "downward"
    else:
        direction = "stable"

    # Extrapolate for tomorrow (x = n)
    projected = past_max_temps[-1] + slope

    return {
        "slope": round(slope, 1),
        "trend_direction": direction,
        "avg_temp": round(avg_temp, 1),
        "min_temp": round(min_temp, 1),
        "max_temp": round(max_temp, 1),
        "recent_temp": round(past_max_temps[-1], 1),
        "projected_temp": round(projected, 1),
    }
