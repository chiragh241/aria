"""Weather skill using OpenWeatherMap API with wttr.in fallback."""

import os
from datetime import datetime, timezone
from typing import Any

import httpx

from ..base import BaseSkill, SkillResult
from ...utils.logging import get_logger

logger = get_logger(__name__)


class WeatherSkill(BaseSkill):
    """
    Skill for getting current weather, forecasts, and weather alerts.

    Uses OpenWeatherMap API when an API key is configured, with
    automatic fallback to wttr.in (no key required).
    """

    name = "weather"
    description = "Get current weather, forecasts, and weather alerts"
    version = "1.0.0"

    OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"
    WTTR_BASE_URL = "https://wttr.in"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.api_key = os.environ.get("OPENWEATHER_API_KEY", "")
        self._client: httpx.AsyncClient | None = None

    def _register_capabilities(self) -> None:
        """Register weather capabilities."""
        self.register_capability(
            name="current",
            description="Get current weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'London', 'New York,US', 'Tokyo,JP'",
                    },
                    "units": {
                        "type": "string",
                        "enum": ["metric", "imperial", "standard"],
                        "default": "metric",
                        "description": "Units of measurement (metric=Celsius, imperial=Fahrenheit, standard=Kelvin)",
                    },
                },
                "required": ["location"],
            },
        )

        self.register_capability(
            name="forecast",
            description="Get a 5-day weather forecast for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'London', 'New York,US', 'Tokyo,JP'",
                    },
                    "units": {
                        "type": "string",
                        "enum": ["metric", "imperial", "standard"],
                        "default": "metric",
                        "description": "Units of measurement (metric=Celsius, imperial=Fahrenheit, standard=Kelvin)",
                    },
                },
                "required": ["location"],
            },
        )

        self.register_capability(
            name="alerts",
            description="Get severe weather alerts for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'London', 'New York,US', 'Tokyo,JP'",
                    },
                    "units": {
                        "type": "string",
                        "enum": ["metric", "imperial", "standard"],
                        "default": "metric",
                        "description": "Units of measurement (metric=Celsius, imperial=Fahrenheit, standard=Kelvin)",
                    },
                },
                "required": ["location"],
            },
        )

    async def initialize(self) -> None:
        """Initialize HTTP client."""
        self._client = httpx.AsyncClient(timeout=15.0)
        self._initialized = True
        if self.api_key:
            logger.info("Weather skill initialized with OpenWeatherMap API key")
        else:
            logger.info("Weather skill initialized with wttr.in fallback (no API key)")

    async def shutdown(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._initialized = False

    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        """Execute a weather capability."""
        start_time = datetime.now(timezone.utc)

        if not self._initialized or not self._client:
            return self._error_result(
                "Weather skill not initialized. Call initialize() first.",
                start_time,
            )

        handlers = {
            "current": self._get_current,
            "forecast": self._get_forecast,
            "alerts": self._get_alerts,
        }

        handler = handlers.get(capability)
        if not handler:
            return self._error_result(f"Unknown capability: {capability}", start_time)

        try:
            result = await handler(**kwargs)
            return self._success_result(result, start_time)
        except httpx.HTTPStatusError as e:
            logger.error(f"Weather API HTTP error: {e.response.status_code} - {e.response.text}")
            return self._error_result(
                f"Weather API error ({e.response.status_code}): {e.response.text}",
                start_time,
            )
        except httpx.RequestError as e:
            logger.error(f"Weather API request error: {e}")
            return self._error_result(
                f"Failed to connect to weather service: {e}",
                start_time,
            )
        except Exception as e:
            logger.error(f"Weather skill error: {e}", exc_info=True)
            return self._error_result(str(e), start_time)

    # ── OpenWeatherMap helpers ──────────────────────────────────────

    async def _owm_current(self, location: str, units: str) -> dict[str, Any]:
        """Fetch current weather from OpenWeatherMap."""
        resp = await self._client.get(
            f"{self.OPENWEATHER_BASE_URL}/weather",
            params={"q": location, "units": units, "appid": self.api_key},
        )
        resp.raise_for_status()
        data = resp.json()

        unit_symbol = self._unit_symbol(units)
        return {
            "location": data.get("name", location),
            "country": data.get("sys", {}).get("country", ""),
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": data["wind"]["speed"],
            "wind_direction": data["wind"].get("deg"),
            "description": data["weather"][0]["description"] if data.get("weather") else "",
            "icon": data["weather"][0]["icon"] if data.get("weather") else "",
            "clouds": data.get("clouds", {}).get("all"),
            "visibility": data.get("visibility"),
            "units": units,
            "unit_symbol": unit_symbol,
            "source": "openweathermap",
        }

    async def _owm_forecast(self, location: str, units: str) -> dict[str, Any]:
        """Fetch 5-day forecast from OpenWeatherMap."""
        resp = await self._client.get(
            f"{self.OPENWEATHER_BASE_URL}/forecast",
            params={"q": location, "units": units, "appid": self.api_key},
        )
        resp.raise_for_status()
        data = resp.json()

        unit_symbol = self._unit_symbol(units)
        forecasts = []
        for item in data.get("list", []):
            forecasts.append({
                "datetime": item["dt_txt"],
                "temperature": item["main"]["temp"],
                "feels_like": item["main"]["feels_like"],
                "temp_min": item["main"]["temp_min"],
                "temp_max": item["main"]["temp_max"],
                "humidity": item["main"]["humidity"],
                "description": item["weather"][0]["description"] if item.get("weather") else "",
                "wind_speed": item["wind"]["speed"],
                "clouds": item.get("clouds", {}).get("all"),
                "pop": item.get("pop", 0),
            })

        return {
            "location": data.get("city", {}).get("name", location),
            "country": data.get("city", {}).get("country", ""),
            "forecasts": forecasts,
            "units": units,
            "unit_symbol": unit_symbol,
            "source": "openweathermap",
        }

    async def _owm_alerts(self, location: str, units: str) -> dict[str, Any]:
        """Fetch weather alerts from OpenWeatherMap One Call API.

        Requires geocoding the location first, then calling One Call 3.0.
        Falls back to wttr.in if geocoding or One Call fails.
        """
        # Geocode location to get lat/lon
        geo_resp = await self._client.get(
            "https://api.openweathermap.org/geo/1.0/direct",
            params={"q": location, "limit": 1, "appid": self.api_key},
        )
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()

        if not geo_data:
            return {
                "location": location,
                "alerts": [],
                "message": "Location not found",
                "source": "openweathermap",
            }

        lat = geo_data[0]["lat"]
        lon = geo_data[0]["lon"]
        resolved_name = geo_data[0].get("name", location)

        # One Call API for alerts
        resp = await self._client.get(
            f"{self.OPENWEATHER_BASE_URL.replace('/2.5', '/3.0')}/onecall",
            params={
                "lat": lat,
                "lon": lon,
                "exclude": "minutely,hourly,daily,current",
                "units": units,
                "appid": self.api_key,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        alerts = []
        for alert in data.get("alerts", []):
            alerts.append({
                "sender": alert.get("sender_name", ""),
                "event": alert.get("event", ""),
                "start": datetime.fromtimestamp(alert["start"], tz=timezone.utc).isoformat()
                if alert.get("start")
                else None,
                "end": datetime.fromtimestamp(alert["end"], tz=timezone.utc).isoformat()
                if alert.get("end")
                else None,
                "description": alert.get("description", ""),
                "tags": alert.get("tags", []),
            })

        return {
            "location": resolved_name,
            "lat": lat,
            "lon": lon,
            "alerts": alerts,
            "alert_count": len(alerts),
            "source": "openweathermap",
        }

    # ── wttr.in fallback helpers ────────────────────────────────────

    async def _wttr_current(self, location: str, units: str) -> dict[str, Any]:
        """Fetch current weather from wttr.in."""
        params = self._wttr_format_params(units)
        resp = await self._client.get(
            f"{self.WTTR_BASE_URL}/{location}",
            params={"format": "j1", **params},
            headers={"User-Agent": "aria-weather-skill"},
        )
        resp.raise_for_status()
        data = resp.json()

        current = data.get("current_condition", [{}])[0]
        unit_symbol = self._unit_symbol(units)

        temp_key = "temp_C" if units == "metric" else "temp_F"
        feels_key = "FeelsLikeC" if units == "metric" else "FeelsLikeF"
        wind_key = "windspeedKmph" if units == "metric" else "windspeedMiles"

        return {
            "location": location,
            "temperature": float(current.get(temp_key, 0)),
            "feels_like": float(current.get(feels_key, 0)),
            "humidity": int(current.get("humidity", 0)),
            "pressure": float(current.get("pressure", 0)),
            "wind_speed": float(current.get(wind_key, 0)),
            "wind_direction": current.get("winddir16Point", ""),
            "description": current.get("weatherDesc", [{}])[0].get("value", ""),
            "visibility": float(current.get("visibility", 0)),
            "clouds": int(current.get("cloudcover", 0)),
            "units": units,
            "unit_symbol": unit_symbol,
            "source": "wttr.in",
        }

    async def _wttr_forecast(self, location: str, units: str) -> dict[str, Any]:
        """Fetch forecast from wttr.in."""
        params = self._wttr_format_params(units)
        resp = await self._client.get(
            f"{self.WTTR_BASE_URL}/{location}",
            params={"format": "j1", **params},
            headers={"User-Agent": "aria-weather-skill"},
        )
        resp.raise_for_status()
        data = resp.json()

        unit_symbol = self._unit_symbol(units)
        temp_max_key = "maxtempC" if units == "metric" else "maxtempF"
        temp_min_key = "mintempC" if units == "metric" else "mintempF"

        forecasts = []
        for day in data.get("weather", []):
            hourly_items = []
            for hour in day.get("hourly", []):
                temp_key = "tempC" if units == "metric" else "tempF"
                feels_key = "FeelsLikeC" if units == "metric" else "FeelsLikeF"
                wind_key = "windspeedKmph" if units == "metric" else "windspeedMiles"
                hourly_items.append({
                    "time": hour.get("time", ""),
                    "temperature": float(hour.get(temp_key, 0)),
                    "feels_like": float(hour.get(feels_key, 0)),
                    "description": hour.get("weatherDesc", [{}])[0].get("value", ""),
                    "wind_speed": float(hour.get(wind_key, 0)),
                    "humidity": int(hour.get("humidity", 0)),
                    "pop": int(hour.get("chanceofrain", 0)),
                })

            forecasts.append({
                "date": day.get("date", ""),
                "temp_max": float(day.get(temp_max_key, 0)),
                "temp_min": float(day.get(temp_min_key, 0)),
                "description": day.get("hourly", [{}])[len(day.get("hourly", [])) // 2]
                .get("weatherDesc", [{}])[0]
                .get("value", "")
                if day.get("hourly")
                else "",
                "hourly": hourly_items,
            })

        return {
            "location": location,
            "forecasts": forecasts,
            "units": units,
            "unit_symbol": unit_symbol,
            "source": "wttr.in",
        }

    async def _wttr_alerts(self, location: str, units: str) -> dict[str, Any]:
        """wttr.in does not support weather alerts directly."""
        return {
            "location": location,
            "alerts": [],
            "alert_count": 0,
            "message": "Weather alerts require an OpenWeatherMap API key (OPENWEATHER_API_KEY). "
            "wttr.in fallback does not support alerts.",
            "source": "wttr.in",
        }

    # ── Dispatch methods ────────────────────────────────────────────

    async def _get_current(self, location: str, units: str = "metric") -> dict[str, Any]:
        """Get current weather, with automatic fallback."""
        if self.api_key:
            try:
                return await self._owm_current(location, units)
            except Exception as e:
                logger.warning(f"OpenWeatherMap failed, falling back to wttr.in: {e}")

        return await self._wttr_current(location, units)

    async def _get_forecast(self, location: str, units: str = "metric") -> dict[str, Any]:
        """Get weather forecast, with automatic fallback."""
        if self.api_key:
            try:
                return await self._owm_forecast(location, units)
            except Exception as e:
                logger.warning(f"OpenWeatherMap failed, falling back to wttr.in: {e}")

        return await self._wttr_forecast(location, units)

    async def _get_alerts(self, location: str, units: str = "metric") -> dict[str, Any]:
        """Get weather alerts, with automatic fallback."""
        if self.api_key:
            try:
                return await self._owm_alerts(location, units)
            except Exception as e:
                logger.warning(f"OpenWeatherMap alerts failed, falling back to wttr.in: {e}")

        return await self._wttr_alerts(location, units)

    # ── Utility ─────────────────────────────────────────────────────

    @staticmethod
    def _unit_symbol(units: str) -> str:
        """Return the temperature unit symbol for the given units system."""
        return {
            "metric": "C",
            "imperial": "F",
            "standard": "K",
        }.get(units, "C")

    @staticmethod
    def _wttr_format_params(units: str) -> dict[str, str]:
        """Return wttr.in query params to match the requested unit system."""
        if units == "imperial":
            return {"u": ""}
        elif units == "standard":
            # wttr.in doesn't support Kelvin natively; use metric and note it
            return {"m": ""}
        else:
            return {"m": ""}
