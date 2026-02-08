"""Weather skill using WeatherAPI.com with IP-based location when not specified."""

import os
from datetime import datetime, timezone
from typing import Any

import httpx

from ..base import BaseSkill, SkillResult
from ...utils.logging import get_logger
from ...utils.request_context import get_client_ip

logger = get_logger(__name__)


class WeatherSkill(BaseSkill):
    """
    Skill for getting current weather, forecasts, and weather alerts.

    Uses WeatherAPI.com. When location is not provided, uses the user's
    IP address for automatic location detection (web requests only).
    """

    name = "weather"
    description = "Get current weather, forecasts, and weather alerts"
    version = "2.0.0"

    BASE_URL = "https://api.weatherapi.com/v1"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.api_key = os.environ.get("WEATHERAPI_KEY", "") or os.environ.get("WEATHER_API_KEY", "")
        self._client: httpx.AsyncClient | None = None

    def _register_capabilities(self) -> None:
        """Register weather capabilities."""
        self.register_capability(
            name="current",
            description="Get current weather. Use user's location (IP-based) when no location given.",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or leave empty for user's location (IP-based)",
                    },
                    "units": {
                        "type": "string",
                        "enum": ["metric", "imperial"],
                        "default": "metric",
                        "description": "metric=Celsius, imperial=Fahrenheit",
                    },
                },
                "required": [],
            },
        )
        self.register_capability(
            name="forecast",
            description="Get weather forecast. Use user's location (IP-based) when no location given.",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or leave empty for user's location (IP-based)",
                    },
                    "days": {
                        "type": "integer",
                        "default": 5,
                        "description": "Number of forecast days (1-14)",
                    },
                    "units": {
                        "type": "string",
                        "enum": ["metric", "imperial"],
                        "default": "metric",
                        "description": "metric=Celsius, imperial=Fahrenheit",
                    },
                },
                "required": [],
            },
        )
        self.register_capability(
            name="alerts",
            description="Get severe weather alerts. Use user's location (IP-based) when no location given.",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or leave empty for user's location (IP-based)",
                    },
                    "units": {
                        "type": "string",
                        "enum": ["metric", "imperial"],
                        "default": "metric",
                        "description": "metric=Celsius, imperial=Fahrenheit",
                    },
                },
                "required": [],
            },
        )

    def _resolve_location(self, location: str | None) -> str:
        """Resolve location: use provided, or client IP, or auto:ip."""
        loc = (location or "").strip()
        if loc:
            return loc
        client_ip = get_client_ip()
        if client_ip:
            return client_ip
        return "auto:ip"

    async def initialize(self) -> None:
        """Initialize HTTP client."""
        self._client = httpx.AsyncClient(timeout=15.0)
        self._initialized = True
        if self.api_key:
            logger.info("Weather skill initialized with WeatherAPI.com")
        else:
            logger.info("Weather skill initialized but no WEATHERAPI_KEY set")

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

        if not self.api_key:
            return self._error_result(
                "WeatherAPI.com API key not configured. Set WEATHERAPI_KEY in .env",
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

    async def _get_current(self, location: str | None = None, units: str = "metric") -> dict[str, Any]:
        """Fetch current weather from WeatherAPI.com."""
        q = self._resolve_location(location)
        resp = await self._client.get(
            f"{self.BASE_URL}/current.json",
            params={"key": self.api_key, "q": q, "aqi": "no"},
        )
        resp.raise_for_status()
        data = resp.json()

        loc = data.get("location", {})
        curr = data.get("current", {})
        cond = curr.get("condition", {})

        use_f = units == "imperial"
        temp = curr.get("temp_f" if use_f else "temp_c", 0)
        feels = curr.get("feelslike_f" if use_f else "feelslike_c", 0)
        wind = curr.get("wind_mph" if use_f else "wind_kph", 0)
        unit_symbol = "F" if use_f else "C"

        return {
            "location": loc.get("name", q),
            "region": loc.get("region", ""),
            "country": loc.get("country", ""),
            "temperature": temp,
            "feels_like": feels,
            "humidity": curr.get("humidity", 0),
            "pressure_mb": curr.get("pressure_mb"),
            "pressure_in": curr.get("pressure_in"),
            "wind_speed": wind,
            "wind_degree": curr.get("wind_degree"),
            "wind_dir": curr.get("wind_dir", ""),
            "description": cond.get("text", ""),
            "icon": cond.get("icon", ""),
            "cloud": curr.get("cloud", 0),
            "visibility_km": curr.get("vis_km"),
            "uv": curr.get("uv"),
            "gust_mph": curr.get("gust_mph"),
            "gust_kph": curr.get("gust_kph"),
            "last_updated": curr.get("last_updated", ""),
            "units": units,
            "unit_symbol": unit_symbol,
            "source": "weatherapi",
        }

    async def _get_forecast(
        self,
        location: str | None = None,
        days: int = 5,
        units: str = "metric",
    ) -> dict[str, Any]:
        """Fetch forecast from WeatherAPI.com."""
        q = self._resolve_location(location)
        resp = await self._client.get(
            f"{self.BASE_URL}/forecast.json",
            params={"key": self.api_key, "q": q, "days": min(max(days, 1), 14), "aqi": "no", "alerts": "no"},
        )
        resp.raise_for_status()
        data = resp.json()

        loc = data.get("location", {})
        forecast_days = data.get("forecast", {}).get("forecastday", [])

        use_f = units == "imperial"
        temp_max_key = "maxtemp_f" if use_f else "maxtemp_c"
        temp_min_key = "mintemp_f" if use_f else "mintemp_c"
        unit_symbol = "F" if use_f else "C"

        forecasts = []
        for day in forecast_days:
            d = day.get("day", {})
            cond = d.get("condition", {})
            forecasts.append({
                "date": day.get("date", ""),
                "temp_max": d.get(temp_max_key, 0),
                "temp_min": d.get(temp_min_key, 0),
                "description": cond.get("text", ""),
                "icon": cond.get("icon", ""),
                "maxwind_kph": d.get("maxwind_kph"),
                "maxwind_mph": d.get("maxwind_mph"),
                "totalprecip_mm": d.get("totalprecip_mm"),
                "totalprecip_in": d.get("totalprecip_in"),
                "daily_chance_of_rain": d.get("daily_chance_of_rain", 0),
                "daily_chance_of_snow": d.get("daily_chance_of_snow", 0),
                "uv": d.get("uv"),
            })

        return {
            "location": loc.get("name", q),
            "region": loc.get("region", ""),
            "country": loc.get("country", ""),
            "forecasts": forecasts,
            "units": units,
            "unit_symbol": unit_symbol,
            "source": "weatherapi",
        }

    async def _get_alerts(self, location: str | None = None, units: str = "metric") -> dict[str, Any]:
        """Fetch weather alerts from WeatherAPI.com (included in forecast with alerts=yes)."""
        q = self._resolve_location(location)
        resp = await self._client.get(
            f"{self.BASE_URL}/forecast.json",
            params={"key": self.api_key, "q": q, "days": 3, "aqi": "no", "alerts": "yes"},
        )
        resp.raise_for_status()
        data = resp.json()

        loc = data.get("location", {})
        alert_data = data.get("alerts") or {}
        alert_list = alert_data.get("alert")
        if isinstance(alert_list, dict):
            alert_list = [alert_list]
        elif not isinstance(alert_list, list):
            alert_list = []

        alerts = []
        for a in alert_list:
            alerts.append({
                "headline": a.get("headline", ""),
                "msgType": a.get("msgType", ""),
                "severity": a.get("severity", ""),
                "event": a.get("event", ""),
                "effective": a.get("effective", ""),
                "expires": a.get("expires", ""),
                "desc": a.get("desc", ""),
                "instruction": a.get("instruction", ""),
            })

        return {
            "location": loc.get("name", q),
            "region": loc.get("region", ""),
            "country": loc.get("country", ""),
            "alerts": alerts,
            "alert_count": len(alerts),
            "source": "weatherapi",
        }
