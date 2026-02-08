"""Finance skill for stock prices, crypto, and market data."""

import os
from datetime import datetime, timezone
from typing import Any

import httpx

from ..base import BaseSkill, SkillResult
from ...utils.logging import get_logger

logger = get_logger(__name__)


class FinanceSkill(BaseSkill):
    """Get stock prices, crypto prices, and market summaries."""

    name = "finance"
    description = "Check stock prices, cryptocurrency prices, and market overview"
    version = "1.0.0"

    def _register_capabilities(self) -> None:
        self.register_capability(
            name="stock_price",
            description="Get current stock price and daily change for a ticker symbol",
            parameters={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. AAPL, GOOGL, TSLA)",
                    },
                },
                "required": ["symbol"],
            },
        )
        self.register_capability(
            name="crypto_price",
            description="Get current cryptocurrency price in USD",
            parameters={
                "type": "object",
                "properties": {
                    "coin": {
                        "type": "string",
                        "description": "Cryptocurrency name or symbol (e.g. bitcoin, ethereum, BTC, ETH)",
                    },
                },
                "required": ["coin"],
            },
        )
        self.register_capability(
            name="market_summary",
            description="Get a brief overview of major market indices and trends",
            parameters={
                "type": "object",
                "properties": {},
            },
        )

    async def execute(self, capability: str, **kwargs: Any) -> SkillResult:
        start = datetime.now(timezone.utc)
        if capability == "stock_price":
            return await self._stock_price(kwargs.get("symbol", ""), start)
        elif capability == "crypto_price":
            return await self._crypto_price(kwargs.get("coin", ""), start)
        elif capability == "market_summary":
            return await self._market_summary(start)
        return self._error_result(f"Unknown capability: {capability}", start)

    async def _stock_price(self, symbol: str, start: datetime) -> SkillResult:
        if not symbol:
            return self._error_result("No stock symbol provided", start)

        symbol = symbol.upper().strip()

        # Try yfinance first
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if not info or "currentPrice" not in info:
                # Try fast_info as fallback
                fast = ticker.fast_info
                price = getattr(fast, "last_price", None)
                prev_close = getattr(fast, "previous_close", None)
                if price:
                    change = price - prev_close if prev_close else 0
                    pct = (change / prev_close * 100) if prev_close else 0
                    return self._success_result(
                        f"**{symbol}**: ${price:.2f} ({'+' if change >= 0 else ''}{change:.2f}, {'+' if pct >= 0 else ''}{pct:.1f}%)",
                        start,
                    )

            price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
            prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose", 0)
            name = info.get("shortName", symbol)
            change = price - prev_close if prev_close else 0
            pct = (change / prev_close * 100) if prev_close else 0
            market_cap = info.get("marketCap", 0)

            result = f"**{name} ({symbol})**\n"
            result += f"Price: ${price:.2f}\n"
            result += f"Change: {'+' if change >= 0 else ''}{change:.2f} ({'+' if pct >= 0 else ''}{pct:.1f}%)\n"
            if market_cap:
                if market_cap >= 1e12:
                    result += f"Market Cap: ${market_cap/1e12:.2f}T"
                elif market_cap >= 1e9:
                    result += f"Market Cap: ${market_cap/1e9:.2f}B"
                else:
                    result += f"Market Cap: ${market_cap/1e6:.0f}M"

            return self._success_result(result, start)
        except ImportError:
            logger.debug("yfinance not installed, trying API fallback")
        except Exception as e:
            logger.debug("yfinance failed", error=str(e))

        # Fallback: free API
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                    params={"interval": "1d", "range": "1d"},
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    meta = data.get("chart", {}).get("result", [{}])[0].get("meta", {})
                    price = meta.get("regularMarketPrice", 0)
                    prev = meta.get("previousClose", 0)
                    change = price - prev if prev else 0
                    pct = (change / prev * 100) if prev else 0
                    return self._success_result(
                        f"**{symbol}**: ${price:.2f} ({'+' if change >= 0 else ''}{change:.2f}, {'+' if pct >= 0 else ''}{pct:.1f}%)",
                        start,
                    )
        except Exception as e:
            logger.debug("Yahoo finance API fallback failed", error=str(e))

        return self._error_result(f"Could not fetch price for {symbol}. Install yfinance for reliable data.", start)

    async def _crypto_price(self, coin: str, start: datetime) -> SkillResult:
        if not coin:
            return self._error_result("No cryptocurrency specified", start)

        # Map common symbols to CoinGecko IDs
        symbol_map = {
            "btc": "bitcoin", "eth": "ethereum", "sol": "solana",
            "ada": "cardano", "dot": "polkadot", "doge": "dogecoin",
            "xrp": "ripple", "bnb": "binancecoin", "avax": "avalanche-2",
            "matic": "matic-network", "link": "chainlink", "ltc": "litecoin",
        }
        coin_id = symbol_map.get(coin.lower().strip(), coin.lower().strip())

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://api.coingecko.com/api/v3/simple/price",
                    params={
                        "ids": coin_id,
                        "vs_currencies": "usd",
                        "include_24hr_change": "true",
                        "include_market_cap": "true",
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if coin_id in data:
                        info = data[coin_id]
                        price = info.get("usd", 0)
                        change_24h = info.get("usd_24h_change", 0)
                        market_cap = info.get("usd_market_cap", 0)

                        result = f"**{coin_id.title()}**\n"
                        result += f"Price: ${price:,.2f}\n"
                        result += f"24h Change: {'+' if change_24h >= 0 else ''}{change_24h:.1f}%"
                        if market_cap:
                            if market_cap >= 1e12:
                                result += f"\nMarket Cap: ${market_cap/1e12:.2f}T"
                            elif market_cap >= 1e9:
                                result += f"\nMarket Cap: ${market_cap/1e9:.2f}B"

                        return self._success_result(result, start)
                    else:
                        return self._error_result(f"Cryptocurrency '{coin}' not found. Try the full name (e.g. 'bitcoin').", start)

                return self._error_result(f"CoinGecko API error: HTTP {resp.status_code}", start)
        except Exception as e:
            return self._error_result(f"Failed to fetch crypto price: {e}", start)

    async def _market_summary(self, start: datetime) -> SkillResult:
        # Fetch major indices
        indices = ["^GSPC", "^DJI", "^IXIC"]  # S&P 500, Dow Jones, NASDAQ
        labels = {"^GSPC": "S&P 500", "^DJI": "Dow Jones", "^IXIC": "NASDAQ"}

        # Try yfinance
        try:
            import yfinance as yf
            lines = ["**Market Summary**\n"]
            for idx in indices:
                ticker = yf.Ticker(idx)
                fast = ticker.fast_info
                price = getattr(fast, "last_price", None)
                prev = getattr(fast, "previous_close", None)
                if price and prev:
                    change = price - prev
                    pct = (change / prev * 100)
                    arrow = "+" if change >= 0 else ""
                    lines.append(f"**{labels[idx]}**: {price:,.2f} ({arrow}{change:,.2f}, {arrow}{pct:.1f}%)")
            if len(lines) > 1:
                return self._success_result("\n".join(lines), start)
        except ImportError:
            pass
        except Exception as e:
            logger.debug("yfinance market summary failed", error=str(e))

        # Fallback: fetch major crypto as a "market summary"
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    "https://api.coingecko.com/api/v3/simple/price",
                    params={
                        "ids": "bitcoin,ethereum,solana",
                        "vs_currencies": "usd",
                        "include_24hr_change": "true",
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    lines = ["**Crypto Market Summary**\n"]
                    for coin in ["bitcoin", "ethereum", "solana"]:
                        if coin in data:
                            price = data[coin].get("usd", 0)
                            change = data[coin].get("usd_24h_change", 0)
                            arrow = "+" if change >= 0 else ""
                            lines.append(f"**{coin.title()}**: ${price:,.2f} ({arrow}{change:.1f}%)")
                    return self._success_result("\n".join(lines), start)
        except Exception:
            pass

        return self._error_result("Could not fetch market data. Install yfinance for stock indices.", start)
