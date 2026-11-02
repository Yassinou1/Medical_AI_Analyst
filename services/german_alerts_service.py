"""Service for fetching German pharmacovigilance alerts (Rote-Hand-Briefe).

Queries the official BfArM RSS feed for drug safety communications
and filters entries by active ingredient name asynchronously.
"""

import logging
import httpx
import feedparser
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings

logger = logging.getLogger(__name__)


class GermanPharmacovigilanceService:
    """Fetches and filters German regulatory drug safety alerts."""

    def __init__(self) -> None:
        self.rss_feeds: list[str] = [settings.BFARM_RSS_URL]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def fetch_german_alerts(self, drug_name: str) -> list[dict]:
        """Search BfArM RSS feeds for alerts mentioning *drug_name* asynchronously.

        Args:
            drug_name: Active ingredient to search for (case-insensitive).

        Returns:
            List of dicts with keys ``title``, ``date``, ``link``, ``summary``
            for every matching alert entry.
        """
        alerts: list[dict] = []
        drug_name_lower = drug_name.lower()

        async with httpx.AsyncClient(timeout=15.0) as client:
            for feed_url in self.rss_feeds:
                try:
                    response = await client.get(feed_url)
                    response.raise_for_status()

                    # Parse the raw XML string
                    feed = feedparser.parse(response.text)

                    for entry in feed.entries:
                        title = entry.title.lower()
                        summary = (
                            entry.description.lower()
                            if hasattr(entry, "description")
                            else ""
                        )

                        if drug_name_lower in title or drug_name_lower in summary:
                            alerts.append(
                                {
                                    "title": entry.title,
                                    "date": (
                                        entry.published
                                        if hasattr(entry, "published")
                                        else "Aktuell"
                                    ),
                                    "link": entry.link,
                                    "summary": (
                                        entry.description[:200] + "..."
                                        if hasattr(entry, "description")
                                        else "Keine Details verfügbar."
                                    ),
                                }
                            )
                except Exception as e:
                    logger.exception(f"Failed to fetch BfArM feed: {feed_url} - {e}")
                    raise

        logger.info(
            "BfArM search for '%s': %d matches found", drug_name, len(alerts)
        )
        return alerts


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)

    async def main():
        service = GermanPharmacovigilanceService()
        results = await service.fetch_german_alerts("ibuprofen")
        print(results)

    asyncio.run(main())

