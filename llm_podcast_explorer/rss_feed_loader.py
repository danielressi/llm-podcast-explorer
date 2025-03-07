import re
from typing import Any, Optional

import feedparser
import requests
from pydantic import AliasChoices, BaseModel, Field, field_validator


class RSSFeedItem(BaseModel):
    index: int = Field(..., description="indext of rss feed")
    title: str = Field(..., description="The title of the episode")
    subtitle: Optional[str] = Field(default=None, description="The subtitle of the episode")
    description: Optional[str] = Field(default=None, description="The description of the episode", alias="summary")
    id: str = Field(..., description="The unique identifier of the episode")
    episode_number: Optional[int] = Field(default=None, description="", alias="podcast_episode")
    link: str = Field(
        ...,
        description="The link to the episode",
        validation_alias=AliasChoices( "links", "link"),
    )
    published: str = Field(..., description="The date the episode was published")
    tags: Optional[list[Any]] = Field(default=None, description="Tags associated with the episode")

    def _extract_link(v):
        if "href" in v:
            return v["href"]
        else:
            return "unknown"

    @field_validator("link", mode="before")
    def validate_link(cls, v):
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            return cls._extract_link(v)
        elif isinstance(v, list):
            audio_link = [link for link in v if link["type"] == "audio/mpeg"]
            if len(audio_link) > 0:
                return cls._extract_link(audio_link[0])
            else:
                return cls._extract_link(v[0])
        else:
            return "unknown"


class InvalidRSSException(Exception):
    pass


class RSSFeedLoader:
    def __init__(self, url):
        self.feed = self._init_feed(url)
        self.description = self.feed.feed.get("description", "")
        self.title = self.feed.feed.get("title", "")
        self.size = len(self.feed.entries)

    @staticmethod
    def _init_feed(url):
        feed_url = ApplePodcastRSS().get_feed_url(url) if "podcasts.apple.com" in url else url

        feed = feedparser.parse(feed_url)
        if feed.bozo:
            raise InvalidRSSException("Invalid RSS feed. Provide valid RSS url")

        return feed

    def lazy_load(self):
        for i, entry in enumerate(self.feed.entries):
            entry["index"] = i
            rss_item = RSSFeedItem(**entry)
            if len(rss_item.description) > 0:
                yield rss_item


class ApplePodcastRSS:
    BASE_LOOKUP_URL = "https://itunes.apple.com/lookup?id={}"

    @staticmethod
    def extract_podcast_id(url: str) -> str:
        """Extracts the podcast ID from an Apple Podcast URL."""
        match = re.search(r"/id(\d+)", url)
        if match:
            return match.group(1)
        raise ValueError("Invalid Apple Podcast URL: Could not extract podcast ID.")

    @classmethod
    def get_feed_url(cls, podcast_url: str) -> str:
        """Extracts the podcast ID and fetches the feed URL from the iTunes API."""
        podcast_id = cls.extract_podcast_id(podcast_url)
        response = requests.get(cls.BASE_LOOKUP_URL.format(podcast_id), timeout=30)

        if response.status_code != 200:
            raise ConnectionError(f"Failed to fetch data from iTunes API. Status code: {response.status_code}")

        data = response.json()
        results = data.get("results", [])

        if not results:
            raise ValueError("No results found for the given podcast ID.")

        feed_url = results[0].get("feedUrl")
        if not feed_url:
            raise ValueError("Feed URL not found in API response.")

        return feed_url

    # Example usage


if __name__ == "__main__":
    podcast_url = "https://podcasts.apple.com/us/podcast/verbrechen/id1374777077"
    fetcher = ApplePodcastRSS()
    feed_url = fetcher.get_feed_url(podcast_url)
    print(feed_url)
