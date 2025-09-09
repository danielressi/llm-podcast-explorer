import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
import os
import time
from openai import APIConnectionError, RateLimitError, Timeout
from src.io_utils import write_to_json, write_to_r2
from src.rss_feed_analyzer import RSSFeedAnalyzer

LOGGER = logging.getLogger(__name__)

SCHEDULE = {"https://geschichten-aus-der-geschichte.podigee.io/feed/mp3": ["wednesday"],
            "https://podcasts.apple.com/us/podcast/99-invisible/id394775318": ["tuesday"],
            "https://podcasts.apple.com/us/podcast/empire/id1639561921": ["tuesday", "thursday"],
            "https://podcasts.apple.com/nl/podcast/revisionist-history/id1119389968": ["thursday"],
            "https://podcasts.apple.com/nl/podcast/data-skeptic/id890348705": ["monday"],
            "https://podcasts.apple.com/nl/podcast/wanging-on-with-graham-norton-and-maria-mcerlane/id1821737353": ["monday"]
            }


def run(rss_url: str, output_path: str, s3_bucket: Optional[str] = None, limit: int = 1000):
    analyzer = RSSFeedAnalyzer(rss_url=rss_url)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found in environment")
    result = analyzer.run(limit=limit)
    filename = f"{analyzer.title}.json"
    output_path = Path(output_path) / filename
    write_to_json(result.model_dump_json(), output_path)
    if s3_bucket:
        write_to_r2(output_path, s3_bucket, filename)

MAX_RETRIES = 3
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze an RSS feed and save the result.")
    parser.add_argument("--output_path", help="Path to save the analysis result")
    parser.add_argument("--s3_bucket", required=False, default=None, help="S3 bucket to upload the result (optional)")
    parser.add_argument(
        "--limit", type=int, default=1000, help="Limit the number of episodes to analyze (default: 100)"
    )
    args = parser.parse_args()

    for rss_url, days in SCHEDULE.items():
        if datetime.now().strftime("%A").lower() in days:
            LOGGER.info(f"Running scheduled analysis for {rss_url}")


            for attempt in range(MAX_RETRIES):
                try:
                    run(rss_url=rss_url, output_path=args.output_path, s3_bucket=args.s3_bucket, limit=args.limit)
                    break
                except (APIConnectionError, RateLimitError, Timeout) as e:
                    LOGGER.warning(f"Attempt {attempt} failed with APIConnectionError: {e}")
                    if attempt >= MAX_RETRIES:
                        LOGGER.exception("Max retries reached, aborting.")
                        continue
                    sleep_seconds = 2 ** attempt
                    LOGGER.info(f"Retrying in {sleep_seconds} seconds...")
                    time.sleep(sleep_seconds)
        else:
            LOGGER.info(f"Skipping {rss_url} for today, scheduled for {', '.join(days)}")