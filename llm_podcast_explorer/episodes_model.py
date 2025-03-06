
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

from rss_feed_loader import RSSFeedItem

class EpisodeInsights(BaseModel):
    """Information about a person."""

    episode_id: str = Field(
        ..., description="The unique identifier of the episode extracted from title or description."
    )
    topic: str = Field(..., description="The topic of the episode.")
    summary: str = Field(..., description="Short summary of the episode")
    topic_year: Optional[int] = Field(
        default=None,
        description="The year related to the topic not the episode (Best guess if not explicitly mentioned)",
    )
    topic_century: Optional[int] = Field(
        default=None, description="The century related to the topic (Best guess if not explicitly mentioned)"
    )
    tags: List[str] = Field(..., description="Enriched tags associated with the episode.")
    inferred_themes: List[str] = Field(..., description="Common themes that seem to fit to the episode.")
    referenced_episodes_id: Optional[List[str]] = Field(
        ..., description="Referenced episodes in this episode. Needs to match episode_id schema"
    )


class ClusteredEpisodeInsights(BaseModel):
    titles: List[str] = Field(..., description="Cluster titles")
    consolidated_titles: Optional[List[str]] = Field(default=None, description="Cluster titles")
    ids: List[int] = Field(..., description="Cluster ids of the episode.")
    embeddings: Optional[Any] = Field(default=None, description="Umap embeddings")
    major_category: Optional[str] = Field(
        default=None, description="major categories are the higher level groupings of clusters"
    )
    attempt: Optional[int] = Field(
        default=None, description="Number of consecutive clustering attempts until noise label was removed"
    )

    @field_validator("titles", mode="before")
    def wrap_string_in_list(cls, v):
        if isinstance(v, str):
            return [v]
        return v


class Episode(BaseModel):
    metadata: RSSFeedItem = Field(..., description="Metadata of the episode.")
    insights: EpisodeInsights = Field(..., description="Insights of the episode.")
    clusters: Optional[ClusteredEpisodeInsights] = Field(default=None, description="Mapping from insights to clusters")


class AnalyzedEpisodes(BaseModel):
    episodes: List[Episode] = Field(..., description="A list of analyzed episodes.")
    category_2_clusters: Optional[Dict[str, List[str]]] = Field(
        default=None, description="Mapping from major categories to clusters"
    )
    distance_map: Optional[Dict[str, float]] = Field(default=None, description="Cosine distances for episode pairs")
    extra: Optional[Dict[str, Any]] = Field(
        default={}, description="Placeholder for storing any other data for analysis"
    )

    def save_episodes(self, file_path: str):
        Path(file_path).parent.mkdir(exist_ok=True, parents=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=4))

    @classmethod
    def load(cls, file_path: str | Path):
        with open(str(file_path), encoding="utf-8") as f:
            data = json.load(f)
        episodes = [Episode(**episode) for episode in data["episodes"]]
        return cls(
            episodes=episodes, category_2_clusters=data["category_2_clusters"], distance_map=data["distance_map"]
        )