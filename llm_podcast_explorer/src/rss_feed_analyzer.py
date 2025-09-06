import itertools
import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import umap
from hdbscan import HDBSCAN as HDBSCAN
from hdbscan.prediction import all_points_membership_vectors
from langchain.embeddings import CacheBackedEmbeddings
from langchain.output_parsers import RetryOutputParser
from langchain.storage import LocalFileStore
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field, RootModel, field_validator

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from tqdm import tqdm

from llm_podcast_explorer.src.episodes_model import AnalyzedEpisodes, ClusteredEpisodeInsights, Episode, EpisodeInsights
from llm_podcast_explorer.src.prompts import CATEGORY_PROMPT, CLUSTER_TITLE_PROMPT, CONSOLIDATION_PROMPT, EXRACTION_PROMPT
from llm_podcast_explorer.src.rss_feed_loader import RSSFeedLoader

COSINE_DISTANCE_THRESHOLD = 0.5


class SimpleList(RootModel):
    root: list[str]


class ClusterTitlesBatch(BaseModel):
    items: list[str] = Field(..., description="Batch of cluster titles")


class Mapping(BaseModel):
    mapping: dict[str, list[str]] = Field(..., description="Mapping")

    @field_validator("mapping", mode="before")
    @classmethod
    def enforce_list(cls, value: Any) -> str:
        return {k: list(v.values()) if isinstance(v, dict) else list(v) for k, v in value.items()}


class MajorCategories(Mapping):
    mapping: dict[str, list[str]] = Field(
        ..., description="Mapping from identified major categories to all the titles that belong to the major category."
    )


class ConsolidatedTitles(Mapping):
    mapping: dict[str, list[str]] = Field(
        ...,
        description="Mapping of consolidated titles to the corresponding titles"
        " that are semantically too similar, duplicates or synonyms",
    )


class TextCatalogEntry(BaseModel):
    themes: str = Field(..., description="Themes of the episode")
    summary: str = Field(..., description="Summaries of the episode")
    tags: str = Field(..., description="Tags of the episode")

    @field_validator("summary", "themes", "tags", mode="before")
    @classmethod
    def convert_list_to_str(cls, v: Any) -> str:
        if isinstance(v, list):
            return ",".join(v)
        return v

    def to_text(self):
        return "\n ".join([f"{k.capitalize()}:{v}" for k, v in self.model_dump().items()])


def get_rate_limiter(model="gpt-4o-mini"):
    if model == "gpt-4o":
        return InMemoryRateLimiter(
            requests_per_second=0.5,  # 1 request every 2 seconds
            check_every_n_seconds=0.1,  # Check every 100 ms
            max_bucket_size=5,  # Allow bursts of 5 requests max
        )
    elif model == "gpt-4o-mini":
        return InMemoryRateLimiter(
            requests_per_second=1.0,  # 1 request per second
            check_every_n_seconds=0.05,  # Check more frequently if you want
            max_bucket_size=10,  # Higher burst capacity
        )
    else:
        return None


class RSSFeedAnalyzer:
    def __init__(
        self,
        rss_url,
        llm_api_key=None,
        extraction_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        analysis_model=None,
        tempature=0.1,
        logger=None,
    ):
        self.rss_loader = RSSFeedLoader(rss_url)
        set_llm_cache(SQLiteCache(database_path=".langchain.db"))
        self.extraction_llm = ChatOpenAI(
            model=extraction_model,
            api_key=llm_api_key,
            temperature=tempature,
            rate_limiter=get_rate_limiter(extraction_model),
        )
        analysis_llm_name = analysis_model if analysis_model is not None else extraction_model
        self.analysis_llm = ChatOpenAI(
            model=analysis_llm_name,
            api_key=llm_api_key,
            temperature=tempature,
            rate_limiter=get_rate_limiter(analysis_llm_name),
        )
        self.embeddings = self._init_embeddings(embedding_model)
        self._noise_title = "Sonstiges" if self.language == "de" else "Other"

        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

    @property
    def title(self):
        return self.rss_loader.title

    @property
    def size(self):
        return self.rss_loader.size

    @property
    def language_prompt(self):
        if hasattr(self.rss_loader.feed.feed, "language"):
            return f"ISO 639={self.rss_loader.feed.feed.language}"
        else:
            return "same language as the input provided by the user"

    @property
    def language(self):
        if hasattr(self.rss_loader.feed.feed, "language"):
            return f"{self.rss_loader.feed.feed.language}"
        else:
            return "unknown"

    def _init_embeddings(self, embedding_model, path="./cache/"):
        embeddings = OpenAIEmbeddings(model=embedding_model)
        os.makedirs(path, exist_ok=True)
        store = LocalFileStore(path)
        return CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=embeddings,
            document_embedding_cache=store,
            namespace=embeddings.model,  # Create a cache-backed embedder using the base embedding and storage
        )

    def analyze_feed(self, limit=10000, batch_size=56):
        episode_loader = self.rss_loader.lazy_load()
        self.logger.info(f"Analyzing {self.size} podcast episdoes")
        parser = PydanticOutputParser(pydantic_object=EpisodeInsights)

        prompt_template = ChatPromptTemplate(EXRACTION_PROMPT).partial(
            format_instructions=parser.get_format_instructions(),
            podcast=self.rss_loader.title,
            tag_limit=5,
            theme_limit=3,
            podcast_description=self.rss_loader.description,
        )

        request_chain = prompt_template | self.extraction_llm
        retry_parser = RetryOutputParser.from_llm(parser=parser, llm=self.extraction_llm, max_retries=2)

        chain = RunnableParallel(
            completion=request_chain,
            prompt_value=prompt_template,  # Add the prompt_value here
        ) | RunnableLambda(
            lambda x: retry_parser.parse_with_prompt(
                completion=x["completion"].content,  # Extract the content from AIMessage
                prompt_value=x["prompt_value"],
            )
        )

        analysis_results = []
        batch_content = []
        batch_metadata = []

        for i, episode in enumerate(episode_loader):
            if i > limit:
                break
            elif (i % batch_size == 0 and i > 0) or (i == limit):
                self.logger.info(f"Running batch {i // batch_size}")
                response = chain.batch(batch_content)

                analysed_episodes = [Episode(metadata=m, insights=r) for m, r in zip(batch_metadata, response)]

                analysis_results.extend(analysed_episodes)
                batch_metadata = []
                batch_content = []
            else:
                if len(episode.description) > 0:
                    batch_content.append({"title": episode.title, "episode_content": episode.description[:1500]})
                    batch_metadata.append(episode.model_dump())

        if len(batch_content) > 0:
            response = chain.batch(batch_content)

            analysed_episodes = [Episode(metadata=m, insights=r) for m, r in zip(batch_metadata, response)]

            analysis_results.extend(analysed_episodes)

        return AnalyzedEpisodes(episodes=analysis_results)

    def _create_episode_text_catalog(self, analysed_episodes):
        text_catalog = {}
        for ep in analysed_episodes:
            text_catalog[ep.metadata.index] = TextCatalogEntry(
                themes=ep.insights.inferred_themes, tags=ep.insights.tags, summary=ep.insights.summary
            )

        return text_catalog

    @staticmethod
    def _predict_clusters(vectors, cluster_offset=0, metric="cosine", **kwargs):
        if metric == "cosine":
            vectors = normalize(vectors, norm="l2")
        # approximate sklearn implementation if no value specified
        min_samples = kwargs.pop("min_samples", kwargs["min_cluster_size"] - 1)
        c_model = HDBSCAN(**kwargs, prediction_data=True, min_samples=min_samples, cluster_selection_method="leaf").fit(
            vectors
        )
        c_labels = c_model.labels_
        c_labels[c_labels >= 0] = c_labels[c_labels >= 0] + cluster_offset
        soft_clusters = all_points_membership_vectors(c_model)

        clusters_top_3 = pd.DataFrame(np.argsort(soft_clusters, axis=1)[:, ::-1][:, :3])
        clusters_top_3_proba = pd.DataFrame(np.sort(soft_clusters, axis=1)[:, ::-1][:, :3])

        clusters_top_3[clusters_top_3_proba < 0.1] = -1
        clusters_top_3[clusters_top_3 != -1] += cluster_offset
        # clusters_top_3.loc[:, 0] = c_labels
        return clusters_top_3

    def _run_umap(self, vectors, metric, scale=True, **kwargs):
        reducer = umap.UMAP(n_jobs=-1, metric=metric, init="pca", **kwargs)
        embedding_2d = reducer.fit_transform(vectors)
        if scale:
            return embedding_2d - embedding_2d.mean(axis=0)
        else:
            return embedding_2d

    def _embedd_cluster_reduce(self, text_catalog, cluster_umap=True, metric="cosine"):
        vectors = np.array(self.embeddings.embed_documents(text_catalog))
        distances = pairwise_distances(vectors, metric=metric)
        embedding_5d = self._run_umap(vectors, metric=metric, n_neighbors=10, min_dist=0.05, n_components=5)
        embedding_2d = self._run_umap(vectors, metric=metric, n_neighbors=10, min_dist=0.05, n_components=2)
        clusters_df = (
            pd.DataFrame({"text_catalog": text_catalog}, index=range(len(text_catalog)))
            .assign(is_extra=False)
            .assign(cluster_attempt=0)
            .assign(umap_0=embedding_2d[:, 0])
            .assign(umap_1=embedding_2d[:, 1])
        )
        initial_min_cluster_size = max(3, min(30, int(len(vectors) * 0.02)))
        min_samples = max(2, int(initial_min_cluster_size * 0.8))
        max_cluster_size = min(75, int(len(vectors) * 0.1))

        cluster_data = embedding_5d if cluster_umap else vectors

        cluster_top_3 = self._predict_clusters(
            vectors=cluster_data,
            min_cluster_size=initial_min_cluster_size,
            max_cluster_size=max_cluster_size,
            min_samples=min_samples,
            metric=metric,
        )
        cluster_top_3.set_index(clusters_df.index, inplace=True)

        clusters_df["cluster"] = cluster_top_3[0]

        unmatched = clusters_df.query("cluster == -1")
        max_iter = 2
        i = 0
        n_unmatched_before = len(unmatched)
        while (len(unmatched) / len(clusters_df)) > 0.15 and int(initial_min_cluster_size / ((i + 1) * 2)) >= 2:
            if i == max_iter:
                print(f"Max iter for clustering reached. {len(unmatched)} points left without cluster")
                break
            extra_clusters = self._predict_clusters(
                vectors=cluster_data[unmatched.index.to_numpy()],
                cluster_offset=cluster_top_3.apply(max).max() + 1,
                max_cluster_size=max(10, int(max_cluster_size * 0.1)),
                min_cluster_size=max(2, int(initial_min_cluster_size / ((i + 1) * 2))),
            )
            extra_clusters.set_index(unmatched.index, inplace=True)
            clusters_df.loc[unmatched.index, "cluster"] = extra_clusters[0].values
            cluster_top_3.loc[unmatched.index, 0] = extra_clusters[0]
            clusters_df.loc[unmatched.index, "is_extra"] = True
            clusters_df.loc[unmatched.index, "cluster_attempt"] = i + 1

            unmatched = clusters_df.query("cluster == -1")
            if len(unmatched) >= n_unmatched_before:
                break
            i += 1

        clusters_df["clusters_fuzzy"] = cluster_top_3.apply(lambda x: x.to_list(), axis=1)
        return clusters_df, distances

    def _cluster_text_catalog(self, text_catalog):
        clusters_df, distances = self._embedd_cluster_reduce([entry.to_text() for entry in text_catalog.values()])
        clusters_df["themes"] = [entry.themes for entry in text_catalog.values()]
        clusters_df["episode_index"] = list(text_catalog.keys())
        distance_map = {}
        for x, y in itertools.combinations(range(len(distances)), 2):
            if distances[x, y] < COSINE_DISTANCE_THRESHOLD:
                distance_map[f"{clusters_df['episode_index'].iloc[x]},{clusters_df['episode_index'].iloc[y]}"] = (
                    distances[x, y]
                )

        cluster_sizes = clusters_df["cluster"].value_counts()

        self.logger.info(f"Clustering results:\n {cluster_sizes}")

        return clusters_df.set_index("episode_index"), distance_map

    @staticmethod
    def _create_clustered_batches(df, batch_size=500, key="text_catalog"):
        batches = []
        cluster_batches = []
        current_batch = []
        current_cluster_batch = []
        current_batch_size = 0

        for cluster, g in df.query("cluster != -1").groupby("cluster"):
            group_size = g[key].apply(lambda x: len(x)).sum()

            if group_size > batch_size * 2:
                if len(current_batch) > 0:
                    batches.append(current_batch)
                    cluster_batches.append(current_cluster_batch)

                sample_fraction = batch_size / group_size
                batches.append([g.sample(frac=sample_fraction)[key].tolist()])
                cluster_batches.append([cluster])

                current_batch = []
                current_cluster_batch = []
                current_batch_size = 0

            elif (current_batch_size + group_size > batch_size) & (current_batch_size > 0):
                batches.append(current_batch)
                cluster_batches.append(current_cluster_batch)
                current_batch = []
                current_cluster_batch = []
                current_batch_size = 0

            current_batch.append(g[key].tolist())
            current_cluster_batch.append(cluster)
            current_batch_size += group_size

        if len(current_batch) > 0:
            batches.append(current_batch)
            cluster_batches.append(current_cluster_batch)

        return batches, cluster_batches

    def _generate_cluster_titles(self, analysed_episodes, clusters_df):
        parser = PydanticOutputParser(pydantic_object=ClusterTitlesBatch)
        prompt_template = ChatPromptTemplate(CLUSTER_TITLE_PROMPT).partial(
            schema=parser.get_format_instructions(), language=self.language_prompt
        )

        self.logger.info("Consolidating episodes")

        retry_parser = RetryOutputParser.from_llm(parser=parser, llm=self.analysis_llm, max_retries=2)

        chain = RunnableParallel(
            completion=prompt_template | self.analysis_llm, prompt_value=prompt_template
        ) | RunnableLambda(
            lambda x: retry_parser.parse_with_prompt(completion=x["completion"].content, prompt_value=x["prompt_value"])
        )

        batched_text_catalog, batched_clusters = self._create_clustered_batches(
            clusters_df, key="text_catalog", batch_size=8000
        )
        batched_prompts = []

        for batch in batched_text_catalog:
            prompt = {"data": json.dumps(batch)}
            batched_prompts.append(prompt)

        cluster_titles = chain.batch(batched_prompts)

        clusters_df["title"] = self._noise_title

        for cluster_title_batch, cluster_batch in zip(cluster_titles, batched_clusters):
            for cluster_title, cluster in zip(cluster_title_batch.items, cluster_batch):
                clusters_df.loc[clusters_df["cluster"] == cluster, "title"] = cluster_title

        clusters_unique_df = clusters_df.groupby("cluster")["title"].first()
        clusters_unique_df.loc[-1] = self._noise_title

        consolidated_episodes = []
        for e in analysed_episodes.episodes:
            insights = EpisodeInsights(**e.insights.model_dump())
            clusters = [
                c for c in clusters_df.loc[e.metadata.index, "clusters_fuzzy"] if c in clusters_unique_df
            ]  # if c != -1
            cluster_titles = [clusters_unique_df.loc[c] for c in clusters]
            cluster_title = clusters_df.loc[e.metadata.index, "title"]
            embeddings2d = clusters_df.loc[[e.metadata.index], ["umap_0", "umap_1"]].values.tolist()
            if len(cluster_titles) > 0:
                clusters = ClusteredEpisodeInsights(
                    titles=cluster_titles,
                    ids=clusters,
                    embeddings=embeddings2d,
                    attempt=clusters_df.loc[e.metadata.index, "cluster_attempt"],
                )
            else:
                clusters = ClusteredEpisodeInsights(titles=[], ids=[], embeddings=embeddings2d)

            consolidated_episodes.append(Episode(metadata=e.metadata, insights=insights, clusters=clusters))
        return AnalyzedEpisodes(episodes=consolidated_episodes), clusters_df

    def _consolidate_clusters(self, analysed_episodes, clusters_df):
        parser = PydanticOutputParser(pydantic_object=ConsolidatedTitles)
        prompt_template = ChatPromptTemplate(CONSOLIDATION_PROMPT).partial(
            schema=parser.get_format_instructions(), language=self.language_prompt
        )

        chain = prompt_template | self.analysis_llm | parser
        self.logger.info("Consolidating clusters")

        cluster_titles = (
            clusters_df.drop_duplicates("cluster").query("cluster != -1").set_index("cluster")["title"].dropna()
        )
        cluster_titles_doc = ",".join(cluster_titles.to_list())

        consolidated_clusters = chain.invoke(cluster_titles_doc)

        clusters_df["consolidated_title"] = clusters_df["title"]
        for c_title, r_titles in consolidated_clusters.mapping.items():
            if len(r_titles) > 0:
                clusters_df.loc[clusters_df["title"].isin(r_titles), "consolidated_title"] = c_title

        clusters_unique_df = clusters_df.groupby("cluster")["consolidated_title"].first()
        clusters_unique_df.loc[-1] = self._noise_title

        consolidated_episodes = []
        for e in analysed_episodes.episodes:
            ep_copy = Episode(**e.model_dump())
            clusters = [c for c in clusters_df.loc[e.metadata.index, "clusters_fuzzy"] if c in clusters_unique_df]
            cluster_titles = [clusters_unique_df.loc[c] for c in clusters]
            ep_clusters_df = pd.DataFrame({
                "clusters": clusters,
                "consolidated_titles": cluster_titles,
            }).drop_duplicates()
            if len(ep_clusters_df) > 1:
                # remove noise cluster from episodes with one or more dedicated clusters
                ep_clusters_df = ep_clusters_df.query("clusters != -1")
            if len(ep_clusters_df) > 0:
                ep_copy.clusters.consolidated_titles = ep_clusters_df["consolidated_titles"].unique().tolist()
                ep_copy.clusters.ids = ep_clusters_df["clusters"].to_list()

            consolidated_episodes.append(ep_copy)

        return AnalyzedEpisodes(episodes=consolidated_episodes), clusters_df

    def _get_major_categories(self, analysed_episodes, clusters_df):
        parser = PydanticOutputParser(pydantic_object=MajorCategories)
        prompt_template = ChatPromptTemplate(CATEGORY_PROMPT).partial(
            schema=parser.get_format_instructions(), language=self.language_prompt
        )

        chain = prompt_template | self.analysis_llm | parser
        self.logger.info("Consolidating episodes")

        cluster_titles = clusters_df["consolidated_title"].unique().tolist()
        cluster_titles_doc = ",".join(cluster_titles)

        major_categories = chain.invoke(cluster_titles_doc)

        clusters_df["major_category"] = self._noise_title
        for m_category, titles in major_categories.mapping.items():
            lost_titles = set(titles).difference(set(clusters_df["consolidated_title"].unique()))
            if len(lost_titles) > 0:
                print(f"chatgpt mispelled or lost original title(s): {lost_titles}")
            clusters_df.loc[clusters_df["consolidated_title"].isin(titles), "major_category"] = m_category

        category_2_clusters = (
            clusters_df.dropna(subset=["major_category", "consolidated_title"], how="any")
            .groupby("major_category")["consolidated_title"]
            .apply(lambda x: list(set(x)))
            .to_dict()
        )

        consolidated_episodes = []
        for e in analysed_episodes.episodes:
            ep_copy = Episode(**e.model_dump())
            major_category = clusters_df.loc[e.metadata.index, "major_category"]

            if not pd.isnull(major_category):
                ep_copy.clusters.major_category = major_category
            else:
                cluster = clusters_df.loc[e.metadata.index, "cluster"]
                major_category = (
                    clusters_df.dropna(subset=["major_category"])
                    .groupby("cluster")["major_category"]
                    .unique()
                    .get(cluster, [self._noise_title])
                )
                ep_copy.clusters.major_category = major_category[0]

            consolidated_episodes.append(ep_copy)
        return AnalyzedEpisodes(episodes=consolidated_episodes, category_2_clusters=category_2_clusters), clusters_df

    def run(self, limit=1000):
        """
        Runs the analysis pipeline with tqdm progress bar (no Streamlit).
        """

        with tqdm(
            total=100, desc="Podcast Analysis", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        ) as pbar:
            pbar.update(20)
            pbar.set_description(
                f"Analyzing {self.title} rss feed ({min(self.size, limit)} episodes)"
                f" with {self.extraction_llm.model_name} ..."
            )
            analysed_episodes = self.analyze_feed(limit)

            pbar.update(30 - pbar.n)
            pbar.set_description("Clustering episode summaries ...")
            text_catalog = self._create_episode_text_catalog(analysed_episodes.episodes)
            summary_clusters, distance_map = self._cluster_text_catalog(text_catalog)

            pbar.update(20)
            pbar.set_description(f"Creating cluster titles with {self.analysis_llm.model_name}...")
            clustered_episodes, titled_clusters = self._generate_cluster_titles(analysed_episodes, summary_clusters)

            pbar.update(10)
            pbar.set_description(f"Consolidating clusters with {self.analysis_llm.model_name}...")
            consolidated_episodes, consolidated_clusters = self._consolidate_clusters(
                clustered_episodes, titled_clusters
            )

            pbar.update(10)
            pbar.set_description(f"Generating major categories with {self.analysis_llm.model_name}...")
            finalized_episodes, final_clusters = self._get_major_categories(
                consolidated_episodes, consolidated_clusters
            )
            finalized_episodes.distance_map = distance_map
            finalized_episodes.extra["consolidation_map"] = (
                final_clusters.groupby("consolidated_title")["title"].apply(lambda x: list(set(x))).to_dict()
            )
            self.logger.info("analysis completed")
            pbar.update(10)
            pbar.set_description("Preparing plot ...")
            return finalized_episodes

    def run_with_streamlit_progress(self, progress_bar, limit=1000):
        """
        @progress_bar: st.progress widget
        """
        progress_bar.progress(
            20,
            f"""
            Analyzing {self.title} rss feed ({min(self.size, limit)} episodes) with {self.extraction_llm.model_name} ...
            """,
        )
        analysed_episodes = self.analyze_feed(limit)

        progress_bar.progress(50, "Clustering episode summaries ...")
        text_catalog = self._create_episode_text_catalog(analysed_episodes.episodes)
        summary_clusters, distance_map = self._cluster_text_catalog(text_catalog)
        progress_bar.progress(70, f"Creating cluster titles with {self.analysis_llm.model_name}...")
        clustered_episodes, titled_clusters = self._generate_cluster_titles(analysed_episodes, summary_clusters)
        progress_bar.progress(80, f"Consolidating clusters with {self.analysis_llm.model_name}...")
        consolidated_episodes, consolidated_clusters = self._consolidate_clusters(clustered_episodes, titled_clusters)

        progress_bar.progress(90, f"Generating major categories with {self.analysis_llm.model_name}...")
        finalized_episodes, final_clusters = self._get_major_categories(consolidated_episodes, consolidated_clusters)
        finalized_episodes.distance_map = distance_map
        finalized_episodes.extra["consolidation_map"] = (
            final_clusters.groupby("consolidated_title")["title"].apply(lambda x: list(set(x))).to_dict()
        )
        self.logger.info("analysis completed")
        progress_bar.progress(95, "Preparing plot ...")

        return finalized_episodes
