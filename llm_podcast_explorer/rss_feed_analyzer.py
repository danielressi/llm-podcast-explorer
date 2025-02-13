#from langchain_community.document_loaders import RSSFeedLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from pydantic import BaseModel, Field, RootModel
from typing import Dict, List, Any, Optional, Set
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import normalize
#from hdbscan import HDBSCAN
#from hdbscan.prediction import all_points_membership_vectors
import pandas as pd
import json
import numpy as np
import os
from pathlib import Path
import logging
import feedparser
import itertools
import collections

class RSSFeedItem(BaseModel):
    title: str = Field(..., description="The title of the episode")
    subtitle: Optional[str] = Field(default=None, description="The subtitle of the episode")
    description: str = Field(..., description="The description of the episode", alias="summary")
    id: str = Field(..., description="The unique identifier of the episode")
    episode_number: Optional[int] = Field(default=None, description="", alias="podcast_episode")
    link: str = Field(..., description="The link to the episode")
    published: str = Field(..., description="The date the episode was published")
    tags: Optional[List[Any]] = Field(default=None, description="Tags associated with the episode")


class RSSFeedLoader:
    def __init__(self, url):
        self.feed = feedparser.parse(url)
        self.description = self.feed.feed.get("description", "")
        self.title = self.feed.feed.get("title", "")
        self.size = len(self.feed.entries)

    def lazy_load(self):
        for entry in self.feed.entries:
            yield RSSFeedItem(**entry)



class WarmupOutput(BaseModel):
    """Extra metadata inferred during warmump"""
    podcast_name: str = Field(..., description="The name of the podcast")
    podcast_description: str = Field(..., description="The description of the podcast in one sentence")
    suggested_extra_fields: Dict[str, Any] = Field(..., description="Characterstic extra fields as keys and examples as values")


class EpisodeInsights(BaseModel):
    """Information about a person."""
    episode_id: str = Field(..., description="The unique identifier of the episode extracted from title or description.")
    topic: str = Field(..., description="The topic of the episode.")
    topic_year: Optional[int] = Field(default=None, description="The year related to the topic not the episode. null if unknown")
    topic_century: Optional[int] = Field(default=None, description="The century related to the topic.")
    tags: List[str] = Field(..., description="Enriched tags associated with the episode.")
    inferred_themes: List[str] = Field(..., description="Common themes that seem to fit to the episode.")
    referenced_episodes_id: List[str] = Field(..., description="Referenced episodes in this episode. Needs to match episode_id schema")
    extra: Dict[str, Any] = Field(..., description="Characterstic extra information unique to this podcast")

class ClusteredEpisodeInsights(BaseModel):
    #tags: Optional[Set[str]] = Field(default=None, description="Clustered tags associated with the episode.")
    themes: Set[int] = Field(..., description="Cluster theme ids of the episode.")
    #topics: Set[str] = Field(..., description="Clustered topics of the episode.")
class Episode(BaseModel):

    metadata: Dict[str, Any] = Field(..., description="Metadata of the episode.")
    insights: EpisodeInsights = Field(..., description="Insights of the episode.")
    clusters: Optional[ClusteredEpisodeInsights] = Field(default=None, description="Mapping from insights to clusters")

class AnalyzedEpisodes(BaseModel):
    episodes: List[Episode] = Field(..., description="A list of analyzed episodes.")
    
    def save_episodes(self, file_path: str):
        Path(file_path).parent.mkdir(exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.model_dump_json(indent=4))
    
    @classmethod
    def load(cls, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        episodes = [Episode(**episode) for episode in data["episodes"]]
        return cls(episodes=episodes)



class Mapping(RootModel):
    root: Dict[str, str]

class SimpleList(RootModel):
    root: List[str]

class ClusterTitlesBatch(BaseModel):
    items: List[str] = Field(..., description="Batch of cluster titles")


class RSSFeedAnalyzer:
    def __init__(self, rss_url, model="gpt-4o-mini", llm_api_key=None, warmup=True, logger=None, embedding_model="text-embedding-3-small"):
        self.rss_loader = RSSFeedLoader(rss_url)
        self.llm = ChatOpenAI(model=model, api_key=llm_api_key)
        self.embeddings = self._init_embeddings(embedding_model)
        self.warmup = warmup
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

    def _init_embeddings(self, embedding_model, path="./cache/"):
        embeddings = OpenAIEmbeddings(model=embedding_model)
        os.makedirs(path, exist_ok=True)
        store = LocalFileStore(path)
        return CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=embeddings, 
            document_embedding_cache=store, 
            namespace=embeddings.model, # Create a cache-backed embedder using the base embedding and storage
            )


    def _run_warmup(self, sample_size=5):
        example_content = "\n\n".join([episode.description for i,episode in enumerate(self.rss_loader.lazy_load()) if i < sample_size] )
        prompt_template = ChatPromptTemplate([
            ("system", """
            You are an information extraction and generalisation specialist for a podcast.
            This is the description of the podcast: {podcast_description}
             
            Given the summary of {sample_size} example episodes, analyse common patterns in the the way the content is structured and how episodes can be linked with each other in a network graph. 
            Based on these patterns recommend extra kev value pairs (in the original language) that generalize well and could be extracted for all other episodes. If field values contains items make sure to use Lists.
            Output your answer as JSON that matches the given schema, but avoid the existing fields {episode_insights}
            """),
            ("user", "{sample_size} example contents of one podcast: {episode_content}"),
        ]).partial(
                   sample_size=sample_size, 
                   podcast_description=self.rss_loader.description,
                   episode_insights=EpisodeInsights.model_fields.keys())
        
        warmup_chain = prompt_template | self.llm 

        return warmup_chain.invoke({"episode_content": example_content})

    def analyze_feed(self, limit=10000):
        episode_loader = self.rss_loader.lazy_load()
        parser = PydanticOutputParser(pydantic_object=EpisodeInsights)
        if self.warmup:
            self.logger.info("Running warmup")
            warmup_output = self._run_warmup()
            prompt_template = ChatPromptTemplate([
                ("system", """
                You are an information extraction and generalisation specialist for a podcast called {podcast_name}. {podcast_description}
                Given the description of an episode, extract the relevant tags and suggest themes that can be used to generalize and cluster all of the episodes later.
                Output your answer (in same language as the input)  JSON that matches the given schema: {format_instructions}. Example for the extra fields and patterns to pay attention to are {extra_schema}
                """),
                ("user", "Episode Title: {title}\n\nEpisode Content: {episode_content}"),
            ]).partial(format_instructions=parser.get_format_instructions(), 
                       extra_schema=warmup_output.suggested_extra_fields,
                       podcast_name=warmup_output.podcast_name,
                       podcast_description=warmup_output.podcast_description
                       )
        else:
            prompt_template = ChatPromptTemplate([
                ("system", """
                You are an information extraction and generalisation specialist for a podcast called {podcast}.
                Given the description of an episode, extract up to {limit} relevant tags and also suggest fitting themes that can be used to describe and generalize the episode.
                The goal is to analyse and cluster all of the episodes in a later stage, so the themes and tags should be consistent across all episodes.
                Constraints:
                    - The tags and themes must be in the same language as the input
                    - Output your answer as JSON that matches the given schema: {format_instructions}.
                """),
                ("user", "Episode Title: {title}\n\n Episode Content: {episode_content}"),
            ]).partial(format_instructions=parser.get_format_instructions(), podcast=self.rss_loader.title, limit=5)
        


        chain = prompt_template | self.llm | parser.with_retry()
        
        analysis_results = []
        batch_content = []
        batch_metadata = []
        batch_size = 20
        for i, episode in enumerate(episode_loader):
            if i > limit:
                break
            elif (i % batch_size == 0 and i > 0) or i == limit:
                self.logger.info(f"Running batch {i//batch_size}") 
                response = chain.batch(batch_content)

                analysed_episodes = [Episode(metadata=m, insights=r) for m, r in zip(batch_metadata, response)]
                
                analysis_results.extend(analysed_episodes)
                batch_metadata = []
                batch_content = []
            else:
                batch_content.append({"title": episode.title, "episode_content": episode.description})
                batch_metadata.append(episode.model_dump(exclude="description"))

        return AnalyzedEpisodes(episodes=analysis_results)
    
    def _create_vocabulary(self, analysed_episodes, keys=["inferred_themes"]):
        vocabulary = collections.Counter()
        for key in keys:
            vocabulary.update(set(itertools.chain.from_iterable([getattr(e.insights,key) for e in analysed_episodes.episodes])))
        return vocabulary
        
    
    @staticmethod
    def _predict_clusters(vectors):
        c_model = HDBSCAN(min_cluster_size=2, 
                    max_cluster_size=50, 
                    metric="cosine",
                    #store_centers="medoid",
                    )
        clusters = c_model.fit_predict(np.array(vectors)) 
        #soft_clusters = all_points_membership_vectors(c_model)
        #assigned_clusters = np.argmax(soft_clusters, axis=1)


        return clusters

    
    def _cluster_vocabulary(self, analysed_episodes):
        vocabulary = self._create_vocabulary(analysed_episodes)
        vectors =  np.array(self.embeddings.embed_documents(vocabulary.keys()))

        clusters = self._predict_clusters(vectors)
        clusters_df = pd.DataFrame({"vocabulary":vocabulary.keys(),
                            "cluster":clusters,
                            }, index=range(len(vocabulary))).assign(is_extra=False)
        #clusters_df = pd.merge(clusters_df, cluster_sizes, how="left", left_on="cluster", right_index=True)
        cluster_sizes = clusters_df["cluster"].value_counts()
        unmatched = clusters_df.query("cluster == -1")
        max_iter = 10
        i = 0
        while (len(unmatched) > 10):
            if i == max_iter:
                print(f"Max iter for clustering reached. {len(unmatched)} points left without cluster")
                
            extra_clusters = self._predict_clusters(vectors[unmatched.index.to_numpy()])
            extra_clusters[extra_clusters != -1] = extra_clusters[extra_clusters != -1] + clusters_df.cluster.max()
            clusters_df.loc[unmatched.index, "cluster"] = extra_clusters
            clusters_df.loc[unmatched.index, "is_extra"] = True

            unmatched = clusters_df.query("cluster == -1")
            i += 1
            
        
        cluster_sizes_final = clusters_df["cluster"].value_counts()

        
        self.logger.info("Clustering results:\n {cluster_sizes}")
        return clusters_df
        
    @staticmethod
    def _create_clustered_batches(df, batch_size=100):
        batches = []
        cluster_batches = []
        current_batch = []
        current_cluster_batch = []
        current_batch_size = 0

        for cluster, g in df.query("cluster != -1").groupby("cluster"):
            group_size = len(g) #['count']
            if (current_batch_size + group_size > batch_size) & (current_batch_size > 0):
                
                batches.append(current_batch)
                cluster_batches.append(current_cluster_batch)
                current_batch = []
                current_cluster_batch = []
                current_batch_size = 0

            current_batch.append(g["vocabulary"].tolist())
            current_cluster_batch.append(cluster)
            current_batch_size += group_size

        if len(current_batch) > 0:
            batches.append(current_batch)
            cluster_batches.append(current_cluster_batch)

        return batches, cluster_batches


    def _consolidate(self, analysed_episodes, clusters_df):
        parser = PydanticOutputParser(pydantic_object=ClusterTitlesBatch)
        prompt_template = ChatPromptTemplate([
                ("system", """
                You are an expert in generalizing sets of {field_name} with a short title. 
                Given a list of sets with related text give each document a poignant title that best describes the items within the set.
                If the content of two or more sets is very similar, apply the same title to all of them

                Constraints: 
                 - The output list must be the same length as the input list
                 - Each item (a set of {field_name}) must be summarized as text
                 - The original language must be maintained
                 - The output must be a valid JSON in the format: {schema}
                
                Example Input: [['Artificial Intelligence','AI', 'Machine Learning'], ['ChatGPT', 'LLM', 'OpenAI'] ]
                Example Output: ['Artificial Intelligence', 'Artificial Intelligence']
                """),
                ("user", "Unique set of {field_name}: {data}"),
            ]).partial(schema=parser.get_format_instructions())

        chain = prompt_template | self.llm | parser
        self.logger.info("Consolidating episodes")
        
        batched_vocabulary, batched_clusters = self._create_clustered_batches(clusters_df)
        batched_prompts = []
        for batch in batched_vocabulary:
            prompt= {
                "field_name": "themes and topics",
                "data": json.dumps(batch)
             }
            batched_prompts.append(prompt)

        
        #tag_mappings = chain.invoke(prompt_data["tags"]).model_dump()
        cluster_titles = chain.with_retry().batch(batched_prompts)
        clusters_df["title"] = None
        
        for cluster_title_batch, cluster_batch in zip(cluster_titles, batched_clusters):
            for cluster_title, cluster in zip(cluster_title_batch.items, cluster_batch):
                clusters_df.loc[clusters_df["cluster"]==cluster, "title"] = cluster_title
        
        clusters_df.loc[clusters_df["title"].isnull(), "title"] = clusters_df.loc[clusters_df["title"].isnull(), "vocabulary"]
        vocabulary_mapping = clusters_df.set_index("vocabulary")["title"].to_dict()

        consolidated_episodes = []
        for e in analysed_episodes.episodes:
            insights = EpisodeInsights(**e.insights.model_dump())
            clustered_episode_themes = {theme: vocabulary_mapping[theme] for theme in e.insights.inferred_themes if theme in vocabulary_mapping}
            cluster_ids = set(clusters_df[clusters_df["vocabulary"].isin(clustered_episode_themes.keys())]["cluster"])
            insights.inferred_themes = set(clustered_episode_themes.values())

            clusters = ClusteredEpisodeInsights(
                themes=cluster_ids
            )

            consolidated_episodes.append(Episode(
                metadata=e.metadata,
                insights=insights,
                clusters=clusters
                )
            )
        return AnalyzedEpisodes(episodes=consolidated_episodes)



    def _cluster(self, analysed_episodes):
        parser = PydanticOutputParser(pydantic_object=Mapping)
        prompt_template = ChatPromptTemplate([
                ("system", """
                You are an expert in consolidating and gernealizing extracted data from podcast episodes. 
                Please review, consolidate, generalize and cluster the items in the set of {field_name} and create a mapping from the original items to the clusters.
                The constraints are:
                 - Each item in the unique set of {field_name} has to be a key in the mapping
                 - Each key maps to a cluster as value in the same language.
                 - The clusters should be general enough so that they to contain multiple keys (inputs)
                 - Th number of clusters (unique values) must not exceed {limit}.
                 - The output must be a valid JSON in the format {schema}
                
                """),
                ("user", "Unique set of {field_name}: {data}"),
            ]).partial(schema=parser.get_format_instructions(), limit=100)

        chain = prompt_template | self.llm | parser.with_retry()
        self.logger.info("Consolidating and clustering episodes")
        

        prompt_data = {field_name:
            {
                "field_name": field_name,
                "data": ",".join(set(itertools.chain.from_iterable([e.insights.model_dump()[field_name] for e in analysed_episodes.episodes])))
            } for field_name in ["inferred_themes"]}
        prompt_data["topics"] = {
            "field_name": "topics",
            "data": ",".join(set([e.insights.topic for e in analysed_episodes.episodes]))
        }
        
        #tag_mappings = chain.invoke(prompt_data["tags"]).model_dump()
        theme_mappings = chain.invoke(prompt_data["inferred_themes"]).model_dump()
        topic_mappings = chain.invoke(prompt_data["topics"]).model_dump()
        
        theme_counter = collections.Counter(theme_mappings.values())
        topic_counter = collections.Counter(topic_mappings.values())
    
        clustered_episodes = []
        for e in analysed_episodes.episodes:
            clusters = ClusteredEpisodeInsights(
                #tags={tag: tag_mappings[tag] for tag in e.insights.tags if tag in tag_mappings},
                themes=set([theme_mappings[theme] for theme in e.insights.inferred_themes if theme in theme_mappings]),
                topics= set([topic_mappings.get(e.insights.topic)]) if topic_mappings.get(e.insights.topic, None) is not None else set()
            )
            clustered_episodes.append(Episode(
                metadata=e.metadata,
                insights=e.insights,
                clusters=clusters
                )
            )
        return AnalyzedEpisodes(episodes=clustered_episodes)

    
    def run(self, limit=10000, checkpoint=True):
        if checkpoint:
            analysed_episodes = AnalyzedEpisodes.load("./dev_data/analysed_episodes.json")
            theme_clusters = analyzer._cluster_vocabulary(analysed_episodes)
            consolidated_episodes = analyzer._consolidate(analysed_episodes, theme_clusters)
        else:
            analysed_episodes = self.analyze_feed(limit)
            #consolidated_results = self._consolidate(analysed_episodes)
            word_clusters = self._cluster_vocabulary(analysed_episodes)
            consolidated_episodes = analysed_episodes
            #clustered_results = self._cluster(consolidated_results)
        return consolidated_episodes



# Example usage:
# rss_urls = ["https://example.com/rss_feed"]
# llm_api_key = "your_openai_api_key"
# analyzer = RSSFeedAnalyzer(rss_urls, llm_api_key)
# analyzer.warmup(sample_size=5)
# results = analyzer.analyze_feeds()
# print(results)

if __name__ == "__main__":
    analyzer = RSSFeedAnalyzer("https://geschichten-aus-der-geschichte.podigee.io/feed/mp3", llm_api_key=os.environ.get('OPENAI_API_KEY'), warmup=False)
    checkpoint = True
    if checkpoint:
        analysed_episodes = AnalyzedEpisodes.load("./dev_data/analysed_episodes.json")
        theme_clusters = analyzer._cluster_vocabulary(analysed_episodes)
        consolidated_episodes = analyzer._consolidate(analysed_episodes, theme_clusters)
        consolidated_episodes.save_episodes("./dev_data/clustered_episodes.json")
    else:
        results = analyzer.run(limit=600)