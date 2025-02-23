#from langchain_community.document_loaders import RSSFeedLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain.output_parsers import RetryOutputParser
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from pydantic import BaseModel, Field, RootModel, field_validator
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
import umap

class RSSFeedItem(BaseModel):
    title: str = Field(..., description="The title of the episode")
    subtitle: Optional[str] = Field(default=None, description="The subtitle of the episode")
    description: Optional[str] = Field(default=None, description="The description of the episode", alias="summary")
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



class EpisodeInsights(BaseModel):
    """Information about a person."""
    episode_id: str = Field(..., description="The unique identifier of the episode extracted from title or description.")
    topic: str = Field(..., description="The topic of the episode.")
    summary: str = Field(..., description="Short summary of the episode")
    topic_year: Optional[int] = Field(default=None, description="The year related to the topic not the episode (Best guess if not explicitly mentioned)")
    topic_century: Optional[int] = Field(default=None, description="The century related to the topic (Best guess if not explicitly mentioned)")
    tags: List[str] = Field(..., description="Enriched tags associated with the episode.")
    inferred_themes: List[str] = Field(..., description="Common themes that seem to fit to the episode.")
    referenced_episodes_id: Optional[List[str]] = Field(..., description="Referenced episodes in this episode. Needs to match episode_id schema")

class ClusteredEpisodeInsights(BaseModel):
    titles: List[str] = Field(..., description="Cluster titles")
    ids: List[int] = Field(..., description="Cluster ids of the episode.")
    embeddings: Optional[Any] = Field(default = None, description="Umap embeddings")

    @field_validator('titles', mode="before")
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
    
    def save_episodes(self, file_path: str):
        Path(file_path).parent.mkdir(exist_ok=True, parents=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.model_dump_json(indent=4))
    
    @classmethod
    def load(cls, file_path: str | Path):
        with open(str(file_path), 'r', encoding='utf-8') as f:
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
    def __init__(self, rss_url, 
                 model="gpt-4o-mini", 
                 llm_api_key=None, 
                 logger=None, 
                 embedding_model="text-embedding-3-small",
                 cluster_themes=False):
        self.rss_loader = RSSFeedLoader(rss_url)
        self.llm = ChatOpenAI(model=model, api_key=llm_api_key)
        self.embeddings = self._init_embeddings(embedding_model)
        self.cluster_themes = cluster_themes

        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

    @property
    def title(self):
        return self.rss_loader.title

    def _init_embeddings(self, embedding_model, path="./cache/"):
        embeddings = OpenAIEmbeddings(model=embedding_model)
        os.makedirs(path, exist_ok=True)
        store = LocalFileStore(path)
        return CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings=embeddings, 
            document_embedding_cache=store, 
            namespace=embeddings.model, # Create a cache-backed embedder using the base embedding and storage
            )
    def analyze_feed(self, limit=10000):
        episode_loader = self.rss_loader.lazy_load()
        self.logger.info(f"Analyzing {self.rss_loader.size} podcast episdoes")
        parser = PydanticOutputParser(pydantic_object=EpisodeInsights)

        prompt_template = ChatPromptTemplate([
            ("system", """
            You are an information extraction and generalisation specialist for a podcast called {podcast}.
            This is the description of the podcast to provide more context: {podcast_description}
            
            Your task:
            
            Given the description of an episode you have the following tasks:
                - give a short and poignant summary of the episode in no more than 30 words
                - extract up to {tag_limit} relevant tags 
                - suggest up to {theme_limit} fitting themes or topic areas that can be used to describe and generalize the topic of the episode.
                - extract year and century of the topic. If not provided in description make a best guess based on the topic.
                - check if there are references to other episodes (episode_id <-> referenced_episode_ids) 
            
            The goal is to analyse and cluster all of the episodes in a later stage, so the themes and tags should be consistent across all episodes.
            Constraints:
                - The tags and themes must be in the same language as the input

                - Output your answer as JSON that matches the given schema: {format_instructions}.
             
            
            """),
            ("user", "Episode Title: {title}\n\n Episode Content: {episode_content}"),
        ]).partial(format_instructions=parser.get_format_instructions(),
                    podcast=self.rss_loader.title, 
                    tag_limit=5,
                    theme_limit=2,
                    podcast_description=self.rss_loader.description)
    


        request_chain = prompt_template | self.llm 
        retry_parser = RetryOutputParser.from_llm(parser=parser, llm=self.llm, max_retries=2)

        chain = RunnableParallel(
            completion=request_chain,
            prompt_value=prompt_template  # Add the prompt_value here
        ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(
            completion=x['completion'].content,  # Extract the content from AIMessage
            prompt_value=x['prompt_value']
        ))

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
    
    def _create_text_catalog(self, analysed_episodes, keys=["inferred_themes"]):
        text_catalog = {}
        for e in analysed_episodes.episodes():
            text_catalog[e.insights.episode_id] = []
            for k in keys():
                text_catalog[e.insights.episode_id].extend(getattr(e.insights,k))
            text_catalog[e.insights.episode_id] = ",".join(text_catalog[e.insights.episode_id]
                                                           )
        return text_catalog
        
    
    @staticmethod
    def _predict_clusters(vectors, **kwargs):
        
        c_model = HDBSCAN(**kwargs,
                    #min_cluster_size=min_cluster_size, 
                    #max_cluster_size=min(50,len(vectors)//4), 
                    metric="cosine",
                    #store_centers="medoid",
                    )
        clusters = c_model.fit_predict(np.array(vectors)) 
        #soft_clusters = all_points_membership_vectors(c_model)
        #assigned_clusters = np.argmax(soft_clusters, axis=1)

        #parents = c_model.dbscan_clustering(0.3, min_cluster_size=2)
        return clusters

    
    def _run_umap(self, vectors):
        reducer = umap.UMAP(n_neighbors=20, n_jobs=-1)
        embedding_2d = reducer.fit_transform(vectors)
        return embedding_2d
    

    def _embedd_cluster_reduce(self, text_catalog, cluster_umap=True):
        vectors =  np.array(self.embeddings.embed_documents(text_catalog))
        
        embedding_2d = self._run_umap(vectors)
        clusters_df = (
            pd.DataFrame({"text_catalog":text_catalog},
                         index=range(len(text_catalog)))
                         .assign(is_extra=False)
                         .assign(umap_0=embedding_2d[:,0])
                         .assign(umap_1=embedding_2d[:,1])
                        
                    )
        initial_min_cluster_size = 5
        max_cluster_size = min(50,len(vectors)//4)
        if cluster_umap:
            clusters_df["cluster"] = self._predict_clusters(vectors=embedding_2d, 
                                                            min_cluster_size=initial_min_cluster_size,
                                                            max_cluster_size=max_cluster_size
                                                            )
            
        else:
            clusters_df["cluster"] = self._predict_clusters(vectors=vectors,
                                                            min_cluster_size=initial_min_cluster_size,
                                                            max_cluster_size=max_cluster_size
                                                            )

        unmatched = clusters_df.query("cluster == -1")
        max_iter = 10
        i = 0
        while (len(unmatched) / len(clusters_df)) > 0.15:
            if i == max_iter:
                print(f"Max iter for clustering reached. {len(unmatched)} points left without cluster")
                
            extra_clusters = self._predict_clusters(vectors=vectors[unmatched.index.to_numpy()],
                                                    max_cluster_size=max_cluster_size,
                                                    min_cluster_size=2)
            extra_clusters[extra_clusters != -1] = extra_clusters[extra_clusters != -1] + clusters_df.cluster.max()
            clusters_df.loc[unmatched.index, "cluster"] = extra_clusters
            clusters_df.loc[unmatched.index, "is_extra"] = True

            unmatched = clusters_df.query("cluster == -1")
            i += 1
            


        return clusters_df

    def _cluster_text_catalog(self, text_catalog):

        clusters_df = self._embedd_cluster_reduce(list(text_catalog.values()))
        clusters_df["episode_id"] = list(text_catalog.keys())
    
        cluster_sizes = clusters_df["cluster"].value_counts()
        
        self.logger.info(f"Clustering results:\n {cluster_sizes}")

        return clusters_df.set_index("episode_id")

        
    @staticmethod
    def _create_clustered_batches(df, batch_size=500):
        batches = []
        cluster_batches = []
        current_batch = []
        current_cluster_batch = []
        current_batch_size = 0

        for cluster, g in df.query("cluster != -1").groupby("cluster"):
            group_size = g["text_catalog"].apply(lambda x: len(x)).sum()

            if group_size > batch_size*2:
                if len(current_batch) > 0:
                    batches.append(current_batch)
                    cluster_batches.append(current_cluster_batch)
 
                
                sample_fraction = batch_size / group_size
                batches.append([g.sample(frac=sample_fraction)["text_catalog"].tolist()])
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

            current_batch.append(g["text_catalog"].tolist())
            current_cluster_batch.append(cluster)
            current_batch_size += group_size

        if len(current_batch) > 0:
            batches.append(current_batch)
            cluster_batches.append(current_cluster_batch)

        return batches, cluster_batches


    def _consolidate_themes(self, analysed_episodes, clusters_df):
        parser = PydanticOutputParser(pydantic_object=ClusterTitlesBatch)
        prompt_template = ChatPromptTemplate([
                ("system", """
                You are an expert in summarizing and generalizing collections of {field_name} with a short title. 
                Given a list of sets with related text give each set a poignant title that best describes the items within the set.
                If the two or more sets are semantically very similar, apply the same title to all of them!

                Constraints: 
                 - The output list must be the same length as the input list
                 - Each item (a set of {field_name}) must be summarized as text
                 - The original language must be maintained
                 - The output must be a valid JSON in the format: {schema}
                
                Example Input: [['Artificial Intelligence','AI', 'Machine Learning'], ['ChatGPT', 'LLM', 'OpenAI'] ]
                Example Output: ['Artificial Intelligence', 'Artificial Intelligence']
                """),
                ("user", "List with sets of {field_name}: {data}"),
            ]).partial(schema=parser.get_format_instructions())

        chain = prompt_template | self.llm | parser
        self.logger.info("Consolidating episodes")
        
        batched_text_catalog, batched_clusters = self._create_clustered_batches(clusters_df)
        batched_prompts = []
    

        for batch in batched_text_catalog:
            prompt= {
                "field_name": "themes",
                "data": json.dumps(batch)
             }
            batched_prompts.append(prompt)

        
        #tag_mappings = chain.invoke(prompt_data["tags"]).model_dump()
        cluster_titles = chain.with_retry().batch(batched_prompts)
        clusters_df["title"] = None
        
        for cluster_title_batch, cluster_batch in zip(cluster_titles, batched_clusters):
            for cluster_title, cluster in zip(cluster_title_batch.items, cluster_batch):
                clusters_df.loc[clusters_df["cluster"]==cluster, "title"] = cluster_title
        
        clusters_df.loc[clusters_df["title"].isnull(), "title"] = clusters_df.loc[clusters_df["title"].isnull(), "text_catalog"]
        text_catalog_mapping = clusters_df.set_index("text_catalog")["title"].to_dict()

        
        title_df = clusters_df.drop_duplicates(subset="title").reset_index().set_index("title")
        title_2_index = title_df["index"].to_dict()

        consolidated_episodes = []
        for e in analysed_episodes.episodes:
            insights = EpisodeInsights(**e.insights.model_dump())
            clustered_episode_themes = {theme: text_catalog_mapping[theme] for theme in e.insights.inferred_themes if theme in text_catalog_mapping}
            insights.inferred_themes = list(set(clustered_episode_themes.values()))
            
            indices = [title_2_index[t] for t in insights.inferred_themes]
            cluster_ids = clusters_df.loc[indices, "cluster"].to_list()
            embeddings2d = clusters_df.loc[indices, ["umap_0","umap_1"]].values.tolist()


            clusters = ClusteredEpisodeInsights(
                titles=list(set(clustered_episode_themes.values())),
                ids=cluster_ids,
                embeddings=embeddings2d
            )

            consolidated_episodes.append(Episode(
                metadata=e.metadata,
                insights=insights,
                clusters=clusters
                )
            )
        return AnalyzedEpisodes(episodes=consolidated_episodes)


    def _consolidate_summaries(self, analysed_episodes, clusters_df):
        parser = PydanticOutputParser(pydantic_object=ClusterTitlesBatch)
        prompt_template = ChatPromptTemplate([
                ("system", """
                You are an expert in gerneralizing semantic content. 
                Your task is to provide a poignant, evocative, and concise title for a group of related documents. 
                The title should capture the essence, themes, and emotional tone of the documents while being engaging. 
                
                Consider the following guidelines:
                    - Generalsation: Capturing the bigger ideas and topics behind the selected documents.
                    - Relevance: The title should reflect the core ideas, topics, or narratives present in the documents.
                    - Emotional Resonance: It should evoke an emotional response or curiosity, drawing the reader in.
                    - Conciseness: Keep the title concise ideally no longer than 5 words.
                                 
                Constraints (Hard rules): 
                 - The output list must be the same length as the input list
                 - The original language must be maintained. Do not change the language!
                 - The output must be a valid JSON in the format: {schema}
                
                Example Input (extract): [['This episode explores how sound design shapes our experiences in ways we often don’t notice...','This episode uncovers the surprising histories and cultural significance behind everyday colors.'], ]
                Example Output: ['The Hidden Designs That Shape Our World']
                """),
                ("user", "Cluster of documents: {data}"),
            ]).partial(schema=parser.get_format_instructions())

        chain = prompt_template | self.llm | parser
        self.logger.info("Consolidating episodes")
        
        batched_text_catalog, batched_clusters = self._create_clustered_batches(clusters_df, batch_size=1000)
        batched_prompts = []
    

        for batch in batched_text_catalog:
            prompt= {
                "data": json.dumps(batch)
             }
            batched_prompts.append(prompt)

        
        cluster_titles = chain.with_retry().batch(batched_prompts)
        clusters_df["title"] = None
        
        for cluster_title_batch, cluster_batch in zip(cluster_titles, batched_clusters):
            for cluster_title, cluster in zip(cluster_title_batch.items, cluster_batch):
                clusters_df.loc[clusters_df["cluster"]==cluster, "title"] = cluster_title
        
        
        consolidated_episodes = []
        for e in analysed_episodes.episodes:
            insights = EpisodeInsights(**e.insights.model_dump())
            # list compatibility with self.cluster_themes mode
            cluster_title = clusters_df.loc[e.insights.episode_id, "title"]
            embeddings2d =  clusters_df.loc[[e.insights.episode_id], ["umap_0","umap_1"]].values.tolist()
            if not pd.isnull(cluster_title):
                cluster_titles = [cluster_title]
                cluster_ids = clusters_df.loc[[e.insights.episode_id], "cluster"]
            

                clusters = ClusteredEpisodeInsights(
                    titles=cluster_titles,
                    ids=cluster_ids,
                    embeddings=embeddings2d
                )
            else:
                clusters = ClusteredEpisodeInsights(
                    titles=[],
                    ids=[],
                    embeddings=embeddings2d
                )

            consolidated_episodes.append(Episode(
                metadata=e.metadata,
                insights=insights,
                clusters=clusters
                )
            )
        return AnalyzedEpisodes(episodes=consolidated_episodes)

    
    def run(self, limit=1000):

        analysed_episodes = self.analyze_feed(limit)
            
        if self.cluster_themes:
            text_catalog = self._create_text_catalog(analysed_episodes, keys=["inferred_themes"])
            theme_clusters = self._cluster_text_catalog(text_catalog)
            consolidated_episodes = self._consolidate_themes(analysed_episodes, theme_clusters)
        else:
            text_catalog = {ep.insights.episode_id: ep.insights.summary for ep in analysed_episodes.episodes}
            summary_clusters = self._cluster_text_catalog(text_catalog)
            consolidated_episodes = self._consolidate_summaries(analysed_episodes, summary_clusters)
            
        return consolidated_episodes
    
    def run_with_streamlit_progress(self, progress_bar, limit=1000):
        """ 
          @progress_bar: st.progress widget
        """
        progress_bar.progress(20, f"Analyzing rss feed with {self.llm.model_name}...")
        analysed_episodes = self.analyze_feed(limit)
            
        if self.cluster_themes:
            keys = ["inferred_themes"]
            progress_bar.progress(50, f"Clustering {','.join(keys)}...")
            text_catalog = self._create_text_catalog(analysed_episodes, keys=keys)
            theme_clusters = self._cluster_text_catalog(text_catalog)
            progress_bar.progress(70, f"Consolidating clusters ...")
            consolidated_episodes = self._consolidate_themes(analysed_episodes, theme_clusters)
        else:
            progress_bar.progress(50, f"Clustering episode summaries ...")
            text_catalog = {ep.insights.episode_id: ep.insights.summary for ep in analysed_episodes.episodes}
            summary_clusters = self._cluster_text_catalog(text_catalog)
            progress_bar.progress(70, f"Consolidating clusters with {self.llm.model_name}...")
            consolidated_episodes = self._consolidate_summaries(analysed_episodes, summary_clusters)
        self.logger.info("analysis completed")
        progress_bar.progress(90, "Preparing plot ...")
        
        return consolidated_episodes

