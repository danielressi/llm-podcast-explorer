#from langchain_community.document_loaders import RSSFeedLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain.output_parsers import RetryOutputParser
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache


#from langchain_community.vectorstores.faiss import FAISS
from pydantic import BaseModel, Field, RootModel, field_validator, AliasChoices
from typing import Dict, List, Any, Optional, Tuple

from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN as HDBSCAN_SKLEARN
from sklearn.metrics import pairwise_distances
from hdbscan import HDBSCAN as HDBSCAN, BranchDetector

from hdbscan.prediction import all_points_membership_vectors
import pandas as pd
import json
import numpy as np
import os
from pathlib import Path
import logging
import itertools
import umap

from rss_feed_loader import RSSFeedLoader, RSSFeedItem

COSINE_DISTANCE_THRESHOLD = 0.5



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
    consolidated_titles: Optional[List[str]] = Field(default=None, description="Cluster titles")
    ids: List[int] = Field(..., description="Cluster ids of the episode.")
    embeddings: Optional[Any] = Field(default = None, description="Umap embeddings")
    major_category: Optional[str] = Field(default = None, description="major categories are the higher level groupings of clusters")
    attempt: Optional[int] = Field(default=None, description="Number of consecutive clustering attempts until noise label was removed")
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
    category_2_clusters: Optional[Dict[str,List[str]]] = Field(default=None, description="Mapping from major categories to clusters")
    distance_map: Optional[Dict[str, float]] = Field(default=None, description="Cosine distances for episode pairs")
    extra: Optional[Dict[str,Any]] = Field(default={}, description="Placeholder for storing any other data for analysis")
    
    def save_episodes(self, file_path: str):
        Path(file_path).parent.mkdir(exist_ok=True, parents=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.model_dump_json(indent=4))
    
    @classmethod
    def load(cls, file_path: str | Path):
        with open(str(file_path), 'r', encoding='utf-8') as f:
            data = json.load(f)
        episodes = [Episode(**episode) for episode in data["episodes"]]
        return cls(episodes=episodes, 
                   category_2_clusters=data["category_2_clusters"],
                   distance_map=data["distance_map"])



class Mapping(RootModel):
    root: Dict[str, str]

class SimpleList(RootModel):
    root: List[str]

class ClusterTitlesBatch(BaseModel):
    items: List[str] = Field(..., description="Batch of cluster titles")

class MajorCategories(BaseModel):
    mapping: Dict[str,List[str]] = Field(..., description="Mapping from identified major categories to all the titles that belong to the major category.")

class ConsolidatedTitles(BaseModel):
    mapping: Dict[str,List[str]] = Field(..., description="Mapping of consolidated titles to the corresponding titles that are semantically too similar, duplicates or synonyms")



class RSSFeedAnalyzer:
    def __init__(self, rss_url, 
                 model="gpt-4o-mini", 
                 llm_api_key=None, 
                 logger=None, 
                 embedding_model="text-embedding-3-small",
                 cluster_themes=False):
        self.rss_loader = RSSFeedLoader(rss_url)
        set_llm_cache(SQLiteCache(database_path=".langchain.db"))
        self.llm = ChatOpenAI(model=model, api_key=llm_api_key, temperature=0.2)
        self.embeddings = self._init_embeddings(embedding_model)
        self.cluster_themes = cluster_themes
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
        if hasattr(self.rss_loader.feed.feed, 'language'):
            return f"ISO 639={self.rss_loader.feed.feed.language}"
        else:
            return "same language as the input provided by the user"

    @property
    def language(self):
        if hasattr(self.rss_loader.feed.feed, 'language'):
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
            namespace=embeddings.model, # Create a cache-backed embedder using the base embedding and storage
            )
    def analyze_feed(self, limit=10000):
        episode_loader = self.rss_loader.lazy_load()
        self.logger.info(f"Analyzing {self.size} podcast episdoes")
        parser = PydanticOutputParser(pydantic_object=EpisodeInsights)

        prompt_template = ChatPromptTemplate([
            ("system", """
            You are an information extraction and generalisation specialist for a podcast called {podcast}.
            This is the description of the podcast to provide more context: {podcast_description}
            
            Your task:
            
            Given the description of an episode you have the following tasks:
                - give a very short and poignant summary of the episode in no more than 15 words. Cut to the chase! 
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
                        )
                    )
    


        analysis_results = []
        batch_content = []
        batch_metadata = []
        batch_size = 56
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
                batch_content.append({"title": episode.title, "episode_content": episode.description[:1500]})
                batch_metadata.append(episode.model_dump(exclude="description"))

        return AnalyzedEpisodes(episodes=analysis_results)
    
    def _create_text_catalog(self, analysed_episodes, keys=["inferred_themes"]):
        text_catalog = {}
        for e in analysed_episodes:
            text_catalog[e.metadata.index] = []
            for k in keys:
                text_catalog[e.metadata.index].extend(getattr(e.insights,k))
            text_catalog[e.metadata.index] = ",".join(text_catalog[e.metadata.index]
                                                           )
        return text_catalog
    
    def _create_episode_text_catalog(self, analysed_episodes):
        text_catalog = {}
        for ep in analysed_episodes:
            ep_themes = ",".join(ep.insights.inferred_themes)
            ep_tags = ",".join(ep.insights.tags)
            text_catalog[ep.metadata.index] = (f"Themes:{ep_themes}\n"
                                                    f"Summary:{ep.insights.summary}\n"
                                                    #f"Tags:{ep_tags}\n"
                                                    )

        return text_catalog
        
    
    @staticmethod
    def _predict_clusters(vectors, cluster_offset=0, **kwargs):
        
        aprox_cosine_vectors = normalize(vectors, norm='l2')
        # approximate sklearn implementation if no value specified
        min_samples = kwargs.pop("min_samples", kwargs["min_cluster_size"] - 1)
        c_model = HDBSCAN(**kwargs, prediction_data=True, min_samples=min_samples).fit(aprox_cosine_vectors)
        soft_clusters = pd.DataFrame(all_points_membership_vectors(c_model))
        clusters_top_3 = pd.DataFrame(soft_clusters.apply(lambda x: np.argsort(-x.values)[:3] + cluster_offset, axis=1).to_list())
        clusters_top_3_proba = pd.DataFrame(soft_clusters.apply(lambda x: np.sort(x.values)[::-1][:3], axis=1).to_list())
        clusters_top_3[clusters_top_3_proba < 0.05] = -1
        clusters_top_3.loc[:, 0] = c_model.labels_ + cluster_offset
        return clusters_top_3


    
    def _run_umap(self, vectors):
        reducer = umap.UMAP(n_neighbors=20, n_jobs=-1)
        embedding_2d = reducer.fit_transform(vectors)
        return StandardScaler().fit_transform(embedding_2d)
    

    def _embedd_cluster_reduce(self, text_catalog, cluster_umap=True):
        vectors =  np.array(self.embeddings.embed_documents(text_catalog))
        distances = pairwise_distances(vectors, metric="cosine")
        embedding_2d = self._run_umap(vectors)
        clusters_df = (
            pd.DataFrame({"text_catalog":text_catalog},
                         index=range(len(text_catalog)))
                         .assign(is_extra=False)
                         .assign(cluster_attempt=0)
                         .assign(umap_0=embedding_2d[:,0])
                         .assign(umap_1=embedding_2d[:,1])
                        
                    )
        initial_min_cluster_size = max(3, min(30,int(len(vectors)*0.02)))
        min_samples = max(2, int(initial_min_cluster_size * 0.75))
        max_cluster_size = min(75,int(len(vectors)*0.12))

        cluster_data = embedding_2d if cluster_umap else vectors

        
        cluster_top_3 = self._predict_clusters(vectors=cluster_data, 
                                                        min_cluster_size=initial_min_cluster_size,
                                                        max_cluster_size=max_cluster_size,
                                                        min_samples=min_samples
                                                        )
        clusters_df["cluster"] = cluster_top_3[0] 
            
        clusters_df["clusters_fuzzy"] = cluster_top_3.apply(lambda x: x.to_list(), axis=1)

        unmatched = clusters_df.query("cluster == -1")
        max_iter = 10
        i = 0
        n_unmatched_before = len(unmatched)
        while (len(unmatched) / len(clusters_df)) > 0.15:
            if i == max_iter:
                print(f"Max iter for clustering reached. {len(unmatched)} points left without cluster")
                break
            extra_clusters = self._predict_clusters(vectors=cluster_data[unmatched.index.to_numpy()],
                                                    cluster_offset=clusters_df.cluster.max() + 1,
                                                    max_cluster_size=max_cluster_size,
                                                    min_cluster_size=max(2,initial_min_cluster_size-5*(i+1))
                                                    )
            clusters_df.loc[unmatched.index, "cluster"] = extra_clusters[0].values
            clusters_df.loc[unmatched.index, "clusters_fuzzy"] = extra_clusters.apply(lambda x: x.to_list(), axis=1).values
            clusters_df.loc[unmatched.index, "is_extra"] = True
            clusters_df.loc[unmatched.index, "cluster_attempt"] = i+1

            unmatched = clusters_df.query("cluster == -1")
            if len(unmatched) >= n_unmatched_before:
                break
            i += 1


        return clusters_df, distances

    def _cluster_text_catalog(self, text_catalog):

        clusters_df, distances = self._embedd_cluster_reduce(list(text_catalog.values()))
        clusters_df["episode_index"] = list(text_catalog.keys())
        distance_map = {}
        for x,y in itertools.combinations(range(len(distances)),2):
            if distances[x,y] < COSINE_DISTANCE_THRESHOLD:
                distance_map[f"{clusters_df["episode_index"].iloc[x]},{clusters_df["episode_index"].iloc[y]}"] = distances[x,y]

        
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

            if group_size > batch_size*2:
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
        prompt_template = ChatPromptTemplate([
                ("system", """
                You are an expert in gerneralizing semantic content into short titles.
                Your task is to provide a poignant, evocative, and concise title in {language} for a group of related documents. 
                The title should capture the essence, themes and emotional tone of the documents while being engaging.  
                
                Consider the following guidelines:
                    - Generalsation: Capturing the bigger ideas and topics behind the selected documents.
                    - Relevance: The title should reflect the core ideas, topics, or narratives present in the documents.
                    - Targeted: Depending on the content the titles should be factual, funny, dramatic etc.
                    - Conciseness: Keep the title concise ideally no longer than 5 words.
                    - Cohesion: The title must be a cohesive and meaningful for humans
                                 
                Constraints (Hard rules): 
                 - The output list must be the same length as the input list
                 - The original language must be maintained. Do not change the language!
                 - The output must be a valid JSON in the format: {schema}
                
                Example Input (extract): [['This episode explores how sound design shapes our experiences in ways we often don’t notice...','This episode uncovers the surprising histories and cultural significance behind everyday colors.'], ]
                Example Output: ['The Hidden Designs That Shape Our World']
                """),
                ("user", "Input: {data}"),
            ]).partial(schema=parser.get_format_instructions(), language=self.language_prompt)

        #chain = prompt_template | self.llm | parser
        self.logger.info("Consolidating episodes")

        retry_parser = RetryOutputParser.from_llm(parser=parser, llm=self.llm, max_retries=2)

        chain = RunnableParallel(
                completion=prompt_template | self.llm,
                prompt_value=prompt_template
            ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(
                completion=x['completion'].content,
                prompt_value=x['prompt_value']
            ))
    

        
        batched_text_catalog, batched_clusters = self._create_clustered_batches(clusters_df, batch_size=1000)
        batched_prompts = []
    

        for batch in batched_text_catalog:
            prompt= {
                "data": json.dumps(batch)
             }
            batched_prompts.append(prompt)

        
        cluster_titles = chain.batch(batched_prompts)
        
        clusters_df["title"] = self._noise_title
        
        for cluster_title_batch, cluster_batch in zip(cluster_titles, batched_clusters):
            for cluster_title, cluster in zip(cluster_title_batch.items, cluster_batch):
                clusters_df.loc[clusters_df["cluster"]==cluster, "title"] = cluster_title

        clusters_unique_df = clusters_df.groupby("cluster")["title"].first()
        clusters_unique_df.loc[-1] = self._noise_title
        
        consolidated_episodes = []
        for e in analysed_episodes.episodes:
            insights = EpisodeInsights(**e.insights.model_dump())
            # list compatibility with self.cluster_themes mode
            clusters = [c for c in clusters_df.loc[e.metadata.index, "clusters_fuzzy"] if c in clusters_unique_df ] #if c != -1
            cluster_titles = [clusters_unique_df.loc[c] for c in clusters]
            cluster_title = clusters_df.loc[e.metadata.index, "title"]
            embeddings2d =  clusters_df.loc[[e.metadata.index], ["umap_0","umap_1"]].values.tolist()
            if len(cluster_titles) > 0:         
                clusters = ClusteredEpisodeInsights(
                    titles=cluster_titles,
                    ids=clusters,
                    embeddings=embeddings2d,
                    attempt=clusters_df.loc[e.metadata.index, "cluster_attempt"]
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
        return AnalyzedEpisodes(episodes=consolidated_episodes), clusters_df
    
    def _consolidate_clusters(self, analysed_episodes, clusters_df):
        parser = PydanticOutputParser(pydantic_object=ConsolidatedTitles)
        prompt_template = ChatPromptTemplate([
                ("system", """
                You are an expert in consolidating and generalizing text. Your task is to analyze a list of titles and identify titles that are semantic duplicates or synonyms.  
                The mapping should look like this:
                    - **Keys**: Consolidated titles.  
                    - **Values**: A list of redundant titles that are unified to the consolidated title. 

                ### **Guidelines for Consolidation:**  
                1. **Redundancy**: If a title does not introduce new information it should be merged with the related title.  
                2. **Uniqueness**: Titles that contain distinct meanings or cannot be grouped should remain unchanged.  
                3. **Partial Overlap**: Titles, which are similar in a sense that they are sub topics of a broader topic, should not be consolidated.
                5. **Simplicity**: The consolidated titles should be concise and must not overstate the orignal meaning
                6. **Precision**: Near semantic duplicates must be consolidated, but do not merge titles if you are unsure or if the consolidated group gets too broad

                ### **Constraints (Hard rules):**
                - The original language ({language}) must be maintained. 
                - The output must be a valid JSON in the format: {schema}
                - Unique titles should be ommitted from the mapping
        

                """),
                ("user", "Input: {data}"),
            ]).partial(schema=parser.get_format_instructions(), language=self.language_prompt)

        chain = prompt_template | self.llm | parser
        self.logger.info("Consolidating clusters")
        
        cluster_titles = clusters_df.drop_duplicates("cluster").query("cluster != -1").set_index("cluster")["title"].dropna()
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
            clusters = [c for c in clusters_df.loc[e.metadata.index, "clusters_fuzzy"]]
            cluster_titles = [clusters_unique_df.loc[c] for c in clusters]
            ep_clusters_df = pd.DataFrame({"clusters":clusters, "consolidated_titles":cluster_titles}).drop_duplicates()
            if len(ep_clusters_df) > 1:
                # remove noise cluster from episodes with one or more dedicated clusters
                ep_clusters_df = ep_clusters_df.query("clusters != -1")
            if len(ep_clusters_df) > 0:
                ep_copy.clusters.consolidated_titles = ep_clusters_df["consolidated_titles"].to_list()
                ep_copy.clusters.ids = ep_clusters_df["clusters"].to_list()

            consolidated_episodes.append(ep_copy)

        return AnalyzedEpisodes(episodes=consolidated_episodes), clusters_df
    
    def _get_major_categories(self, analysed_episodes, clusters_df):
        parser = PydanticOutputParser(pydantic_object=MajorCategories)
        prompt_template = ChatPromptTemplate([
                ("system", """
                You are an expert in consolidating and generalizing documents.
                You will be provided with a list of titles and your task is to group related titles together into higher level topics (major categories).
                
                Consider the following guidelines:
                    - Generalsation: Capturing the bigger picture of groups of titles is essential. It should provide a broad grouping
                    - Relevance: The identified parent topics should be unique and only related titles should be grouped together
                    - Completeness: Every title should be assigned to a category. You can create also one or more category for diverse leftover titles, as long as the category should reflect this.
                    - Conciseness: Keep the title concise ideally no longer than 5 words.
                    - Reduction: The number of major categories should be much smaller than the child titles (Ideally around 5-10 % of the child titles, but one major category should also not contain more than 7-10 titles)
                                 
                Constraints (Hard rules): 
                 - The original language ({language}) must be maintained.
                 - The output must be a valid JSON in the format: {schema}

                """),
                ("user", "Input: {data}"),
            ]).partial(schema=parser.get_format_instructions(), language=self.language_prompt)

        chain = prompt_template | self.llm | parser
        self.logger.info("Consolidating episodes")
        
        cluster_titles = clusters_df.groupby("cluster")["consolidated_title"].first()
        cluster_titles_doc = ",".join(cluster_titles.to_list())
        
        major_categories = chain.invoke(cluster_titles_doc)

        clusters_df["major_category"] = None  
        for m_category, titles in major_categories.mapping.items():
            lost_titles = set(titles).difference(set(clusters_df["consolidated_title"].unique()))
            if len(lost_titles) > 0:
                print(f"chatgpt mispelled or lost original title(s): {lost_titles }")
            clusters_df.loc[clusters_df["consolidated_title"].isin(titles), "major_category"] = m_category

        category_2_clusters = (clusters_df
                                 .dropna(subset=["major_category","consolidated_title"], how="any")
                                 .groupby("major_category")["consolidated_title"]
                                 .apply(lambda x: list(set(x)))
                                 .to_dict())
        
        consolidated_episodes = []
        for e in analysed_episodes.episodes:
            ep_copy = Episode(**e.model_dump())
            major_category = clusters_df.loc[e.metadata.index, "major_category"]
            
            if not pd.isnull(major_category):
                ep_copy.clusters.major_category = major_category
            else:
                cluster = clusters_df.loc[e.metadata.index, "cluster"]
                major_category = clusters_df.dropna(subset=["major_category"]).groupby("cluster")["major_category"].unique().get(cluster,[self._noise_title]) 
                ep_copy.clusters.major_category = major_category[0] 

            consolidated_episodes.append(ep_copy)
        return AnalyzedEpisodes(episodes=consolidated_episodes, category_2_clusters=category_2_clusters), clusters_df

    
    # def run(self, limit=1000):
    #     raise NotImplementedError("todos from run with streamlit")
    #     analysed_episodes = self.analyze_feed(limit)
            
    #     if self.cluster_themes:
    #         text_catalog = self._create_text_catalog(analysed_episodes, keys=["inferred_themes"])
    #         theme_clusters = self._cluster_text_catalog(text_catalog)
    #         consolidated_episodes = self._consolidate_themes(analysed_episodes, theme_clusters)
    #     else:
    #         text_catalog = {ep.metadata.index: ep.insights.summary for ep in analysed_episodes.episodes}
    #         summary_clusters = self._cluster_text_catalog(text_catalog)
    #         consolidated_episodes = self._consolidate_summaries(analysed_episodes, summary_clusters)
            
    #     return consolidated_episodes
    
    def run_with_streamlit_progress(self, progress_bar, limit=1000):
        """ 
          @progress_bar: st.progress widget
        """
        progress_bar.progress(20, f"Analyzing {self.title} rss feed ({min(self.size,limit)} episodes) with {self.llm.model_name}...")
        analysed_episodes = self.analyze_feed(limit)
            
        if self.cluster_themes:
            keys = ["inferred_themes"]
            progress_bar.progress(50, f"Clustering {','.join(keys)}...")
            text_catalog = self._create_text_catalog(analysed_episodes.episodes, keys=keys)
            theme_clusters = self._cluster_text_catalog(text_catalog)
            progress_bar.progress(70, f"Consolidating clusters ...")
            consolidated_episodes = self._consolidate_themes(analysed_episodes, theme_clusters)
        else:
            progress_bar.progress(50, f"Clustering episode summaries ...")
            text_catalog = self._create_episode_text_catalog(analysed_episodes.episodes)
            summary_clusters, distance_map = self._cluster_text_catalog(text_catalog)
            progress_bar.progress(70, f"Creating cluster titles with {self.llm.model_name}...")
            clustered_episodes, titled_clusters = self._generate_cluster_titles(analysed_episodes, summary_clusters)
            progress_bar.progress(80, f"Consolidating clusters with {self.llm.model_name}...")
            consolidated_episodes, consolidated_clusters = self._consolidate_clusters(clustered_episodes, titled_clusters)
            progress_bar.progress(90, f"Generating major categories with {self.llm.model_name}...")
            finalized_episodes, final_clusters = self._get_major_categories(consolidated_episodes, consolidated_clusters)
            finalized_episodes.distance_map = distance_map
            finalized_episodes.extra["consolidation_map"] = final_clusters.groupby("consolidated_title")["title"].apply(lambda x: list(set(x))).to_dict()
            #cluster_id_lookup = final_clusters.groupby("cluster")["consolidated_title"].first()
        self.logger.info("analysis completed")
        progress_bar.progress(95, "Preparing plot ...")
        
        return finalized_episodes

