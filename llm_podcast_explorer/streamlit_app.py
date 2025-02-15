import streamlit as st
import json
from rss_feed_analyzer import RSSFeedAnalyzer, AnalyzedEpisodes
import os
from network_viz import build_networkx_graph, create_figure


llm_api_key = os.environ.get('OPENAI_API_KEY')
if llm_api_key is None:
    raise ValueError("Please provide an OpenAI API Key. Set it as Environment Variable 'OPENAI_API_KEY'")

@st.cache_data
def get_analysis(url):
    #analyzer = RSSFeedAnalyzer(url, llm_api_key=llm_api_key, warmup=False)
    #analysed_episodes = analyzer.run(limit=100)
    analysed_episodes = AnalyzedEpisodes.load("./dev_data/clustered_episodes.json").model_dump()

    return analysed_episodes["episodes"]

def main():
    st.title("RSS Podcast Analyzer")
    
    rss_url = st.text_input("Enter Podcast RSS Feed URL:")
    
    if rss_url:
        analysed_episodes = get_analysis(rss_url)   

        #st.title("Subgraph Layout by Time Buckets, with Cluster Highlighting")

        G, global_positions, clusters = build_networkx_graph(analysed_episodes)
        
        selected_cluster = st.sidebar.selectbox(
            "Select a cluster to highlight:",
            options=["<None>"] + list(clusters),
            index=0
        )

        fig = create_figure(G, global_positions, selected_cluster)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
