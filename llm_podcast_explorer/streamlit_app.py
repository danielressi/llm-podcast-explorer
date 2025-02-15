import streamlit as st
import json
from rss_feed_analyzer import RSSFeedAnalyzer, AnalyzedEpisodes
import os
from network_viz import build_networkx_graph, create_figure, update_figure
import copy

llm_api_key = os.environ.get('OPENAI_API_KEY')
if llm_api_key is None:
    raise ValueError("Please provide an OpenAI API Key. Set it as Environment Variable 'OPENAI_API_KEY'")

@st.cache_data
def run_analysis(url):
    #analyzer = RSSFeedAnalyzer(url, llm_api_key=llm_api_key, warmup=False)
    #analysed_episodes = analyzer.run(limit=100)
    analysed_episodes = AnalyzedEpisodes.load("./dev_data/clustered_episodes.json").model_dump()
    G, global_positions, clusters = build_networkx_graph(analysed_episodes["episodes"])
    fig, cluster_edge_indices, cluster_node_indices, ranges = create_figure(G, global_positions, clusters)
    cluster_data = {"clusters": clusters, 
                    "cluster_edge_indices": cluster_edge_indices,
                    "cluster_node_indices": cluster_node_indices,
                    "node_index_range": ranges["nodes"],
                    "edge_index_range": ranges["edges"],
                    }
    
    return fig, cluster_data


def main():
    st.title("RSS Podcast Analyzer")
    
    rss_url = st.text_input("Enter Podcast RSS Feed URL:")
    
    if rss_url:
        base_fig, cluster_data = run_analysis(rss_url)           
        
        selected_cluster = st.sidebar.selectbox(
            "Select a cluster to highlight:",
            options=["<None>"] + list(cluster_data["clusters"]),
            index=0
        )

        
        fig = copy.deepcopy(base_fig)
        updated_fig = update_figure(fig, selected_cluster, cluster_data)
        selection = st.plotly_chart(updated_fig , use_container_width=True, key="network", on_select="rerun")
        if len(selection["selection"]["points"]) > 0:
            selection["selection"]["points"][0]["customdata"][-1]

        #plotly_chart.plotly_chart(fig, use_container_width=True)
        #fig = create_figure2(G, global_positions, selected_cluster)
        

if __name__ == "__main__":
    main()
