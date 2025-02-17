import streamlit as st
import json
from rss_feed_analyzer import RSSFeedAnalyzer, AnalyzedEpisodes
import os
from network_viz import build_networkx_graph, create_figure, update_figure
import copy
import time
from typing import Dict, Union, List
from pathlib import Path

llm_api_key = os.environ.get('OPENAI_API_KEY')
if llm_api_key is None:
    raise ValueError("Please provide an OpenAI API Key. Set it as Environment Variable 'OPENAI_API_KEY'")

@st.cache_data
def load_data(url, checkpoint):
    if url:
        analyzer = RSSFeedAnalyzer(url, llm_api_key=llm_api_key)
        checkpoint_path = f"./dev_data/{analyzer.title}.json"
        if checkpoint and Path(checkpoint_path).exists():
            analysed_episodes = AnalyzedEpisodes.load(checkpoint_path)
        else:
            analysed_episodes = analyzer.run(limit=500)
            analysed_episodes.save_episodes(checkpoint_path)
        return analysed_episodes.model_dump()


@st.cache_data
def run_analysis(analysed_episodes, timeline):

    G, global_positions, clusters = build_networkx_graph(analysed_episodes["episodes"], timeline)
    fig, cluster_edge_indices, cluster_node_indices, ranges = create_figure(G, global_positions, clusters)
    cluster_data = {"clusters": clusters, 
                    "cluster_edge_indices": cluster_edge_indices,
                    "cluster_node_indices": cluster_node_indices,
                    "node_index_range": ranges["nodes"],
                    "edge_index_range": ranges["edges"],
                    }
    
    return fig, cluster_data



def format_dict_to_markdown(data: Dict[str, Union[str, List[str]]]) -> str:
    """
    Formats a dictionary into markdown text with keys as headers and lists as bullet points.
    
    Args:
        data: Dictionary with string keys and values that are either strings or lists of strings
        
    Returns:
        Markdown formatted string
    """
    markdown = []
    for key, value in data.items():
        # Add header for the key
        markdown.append(f"**{key}**  \n")  # Two spaces at end for line break
        
        # Handle list values
        if isinstance(value, list):
            markdown.extend([f"- {item}" for item in value])
        # Handle string values
        else:
            markdown.append(f"{value}")
            
        markdown.append("\n")  # Add spacing between sections
    
    return "\n".join(markdown)

def main():
    st.title("RSS Podcast Analyzer")
    if 'timeline_mode' not in st.session_state:
        st.session_state.timeline_mode = True
    if "timeline_toggle_disabled" not in st.session_state:
        st.session_state.timeline_toggle_disabled = True
    
    rss_url = st.text_input("Enter Podcast RSS Feed URL:", value=None)
    
    with st.sidebar:
        timeline = st.toggle("Timline mode",
                            value=st.session_state.timeline_mode,
                            disabled=st.session_state.timeline_toggle_disabled)
    placeholder = st.empty()
    if rss_url:
        checkpoint = True
        with st.sidebar:
            reset = st.button("Reset analysis")
            if reset:
                checkpoint = False
        placeholder.progress(0, "Loading data ...")
        analysed_episodes = load_data(rss_url, checkpoint)

        st.session_state.timeline_toggle_disabled = False
        placeholder.progress(50, "Preparing plot ...")
        base_fig, cluster_data = run_analysis(analysed_episodes, timeline)           
        selected_cluster = st.sidebar.selectbox(
            "Select a cluster to highlight:",
            options=["<None>"] + list(cluster_data["clusters"]),
            index=0
        )
        start_t = time.time()

        
        fig = copy.deepcopy(base_fig)
        
        placeholder.progress(90, "Rendering ...")
        with placeholder.container():
            updated_fig = update_figure(fig, selected_cluster, cluster_data, timeline)
            selection = st.plotly_chart(updated_fig,
                                        use_container_width=True,
                                        key="network",
                                        selection_mode=('points',),
                                        on_select="rerun")
        
        with st.sidebar:
            st.write(f"Request Took {time.time() - start_t:.2f} seconds")
            if len(selection["selection"]["points"]) > 0:
                
                display_data = copy.deepcopy(selection["selection"]["points"][0]["customdata"][-1])
                url = display_data.pop("link")
                st.page_link(url, label="Go to episode", icon="🎧")
                st.write(format_dict_to_markdown(display_data))
        

  
        

if __name__ == "__main__":
    main()
