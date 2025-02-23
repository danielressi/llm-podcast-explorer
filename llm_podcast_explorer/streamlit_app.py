import streamlit as st
import json
from rss_feed_analyzer import RSSFeedAnalyzer, AnalyzedEpisodes
import os
from network_viz import build_networkx_graph, create_figure, update_figure
import copy
import time
from typing import Dict, Union, List
from pathlib import Path
from streamlit.runtime.scriptrunner import StopException


CHECKPOINT_PATH = Path("./static")

@st.cache_data(show_spinner = False)
def load_static_data(checkpoint_path):
    analysed_episodes = AnalyzedEpisodes.load(checkpoint_path)
    return analysed_episodes.model_dump()

@st.cache_data(show_spinner = False)
def load_data(url, checkpoint):
    progress_bar = st.progress(0, "Loading data .. ")
    if url:
        llm_api_key = os.environ.get('OPENAI_API_KEY')
        analyzer = RSSFeedAnalyzer(url,llm_api_key=llm_api_key)
        checkpoint_path = CHECKPOINT_PATH / f"{analyzer.title}.json"
        if checkpoint and checkpoint_path.exists():
            analysed_episodes = AnalyzedEpisodes.load(checkpoint_path)
        else:
            
            if llm_api_key is None:
                raise ValueError("Please provide an OpenAI API Key. Set it as Environment Variable 'OPENAI_API_KEY'")

            analysed_episodes = analyzer.run_with_streamlit_progress(progress_bar,limit=500)
            analysed_episodes.save_episodes(checkpoint_path)
        progress_bar.empty()
        return analysed_episodes.model_dump()
    else:
        progress_bar.progress(90, "No data found ..")
        progress_bar.empty()
        


@st.cache_data(show_spinner = False)
def create_network_graph(analysed_episodes, timeline):

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
        markdown.append(f"#### {key}  \n")  # Two spaces at end for line break
        
        # Handle list values
        if isinstance(value, list):
            markdown.extend([f"- {item}" for item in value])
        # Handle string values
        else:
            markdown.append(f"{value}")
            
        markdown.append("\n")  # Add spacing between sections
    
    return "\n".join(markdown)

def on_select():

    if "plotly_state" in st.session_state:
        selection = st.session_state.plotly_state
        if len(selection["selection"]["points"]) > 0:
            st.session_state.click_selection = True
        
            clusters = list(selection["selection"]["points"][0]["customdata"][3].values())
            if len(clusters) > 0:
                st.session_state.selected_cluster = list(selection["selection"]["points"][0]["customdata"][3].values())[0]
            else:
                st.session_state.selected_cluster =  "All"
            st.session_state.selection_data = selection["selection"]["points"][0]["customdata"][-1]
        else:
            st.session_state.selected_cluster =  "All"

    

def main(analyis_mode):
    title = "RSS Podcast Explorer"
    st.set_page_config(page_title=title,layout="wide", initial_sidebar_state="expanded")
    st.title(title)

    st.caption("✨ Using AI to explore content visually instead of generating it ✨")
    if 'timeline_mode' not in st.session_state:
        st.session_state.timeline_mode = False
    if "timeline_toggle_disabled" not in st.session_state:
        st.session_state.timeline_toggle_disabled = False
    # Initialize session state variables
    if "selected_podcast" not in st.session_state:
        st.session_state.selected_podcast = None
    if "rss_url" not in st.session_state:
        st.session_state.rss_url = None
    if "checkpoint" not in st.session_state:
        st.session_state.checkpoint = True
    if "selected_cluster" not in st.session_state:
        st.session_state.selected_cluster = "All"
    if "selection_data" not in st.session_state:
        st.session_state.selection_data = None
    if "click_selection" not in st.session_state:
        st.session_state.click_selection = False
    
    if analyis_mode == "active":
        rss_url = st.text_input("Enter Podcast RSS Feed URL:", value=st.session_state.rss_url)
            # Update session state when RSS URL is provided
        if rss_url:
            st.session_state.rss_url = rss_url
            st.session_state.checkpoint = True
            analysed_episodes = load_data(rss_url, st.session_state.checkpoint)
    else:
        podcasts = {p.stem: str(p) for p in CHECKPOINT_PATH.glob("*.json")}
        
        selected_podcast = st.selectbox("Choose a podcast:", options=podcasts.keys(), index=0)
        #st.session_state.rss_url 
        st.session_state.selected_podcast = selected_podcast
        analysed_episodes = load_static_data(podcasts[st.session_state.selected_podcast])


    
    placeholder = st.empty()

    with st.sidebar:
        reset = st.button("Rerun analysis")
        if reset:
            st.session_state.checkpoint = False
            #st.rerun()  # Force a rerun to reset the app

    #if st.session_state.rss_url is None:
    #    with placeholder.container():
    #        st.write("Please enter an RSS URL to begin analysis.")
    if True:
        with st.sidebar:

            timeline = st.toggle("Timline mode",
                                value=st.session_state.timeline_mode,
                                disabled=st.session_state.timeline_toggle_disabled)
        
        

        st.session_state.timeline_toggle_disabled = False
        
        base_fig, cluster_data = create_network_graph(analysed_episodes, timeline)

           
        try:
            selected_cluster = st.sidebar.selectbox(
                "Select a cluster to highlight:",
                options=["All"] + list(cluster_data["clusters"]),
                index=0
            )
        except StopException:
            selected_cluster = "All"

        start_t = time.time()

        if st.session_state.click_selection:
            selection_value = st.session_state.selected_cluster
            #reset
            st.session_state.click_selection = False
        else:
            selection_value = selected_cluster
        
        fig = copy.deepcopy(base_fig)
        
        with placeholder.container():
            updated_fig = update_figure(fig, selection_value, cluster_data, timeline)
            st.plotly_chart(updated_fig,
                            use_container_width=True,
                            key="plotly_state",
                            selection_mode=('points',),
                            on_select=on_select)
            
        if analyis_mode == "static":
            st.info("This app is running with static data to keep the costs at a minimim")
            st.page_link("https://github.com/danielressi/llm-podcast-explorer",
                                  label="Go to the github page for more infos")
        
        with st.sidebar:
            st.write(f"Visualized {len(analysed_episodes["episodes"])} episodes in {time.time() - start_t:.2f} seconds, ")
            #st.write(f"Visualized {} episodes.")
            if "plotly_state" in st.session_state:                
                if st.session_state.selection_data:
                    display_data = st.session_state.selection_data
                    st.write("### Episode Info ")
                    if "link" in display_data:
                        url = display_data.pop("link")
                        st.page_link(url, label="Go to episode", icon="🎧")
                    st.write(format_dict_to_markdown(display_data))
        

  
        

if __name__ == "__main__":
    analyis_mode = os.environ.get("ANALYSIS_MODE","static")
    if analyis_mode not in  ["static","active"]:
        raise ValueError(f"Environment variable ANALYSIS_MODE has to be 'static' or 'active', but got {analyis_mode} ")
    main(analyis_mode)

    

