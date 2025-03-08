import copy
import os
from pathlib import Path
from typing import Dict, List, Union

import streamlit as st
from network_viz import build_networkx_graph, create_figure, update_figure
from rss_feed_analyzer import AnalyzedEpisodes, RSSFeedAnalyzer
from rss_feed_loader import InvalidRSSException
from streamlit.runtime.scriptrunner import StopException

CHECKPOINT_PATH = Path("./static")
ALL_KEY = "All"
EPISODE_LIMIT = 700
DEFAULT_MODE = "active"


@st.cache_data(show_spinner=False)
def load_static_data(checkpoint_path):
    analysed_episodes = AnalyzedEpisodes.load(checkpoint_path)
    return analysed_episodes.model_dump()


@st.cache_data(show_spinner=False)
def load_data(url, checkpoint):
    progress_bar = st.progress(0, "Loading data .. ")
    if url:
        llm_api_key = os.environ.get("OPENAI_API_KEY")

        analyzer = RSSFeedAnalyzer(url, llm_api_key=llm_api_key)

        checkpoint_path = CHECKPOINT_PATH / f"{analyzer.title}.json"
        if checkpoint and checkpoint_path.exists():
            analysed_episodes = AnalyzedEpisodes.load(checkpoint_path)
        else:
            if llm_api_key is None:
                raise ValueError("Please provide an OpenAI API Key. Set it as Environment Variable 'OPENAI_API_KEY'")

            analysed_episodes = analyzer.run_with_streamlit_progress(progress_bar, limit=EPISODE_LIMIT)
            analysed_episodes.save_episodes(checkpoint_path)
        progress_bar.empty()
        return analysed_episodes.model_dump()
    else:
        progress_bar.progress(90, "No data found ..")
        progress_bar.empty()


@st.cache_data(show_spinner=False)
def create_network_graph(analysed_episodes, timeline):
    G, global_positions, clusters = build_networkx_graph(analysed_episodes, timeline)
    fig, cluster_edge_indices, cluster_node_indices, ranges = create_figure(G, global_positions, clusters)
    cluster_data = {
        "clusters": clusters,
        "cluster_edge_indices": cluster_edge_indices,
        "cluster_node_indices": cluster_node_indices,
        "node_index_range": ranges["nodes"],
        "edge_index_range": ranges["edges"],
    }

    return fig, cluster_data


def format_dict_to_markdown(display_data: Dict[str, Union[str, List[str]]]) -> str:
    """
    Formats a dictionary into markdown text with keys as headers and lists as bullet points.

    Args:
        data: Dictionary with string keys and values that are either strings or lists of strings

    Returns:
        Markdown formatted string
    """
    episode_title = display_data.pop("title")
    summary =  display_data.pop("summary")
    url = display_data.pop("link")
    
    
    st.write(f"### {episode_title} ")
    st.write(f"{summary}")

    if url != "unknown":
        st.page_link(url, label="Listen to episode", icon="🎧")
    else:
        st.write("Could not parse link to episode")

    markdown = []
    for key, value in display_data.items():
        # Add header for the key
        if len(value) > 0:
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
        if st.session_state.selection_state == selection["selection"]["points"]:
            return
        elif len(selection["selection"]["points"]) > 0:
            st.session_state.click_selection = True
            st.session_state.click_reset = False
            st.session_state.selection_state = selection["selection"]["points"]
            category = list(selection["selection"]["points"][0]["customdata"][4].values())
            if len(category) > 0:
                st.session_state.selected_category = category[0]
                filtered_clusters = {}
                for c in st.session_state.major_categories[st.session_state.selected_category]:
                    filtered_clusters[c] = (
                        True
                        if c in list(selection["selection"]["points"][0]["customdata"][3].values())
                        else "legendonly"
                    )
                st.session_state.filtered_clusters = filtered_clusters
            else:
                st.session_state.selected_category = ALL_KEY
                st.session_state.filtered_clusters = list(selection["selection"]["points"][0]["customdata"][3].values())
            st.session_state.selection_data = selection["selection"]["points"][0]["customdata"][-1]

        else:
            st.session_state.click_selection = False
            st.session_state.click_reset = True

def _init_sesion_state():
    if "timeline_mode" not in st.session_state:
        st.session_state.timeline_mode = False
    if "timeline_toggle_disabled" not in st.session_state:
        st.session_state.timeline_toggle_disabled = False
    # Initialize session state variables
    if "selected_podcast" not in st.session_state:
        st.session_state.selected_podcast = None
    if "checkpoint" not in st.session_state:
        st.session_state.checkpoint = True
    if "filtered_clusters" not in st.session_state:
        st.session_state.filtered_clusters = {}
    if "selected_category" not in st.session_state:
        st.session_state.selected_category = ALL_KEY
    if "selected_cluster" not in st.session_state:
        st.session_state.selected_cluster = None
    if "selection_state" not in st.session_state:
        st.session_state.selection_state = None

    if "selection_data" not in st.session_state:
        st.session_state.selection_data = None
    if "click_selection" not in st.session_state:
        st.session_state.click_selection = False
    if "click_reset" not in st.session_state:
        st.session_state.click_reset = False
    if "zoom_state" not in st.session_state:
        st.session_state.zoom_state = None
    if "major_categories" not in st.session_state:
        st.session_state.major_categories = None


def main(analyis_mode):
    title = "Podcasts Explored"
    st.set_page_config(page_title=title, layout="wide", initial_sidebar_state="expanded")
    st.title(f"{title}")

    _init_sesion_state()


    if analyis_mode == "active":
        reset_disabled = False
        rss_url = st.text_input("Enter Apple Podcast URL or RSS Feed URL:", value=st.session_state.selected_podcast)
        # Update session state when RSS URL is provided
        if rss_url not in [None, "", " "]:
            st.session_state.selected_podcast = rss_url
            try:
                analysed_episodes = load_data(st.session_state.selected_podcast, st.session_state.checkpoint)
                # enable cache and checkpoint until reset button is clicked again

                st.session_state.checkpoint = True
            except InvalidRSSException as e:
                st.error(e)
                st.session_state.selected_podcast = None
    else:
        podcasts = {p.stem: str(p) for p in CHECKPOINT_PATH.glob("*.json")}
        reset_disabled = True
        selected_podcast = st.selectbox("Choose a podcast:", options=sorted(podcasts.keys()), index=None)
        st.session_state.selected_podcast = selected_podcast

        # st.session_state.rss_url
        if st.session_state.selected_podcast is not None:
            analysed_episodes = load_static_data(podcasts[st.session_state.selected_podcast])

    with st.sidebar:
        reset = st.button("Rerun analysis", disabled=reset_disabled)
        if reset and st.session_state.selected_podcast is not None:
            st.session_state.checkpoint = False
            load_data.clear()
    
    placeholder = st.empty()

    if st.session_state.selected_podcast is None:
        with placeholder.container():
            st.markdown("")
            st.markdown("")
            st.markdown("")
            st.markdown("")
            st.markdown("")
            st.markdown("")
            st.markdown(
                f""" 
            #### Explore the Big Picture Behind Every Podcast

            Podcasts are full of ideas, connections, and themes—but they’re not always easy to navigate. {title} helps you break down, explore, and visualize the hidden patterns inside your favorite shows.

            ✨ Discover the core themes – See what a podcast is really about.

            🔗 Follow the connections – Trace how episodes link together.

            🚀 Find the best episodes – Get straight to the topics that matter to you.


            🎧 Start Exploring Now!

            """
            )
    else:
        with st.sidebar:
            timeline = st.toggle(
                "Timline mode", value=st.session_state.timeline_mode, disabled=st.session_state.timeline_toggle_disabled
            )

        st.session_state.timeline_toggle_disabled = False

        base_fig, cluster_data = create_network_graph(analysed_episodes, timeline)

        try:
            major_categories = analysed_episodes["category_2_clusters"]
            st.session_state.major_categories = major_categories
            category_options = [ALL_KEY] + list(sorted(major_categories, key=lambda k: len(major_categories[k]), reverse=True)) #list(major_categories.keys())

            selected_category = st.sidebar.selectbox(
                "Select a category:", options=category_options, key="category_selection", index=0
            )

            if st.session_state.click_reset:
                st.session_state.selected_category = ALL_KEY
                st.session_state.click_reset = False
            elif st.session_state.click_selection:
                # st.session_state.selected_cluster already set on_select call
                # st.session_state.click_selection = False
                pass
            else:
                st.session_state.selected_category = selected_category

            if st.session_state.click_selection:
                # filtered_clusters already set on_select call
                pass
            elif st.session_state.selected_category == ALL_KEY:
                st.session_state.filtered_clusters = {}
            else:
                st.session_state.filtered_clusters = {
                    c: True for c in major_categories[st.session_state.selected_category]
                }

        except StopException:
            st.session_state.click_selection = False
            st.session_state.selected_category = ALL_KEY

        fig = copy.deepcopy(base_fig)

        with placeholder.container():
            updated_fig, zoom_state = update_figure(
                fig,
                st.session_state.selected_category,
                st.session_state.filtered_clusters,
                cluster_data,
                timeline,
                st.session_state.click_selection,
                previous_zoom=st.session_state.zoom_state,
                selection_state=st.session_state.selection_state 
            )

            st.session_state.zoom_state = zoom_state

            st.plotly_chart(
                updated_fig,
                use_container_width=True,
                key="plotly_state",
                selection_mode=("points",),
                on_select=on_select,
                config=dict(scrollZoom=False, doubleClick="reset+autosize", doubleClickDelay=1000),
            )

        if analyis_mode == "static":
            st.info(
                "This app is running with static data. Go to [Github](https://github.com/danielressi/llm-podcast-explorer) page for more options"
            )

        with st.sidebar:
            info_placeholder = st.empty()
            st.session_state.click_reset = False
            with info_placeholder.container():
                if st.session_state.click_selection and st.session_state.selection_data:
                    display_data = st.session_state.selection_data
                    st.write(format_dict_to_markdown(display_data))
                    st.session_state.click_selection = False

                elif st.session_state.selected_cluster in [None, ALL_KEY]:
                    st.write(
                    """
                    **Tips:**

                    - Select a category to start unravelling the themes and topics of the podcast
                    - Click on a point to show episode details.
                    """
                    )
                    
                else:
                                        """
                    **Tips:**

                    - Enable/disable clusters by clicking the names in the legend.
                    -  Select a cluster by double clicking the name on the legend.
                    - Click on a point to show episode details.
                    """

        st.caption("✨ Leveraging AI to explore content instead of generating it ✨")


if __name__ == "__main__":
    analyis_mode = os.environ.get("ANALYSIS_MODE", DEFAULT_MODE)
    if analyis_mode not in ["static", "active"]:
        raise ValueError(f"Environment variable ANALYSIS_MODE has to be 'static' or 'active', but got {analyis_mode} ")
    main(analyis_mode)
