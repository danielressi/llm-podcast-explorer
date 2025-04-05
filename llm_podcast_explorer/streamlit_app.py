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
EPISODE_LIMIT = 1000
DEFAULT_MODE = "active"


PODCAST_QUERY_LOOKUP = {"GAG": "Geschichten aus der Geschichte",
                        "99pi": "99% Invisible",
                        "Verbrechen": "Verbrechen",
                        "Atlas-Obscura": "The Atlas Obscura Podcast"
                        }


@st.cache_data(show_spinner=False)
def load_static_data(checkpoint_path):
    analysed_episodes = AnalyzedEpisodes.load(checkpoint_path)
    return analysed_episodes.model_dump()


@st.cache_data(show_spinner=False)
def load_data(url, checkpoint):
    progress_bar = st.progress(0, "Loading data .. ")
    if url:
        llm_api_key = os.environ.get("OPENAI_API_KEY")
        extraction_model = os.environ.get("EXTRACTION_MODEL", "gpt-4o-mini")
        analyis_model = os.environ.get("ANALYSIS_MODEL", "gpt-4o")
        analyzer = RSSFeedAnalyzer(rss_url=url, 
                                   llm_api_key=llm_api_key,
                                   extraction_model=extraction_model,
                                   analysis_model=analyis_model)

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
def create_network_graph(analysed_episodes, timeline, hover_enabled):
    G, global_positions, clusters, episode_lookup = build_networkx_graph(analysed_episodes, timeline)
    fig, cluster_edge_indices, cluster_node_indices = create_figure(G, global_positions, clusters, hover_enabled=hover_enabled)
    cluster_data = {
        "clusters": clusters,
        "cluster_edge_indices": cluster_edge_indices,
        "cluster_node_indices": cluster_node_indices,
    }
    return fig, cluster_data, episode_lookup


def pretty_key(v):
    return str(v).replace("_", " ").capitalize()

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
    description = display_data.pop("description", "Coming Soon!")
    
    
    st.write(f"### {episode_title} ")
    st.write(f"{summary}")

    if url != "unknown":
        st.page_link(url, label="Listen to episode", icon="🎧")
    else:
        st.write("Could not parse link to episode")

    markdown = []
    for key, value in display_data.items():
        # Add header for the key
        
        markdown.append(f"#### {pretty_key(key)}:  \n") 

        # Handle list values
        if isinstance(value, list):
            markdown.extend([f"- {item}" for item in value])
        # Handle string values
        else:
            markdown.append(f"{value}")

        markdown.append("\n")  # Add spacing between sections

    st.write("\n".join(markdown))
        
    with st.popover("Full Description"):
        #st.markdown(f'<div style="max-height:400px; overflow:auto;">{description}</div>', unsafe_allow_html=True)
        st.markdown(description)

def on_select():
    if "plotly_state" in st.session_state:
        st.session_state.click_selection = True
        selection = st.session_state.plotly_state
        if len(selection["selection"]["points"]) > 0:
            st.session_state.searched_episode = selection["selection"]["points"][0]["customdata"][0]
        else:
            st.session_state.searched_episode = None
            st.session_state.click_selection = False
            #st.session_state.click_reset = True


def _init_sesion_state():
     # Initialize session state variables
    if "timeline_mode" not in st.session_state:
        st.session_state.timeline_mode = False
    if "podcast_query" not in st.session_state:
        st.session_state.podcast_query = False
    if "selected_podcast" not in st.session_state:
        st.session_state.selected_podcast = None
    if "checkpoint" not in st.session_state:
        st.session_state.checkpoint = True
    if "filtered_clusters" not in st.session_state:
        st.session_state.filtered_clusters = {}
    if "selected_category" not in st.session_state:
        st.session_state.selected_category = ALL_KEY
    if "selection_state" not in st.session_state:
        st.session_state.selection_state = None
    if "click_selection" not in st.session_state:
        st.session_state.click_selection = False
    if "click_reset" not in st.session_state:
        st.session_state.click_reset = False
    if "zoom_state" not in st.session_state:
        st.session_state.zoom_state = None
    if "major_categories" not in st.session_state:
        st.session_state.major_categories = None
    if "searched_episode" not in st.session_state:
        st.session_state.searched_episode = None



def set_title_on_top(title):
    st.markdown("""
        <style>
               /* Remove blank space at top and bottom */ 
               .block-container {
                   padding-top: 2rem;
                   padding-bottom: 0rem;
                }

        </style>
        """, unsafe_allow_html=True)
    
    st.markdown(
        f"""
        <h1 style="text-align: left; margin-top: 0;">
            {title} <span style="font-size: 14px;font-weight: normal">by</span> <span style="font-size: 14px;">FeedPAM</span>
        </h1>
        """,
        unsafe_allow_html=True
    )

def reset_search():
    st.session_state.searched_episode = None
    st.session_state.selection_state = None
    st.session_state.episode_selection = None
    st.session_state.checkpoint = True

def reset_category_selection():
    st.session_state.category_selection = ALL_KEY
    st.session_state.selection_state = None
    st.session_state.zoom_state = None
    st.session_state.checkpoint = True


def click_reset():
    """ User clicked on reset view or double clicked graph"""
    st.session_state.selected_category = ALL_KEY
    st.session_state.click_reset = False
    st.session_state.searched_episode = None

def main(analyis_mode):
    title = "Podcasts | Explored"
    st.set_page_config(page_title=title, layout="centered", initial_sidebar_state="expanded")
    set_title_on_top(title)

    _init_sesion_state()
    podcasts = {p.stem: str(p) for p in CHECKPOINT_PATH.glob("*.json")}
    podcast_query = st.query_params.get("podcast", None)
    if podcast_query is not None:
        if podcast_query in PODCAST_QUERY_LOOKUP:
            st.session_state.selected_podcast = PODCAST_QUERY_LOOKUP[podcast_query]
            st.session_state.podcast_query = True
        elif podcast_query.replace("-", " ") in podcasts:
            st.session_state.podcast_query = True
            st.session_state.selected_podcast = podcast_query.replace("-", " ")

        


    if analyis_mode == "active" and not st.session_state.podcast_query:
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
        
        reset_disabled = True
        podcast_options = sorted(podcasts.keys())
        if st.session_state.podcast_query:
            index = podcast_options.index(st.session_state.selected_podcast)
        else:
            index = None
        selected_podcast = st.selectbox("Choose a podcast:", options=podcast_options, index=index)
        
        
        st.session_state.selected_podcast = selected_podcast

        # st.session_state.rss_url
        if st.session_state.selected_podcast is not None:
            analysed_episodes = load_static_data(podcasts[st.session_state.selected_podcast])

    with st.sidebar:
        col1, col2 = st.columns([1,1])
        with col1:
            reset = st.button("Rerun analysis", disabled=reset_disabled)
            if reset and st.session_state.selected_podcast is not None:
                st.session_state.checkpoint = False
                load_data.clear()
                st.rerun()
        with col2:
            reset_view = st.button("Reset view", disabled=False)
            if not st.session_state.click_reset:
                st.session_state.click_reset = reset_view

    
    select_box_placeholder = st.empty()
    placeholder = st.empty()

    if st.session_state.selected_podcast is None:
        with placeholder.container():
            st.markdown("")
            st.markdown("")
            st.markdown("")
            st.markdown(
                f""" 
            #### Explore the Big Picture Behind Every Podcast

            Podcasts are full of ideas, connections, and themes — but they’re not always easy to navigate. **FeedPAM** helps you break down, explore, and visualize the hidden patterns inside your favorite shows.

            ✨ Discover the core themes – See what a podcast is really about.

            🔗 Follow the connections – Trace how episodes link together.

            🚀 Find the best episodes – Get straight to the topics that matter to you.


            🎧 Start Exploring Now!

            """
            )
    else:
        with st.sidebar:
            timeline = st.toggle(
                "Timline mode", value=st.session_state.timeline_mode, disabled=False
            )

        base_fig, cluster_data, episode_lookup = create_network_graph(analysed_episodes, timeline, hover_enabled=os.getenv("HOVER_ENABLED", "true").lower() in ('true', '1', 't'))

        try:
            
            major_categories = analysed_episodes["category_2_clusters"]
            st.session_state.major_categories = major_categories
            category_options = [ALL_KEY] + list(sorted(major_categories, key=lambda k: len(major_categories[k]), reverse=True)) #list(major_categories.keys())
            with select_box_placeholder.container():
                selected_category = st.selectbox(
                    "Select a category:", 
                    options=category_options, 
                    key="category_selection", 
                    index=0, 
                    on_change=reset_search
                )

            with st.sidebar:
                search_episode = st.selectbox(
                    "Search episodes", 
                    options=episode_lookup.keys(), 
                    key="episode_selection", 
                    on_change=reset_category_selection,
                    index=None,
                    placeholder="Search"
                )
            
            

            # Reset on double click
            if st.session_state.click_reset:
                click_reset()
            elif not st.session_state.click_selection:
                st.session_state.selected_category = selected_category
                st.session_state.searched_episode = search_episode

            if st.session_state.searched_episode  is not None:
                ep_data = episode_lookup[st.session_state.searched_episode]
                category = ep_data["category"][0] if len(ep_data["category"]) > 0 else None
                category_clusters = major_categories[category] if category is not None else ep_data["clusters"]
                
                st.session_state.filtered_clusters = {
                    c: True if c in ep_data["clusters"] else "legendonly" for c in category_clusters
                }

                st.session_state.selection_state = [ep_data]
                st.session_state.selected_category = ep_data["category"]
                st.session_state.click_selection = True
            # filtered_clusters already set on_select call
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
                config=dict(scrollZoom=True, 
                            doubleClick="reset+autosize", 
                            doubleClickDelay=1000,
                            toImageButtonOptions={
                                'format': 'png', # one of png, svg, jpeg, webp
                                'filename': 'network_view',
                                'height': 500,
                                'width': 700,
                                'scale':6 # Multiply title/legend/axis/canvas sizes by this factor
                            }),
            )

        if analyis_mode == "static":
            st.info(
                "This app is running with static data. Go to [Github](https://github.com/danielressi/llm-podcast-explorer) page for more options"
            )

        with st.sidebar:
            info_placeholder = st.empty()
            st.session_state.click_reset = False
            with info_placeholder.container():
                if st.session_state.click_selection and st.session_state.selection_state:
                    display_data = st.session_state.selection_state[0]["customdata"][-1]
                    format_dict_to_markdown(display_data)
                    
                    

                    st.session_state.click_selection = False

                elif len(st.session_state.filtered_clusters) == 0:
                    st.write(
                    """
                    **Tips:**
                    - Select a category to start exploring the themes and topics of the podcast
                    - Each point represents an episode and similar episodes are visualised closer to each other.
                    - Click on a point to show episode details.

                    **Note:** 
                    - All insights are generated automatically with AI and may contain inaccuracies.
                    - For better user experience use a tablet, laptop or computer
                    """
                    )
                    
                else:
                    """
                    **Tips:**

                    - Enable/disable clusters by clicking the names in the legend.
                    - Select a cluster by double clicking the name on the legend.
                    - Click on a point to show episode details.

                    **Note:** 
                    - All insights are generated automatically with AI and may contain inaccuracies.
                    - For better user experience use a tablet, laptop or computer.
                    """

        st.caption("✨ Leveraging AI to explore content instead of generating it ✨")


if __name__ == "__main__":
    analyis_mode = os.environ.get("ANALYSIS_MODE", DEFAULT_MODE)
    if analyis_mode not in ["static", "active"]:
        raise ValueError(f"Environment variable ANALYSIS_MODE has to be 'static' or 'active', but got {analyis_mode} ")
    main(analyis_mode)
