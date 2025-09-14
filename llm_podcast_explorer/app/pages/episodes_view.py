import copy
import os
from typing import Union

import pandas as pd
import streamlit as st
from streamlit.runtime.scriptrunner import StopException

from llm_podcast_explorer.app.network_viz import build_networkx_graph, create_figure, update_figure
from llm_podcast_explorer.app.streamlit_app import (
    ALL_KEY,
    CACHE_TIMEOUT,
    DEFAULT_MODE,
    load_data,
    set_title_on_top,
    show_social,
)
from llm_podcast_explorer.app.table_viz import prepare_table, show_table


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


def reset_zoom():
    st.session_state.zoom_state = None


def click_reset():
    """User clicked on reset view or double clicked graph"""
    st.session_state.selected_category = ALL_KEY
    st.session_state.click_reset = False
    st.session_state.searched_episode = None


def st_category_selection(major_categories):
    st.session_state.major_categories = major_categories
    major_category_options = sorted(major_categories, key=lambda k: len(major_categories[k]), reverse=True)
    with st.container():
        selected_category = st.selectbox(
            "Select a category:",
            options=[ALL_KEY, *major_category_options],
            key="category_selection",
            index=0,
            on_change=reset_search,
        )
    return selected_category


def show_infos(analysis_mode):
    if analysis_mode == "static":
        st.info(
            "This app is running with static data. "
            "Go to [Github](https://github.com/danielressi/llm-podcast-explorer) page for more options"
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


def update_and_render_fig(base_fig, cluster_data, timeline, animation_mode):
    fig = copy.deepcopy(base_fig)

    with st.container():
        updated_fig, zoom_state = update_figure(
            fig,
            st.session_state.selected_category,
            st.session_state.filtered_clusters,
            cluster_data,
            timeline,
            st.session_state.click_selection,
            previous_zoom=st.session_state.zoom_state,
            selection_state=st.session_state.selection_state,
            animation_mode=animation_mode,
        )

        st.session_state.zoom_state = zoom_state

        st.plotly_chart(
            updated_fig,
            use_container_width=True,
            key="plotly_state",
            selection_mode=("points",),
            on_select=on_select,
            autoplay=True,
            config={
                "scrollZoom": True,
                "doubleClick": "reset+autosize",
                "doubleClickDelay": 1000,
                "displayModeBar": not animation_mode,
                "toImageButtonOptions": {
                    "format": "png",  # one of png, svg, jpeg, webp
                    "filename": "network_view",
                    "height": 500,
                    "width": 700,
                    "scale": 3,  # Multiply title/legend/axis/canvas sizes by this factor
                },
            },
        )


def on_select():
    if "plotly_state" in st.session_state:
        st.session_state.click_selection = True
        selection = st.session_state.plotly_state
        if len(selection["selection"]["points"]) > 0:
            st.session_state.searched_episode = selection["selection"]["points"][0]["customdata"][0]
        else:
            st.session_state.searched_episode = None
            st.session_state.click_selection = False
            # st.session_state.click_reset = True


def go_to_home():
    st.session_state.selected_podcast = None
    st.switch_page("./streamlit_app.py")


@st.cache_data(show_spinner=False, ttl=CACHE_TIMEOUT)
def create_network_graph(analysed_episodes, timeline, animation_mode):
    G, global_positions, clusters, episode_lookup = build_networkx_graph(analysed_episodes, timeline)
    fig, cluster_edge_indices, cluster_node_indices = create_figure(
        G, global_positions, clusters, animation_mode=animation_mode
    )
    cluster_data = {
        "clusters": clusters,
        "cluster_edge_indices": cluster_edge_indices,
        "cluster_node_indices": cluster_node_indices,
    }
    return fig, cluster_data, episode_lookup


def pretty_key(v):
    return str(v).replace("_", " ").capitalize()


def format_dict_to_markdown(display_data: dict[str, Union[str, list[str]]]) -> str:
    """
    Formats a dictionary into markdown text with keys as headers and lists as bullet points.

    Args:
        data: Dictionary with string keys and values that are either strings or lists of strings

    Returns:
        Markdown formatted string
    """
    episode_title = display_data.pop("title")
    summary = display_data.pop("summary")
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
        # st.markdown(f'<div style="max-height:400px; overflow:auto;">{description}</div>', unsafe_allow_html=True)
        st.markdown(description)


if "analysed_episodes" not in st.session_state or st.session_state.get("analysed_episodes") is None:
    try:
        load_data()  # cached; should re-create st.session_state items the root page sets
    except Exception:
        st.warning("Session state expired or not initialized. Click Home to re-initialize the app.")
        st.switch_page("./streamlit_app.py")

analysis_mode = os.environ.get("ANALYSIS_MODE", DEFAULT_MODE)
animation_mode = os.getenv("ANIMATION_MODE", "false").lower() in ("true", "1", "t")


reset_disabled = analysis_mode in ["static", "s3-scheduled"]

title = "Podcasts | Explored"
st.set_page_config(page_title=title, layout="centered", initial_sidebar_state="collapsed")
set_title_on_top(title)
st.markdown(
    """
    <style>
    .category-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding: 1rem;
        border-radius: 12px;
        color: #111;
        box-shadow: 0 3px 8px rgba(0,0,0,0.05);
    }
    .cluster-title {
        font-size: 1rem;
        font-weight: 600;
        margin: 1.5rem 0 0.5rem 0;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .episode-card {
        background: white;
        border-radius: 16px;
        padding: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        transition: all 0.2s ease-in-out;
    }
    .episode-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(0,0,0,0.12);
    }
    .episode-link {
        text-decoration: none;
        color: #1e88e5;
        font-weight: 600;
        font-size: 1.05rem;
        display: block;
        margin-bottom: 0.5rem;
    }
    .tags-container {
        margin-top: 0.5rem;
    }
    .tag {
        display: inline-block;
        margin: 0.2rem 0.3rem 0 0;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
        background: #f5f5f5;
        color: #333;
    }
    .tag:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(0,0,0,0.12);
    }
    .custom-expander {
        margin-top: 0.8rem;
        padding: 0.6rem 0.8rem;
        background: #fafafa;
        border-radius: 10px;
        font-size: 0.9rem;
        color: #444;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# st.sidebar.page_link('streamlit_app.py', label='Home')

with st.sidebar:
    col1, col2 = st.columns([1, 1])
    with col1:
        home_clicked = st.button("Home", key="home_button")
        if home_clicked:
            go_to_home()
        reset = st.button("Rerun analysis", disabled=reset_disabled) if not reset_disabled else False
    with col2:
        reset_view = st.button("Reset", disabled=False)
        if not st.session_state.click_reset:
            st.session_state.click_reset = reset_view

        if reset and st.session_state.selected_podcast is not None:
            st.session_state.checkpoint = False
            st.session_state.reset_podcasts = False
            load_data.clear()
            st.switch_page("./streamlit_app.py")
            # st.rerun()

    timeline = st.toggle("Timline mode", value=st.session_state.timeline_mode, disabled=False, on_change=reset_zoom)


if st.session_state.analysed_episodes is not None:
    analysed_episodes = st.session_state.analysed_episodes
    base_fig, cluster_data, episode_lookup = create_network_graph(
        analysed_episodes, timeline, animation_mode=animation_mode
    )

    try:
        major_categories = analysed_episodes["category_2_clusters"]
        selected_category = st_category_selection(major_categories)

        with st.sidebar:
            search_episode = st.selectbox(
                "Search episodes",
                options=episode_lookup.keys(),
                key="episode_selection",
                on_change=reset_category_selection,
                index=None,
                placeholder="Search",
            )

        # Reset on double click
        if st.session_state.click_reset:
            click_reset()
        elif not st.session_state.click_selection:
            st.session_state.selected_category = selected_category
            st.session_state.searched_episode = search_episode

        if st.session_state.searched_episode is not None:
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
            selected_category = selected_category[0] if isinstance(st.session_state.selected_category, list) else st.session_state.selected_category
            selected_category_clusters = major_categories[selected_category]
            st.session_state.filtered_clusters = dict.fromkeys(selected_category_clusters, True)

        tab1, tab2 = st.tabs(["Table", "Graph"])
        with tab1:
            df = prepare_table(st.session_state.analysed_episodes["episodes"])
            if st.session_state.searched_episode is not None:
                mask = df["title"] == st.session_state.searched_episode
                if mask.any():
                    df = pd.concat([df[mask], df[~mask]], ignore_index=True)
            show_table(df, st.session_state.selected_category, st.session_state.filtered_clusters)

        with tab2:
            update_and_render_fig(base_fig, cluster_data, timeline, animation_mode)

    except StopException:
        st.session_state.click_selection = False
        st.session_state.selected_category = ALL_KEY

    show_infos(analysis_mode)

    show_social()
