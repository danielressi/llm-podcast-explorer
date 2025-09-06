from llm_podcast_explorer.src.episodes_model import AnalyzedEpisodes
from llm_podcast_explorer.app.streamlit_app import CACHE_TIMEOUT
import pandas as pd
import streamlit as st
import itertools
import hashlib

if "css_injected" not in st.session_state:
        # --- CSS ---
    st.markdown("""
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
        """, unsafe_allow_html=True)

    st.session_state["css_injected"] = True

# 🎨 Define a rotating palette (inspired by NTS / ChatGPT tones)
# 🎨 Define a rotating palette (inspired by NTS / ChatGPT tones)
PALETTE = [
    "#E3F2FD",  # light blue
    "#FCE4EC",  # light pink
    "#E8F5E9",  # light green
    "#FFF3E0",  # light orange
    "#F3E5F5",  # light purple
    "#E0F7FA",  # aqua
    "#F9FBE7",  # lemon
]

def assign_cluster_colors(clusters):
    """Assign distinct palette colors to clusters."""
    color_cycle = itertools.cycle(PALETTE)
    return {cluster: next(color_cycle) for cluster in clusters}


def show_table(df: pd.DataFrame, selected_category: str, filtered_clusters: list[str] | None = None):
    """Render episodes from a given category in a mobile-friendly layout with tags & expandable description."""

    # Pick data

    category_df = df if selected_category == "All" else df[df["major_category"] == selected_category]

    if len(filtered_clusters) == 0:
        filtered_clusters = set(df["consolidated_titles"].explode())


    if category_df.empty:
        st.info(f"No episodes found for category: {selected_category}")
        return

    cluster_colors = assign_cluster_colors(filtered_clusters)
            # Category header with auto color
    
    if selected_category != "All":
        bg_color = "#BABEC0"
        st.markdown(
            f"""
            <div style="background:{bg_color}; padding:1rem; border-radius:12px; margin:1rem 0;">
                <h2 style="margin:0; color:#111;">{selected_category}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    

    # Iterate clusters (keep loop unchanged as requested)
    for cluster in filtered_clusters:
        df_cluster = category_df[category_df["consolidated_titles"].apply(lambda x: cluster in x if x else False)]
        if df_cluster.empty:
            continue
        cluster_color = cluster_colors.get(cluster, "#DDD")
        #st.markdown(f"<div class='cluster-title'>{cluster}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='cluster-title' style='background:{cluster_color}; padding:0.4rem 0.8rem; border-radius:8px;'>{cluster}</div>",
            unsafe_allow_html=True
        )
        for row in df_cluster.itertuples():
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(
                    f"""
                    <div class='episode-card'>
                        <a class='episode-link' href='{row.podlink}' target='_blank'>
                            🎧 {row.title}
                        </a>
                    </div>
                    """, unsafe_allow_html=True
                )

            
                if hasattr(row, "description") and row.description:
                    with st.expander("More info", expanded=False):
                        st.write(row.description)

            with col2:
                if hasattr(row, "tags") and row.tags:
                    tag_html = " ".join(
                        [f"<span class='tag'>{tag}</span>" for tag in row.tags]
                    )
                    st.markdown(tag_html, unsafe_allow_html=True)


@st.cache_data(show_spinner=False, ttl=CACHE_TIMEOUT)
def prepare_table(episodes: AnalyzedEpisodes) -> pd.DataFrame:
    df_meta = pd.DataFrame([ep["metadata"] for ep in episodes])
    df_insights = pd.DataFrame([ep["insights"] for ep in episodes])
    df_clusters = pd.DataFrame([ep["clusters"] for ep in episodes])

    df_1 = pd.merge(
        df_meta[["title", "podlink", "description"]],
        df_clusters[["major_category", "consolidated_titles"]],
        left_index=True,
        right_index=True,
    )
    df = pd.merge(
        df_1,
        df_insights[["tags", "inferred_themes"]],
        left_index=True,
        right_index=True,
    )
    return df
