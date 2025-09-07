from llm_podcast_explorer.src.episodes_model import AnalyzedEpisodes
from llm_podcast_explorer.app.streamlit_app import CACHE_TIMEOUT
import pandas as pd
import streamlit as st
import itertools
import hashlib



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
    
    # if selected_category != "All":
    #     bg_color = "#BABEC0"
    #     st.markdown(
    #         f"""
    #         <div style="background:{bg_color}; padding:1rem; border-radius:12px; margin:1rem 0;">
    #             <h2 style="margin:0; color:#111;">{selected_category}</h2>
    #         </div>
    #         """,
    #         unsafe_allow_html=True
    #     )
    

    # Iterate clusters (keep loop unchanged as requested)
    for cluster in filtered_clusters:
        df_cluster = category_df[category_df["consolidated_titles"].apply(lambda x: cluster in x if x else False)]
        if df_cluster.empty:
            continue
        #st.markdown(f"<div class='cluster-title'>{cluster}</div>", unsafe_allow_html=True)
        cluster_color = cluster_colors.get(cluster, "#E0E0E0")
        st.markdown(
            f"<div class='cluster-title' style='background:{cluster_color}; padding:0.4rem 0.8rem; border-radius:8px;'>{cluster}</div>",
            unsafe_allow_html=True
        )

        for row in df_cluster.itertuples():
            # small helper to convert hex to rgba for a subtle background tint
            def hex_to_rgba(h: str, a: float = 0.06) -> str:
                h = h.lstrip("#")
                r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                return f"rgba({r},{g},{b},{a})"

            # tags HTML
            tags_html = ""
            if hasattr(row, "tags") and row.tags:
                tags_html = " ".join([f"<span class='tag'>{tag}</span>" for tag in row.tags])

            # description HTML using native <details> for a compact, sleek expander
            desc_html = ""
            if hasattr(row, "description") and row.description:
                safe_desc = str(row.description).replace("\n", "<br/>")
                desc_html = (
                    f"<details class='custom-expander'><summary>More info</summary>"
                    f"<div style='margin-top:0.5rem'>{safe_desc}</div></details>"
                )

            accent_bg = hex_to_rgba(cluster_color, 0.06)

            # Single HTML card that visually spans both "columns" with a left accent,
            # subtle tinted background and tags aligned to the right.
            st.markdown(
                f"""
                  <div class='episode-card' style='border-left:6px solid {cluster_color};
                                                  background:{accent_bg};
                                                  padding:1rem;
                                                  margin:1rem 0;
                                                  display:flex;
                                                  gap:1rem;
                                                  align-items:flex-start;'>
                    <div style='flex:2; min-width:0;'>
                        <a class='episode-link' href='{row.podlink}' target='_blank' style='color:{cluster_color};'>
                            🎧 {row.title}
                        </a>
                        {desc_html}
                    </div>
                    <div style='flex:1; display:flex; justify-content:flex-end; gap:0.4rem; flex-wrap:wrap;'>
                        {tags_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

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
