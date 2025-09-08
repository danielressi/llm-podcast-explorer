from llm_podcast_explorer.src.episodes_model import AnalyzedEpisodes
from llm_podcast_explorer.app.streamlit_app import CACHE_TIMEOUT
import pandas as pd
import streamlit as st
import itertools
import hashlib
import random



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

# small helper to convert hex to rgba for a subtle background tint
def hex_to_rgba(h: str, a: float = 0.06) -> str:
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"

def show_table(df: pd.DataFrame, selected_category: str, filtered_clusters: list[str] | None = None):
    """Render episodes from a given category in a mobile-friendly layout with tags & expandable description."""

    # Pick data
    if isinstance(selected_category, list):
        selected_category = selected_category[0]

    category_df = df if selected_category == "All" else df[df["major_category"] == selected_category]



    if len(filtered_clusters) == 0:
        filtered_clusters = {k: True for k in df["consolidated_titles"].explode().unique()}


    if category_df.empty:
        st.info(f"No episodes found for category: {selected_category}")
        return

    cluster_colors = assign_cluster_colors(filtered_clusters)

    category_color = "#BABEC0"
    category_bg = hex_to_rgba(category_color, 0.12)

    # prepare a compact set of cluster chips (show up to 8)
    visible_clusters = list(filtered_clusters) if filtered_clusters else []

    display_name = "Showing All Categories" if selected_category == "All" else selected_category
    # --- Category header ---
    st.markdown(
        f"""
        <div style="background:{category_bg}; padding:1rem; border-radius:12px; margin:0.6rem 0;">
          <div style="min-width:0;">
            <div style="font-weight:800; font-size:1.35rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{display_name}</div>
            <div style="font-size:0.9rem; color:#555;">{len(visible_clusters)} clusters • {len(category_df)} episodes</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    show_clusters = [k for k, show in filtered_clusters.items() if show and show != "legendonly"]
    random.shuffle(show_clusters)

    # Ensure the first non-empty cluster starts expanded
    cluster_index = 0

    # Iterate clusters (keep loop unchanged as requested)
    for cluster in show_clusters:
        df_cluster = category_df[category_df["consolidated_titles"].apply(lambda x: cluster in x if x else False)]
        if df_cluster.empty:
            continue

        # Build an expandable <details> per cluster (first non-empty opened)
        cluster_color = cluster_colors.get(cluster, "#E0E0E0")
        open_attr = " open" if cluster_index == 0 else ""
        cluster_index += 1

        cluster_html = []
        cluster_html.append(
            f"<details class='cluster-details' style='margin:0.6rem 0;'{open_attr}>"
            f"<summary class='cluster-title' style='background:{cluster_color}; padding:0.4rem 0.8rem; border-radius:8px; cursor:pointer;'>"
            f"{cluster} &nbsp; <span style='color:#666; font-weight:600;'>({len(df_cluster)})</span>"
            f"</summary>"
        )

        for row in df_cluster.itertuples():
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

            # Single HTML card inside the cluster details (avoid leading indentation/newline)
            card_html = (
                f"<div class='episode-card' style='border:6px solid {cluster_color};"
                f"background:{accent_bg};"
                f"padding:1rem;"
                f"margin:1rem 0;"
                f"border-radius:10px;"
                f"display:flex;"
                f"gap:1rem;"
                f"align-items:flex-start;'>"
                f"<div style='flex:2; min-width:0;'>"
                f"<a class='episode-link' href='{row.podlink}' target='_blank' style='color:{cluster_color}; text-decoration:none;'>🎧 {row.title}</a>"
                f"{desc_html}"
                f"</div>"
                f"<div style='flex:1; display:flex; justify-content:flex-end; gap:0.4rem; flex-wrap:wrap;'>"
                f"{tags_html}"
                f"</div>"
                f"</div>"
            )
            cluster_html.append(card_html)

        cluster_html.append("</details>")
        st.markdown("".join(cluster_html), unsafe_allow_html=True)
        



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