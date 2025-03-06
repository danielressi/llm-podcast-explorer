import networkx as nx
import plotly.graph_objects as go
import json
import itertools
from collections import Counter
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.express as px
import textwrap

SCALE = 1
HIGHLIGHT_COLOR = "rgba(214, 39, 40, 1)"
SELECT_COLOR = "rgba(232, 228, 215, 0.8)"
HIGHLIGHT_COLOR_EDGE = "rgba(214, 39, 40, 0.4)"
BACKGROUND_COLOR = "rgba(138, 138, 138, 0.2)"
DEFAULT_NODE_COLOR = "rgba(138, 138, 138, 0.8)"
EDGE_COSINE_THRESHOLD = 0.35
COLOR_OPACITY = 0.6
COLOR_CYCLE = [
    f"rgba(214, 39, 40, {COLOR_OPACITY})",  # Red
    f"rgba(44, 160, 44, {COLOR_OPACITY})",  # Green
    f"rgba(31, 119, 180, {COLOR_OPACITY})",  # Blue
    f"rgba(255, 127, 14, {COLOR_OPACITY})",  # Orange
    f"rgba(148, 103, 189, {COLOR_OPACITY})",  # Purple
    f"rgba(140, 86, 75, {COLOR_OPACITY})",  # Brown
    f"rgba(227, 119, 194, {COLOR_OPACITY})",  # Pink
    f"rgba(127, 127, 127, {COLOR_OPACITY})",  # Gray
    f"rgba(188, 189, 34, {COLOR_OPACITY})",  # Yellow-green
    f"rgba(23, 190, 207, {COLOR_OPACITY})",  # Cyan
]

HOVERTEMPLATE = (
    "<b>%{customdata[0]}</b><br>"
    "%{customdata[5]}<br>"
    "Category: %{customdata[4]}<br>"
    "Clusters: %{customdata[3]}<br>"
    "Themes: %{customdata[1]}<br>"
    "<i>Century: %{customdata[2]}</i><br>"
    "<extra></extra>"
)


def build_networkx_graph(episodes, timeline=True, weight_threshold=0.8):
    # -------------------------------
    # 2) Build the Main Graph
    # -------------------------------
    G = nx.Graph()
    all_clusters = Counter()
    distance_map = episodes["distance_map"]
    category_2_clusters = episodes["category_2_clusters"]
    clusters_2_category = {c: k for k, v in category_2_clusters.items() for c in v}
    # all_clusters = list(clusters_2_category.keys())
    for ep in episodes["episodes"]:
        century = ep["insights"]["topic_century"] if ep["insights"]["topic_century"] else 21
        cluster_titles = ep["clusters"]["consolidated_titles"]

        category = [clusters_2_category[cluster] for cluster in cluster_titles if cluster in clusters_2_category]
        G.add_node(
            ep["insights"]["episode_id"],
            title=ep["metadata"]["title"],
            subtitle=ep["metadata"]["subtitle"],
            century=century,
            embedding=ep["clusters"]["embeddings"],
            on_click=dict(
                title=ep["metadata"]["title"],
                summary=ep["insights"]["summary"],
                tags=ep["insights"]["tags"],
                themes=ep["insights"]["inferred_themes"],
                category=category,
                clusters=cluster_titles,
                clusters_raw=ep["clusters"]["titles"],
                cluster_attempt=ep["clusters"]["attempt"],
                referenced_episodes=ep["insights"]["referenced_episodes_id"],
                link=ep["metadata"]["link"],
                year=f"{ep['insights']['topic_year']} ({ep['insights']['topic_century']} Century)",
            ),
            cluster=cluster_titles,
            category=category,
        )
        all_clusters.update(cluster_titles)

    sorted_clusters = pd.Series(all_clusters).sort_values(ascending=False)
    min_cluster_size = 2
    # relevant_clusters = sorted_clusters[(sorted_clusters >= min_cluster_size) & (sorted_clusters < len(episodes["episodes"])*0.8)].index.to_list()
    relevant_clusters = sorted_clusters.index.to_list()

    edges_data = ()
    referenced_edges = ()
    for ep_x, ep_y in itertools.combinations(episodes["episodes"], 2):
        # ep_x_clusters = set(relevant_clusters).intersection(set(ep_x["clusters"]["titles"]))
        # ep_y_clusters = set(relevant_clusters).intersection(set(ep_y["clusters"]["titles"]))
        ep_x_id = ep_x["metadata"]["index"]
        ep_y_id = ep_y["metadata"]["index"]
        if f"{ep_x_id},{ep_y_id}" in distance_map:
            distance = distance_map[f"{ep_x_id},{ep_y_id}"]
        elif f"{ep_y_id},{ep_x_id}" in distance_map:
            distance = distance_map[f"{ep_y_id},{ep_x_id}"]
        else:
            distance = 2

        references_x = ep_x["insights"]["referenced_episodes_id"] if ep_x["insights"]["referenced_episodes_id"] else []
        references_y = ep_y["insights"]["referenced_episodes_id"] if ep_y["insights"]["referenced_episodes_id"] else []
        # shared_themes = ep_x_clusters.intersection(ep_y_clusters)
        # union_themes = ep_x_clusters.union(ep_y_clusters)
        # edge_weight = len(shared_themes) / len(union_themes) if len(union_themes) > 0 else 0
        if distance <= EDGE_COSINE_THRESHOLD:
            edges_data += ((ep_x["insights"]["episode_id"], ep_y["insights"]["episode_id"], 2 - distance),)
        elif ep_x["insights"]["episode_id"] in references_y or ep_y["insights"]["episode_id"] in references_x:
            referenced_edges += ((ep_x["insights"]["episode_id"], ep_y["insights"]["episode_id"], 1),)

    G.add_weighted_edges_from(edges_data, attr="themes")
    G.add_weighted_edges_from(referenced_edges, attr="references")
    # -------------------------------
    # 3) Identify century Buckets
    # -------------------------------
    unique_centuries = sorted({
        century for century in nx.get_node_attributes(G, "century").values() if century is not None
    })
    global_positions = {}
    embedding_positions = {node: np.array(G.nodes[node]["embedding"]).mean(axis=0) for node in G.nodes()}

    if timeline:
        scale = SCALE
        x_positions = {century: century * scale for century in unique_centuries}

        for century in unique_centuries:
            nodes_in_century = [n for n in G.nodes if G.nodes[n].get("century") == century]
            G_sub = G.subgraph(nodes_in_century).copy()

            if not G_sub.nodes:
                continue  # Skip empty subgraphs

            for node in G_sub.nodes():
                meta = G.nodes[node]
                year = meta.get("year", None)
                emb_x, emb_y = embedding_positions[node]

                # Base x-position: spaced by century
                base_x = x_positions[century]

                # Fine-tune within century
                year_offset = (year % 100) / 100 if year else 0.5  # Normalized year position
                emb_x_offset = emb_x * (scale / 5)  # Additional x-axis spacing

                final_x = base_x + year_offset + emb_x_offset
                final_y = emb_y * scale  # Maintain embedding structure on y-axis

                global_positions[node] = (final_x, final_y)

    else:
        global_positions = embedding_positions

    return G, global_positions, relevant_clusters


def create_figure(G, global_positions, clusters):
    fig = go.Figure()
    # fig = make_subplots(specs=[[{"secondary_y": True}]])
    cluster_edges_indices = {c: set() for c in clusters}
    cluster_node_indices = {c: set() for c in clusters}
    edge_x = []
    edge_y = []
    offset = 0
    for i, (u, v) in enumerate(G.edges()):
        # get the positions
        x0, y0 = global_positions[u]
        x1, y1 = global_positions[v]
        edge_x.extend([x0, x1])
        edge_y.extend([y0, y1])
        shared_clusters = set(G.nodes[u]["cluster"]).intersection(set(G.nodes[v]["cluster"]))
        for sc in shared_clusters:
            # not all clusters passed on to selector
            if sc in cluster_edges_indices:
                cluster_edges_indices[sc].add(i + offset)
                cluster_edges_indices[sc].add(i + 1 + offset)

        offset += 1

    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color=BACKGROUND_COLOR, width=1),
            hoverinfo="none",
            visible=True,
            showlegend=False,
            name="Edges",
        ),
        # secondary_y=True,
    )

    index_offset = 0  # len(G.edges())
    edge_range = (0, index_offset - 1)
    node_range = (index_offset, index_offset + len(G.nodes()))
    index_ranges = dict(edges=edge_range, nodes=node_range)
    nodes_x = []
    nodes_y = []
    metadata_list = []
    for j, node_name in enumerate(G.nodes()):
        node = G.nodes[node_name].copy()
        x, y = global_positions[node_name]
        nodes_x.append(x)
        nodes_y.append(y)
        metadata_list.append([
            node["title"],
            ",".join(node["on_click"]["themes"]),
            node["century"],
            node["cluster"],
            list(set(node["category"])),
            node["on_click"]["summary"],
            node["on_click"],
        ])

        for c in node["cluster"]:
            # not all clusters passed on to selector
            if c in cluster_node_indices:
                cluster_node_indices[c].add(index_offset + j)

    ### Nodes
    fig.add_trace(
        go.Scatter(
            x=nodes_x,
            y=nodes_y,
            mode="markers",
            visible=True,
            showlegend=False,
            marker=dict(size=12, color=DEFAULT_NODE_COLOR, line=dict(color="black", width=1)),
            selected=dict(marker=dict(size=25, color=SELECT_COLOR)),
            customdata=metadata_list,  # Store node_text in customdata
            hoverinfo="none",
            hovertemplate=HOVERTEMPLATE,
            name="Nodes",
        ),
        # secondary_y=True,
    )

    fig.update_layout(
        # title_font=dict(size=20, color="#333"),  # Dark gray for a clean look
        # showlegend=True,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(l=20, r=20, t=10, b=10),
        # plot_bgcolor="#F0F2F6",  # Matches Streamlit's default background
        # paper_bgcolor="#F0F2F6",  # Ensures smooth blending
        legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="left", x=0.0),
    )

    return fig, cluster_edges_indices, cluster_node_indices, index_ranges


def update_figure(fig, selected_category, filtered_clusters, cluster_data, timeline, clicked, previous_zoom):
    if timeline:
        fig.update_xaxes(title_text="Century")
        fig.update_layout(xaxis=dict(showgrid=True, zeroline=True, visible=True))

    if selected_category == "All":
        return fig, None

    selected_clusters = list(filtered_clusters.keys())

    # fig.update_traces(hovertemplate=None)
    min_x, max_x = np.inf, -np.inf
    min_y, max_y = np.inf, -np.inf
    for selected_cluster, highlight_color in zip(selected_clusters, itertools.cycle(COLOR_CYCLE)):
        if selected_cluster not in cluster_data["cluster_edge_indices"]:
            continue
        edge_indices = cluster_data["cluster_edge_indices"][selected_cluster]

        edges_x = []
        edges_y = []
        for idx in edge_indices:
            edges_x.append(fig.data[0].x[idx])
            edges_y.append(fig.data[0].y[idx])

        fig.add_trace(
            go.Scatter(
                x=edges_x,
                y=edges_y,
                mode="lines",
                line=dict(color=highlight_color, width=2),
                showlegend=False,
                legendgroup=selected_cluster,
                hoverinfo="none",
                visible=filtered_clusters[selected_cluster],
                name="Edges-Highlight",
            ),
            # secondary_y=False,
        )

        # update nodes
        node_indices = cluster_data["cluster_node_indices"][selected_cluster]
        nodes_x = []
        nodes_y = []
        nodes_customdata = []
        for idx in node_indices:
            nodes_x.append(fig.data[1].x[idx])
            nodes_y.append(fig.data[1].y[idx])
            nodes_customdata.append(fig.data[1].customdata[idx])

        if filtered_clusters[selected_cluster] not in [False, "legendonly"]:
            min_x, max_x = min(*nodes_x, min_x), max(*nodes_x, max_x)
            min_y, max_y = min(*nodes_y, min_y), max(*nodes_y, max_y)

        fig.add_trace(
            go.Scatter(
                x=nodes_x,
                y=nodes_y,
                mode="markers",
                visible=filtered_clusters[selected_cluster],
                showlegend=True,
                legendgroup=selected_cluster,
                marker=dict(size=20, color=highlight_color, line=dict(color="black", width=1)),
                selected=dict(marker=dict(size=25, color=SELECT_COLOR)),
                customdata=nodes_customdata,  # Store node_text in customdata
                # hoverinfo='none',
                hovertemplate=HOVERTEMPLATE,
                hoverlabel=dict(
                    bordercolor=highlight_color  # Border color
                ),
                name=selected_cluster,
            ),
            # secondary_y=False
        )
    # nodes_df = pd.concat(nodes_df)
    # fig.add_trace(px.scatter(nodes_df, x="x",y="y",color="cluster"))

    zoom_info = None
    if timeline:
        x_min = max([min_x - 5 * SCALE, min(fig.data[1].x) - SCALE])
        x_max = min([max_x + 5 * SCALE, max(fig.data[1].x) + SCALE])
        zoom_info = dict(xaxis_range=[x_min, x_max], yaxis_range=None)
        fig.update_xaxes(title_text="Century", range=[x_min, x_max])

    elif clicked and previous_zoom:
        fig.update_layout(xaxis_range=previous_zoom["xaxis_range"], yaxis_range=previous_zoom["yaxis_range"])
    elif not clicked or not previous_zoom:
        x_margin = (max_x - min_x) * 0.05
        x_min = max([min_x - x_margin, min(fig.data[1].x) - x_margin])
        x_max = min([max_x + x_margin, max(fig.data[1].x) + x_margin])

        y_margin = (max_x - min_x) * 0.05
        y_min = max([min_y - y_margin, min(fig.data[1].y) - y_margin])
        y_max = min([max_y + y_margin, max(fig.data[1].y) + y_margin])

        zoom_info = dict(xaxis_range=[x_min, x_max], yaxis_range=[y_min, y_max])
        fig.update_layout(xaxis_range=[x_min, x_max], yaxis_range=[y_min, y_max])
    return fig, zoom_info
