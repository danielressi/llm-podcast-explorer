import networkx as nx
import plotly.graph_objects as go
import json
import itertools
from collections import Counter
import pandas as pd
import numpy as np

def build_network_graph(episodes):
    
    episode_ids = set([ep["insights"]['episode_id'] for ep in episodes])
    
    nodes = [{'data': {'id': ep["insights"]['episode_id'], 
                       'label': ep["insights"]['episode_id'], 
                       "insights":ep["insights"],
                       "position": {"x": ep["insights"]["topic_century"]}
                }   
            } for ep in episodes]
    edges = [{'data': {'source': ep["insights"]['episode_id'], 'target': ref_ep}} for ep in episodes for ref_ep in ep["insights"]['referenced_episodes_id'] if ref_ep in episode_ids]
    
    return nodes, edges


def build_networkx_graph(episodes):
        # -------------------------------
    # 2) Build the Main Graph
    # -------------------------------
    G = nx.Graph()
    all_clusters = Counter()
    for ep in episodes:
        century = ep["insights"]["topic_century"] if ep["insights"]["topic_century"] else 22
        G.add_node(ep["insights"]['episode_id'], 
                   title=ep["metadata"]["title"], 
                   subtitle=ep["metadata"]["subtitle"],
                   century=century,
                   cluster=ep["insights"]["inferred_themes"])
        all_clusters.update(ep["insights"]["inferred_themes"])
    edges_data = ()
    for ep_x, ep_y in itertools.combinations(episodes, 2):
        ep_x_clusters = set(ep_x["insights"]["inferred_themes"])
        ep_y_clusters = set(ep_y["insights"]["inferred_themes"])
        if len(ep_x_clusters.intersection(ep_y_clusters)) > 0:
            edges_data += ((ep_x["insights"]['episode_id'], ep_y["insights"]['episode_id']),)
        elif ep_x["insights"]['episode_id'] in ep_y["insights"]['referenced_episodes_id'] or ep_y["insights"]['episode_id'] in ep_x["insights"]['referenced_episodes_id']:
            edges_data += ((ep_x["insights"]['episode_id'], ep_y["insights"]['episode_id']),)            
    
    G.add_edges_from(edges_data)

    # -------------------------------
    # 3) Identify century Buckets
    # -------------------------------
    unique_centuries = sorted(
        {century for century in nx.get_node_attributes(G, 'century').values() if century is not None}
    )
    global_positions = {}
    timeline=True

    if timeline:
        x_positions = {century: century * 10 for century in unique_centuries}

        for century in unique_centuries:
            nodes_in_century = [n for n in G.nodes if G.nodes[n].get('century') == century]
            G_sub = G.subgraph(nodes_in_century).copy()

            if len(G_sub.nodes) == 1:
                # Place single nodes directly
                global_positions[nodes_in_century[0]] = (x_positions[century], 0)
                continue
            
            # Use Kamada-Kawai layout for better spacing
            pos = nx.kamada_kawai_layout(G_sub)  

            # Compute centroid
            x_vals, y_vals = zip(*pos.values()) if pos else ([0], [0])
            mean_x, mean_y = sum(x_vals) / len(x_vals), sum(y_vals) / len(y_vals)

            # Adjust layout to align centuries
            shift_x = x_positions[century] - mean_x
            shift_y = -mean_y  

            # Add slight random jitter to avoid perfect circles
            jitter = lambda: np.random.uniform(-2, 2)
            
            for node, (raw_x, raw_y) in pos.items():
                global_positions[node] = (raw_x + shift_x + jitter(), raw_y + shift_y + jitter())

    else:
        global_positions = nx.spring_layout(G, seed=42, k=10)

        

    sorted_clusters = pd.Series(all_clusters).sort_values(ascending=False)
    min_cluster_size = 3
    relevant_clusters = sorted_clusters[(sorted_clusters >= min_cluster_size) & (sorted_clusters < len(sorted_clusters)*0.75)].index

    return G, global_positions, relevant_clusters


def create_figure(G, global_positions, selected_cluster):

    fig = go.Figure()

    # a) Add edges

    edge_visible = []
    for (u, v) in G.edges():
        # get the positions
        x0, y0 = global_positions[u]
        x1, y1 = global_positions[v]
        edge_x = [x0, x1, None]
        edge_y = [y0, y1, None]
        
        edge_visible = True
        if selected_cluster == "<None>":
            line=dict(color="#D3D3D3", width=1)
        elif (selected_cluster in G.nodes[u]['cluster']) and (selected_cluster in G.nodes[v]['cluster']):
            line=dict(color="#A9A9A9", width=1)
        else: 
            line=dict(color="#E6E6E6", width=1)
            edge_visible=False
        

        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=line,
                hoverinfo='none',
                visible=edge_visible,
                name='Edges'
            )
        )

    # b) Cluster selection in Streamlit


    # c) Add nodes (color highlight if cluster is selected)

    for node in G.nodes():
        x, y = global_positions[node]
        
        data = G.nodes[node].copy()
        c = data['cluster']
        
        node_text = f"Century: {data['century']}"  # Adjusted formatting

        visible = True
        if  selected_cluster in c:
            node_color =[ "#d62728"]  # highlight color
        elif selected_cluster == "<None>":
            node_color = ["#BDBDBD"]  # gray out
        else:
            node_color =[ "#BDBDBD"]  # gray out
            visible=True
        
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                visible=visible,
                marker=dict(
                    size=12,
                    color=node_color,
                    line=dict(color="black",width=1)
                ),
                customdata=[[data["title"], data["subtitle"], node_text]],  # Store node_text in customdata
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br><br><i>%{customdata[2]}</i>",  
                name='Episodes'
            )
        )

    # d) Final layout touches
    fig.update_layout(
        title="<b>Podcast Episode Network</b>",
        title_font=dict(size=20, color="#333"),  # Dark gray for a clean look
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="#F0F2F6",  # Matches Streamlit's default background
        paper_bgcolor="#F0F2F6",  # Ensures smooth blending
    )

    return fig


if __name__ == "__main__":
    with open('data.json', 'r') as f:
        analysed_episodes = json.load(f)
    G, global_positions, clusters = build_networkx_graph(analysed_episodes["episodes"])
    fig = create_figure(G, global_positions, "<None>")