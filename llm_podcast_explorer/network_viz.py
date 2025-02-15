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
                   on_click=dict(
                                #title=ep["metadata"]["title"],
                                 #description=ep["metadata"]["description"],
                                 link=f'<a target="_blank" href="{ep["metadata"]["link"]}">{ep["metadata"]["title"]}</a>'),
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
        scale = 100
        x_positions = {century: century * scale for century in unique_centuries}

        for century in unique_centuries:
            nodes_in_century = [n for n in G.nodes if G.nodes[n].get('century') == century]
            G_sub = G.subgraph(nodes_in_century).copy()

            if len(G_sub.nodes) == 1:
                # Place single nodes directly
                global_positions[nodes_in_century[0]] = (x_positions[century], 0)
                
            
            # Use Kamada-Kawai layout for better spacing
            pos = nx.kamada_kawai_layout(G_sub)  

            # Compute centroid
            x_vals, y_vals = zip(*pos.values()) if pos else ([0], [0])
            mean_x, mean_y = sum(x_vals)*scale / len(x_vals), sum(y_vals)*scale / len(y_vals)

            # Adjust layout to align centuries
            shift_x = x_positions[century] - mean_x
            shift_y = -mean_y  

            # Add slight random jitter to avoid perfect circles
            jitter = lambda: np.random.uniform(-2*scale/4, 2*scale/4)
            
            for node, (raw_x, raw_y) in pos.items():
                global_positions[node] = (raw_x + shift_x + jitter(), raw_y + shift_y + jitter())

    else:
        global_positions = nx.spring_layout(G, seed=42, k=10)

        

    sorted_clusters = pd.Series(all_clusters).sort_values(ascending=False)
    min_cluster_size = 2
    relevant_clusters = sorted_clusters[(sorted_clusters >= min_cluster_size) & (sorted_clusters < len(sorted_clusters)*0.75)].index.to_list()

    return G, global_positions, relevant_clusters



HIGHLIGHT_COLOR = "#d62728"
BACKGROUND_COLOR = "rgba(138, 138, 138, 0.2)"
DEFAULT_NODE_COLOR = "rgba(138, 138, 138, 0.8)"

def create_figure(G, global_positions, clusters):
    fig = go.Figure()
    cluster_edges_indices = {c: set() for c in clusters}
    cluster_node_indices = {c: set() for c in clusters}
    for i, (u, v) in enumerate(G.edges()):
        
        # get the positions
        x0, y0 = global_positions[u]
        x1, y1 = global_positions[v]
        edge_x = [x0, x1, None]
        edge_y = [y0, y1, None]                
        shared_clusters = set(G.nodes[u]["cluster"]).intersection(set(G.nodes[v]["cluster"]))
        for sc in shared_clusters:
            # not all clusters passed on to selector
            if sc in cluster_edges_indices:
                cluster_edges_indices[sc].add(i)
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(color=BACKGROUND_COLOR, width=1),
                hoverinfo='none',
                visible=True,
                name="Edges"
            )
        )

    index_offset = len(G.edges())
    edge_range = (0, index_offset-1)
    node_range = (index_offset, index_offset+len(G.nodes()))
    index_ranges = dict(edges=edge_range,nodes=node_range)
    for j, node_name in enumerate(G.nodes()):
        node = G.nodes[node_name].copy()
        x, y = global_positions[node_name]
        metadata_list = [[node["title"], node["subtitle"],node["century"], node["on_click"]]]

        for c in node["cluster"]:
            # not all clusters passed on to selector
            if c in cluster_node_indices:
                cluster_node_indices[c].add(index_offset+j)
            

        ### Nodes
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                visible=True,
                marker=dict(
                    size=12,
                    color=DEFAULT_NODE_COLOR,
                    line=dict(color="black",width=1)
                ),
                customdata=metadata_list,  # Store node_text in customdata
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br><br><i>Century: %{customdata[2]}</i>",  
                name="Nodes"
            )
        )


    
        fig.update_layout(
        title="<b>Podcast Episode Network</b>",
        #title_font=dict(size=20, color="#333"),  # Dark gray for a clean look
        showlegend=False,
        xaxis=dict(showgrid=True, zeroline=True, visible=True),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        #margin=dict(l=20, r=20, t=40, b=20),
        #plot_bgcolor="#F0F2F6",  # Matches Streamlit's default background
        #paper_bgcolor="#F0F2F6",  # Ensures smooth blending
    )

    
    return fig, cluster_edges_indices, cluster_node_indices, index_ranges


def update_figure(fig, selected_cluster, cluster_data):
    if selected_cluster == "<None>":
        return fig
    else:
        # update edges
        edge_indices = cluster_data["cluster_edge_indices"][selected_cluster]
        
        """
        edges_x = []
        edges_y = []
        edges_customdata = [] 
        for idx in edge_indices:
            edges_x.extend(fig.data[idx].x)
            fig.data[idx].visible = True
            fig.data[idx].line.update(width=2, color=HIGHLIGHT_COLOR)
        """


        # update nodes
        node_indices = cluster_data["cluster_node_indices"][selected_cluster]
        nodes_x = []
        nodes_y = []
        nodes_customdata = [] 
        for idx in node_indices:          
            #fig.data[idx].marker.update(dict(size=20, color=HIGHLIGHT_COLOR))
            nodes_x.extend(fig.data[idx].x)
            nodes_y.extend(fig.data[idx].y)
            nodes_customdata.extend(fig.data[idx].customdata)

        fig.add_trace(
            go.Scatter(
                x=nodes_x,
                y=nodes_y,
                mode='markers',
                visible=True,
                marker=dict(
                    size=20,
                    color=HIGHLIGHT_COLOR,
                    line=dict(color="black",width=1)
                ),
                customdata=nodes_customdata,  # Store node_text in customdata
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br><br><i>Century: %{customdata[2]}</i>",  
                name="Nodes"
            )
        )
        return fig




if __name__ == "__main__":
    with open('data.json', 'r') as f:
        analysed_episodes = json.load(f)
    G, global_positions, clusters = build_networkx_graph(analysed_episodes["episodes"])
    fig = create_figure(G, global_positions, "<None>")