import networkx as nx
import plotly.graph_objects as go
import json
import itertools
from collections import Counter
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots

SCALE=1
def build_networkx_graph(episodes, timeline=True, weight_threshold=0.8):
        # -------------------------------
    # 2) Build the Main Graph
    # -------------------------------
    G = nx.Graph()
    all_clusters = Counter()
    for ep in episodes:
        century = ep["insights"]["topic_century"] if ep["insights"]["topic_century"] else 21
        cluster_titles = ep["clusters"]["titles"] 
        G.add_node(ep["insights"]['episode_id'], 
                   title=ep["metadata"]["title"], 
                   subtitle=ep["metadata"]["subtitle"],
                   century=century,
                   embedding=ep["clusters"]["embeddings"],
                   on_click=dict(
                                 title=ep["metadata"]["title"],
                                 summary=ep["insights"]["summary"],
                                 #topic=ep["insights"]["topic"],
                                 tags=ep["insights"]["tags"],
                                 themes=ep["insights"]["inferred_themes"],
                                 clusters=cluster_titles,
                                 referenced_episodes=ep["insights"]["referenced_episodes_id"],
                                 link=ep["metadata"]["link"],
                                 year= f'{ep["insights"]["topic_year"]} ({ep["insights"]["topic_century"]} Century)'
                                 ),
                   cluster=cluster_titles)
        all_clusters.update(cluster_titles)

    sorted_clusters = pd.Series(all_clusters).sort_values(ascending=False)
    min_cluster_size = 3
    relevant_clusters = sorted_clusters[(sorted_clusters >= min_cluster_size) & (sorted_clusters < len(episodes)*0.8)].index.to_list()


    edges_data = ()
    referenced_edges = ()
    for ep_x, ep_y in itertools.combinations(episodes, 2):
        ep_x_clusters = set(relevant_clusters).intersection(set(ep_x["clusters"]["titles"]))
        ep_y_clusters = set(relevant_clusters).intersection(set(ep_y["clusters"]["titles"]))
        references_x = ep_x["insights"]['referenced_episodes_id'] if ep_x["insights"]['referenced_episodes_id'] else []
        references_y = ep_y["insights"]['referenced_episodes_id'] if ep_y["insights"]['referenced_episodes_id'] else []
        shared_themes = ep_x_clusters.intersection(ep_y_clusters)
        union_themes = ep_x_clusters.union(ep_y_clusters)
        edge_weight = len(shared_themes) / len(union_themes) if len(union_themes) > 0 else 0
        if len(shared_themes) > weight_threshold:
            edges_data += ((ep_x["insights"]['episode_id'], ep_y["insights"]['episode_id'], edge_weight),)
        elif ep_x["insights"]['episode_id'] in references_y or ep_y["insights"]['episode_id'] in references_x:
            referenced_edges += ((ep_x["insights"]['episode_id'], ep_y["insights"]['episode_id'], 1),)            
    
    G.add_weighted_edges_from(edges_data,attr="themes")
    G.add_weighted_edges_from(referenced_edges, attr="references")
    # -------------------------------
    # 3) Identify century Buckets
    # -------------------------------
    unique_centuries = sorted(
        {century for century in nx.get_node_attributes(G, 'century').values() if century is not None}
    )
    global_positions = {}
    embedding_positions = {node: np.array(G.nodes[node]["embedding"]).mean(axis=0) for node in  G.nodes()}
    if timeline:
        use_y_embedding = True
        scale = SCALE
        x_positions = {century: century * scale for century in unique_centuries}

        for century in unique_centuries:
            nodes_in_century = [n for n in G.nodes if G.nodes[n].get('century') == century]
            G_sub = G.subgraph(nodes_in_century).copy()

            if len(G_sub.nodes) == 1:
                # Place single nodes directly
                if use_y_embedding:
                    node = list(G_sub.nodes())[0]
                    pos_y = embedding_positions[node][0]*scale
                else:
                    pos_y = 0 
                global_positions[nodes_in_century[0]] = (x_positions[century], pos_y)
                
            else:
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
                    if use_y_embedding:
                        pos_y = embedding_positions[node][0] *scale
                    else:
                        pos_y = raw_y + shift_y + jitter()
                    global_positions[node] = (raw_x + shift_x + jitter(), pos_y)

    else:
        global_positions = embedding_positions
        #global_positions = nx.spring_layout(G, seed=42, k=10)

        


    return G, global_positions, relevant_clusters



HIGHLIGHT_COLOR = "#d62728"
BACKGROUND_COLOR = "rgba(138, 138, 138, 0.2)"
DEFAULT_NODE_COLOR = "rgba(138, 138, 138, 0.8)"

HOVERTEMPLATE = (
                    "<b>%{customdata[0]}</b><br>"
                    "Custers: %{customdata[3]}<br>"
                    "Themes: %{customdata[1]}<br>"
                    "<i>Century: %{customdata[2]}</i><br>"
                    "<extra></extra>"
                )

def create_figure(G, global_positions, clusters):
    fig = go.Figure()
    #fig = make_subplots(specs=[[{"secondary_y": True}]])
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
                cluster_edges_indices[sc].add(i+offset)
                cluster_edges_indices[sc].add(i+ 1 + offset)

        offset += 1

    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(color=BACKGROUND_COLOR, width=1),
            hoverinfo='none',
            visible=True,
            name="Edges"
        ),
        #secondary_y=True,
    )

    index_offset = 0 # len(G.edges())
    edge_range = (0, index_offset-1)
    node_range = (index_offset, index_offset+len(G.nodes()))
    index_ranges = dict(edges=edge_range,nodes=node_range)
    nodes_x = []
    nodes_y = []
    metadata_list = []
    for j, node_name in enumerate(G.nodes()):
        node = G.nodes[node_name].copy()
        x, y = global_positions[node_name]
        nodes_x.append(x)
        nodes_y.append(y)
        metadata_list.append([node["title"], ",".join(node["on_click"]["themes"]),node["century"], node["cluster"], node["on_click"]])

        for c in node["cluster"]:
            # not all clusters passed on to selector
            if c in cluster_node_indices:
                cluster_node_indices[c].add(index_offset+j)
            

    ### Nodes
    fig.add_trace(
        go.Scatter(
            x=nodes_x,
            y=nodes_y,
            mode='markers',
            visible=True,
            marker=dict(
                size=12,
                color=DEFAULT_NODE_COLOR,
                line=dict(color="black",width=1)
            ),
            customdata=metadata_list,  # Store node_text in customdata
            hoverinfo='none',
            hovertemplate=HOVERTEMPLATE,
            name="Nodes"
        ),
        #secondary_y=True,
    )


    
    fig.update_layout(
        title="<b>Podcast Episode Network</b>",
        #title_font=dict(size=20, color="#333"),  # Dark gray for a clean look
        showlegend=False,
        xaxis=dict(showgrid=True, zeroline=True, visible=True),
        yaxis=dict(showgrid=True, zeroline=False, visible=True),
        #margin=dict(l=20, r=20, t=10, b=10),
        #plot_bgcolor="#F0F2F6",  # Matches Streamlit's default background
        #paper_bgcolor="#F0F2F6",  # Ensures smooth blending
    )

    
    return fig, cluster_edges_indices, cluster_node_indices, index_ranges


def update_figure(fig, selected_cluster, cluster_data, timeline):
    if timeline:
        fig.update_xaxes(title_text='Century')
    if (selected_cluster == "All") or (selected_cluster not in cluster_data["cluster_edge_indices"]):
        return fig
    else:
        # update edges
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
                mode='lines',
                line=dict(color=HIGHLIGHT_COLOR, width=2),
                hoverinfo='none',
                visible=True,
                name="Edges-Highlight"
                ),
            #secondary_y=False,
        )

        # update nodes
        node_indices = cluster_data["cluster_node_indices"][selected_cluster]
        nodes_x = []
        nodes_y = []
        nodes_customdata = [] 
        for idx in node_indices:          
            #fig.data[idx].marker.update(dict(size=20, color=HIGHLIGHT_COLOR))
            nodes_x.append(fig.data[1].x[idx])
            nodes_y.append(fig.data[1].y[idx])
            nodes_customdata.append(fig.data[1].customdata[idx])

        #fig.update_traces(hovertemplate="Name: %{customdata[0]}")
        fig.update_traces(hovertemplate=None)
        fig.add_trace(
            go.Scatter(
                x=nodes_x,
                y=nodes_y,
                mode='markers',
                visible=True,
                marker=dict(
                    size=18,
                    color=HIGHLIGHT_COLOR,
                    line=dict(color="black",width=1)
                ),
                customdata=nodes_customdata,  # Store node_text in customdata
                #hoverinfo='none',
                hovertemplate=HOVERTEMPLATE,
                name="Nodes-Highlight"
            ),
            #secondary_y=False
        )
        if timeline:
            x_min = max([min(nodes_x) - 5*SCALE, min(fig.data[1].x)-SCALE])
            x_max = min([max(nodes_x) + 5*SCALE, max(fig.data[1].x)+SCALE])
            #fig.update_layout(xaxis_range=[x_min, x_max])
            fig.update_xaxes(title_text='Century', range=[x_min, x_max])
        else:
            x_std = np.array(fig.data[1].x).std()
            x_min = max([min(nodes_x) - x_std, min(fig.data[1].x)-(x_std/2)])
            x_max = min([max(nodes_x) + x_std, max(fig.data[1].x)+(x_std/2)])
            
            y_std = np.array(fig.data[1].y).std()
            y_min = max([min(nodes_y) - y_std, min(fig.data[1].y)-(y_std/2)])
            y_max = min([max(nodes_y) + y_std, max(fig.data[1].y)+(y_std/2)])
            fig.update_layout(xaxis_range=[x_min, x_max], yaxis_range=[y_min, y_max])
        return fig
    