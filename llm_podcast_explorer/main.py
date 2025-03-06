import dash
import dash_cytoscape as cyto

from dash.dependencies import Input, Output, State
from dash import Dash, html, Input, Output, callback, dcc
import os
from flask_caching import Cache
import json
from rss_feed_analyzer import RSSFeedAnalyzer
from network_viz import build_network_graph


# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

LOGGER = app.logger

# Configure cache
cache = Cache(app.server, config={"CACHE_TYPE": "simple"})


default_stylesheet = [{"selector": "node", "style": {"background-color": "#BFD7B5", "label": "data(label)"}}]

# Layout of the app
app.layout = html.Div([
    dcc.Input(id="url-input", type="text", placeholder="Enter RSS Feed URL"),
    html.Button("Submit", id="submit-button", n_clicks=0),
    cyto.Cytoscape(
        id="cytoscape-graph",
        layout={"name": "cose"},
        style={"width": "100%", "height": "600px"},
        # stylesheet=default_stylesheet,
        elements=[],
    ),
    html.P(id="cytoscape-tapNodeData-output"),
    html.P(id="cytoscape-tapEdgeData-output"),
    html.P(id="cytoscape-mouseoverNodeData-output"),
    html.P(id="cytoscape-mouseoverEdgeData-output"),
])


@callback(Output("cytoscape-tapNodeData-output", "children"), Input("cytoscape-graph", "tapNodeData"))
def displayTapNodeData(data):
    LOGGER.warning("display node")
    if data:
        return f"Episode Info: {data['insights']}"


@callback(Output("cytoscape-tapEdgeData-output", "children"), Input("cytoscape-graph", "tapEdgeData"))
def displayTapEdgeData(data):
    LOGGER.warning("display edge")
    if data:
        return (
            "You recently clicked/tapped the edge between " + data["source"].upper() + " and " + data["target"].upper()
        )


@callback(Output("cytoscape-mouseoverNodeData-output", "children"), Input("cytoscape-graph", "mouseoverNodeData"))
def displayTapNodeData(data):
    LOGGER.warning("tap node")
    if data:
        return f"Episode Info: {data['insights']}"


@callback(Output("cytoscape-mouseoverEdgeData-output", "children"), Input("cytoscape-graph", "mouseoverEdgeData"))
def displayTapEdgeData(data):
    LOGGER.warning("tap edge")
    if data:
        return "You recently hovered over the edge between " + data["source"].upper() + " and " + data["target"].upper()


# Cache the analyzer output
@cache.memoize(timeout=300)
def analyze_feed(url):
    llm_api_key = os.environ.get("OPENAI_API_KEY")
    if llm_api_key is None:
        raise ValueError("Please provide an OpenAI API Key. Set it as Environment Variable 'OPENAI_API_KEY'")
    analyzer = RSSFeedAnalyzer(url, llm_api_key=llm_api_key, warmup=False, logger=LOGGER)
    data = analyzer.run(limit=100)
    with open("data.json", "w") as f:
        f.write(data.model_dump_json(indent=4))
    return data


# Update the graph based on the input URL
@app.callback(
    Output("cytoscape-graph", "elements"), [Input("submit-button", "n_clicks")], [State("url-input", "value")]
)
def update_graph(n_clicks, url):
    LOGGER.warning(f"Button {n_clicks}")
    if n_clicks > 0 and url is not None:
        LOGGER.warning(f"Loading data for {url}")
        # data = analyze_feed(url)
        with open("data.json", "r") as f:
            data = json.load(f)
        LOGGER.warning("Preparing Graph")
        nodes, edges = build_network_graph(data["episodes"])
        elements = nodes + edges
        return elements
    return []


if __name__ == "__main__":
    app.run(debug=False)
