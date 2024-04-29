import numpy as np


# COLOUR_DICT = ['#ffff00', '#1ce6ff', '#ff34ff', '#ff4a46', '#008941', '#006fa6', '#a30059', '#ffdbe5', '#7a4900', '#0000a6', '#63ffac', '#b79762', '#004d43', '#8fb0ff', '#997d87', '#5a0007', '#809693', '#6a3a4c', '#1b4400', '#4fc601', '#3b5dff', '#4a3b53', '#ff2f80', '#61615a', '#ba0900', '#6b7900', '#00c2a0', '#ffaa92', '#ff90c9', '#b903aa']
COLOUR_DICT = ['#ffff00', '#1ce6ff', '#ff34ff', '#ff4a46', '#008941', '#006fa6', '#a30059', '#ffdbe5', '#7a4900', '#0000a6', '#63ffac', '#b79762', '#004d43', '#8fb0ff', '#997d87', '#5a0007', '#809693', '#6a3a4c', '#1b4400', '#4fc601', '#3b5dff', '#4a3b53', '#ff2f80', '#61615a', '#ba0900', '#6b7900', '#00c2a0', '#ffaa92', '#ff90c9', '#b903aa', '#d16100', '#ddefff', '#000035', '#7b4f4b']
# COLOUR_DICT = [
#     "#e6194b",
#     "#3cb44b",
#     "#ffe119",
#     "#4363d8",
#     "#f58231",
#     "#911eb4",
#     "#46f0f0",
#     "#f032e6",
#     "#fabebe",
#     "#008080",
#     "#e6beff",
#     "#9a6324",
#     "#fffac8",
#     "#800000",
#     "#aaffc3",
#     "#808000",
#     "#000075",
#     "#ffd8b1",
#     "#000000",
#     "#808080",
#     "grey",
#     "purple",
#     "yellow",
#     "dark gray",
#     "dark orchid",
#     "light pink",
#     "light grey",
#     "light green",
#     "light sky blue",
#     "navy",
#     "orange",
#     "orchid",
#     "silver",
#     "snow",
#     "sea green",
#     "wheat",
#     "peru",
#     "medium blue",
#     "lime",
#     "magenta",
#     "indigo",
#     "ivory",
#     "honeydew",
#     "green yellow",
#     "fuchsia",
#     "gold",
#     "dark orange",
#     "dark green",
#     "dark red",
#     "dark magenta",
#     "dark salmon",
#     "dark khaki",
#     "cyan",
#     "coral",
#     "azure",
#     "aqua",
#     "aliceblue",
#     "light salmon",
#     "light cyan",
#     "light steel blue",
#     "lightskyblue",
#     "lightyellow",
#     "#ffffff",
# ]

def update_graph_labels(graph, labels):
    """Update labels(bin_id) to graph and add colours list to
    graph object.

    Args:
        graph (igraph.graph): graph object.
        labels (2D list): labels of nodes.

    Returns:
        image (np.ndarray): image to logging.
    """

    assert len(np.unique(labels)) <= len(COLOUR_DICT)
    graph.vs["color"] = [COLOUR_DICT[i] for i in labels]
    # return graph
