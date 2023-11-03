import numpy as np

COLOUR_DICT = [
    "#e6194b", 
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231", 
    "#911eb4", 
    "#46f0f0", 
    "#f032e6", 
    "#fabebe", 
    "#008080", 
    "#e6beff", 
    "#9a6324", 
    "#fffac8", 
    "#800000", 
    "#aaffc3", 
    "#808000", 
    "#000075", 
    "#ffffff", 
    "#ffd8b1", 
    "#000000", 
    "#808080", 
    "grey", 
    "purple", 
    "yellow", 
    "dark gray", 
    "dark orchid", 
    "light pink", 
    "light grey", 
    "light green", 
    "light sky blue", 
    "navy", 
    "orange", 
    "orchid", 
    "silver", 
    "snow", 
    "sea green", 
    "wheat", 
    "peru", 
    "medium blue", 
    "lime", 
    "magenta", 
    "indigo", 
    "ivory", 
    "honeydew", 
    "green yellow", 
    "fuchsia", 
    "gold", 
    "dark orange", 
    "dark green", 
    "dark red", 
    "dark magenta", 
    "dark salmon", 
    "dark khaki", 
    "cyan", 
    "coral", 
    "azure", 
    "aqua", 
    "aliceblue", 
    "light salmon", 
    "light cyan", 
    "light steel blue", 
    "lightskyblue", 
    "lightyellow"
]

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
