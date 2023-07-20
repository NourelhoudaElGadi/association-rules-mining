### Community detection algorithm (Walk Trap)
import numpy as np
from assocrulext.clustering.cooc import coocc_matrix_Label

import networkx as nx
from cdlib import algorithms


def walk_trap(one_hot_label):
    ## Co-occurrence matrix
    coooc_s = coocc_matrix_Label(one_hot_label)

    ## Creating tuples with co-occurrence frequencies higher than 0##
    labels = one_hot_label.columns
    tuple_list = []
    for i in range(len(coooc_s)):
        no_zero = np.where(coooc_s[i] != 0)
        tuple_list.extend([labels[i], labels[j], coooc_s[i][j]] for j in no_zero[0])
    ## Create Graph ##
    G = nx.Graph()

    for edge in tuple_list:
        G.add_edge(edge[0], edge[1], weight=edge[2])

    com_wt = algorithms.walktrap(G)
    return com_wt.communities
