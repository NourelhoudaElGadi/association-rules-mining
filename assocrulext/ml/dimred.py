from scipy.sparse import coo_matrix
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE, Isomap, MDS, LocallyLinearEmbedding
import pandas as pd

import umap

from assocrulext.ml.autoencoder import AutoEncoderDimensionalityReduction


# Reduction of the number of variables + Clustering
# The autoencoder allows to reduce the dimension and to be able to apply the CAH which is not robust in the face of too many variables
def dimensionality_reduction(one_hot_matrix, n_components, method, score=None):
    methods = {
        "autoencoder": AutoEncoderDimensionalityReduction(
            input_dim=one_hot_matrix.shape[1],
            encoding_dim=n_components,
            epochs=100,
            batch_size=128,
            learning_rate=1e-1,
            novelty_score=score,
        ),
        "pca": PCA(n_components=n_components),
        "tsne": TSNE(
            n_components=n_components,
            method="exact",
            perplexity=30,
            learning_rate=200,
            n_iter=1000,
        ),
        "umap": umap.UMAP(n_components=n_components),
        "isomap": Isomap(n_neighbors=5, n_components=n_components),
        "mds": MDS(n_components=n_components),
        "ica": FastICA(n_components=n_components),
        "lle": LocallyLinearEmbedding(n_components=n_components),
    }

    # The above code is implementing dimensionality reduction techniques such as autoencoder, PCA, ICA,
    # LLE, etc. on a given input matrix `one_hot_matrix`. If the chosen method is autoencoder, it creates
    # an autoencoder model using Keras and trains it on the input matrix. If the chosen method is any
    # other dimensionality reduction technique, it applies the chosen method on the input matrix and
    # returns the reduced data in a pandas DataFrame format. The number of components for the
    # dimensionality reduction is specified by the `n_components` parameter. The resulting reduced data is
    # returned with column names based

    # Dimensionality reduction
    reducer = methods[method]
    reduced_data = pd.DataFrame(reducer.fit_transform(one_hot_matrix.to_numpy()))

    # Define column names for the reduced data
    reduced_data.columns = [f"{method}_{str(i + 1)}" for i in range(n_components)]
    reduced_data.index = one_hot_matrix.index

    return reduced_data
