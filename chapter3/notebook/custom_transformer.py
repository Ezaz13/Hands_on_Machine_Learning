from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils.validation import validate_data, check_is_fitted

class ClusterSimilarity(BaseEstimator, TransformerMixin):
# class ClusterSimilarity():
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        # Validate the input data and set n_features_in_
        # X = validate_data(X, accept_sparse=False)
        self.n_features_in_ = X.shape[1]  # Explicitly set n_features_in_
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        # Ensure the estimator is fitted before transforming
        # check_is_fitted(self, 'kmeans_')

        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]