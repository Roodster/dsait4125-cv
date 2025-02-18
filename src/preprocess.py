from sklearn.base import BaseEstimator, TransformerMixin

class ExampleTransformer(BaseEstimator, TransformerMixin):
    """
    Example class to transform data
    Returns transformed data.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data, labels = X
        return (data, labels)
    