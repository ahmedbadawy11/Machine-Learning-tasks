from sklearn.decomposition import PCA


class DR:
    def __init__(self):
        ...

    def pca(self, data, n_components=0, random_state=0):
        pca = PCA(n_components=n_components, random_state=random_state)
        return pca.fit_transform(data)
