from sklearn.neighbors import KNeighborsClassifier
from enum import Enum


class Metric(Enum):
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    CHEBYSHEV = "chebyshev"


class Knn:
    def __init__(self, X_train, Y_train, X_test, X_valid):
        self.x_train = X_train
        self.y_train = Y_train
        self.x_test = X_test
        self.x_valid = X_valid

    def knn(self, k, metric=Metric.EUCLIDEAN):
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric.value)
        knn.fit(self.x_train, self.y_train, )
        y_test_pred = knn.predict(self.x_test)
        if self.x_valid is not None:
            y_valid_pred = knn.predict(self.x_valid)
            return y_valid_pred, y_test_pred

        return y_test_pred
