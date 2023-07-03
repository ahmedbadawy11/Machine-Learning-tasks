from knn import Knn
from nb import NaiveBayes


class ClassificationModel(Knn, NaiveBayes):
    def __init__(self, X_train, Y_train, X_test, X_valid=None):
        super().__init__(X_train=X_train, Y_train=Y_train, X_test=X_test, X_valid=X_valid)
