from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB


class NaiveBayes:
    def __init__(self, X_train, Y_train, X_test):
        self.x_train = X_train
        self.y_train = Y_train
        self.x_test = X_test

    def naiveBayesModels(self, classifier):
        classifier.fit(self.x_train, self.y_train)
        classifier_predictions = classifier.predict(self.x_test)
        return classifier_predictions

    def gaussianNB(self):
        y_predict = self.naiveBayesModels(GaussianNB)
        return y_predict

    def multinomialNB(self):
        y_predict = self.naiveBayesModels(MultinomialNB)
        return y_predict

    def complementNB(self):
        y_predict = self.naiveBayesModels(ComplementNB)
        return y_predict

    def bernoulliNB(self):
        y_predict = self.naiveBayesModels(BernoulliNB)
        return y_predict
