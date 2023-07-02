from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


class N_Gram:
    def __init__(self, documents):
        self.documents = documents

    def n_gram(self, n, vectorizer=None):
        if vectorizer is None:
            vectorizer = CountVectorizer(ngram_range=(n, n))

        ngram_vectors = vectorizer.fit_transform(self.documents)
        feature_names = vectorizer.get_feature_names_out()
        df_ngram = pd.DataFrame(ngram_vectors.toarray(), columns=feature_names)

        return df_ngram
