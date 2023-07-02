from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


class Tf_Idf:
    def __init__(self, documents):
        self.documents = documents
    """
    Tf-Idf Transformer.

    This class provides a method to transform a list of documents into a Tf-Idf representation.

    Args:
        documents (list): A list of strings representing the documents or partitions to transform.
        vectorizer (sklearn.feature_extraction.text.TfidfVectorizer, optional): An instance of a TfidfVectorizer
            class from scikit-learn. If not provided, a default TfidfVectorizer will be used.

    Returns:
        pandas.DataFrame: A DataFrame representing the Tf-Idf representation of the input documents.

    Example:
        >>> documents = ["This is the first document.", "This document is the second document."]
        >>> tfidf_transformer = Tf_Idf()
        >>> tfidf_transformer.tf_idf(documents)
           document     first        is    second       the      this
        0   0.417021  0.417021  0.417021  0.000000  0.417021  0.417021
        1   0.851558  0.000000  0.425779  0.851558  0.425779  0.425779
    """

    def tf_idf(self, vectorizer=None):
        if vectorizer is None:
            vectorizer = TfidfVectorizer()

        tf_idf_out = vectorizer.fit_transform(self.documents)
        feature_names = vectorizer.get_feature_names_out()
        dense = tf_idf_out.todense()
        dense_list = dense.tolist()
        df = pd.DataFrame(dense_list, columns=feature_names)
        return df
