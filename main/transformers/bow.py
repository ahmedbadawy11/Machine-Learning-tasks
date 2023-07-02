from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


class Bow:
    def __init__(self, documents):
        self.documents = documents

    """
    Bag-of-Words Transformer.

    This class provides a method to transform a list of documents into a bag-of-words representation.

    Args:
        documents (list): A list of strings representing the documents or partitions to transform.
        vectorizer (sklearn.feature_extraction.text.VectorizerMixin, optional): An instance of a vectorizer class
            from scikit-learn. If not provided, a CountVectorizer will be used by default.

    Returns:
        pandas.DataFrame: A DataFrame representing the bag-of-words representation of the input documents.

    Example:
        >>> documents = ["This is the first document.", "This document is the second document."]
        >>> bow_transformer = Bow()
        >>> bow_transformer.bow(documents)
           document  first  is  second  the  this
        0         1      1   1       0    1     1
        1         2      0   1       1    1     1
    """

    def bow(self, vectorizer=None):
        if vectorizer is None:
            vectorizer = CountVectorizer()

        X = vectorizer.fit_transform(self.documents)
        self.df_bow = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        return self.df_bow
