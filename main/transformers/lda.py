from gensim.models import LdaModel
from gensim.corpora import Dictionary
import numpy as np


class Lda:
    def __init__(self, documents):
        self.documents = documents

    def apply_lda(self, num_topics, preprocess_fn=None, pad_value=0, **kwargs):
        """
        Apply LDA topic modeling to a list of documents.

        This function preprocesses the text data, applies LDA topic modeling, and pads the topic distributions.

        Args:
            documents (list): A list of documents to apply LDA on.
            num_topics (int): The number of topics to discover.
            preprocess_fn (callable, optional): A function to preprocess the text data. If not provided, the default
                preprocessing function splits the text on whitespace. The function should accept a string as input and
                return a list of preprocessed tokens.
            pad_value (int, optional): The value used for padding the topic distributions. Defaults to 0.
            **kwargs: Additional keyword arguments to be passed to the LdaModel constructor.

        Returns:
            numpy.ndarray: An array representing the padded topic distributions.

        Example:
            >>> documents = ["This is the first document.", "This document is the second document."]
            >>> topics = apply_lda(documents, num_topics=5)
            >>> print(topics)
            [[0.2, 0.1, 0.4, 0.2, 0.1, 0.0],
             [0.1, 0.3, 0.2, 0.2, 0.1, 0.1]]
        """
        if preprocess_fn is None:
            preprocess_fn = lambda text: text.split()

        preprocessed_data = [preprocess_fn(doc) for doc in self.documents]
        dictionary = Dictionary(preprocessed_data)
        corpus = [dictionary.doc2bow(text) for text in preprocessed_data]
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, **kwargs)

        document_topics = []
        for doc in corpus:
            topic_dist = lda_model.get_document_topics(doc)
            document_topics.append([topic[1] for topic in topic_dist])
        max_length = max(len(seq) for seq in document_topics)

        padded_sequences = [seq + [pad_value] * (max_length - len(seq)) for seq in document_topics]

        X = np.array(padded_sequences)
        return X
