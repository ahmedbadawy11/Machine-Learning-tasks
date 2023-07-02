from bow import Bow
from tf_idf import Tf_Idf
from n_gram import N_Gram
from lda import Lda


class Transformers(Bow, Tf_Idf, N_Gram, Lda):
    def __init__(self,document):
        super().__init__(documents=document)


documents = ["This is the first document.", "This document is the second document."]
t = Transformers(documents)
print(t.bow())
print(t.tf_idf())
print(t.n_gram(2))
print(t.apply_lda(3))


# t.bow(["anas", "Ibrahm"])
# t.apply_lda(documents,5)

