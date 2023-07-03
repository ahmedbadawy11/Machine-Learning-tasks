from evaluation.confusion import Evaluation
from modeling.classification_models import ClassificationModel
from preprocessing.dimensionality_reduction import DR
from preprocessing.read_data import ReadData, Format
from sklearn.naive_bayes import MultinomialNB,ComplementNB,BernoulliNB,GaussianNB


# ---------- Preprocessing--------------
read_data = ReadData()
dr = DR()

df = read_data.read_data_from_local_path("MCSDatasetNEXTCONLab.csv", Format.CSV)
print(df.head(10))
print (df['Day'].value_counts())
print (df.shape)

train_dataset = df[df['Day'].isin([0, 1, 2])]
test_dataset = df[df['Day'] == 3]

# print (train_dataset.shape)
# print (test_dataset.shape)
train_dataset = train_dataset.drop(['ID', 'Day'], axis=1)
test_dataset = test_dataset.drop(['ID', 'Day'], axis=1)


X_train = train_dataset.drop('Ligitimacy', axis=1)
y_train = train_dataset['Ligitimacy']

X_test = test_dataset.drop('Ligitimacy', axis=1)
y_test = test_dataset['Ligitimacy']

print (df['Ligitimacy'].value_counts())

# ---------- Modeling--------------
modeling = ClassificationModel(X_train, y_train, X_test)
y_test_pred=modeling.knn(2)
def Naive_Bayes_Models(classifier,X__train,Y__train,X__test):
    classifier.fit(X__train, Y__train)
    classifier_predictions = classifier.predict(X__test)
    return classifier_predictions
MultinomialNB_predictions=Naive_Bayes_Models(GaussianNB(),X_train,y_train,X_test)

y_predict=modeling.bernoulliNB()

# ---------- Visualization--------------

# ---------- Evaluation--------------
evaluation_NB = Evaluation(y_test, y_test_pred)
evaluation_KNN = Evaluation(y_test, MultinomialNB_predictions)
evaluation_Ber = Evaluation(y_test, y_predict)

evaluation_NB.conf_matrix("Multinomial Naive Bayes Classifiers for Part 2 (A)")
evaluation_KNN.conf_matrix("KNN")
evaluation_Ber.conf_matrix("evaluation_Ber")

