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

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from preprocessing.dimensionality_reduction import DR
from modeling.knn import Knn
from modeling.nb import NaiveBayes

dr = DR()
n_cmops = [2, 3, 4, 5, 6, 7, 8, 9, 10]
f1_score_pca_knn = []
f1_score_autoencoder_knn = []
f1_score_pca_nb = []
f1_score_autoencoder_nb = []

for i in n_cmops:
    transformed_pca_train_data = dr.create_pca_model(X_train, i, 42)
    transformed_pca_test_data = dr.create_pca_model(X_test, i, 42)
    transformed_autoencoder_train_data = dr.create_autoencoder_model(X_train, i)
    transformed_autoencoder_test_data = dr.create_autoencoder_model(X_test, i)
    knn_model_pca = Knn(transformed_pca_train_data, y_train, transformed_pca_test_data)
    knn_model_autoencoder = Knn(transformed_autoencoder_train_data, y_train, transformed_pca_test_data)
    naive_bays_model_pca = NaiveBayes(transformed_pca_train_data, y_train, transformed_autoencoder_test_data)
    naive_bays_model_autoencoder = NaiveBayes(transformed_autoencoder_train_data, y_train, transformed_autoencoder_test_data)

    y_pred_knn_pca = knn_model_pca.knn(2)
    classification_report_knn_pca = classification_report(y_test, y_pred_knn_pca, output_dict=True)
    f1_score_pca_knn.append(classification_report_knn_pca["macro avg"]["f1-score"])

    y_pred_knn_autoencoder = knn_model_autoencoder.knn(2)
    classification_report_knn_autoencoder = classification_report(y_test, y_pred_knn_autoencoder, output_dict=True)
    f1_score_autoencoder_knn.append(classification_report_knn_autoencoder["macro avg"]["f1-score"])

    y_pred_nb_pca = naive_bays_model_pca.gaussianNB()
    classification_report_nb_pca = classification_report(y_test, y_pred_nb_pca, output_dict=True)
    f1_score_pca_nb.append(classification_report_nb_pca["macro avg"]["f1-score"])

    y_pred_nb_autoencoder = naive_bays_model_autoencoder.gaussianNB()
    classification_report_nb_autoencoder = classification_report(y_test, y_pred_nb_autoencoder, output_dict=True)
    f1_score_autoencoder_nb.append(classification_report_nb_autoencoder["macro avg"]["f1-score"])

    print("PCA - KNN - Classification Report (n_components = {})".format(i))
    print(classification_report(y_test, y_pred_knn_pca))
    print("Autoencoder - KNN - Classification Report (n_components = {})".format(i))
    print(classification_report(y_test, y_pred_knn_autoencoder))
    print("PCA - Naive Bayes - Classification Report (n_components = {})".format(i))
    print(classification_report(y_test, y_pred_nb_pca))
    print("Autoencoder - Naive Bayes - Classification Report (n_components = {})".format(i))
    print(classification_report(y_test, y_pred_nb_autoencoder))

plt.plot(n_cmops, f1_score_pca_knn, label="PCA - KNN")
plt.plot(n_cmops, f1_score_autoencoder_knn, label="Autoencoder - KNN")
plt.plot(n_cmops, f1_score_pca_nb, label="PCA - Naive Bayes")
plt.plot(n_cmops, f1_score_autoencoder_nb, label="Autoencoder - Naive Bayes")
plt.xlabel("Number of components")
plt.ylabel("F1 score")
plt.legend()
plt.show()

# ---------- Visualization--------------

# ---------- Evaluation--------------
evaluation_NB = Evaluation(y_test, y_test_pred)
evaluation_KNN = Evaluation(y_test, MultinomialNB_predictions)
evaluation_Ber = Evaluation(y_test, y_predict)

evaluation_NB.conf_matrix("Multinomial Naive Bayes Classifiers for Part 2 (A)")
evaluation_KNN.conf_matrix("KNN")
evaluation_Ber.conf_matrix("evaluation_Ber")


