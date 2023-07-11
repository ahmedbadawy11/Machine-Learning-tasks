<<<<<<< Updated upstream
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
=======
from Assignment_sections.Part_4 import clustering_Kmeans,clustering_SOFM,clustering_DBSCAN
from Assignment_sections.Part_3 import Part_3_Filter_Methods



from sklearn.metrics import accuracy_score

from Assignment_sections.Part_3 import Part_3_Filter_Methods
from Assignment_sections.Part_4 import clustering_Kmeans, clustering_SOFM, clustering_DBSCAN
from evaluation.confusion import Evaluation

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, mutual_info_classif

import pandas as pd
from modeling.classification_models import ClassificationModel
from preprocessing.dimensionality_reduction import DR
from preprocessing.read_data import ReadData, Format
from preprocessing.Feature_Selection import select_feature, apply_wrapper_RFECV_methods, \
    apply_wrapper_feature_elimination_methods
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB, GaussianNB
import matplotlib.pyplot as plt

from visualization.tsne import draw_TSNE, draw_TSNE_method_2
from visualization.plot import Draw_plots

from sklearn.exceptions import UndefinedMetricWarning

import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
# ---------- Preprocessing--------------
read_data = ReadData()
dr = DR()
df = read_data.read_data_from_local_path("MCSDatasetNEXTCONLab.csv", Format.CSV)
print(df.head(5))
print(df['Day'].value_counts())
print(df.shape)

train_dataset = df[df['Day'].isin([0, 1, 2])]
test_dataset = df[df['Day'] == 3]

# print (train_dataset.shape)
# print (test_dataset.shape)
train_dataset = train_dataset.drop(['ID', 'Day'], axis=1)
test_dataset = test_dataset.drop(['ID', 'Day'], axis=1)


print("train set Shape",train_dataset.shape)

X_train = train_dataset.drop('Ligitimacy', axis=1)
y_train = train_dataset['Ligitimacy']

X_test = test_dataset.drop('Ligitimacy', axis=1)
y_test = test_dataset['Ligitimacy']

print(df['Ligitimacy'].value_counts())

latitude = df['Latitude']
longitude = df['Longitude']

# Filter legitimate data
legitimate_data = df[df['Ligitimacy'] == 1]

# ------------------------ Modeling--------------------------------------------

modeling = ClassificationModel(X_train, y_train, X_test)
y_test_pred = modeling.knn(2)
gaussianNB_predictions = modeling.gaussianNB()

# y_predict=modeling.bernoulliNB()
# ---------- Visualization--------------
# draw_TSNE(X_train,y_train,"Training Set")
# draw_TSNE(X_test,y_test,"Test Set")

# draw_TSNE_method_2(X_train,y_train)
# ---------- Evaluation--------------
evaluation_NB = Evaluation(y_test, gaussianNB_predictions)
evaluation_KNN = Evaluation(y_test, y_test_pred)
# evaluation_Ber = Evaluation(y_test, y_predict)

evaluation_NB.conf_matrix(" Gaussian Naive Bayes Classifiers for Part 2 (A)")
evaluation_KNN.conf_matrix("KNN")

# evaluation_Ber.conf_matrix("evaluation_Ber")


# part (2) _3
#
# Part_3_Filter_Methods(X_train,y_train, X_test, y_test)

# part (2) _3



#-------------------------------------------------------------------------------------------------------




# part (2) _4


# clustering_Kmeans(X_train,y_train,X_test,df,test_dataset)
# clustering_SOFM(X_train,y_train,X_test,df,train_dataset)
clustering_DBSCAN(X_train,y_train,X_test,df,train_dataset)




#-part (2) _4








>>>>>>> Stashed changes


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
