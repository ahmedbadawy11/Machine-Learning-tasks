from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from decorator import timer_decorator


class Evaluation:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @timer_decorator
    def conf_matrix(self, title, show_report=False):
        cm = confusion_matrix(self.x, self.y)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Oranges", cbar=False)
        plt.title(f'Confusion Matrix - {title} Data')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.show()
        if not show_report:
            print("Accuracy: ", accuracy_score(self.x, self.y))

        if show_report:
            report = classification_report(self.x, self.y)
            print("Classification Report:")
            print(report)
            plt.show()
