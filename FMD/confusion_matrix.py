from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


#confusion matrix
def show_confusion_matrix(testY, prediction):
    matrix = metrics.confusion_matrix(testY, prediction)
    plt.figure(figsize=(6, 4))
    labels = ['With Mask, Without Mask']
    sns.heatmap(matrix,
        cmap='coolwarm',
        linecolor='white',
        linewidths=1,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()