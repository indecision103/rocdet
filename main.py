from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import sklearn.metrics as metrics
import matplotlib
from matplotlib import pyplot as plt



# Compute micro-average ROC curve and ROC area
y_class = [1,1,0,1,1,1,0,0,1,0,1,0,1,0,0,0,1,0,1,0]
print("y class length: ", len(y_class))
y_scores= [0.9,0.8, 0.7, 0.6, 0.55, 0.54, 0.53, 0.52, 0.51, 0.505, 0.4, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33, 0.30, 0.1]
print("y scores length: ", len(y_scores))

fpr, tpr, threshold = metrics.roc_curve(y_class, y_scores)
roc_auc = metrics.auc(fpr, tpr)
fnr = []
fnr = 1 - tpr
det_auc = metrics.auc(fpr, fnr)

print("fpr : ",fpr)
print("tpr :", tpr)
print("fnr :", fnr)
#print(roc_auc)
def ShowROC() :
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

ShowROC()

def ShowDET(fpr,fnr):
    """
    Given false positive and false negative rates, produce a DET Curve.
    The false positive rate is assumed to be increasing while the false
    negative rate is assumed to be decreasing.
    """
    plt.figure()
    lw = 2
    plt.plot(fpr, fnr, color='red',
             lw=lw, label='DET curve (area = %0.2f)' % det_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


ShowDET(fpr, fnr)