# NOTE: Author - Hamza Idelcaid
# NOTE: The file contains different functions, needed to prepare data for
#       SVM classification model .

# TODO: Select relevant data from the csv file -> Yes
# TODO: Apply Quantile Normalization on numerical data of csv -> Yes
# TODO: ...

# Avoiding sklearn validation warnings to have only results in terminal. . .
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from plot_svm import make_meshgrid, plot_contours

import numpy as np
import pandas as ps
from math import sqrt
from scipy import stats

from sklearn import svm
from sklearn.utils.validation import column_or_1d

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


# Global variables
C = 1.0
K_FOLD = 5
TEST_SET_SIZE = .3
MODELS = {
    "SVM": svm.SVC(kernel='linear', C=C),
    "DTC": DecisionTreeClassifier(random_state=0,),
    "RFC": RandomForestClassifier(max_depth=2, random_state=0),
}

def quantilenormalize(df_input):
    """
        Return new matrix (df) of Quantile Normalization values of (df_input)
        origin matrix, using pandas and numpy functions .
    """
    df = df_input.copy()
    # compute rank
    dic = {}
    for col in df:
        dic.update({col: sorted(df[col])})
    sorted_df = ps.DataFrame(dic)
    rank = sorted_df.mean(axis=1).tolist()
    # sort
    for col in df:
        t = np.searchsorted(np.sort(df[col]), df[col])
        df[col] = [rank[i] for i in t]
    return df

def t_test(data_1, data_2):

    """
        This function calculates t_value and p_value from two datasets .
        It returns (t_value and a p_value)
    """

    # Calculate t_value from this formula : abs(mean(data_1) - mean(data_2)) / sqrt((s^2_1/n_1)+(s^2_2/n_2))
    # where: s^2_1 and s^2_2 are respectively variances of data_1 and data_2, and n_(1 or 2)
    # are respectively number of elements in data_(1 or 2)

    # print(data_1)
    # first value of numerator
    mean_1 = np.average(data_1)
    mean_2 = np.average(data_2)
    # print(mean_1, mean_2)
    numerator = abs(mean_1 - mean_2)

    # now the denominator
    var_1 = np.var(data_1, ddof=1)
    var_2 = np.var(data_2, ddof=1)
    n_1 = len(data_1)
    n_2 = len(data_2)
    # denominator = sqrt((var_1/n_1)+(var_2/n_2))
    denominator = sqrt((var_1+var_2)/(n_2))

    # ...then the t_value
    t_value = numerator/denominator

    # now the p_value...
    # degrees of freedom
    df = 2*n_1 - 2

    #p-value after comparison with the t_value
    p_value = 2*(1 - stats.t.cdf(t_value, df=df))

    # print("---------@@line@@---------")
    # print("t_value = " + str(t_value))
    # print("p_value = " + str(p_value))

    return (t_value, p_value)

def filter_data(data):

    # working with data copy...
    data_f = data.copy()

    # add two columns for t_ and p_ values
    data_f["t_value"] = np.nan
    data_f["p_value"] = np.nan

    # all 0H data list
    probes_0H = data_f.iloc[:, 1:208]


    # all 6H data list
    probes_6H = data_f.iloc[:, 208:415]

    for i in range(len(probes_0H)):
        t_value , p_value = t_test(probes_0H.iloc[i], probes_6H.iloc[i])
        if p_value <= 0.05:
            data_f.set_value(i, 't_value', t_value)
            data_f.set_value(i, 'p_value', p_value)

    final_data = data_f.dropna(how='any').reset_index(drop=True)

    return final_data

def get_X_y_data(data, test_set_size):
    X = []
    y = []

    for j,col in enumerate(data.columns):
        X.append(data[col].tolist())
        if j >= 0 and j<207:
            y.append(0)
        elif j>=207:
            y.append(1)

    y = np.array(y)

    # Binarize the output
    y = label_binarize(y, classes=[0,1])
    n_classes = y.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size,
                                                    random_state=0)

    return (X_train, X_test, y_train, y_test, n_classes)

def training_models(X_train, y_train, X_test, y_test, k_fold):

    # fitting the models with training data and getting cross_val_score
    for key in MODELS:
        MODELS[key] = MODELS[key].fit(X_train, y_train)
        model = MODELS[key]
        scores = cross_val_score(model, X_train, y_train, cv=k_fold)
        print("Results for "+key+" classifier:")
        print("\t- Cross Validation Score with "+str(k_fold)+"-Fold")
        print("\t\t- Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print("\t\t- Score(X_test, y_test) = "+str(model.score(X_test, y_test)))


def roc_graph(model, X_test, y_test, n_classes):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_score = model.decision_function(X_test)
    y_score = [[elem] for elem in y_score]
    y_score = np.array(y_score)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    # The graph of ROC with matplotlib
    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def svm_graph(X, Y):
    """"we create an instance of SVM and fit out data. We do not scale our
    data since we want to plot the support vectors"""
    C = 1.0  # SVM regularization parameter
    models = (svm.SVC(kernel='linear', C=C),
              # svm.LinearSVC(C=C),
              svm.SVC(kernel='rbf', gamma=0.7, C=C),
              svm.SVC(kernel='poly', degree=3, C=C))
    models = (clf.fit(X_train, y_train) for clf in models)

    # title for the plots
    titles = ('SVC with linear kernel',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel')

    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    X0, X1 = X_train[:, 0], X_train[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        # plot_contours(ax, clf, xx, yy,
        #               cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y_train.ravel(), cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('0H after radiation')
        ax.set_ylabel('6H after radiation')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)

    plt.show()



def main():
    df = ps.read_csv(
        "/Users/idelhamza/Desktop/Machine_Learning/MiniMemoire/datasets/dataset.csv")
    data = df[[col for col in df.columns if ("2hr" not in col)]]

    probes_names = data.iloc[:, :1]
    numeric_data = data.iloc[:, 6:].fillna(0)

    # Normalized numeric data
    normalized_numeric_data = quantilenormalize(numeric_data)

    # filtered probes and numeric data
    data_to_filter = probes_names.join(normalized_numeric_data)
    filtered_data = filter_data(data_to_filter)

    # Filtered training and test datasets
    dataset = filtered_data.iloc[:, 1:415]

    # Getting training and test datasets
    # Takes a dataset and test data size as argument
    X_train, X_test, y_train, y_test, n_classes = get_X_y_data(dataset, TEST_SET_SIZE)

    # Training and test process with a K-Fold value given in argument
    training_models(X_train, y_train, X_test, y_test, K_FOLD)


    # Plot of SVM Classifiers. . .
    # svm_graph(np.array(X_train), y_train)

    # Plot of ROC. . .
    # model = MODELS["SVM"].fit(X_train, y_train)
    # roc_graph(model, X_test, y_test, n_classes)

if __name__ == '__main__':
    main()
