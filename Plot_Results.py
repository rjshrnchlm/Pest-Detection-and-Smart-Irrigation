import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from itertools import cycle
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score
import warnings

warnings.filterwarnings('ignore')


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'CWO-ARD-SA', 'TOT-ARD-SA', 'WOA-ARD-SA', 'SSA-ARD-SA', 'GSSA-ARD-SA']

    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
    for j in range(len(Algorithm) - 1):  # for 5 algms
        Conv_Graph[j, :] = Statistical(Fitness[0, j, :])

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- Statistical Analysis  ',
          '--------------------------------------------------')
    print(Table)

    fig = plt.figure()
    fig.canvas.manager.set_window_title('Convergence Curve')
    length = np.arange(Fitness.shape[2])
    Conv_Graph = Fitness[0]
    plt.plot(length, Conv_Graph[0, :], color='#FF69B4', linewidth=3, marker='*', markerfacecolor='red',
             markersize=12, label='CWO-ARD-SA')
    plt.plot(length, Conv_Graph[1, :], color='#7D26CD', linewidth=3, marker='*', markerfacecolor='#00FFFF',
             markersize=12, label='TOT-ARD-SA')
    plt.plot(length, Conv_Graph[2, :], color='#FF00FF', linewidth=3, marker='*', markerfacecolor='blue',
             markersize=12, label='WOA-ARD-SA')
    plt.plot(length, Conv_Graph[3, :], color='#43CD80', linewidth=3, marker='*', markerfacecolor='magenta',
             markersize=12, label='SSA-ARD-SA')
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
             markersize=12, label='GSSA-ARD-SA')
    plt.xlabel('No. of Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    plt.savefig("./Results/Conv.png")
    plt.show()


def Plot_ROC_Curve():
    cls = ['LSTM', 'Faster R-CNN', 'GRU', 'RD-SA', 'GSSA-ARD-SA']
    Actual = np.load('Crop_Target.npy', allow_pickle=True)
    lenper = round(Actual.shape[0] * 0.75)
    Actual = Actual[lenper:, :]
    fig = plt.figure(facecolor='#F9F9F9')
    fig.canvas.manager.set_window_title('ROC Curve')
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
    ax.set_facecolor("#F9F9F9")
    colors = cycle(["blue", "darkorange", "limegreen", "deeppink", "black"])
    for i, color in zip(range(len(cls)), colors):  # For all classifiers
        Predicted = np.load('Y_Score.npy', allow_pickle=True)[i]
        false_positive_rate, true_positive_rate, _ = roc_curve(Actual.ravel(), Predicted.ravel())
        roc_auc = roc_auc_score(Actual.ravel(), Predicted.ravel())
        roc_auc = roc_auc * 100

        plt.plot(
            false_positive_rate,
            true_positive_rate,
            color=color,
            lw=2,
            label=f'{cls[i]} (AUC = {roc_auc:.2f} %)',
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Accuracy')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    path = "./Results/ROC.png"
    plt.savefig(path)
    plt.show()


def Table():
    eval = np.load('Evaluate_all.npy', allow_pickle=True)
    Algorithm = ['No. of Epochs', 'CWO-ARD-SA', 'TOT-ARD-SA', 'WOA-ARD-SA', 'SSA-ARD-SA', 'GSSA-ARD-SA']
    Classifier = ['No. of Epochs', 'LSTM', 'Faster R-CNN', 'GRU', 'RD-SA', 'GSSA-ARD-SA']
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = np.array([0, 2, 4, 6, 9, 15]).astype(int)
    Table_Terms = [0, 2, 4, 6, 9, 15]
    table_terms = [Terms[i] for i in Table_Terms]
    Kfold = [50, 100, 150, 200, 250]
    for i in range(eval.shape[0]):
        for k in range(len(Table_Terms)):
            value = eval[i, :, :, 4:]

            Table = PrettyTable()
            Table.add_column(Algorithm[0], Kfold)
            for j in range(len(Algorithm) - 1):
                Table.add_column(Algorithm[j + 1], value[:, j, Graph_Terms[k]])
            print('-------------------------------', table_terms[k], '  Algorithm Comparison',
                  '---------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Classifier[0], Kfold)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, len(Algorithm) + j - 1, Graph_Terms[k]])
            print('------------------------------- Dataset- ', i + 1, table_terms[k], '  Classifier Comparison',
                  '---------------------------------------')
            print(Table)


def Plots_Results():
    eval = np.load('Evaluates.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 17, 18]
    bar_width = 0.15
    colors = ['#219ebc', '#f77f00', '#8ac926']
    Classifier = ['GRU', 'RD-SA', 'GSSA-ARD-SA']
    Kfold = [1, 2, 3, 4, 5]
    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.15, 0.6, 0.6])
            X = np.arange(len(Kfold))
            ax.barh(X + 0.00, Graph[:, 5], height=0.15, color='darkmagenta', label="LSTM")
            ax.barh(X + 0.15, Graph[:, 6], height=0.15, color='darkorange', label="Faster R-CNN")
            ax.barh(X + 0.30, Graph[:, 7], height=0.15, color='darkslateblue', label="GRU")
            ax.barh(X + 0.45, Graph[:, 8], height=0.15, color='#80B918', label="RD-SA")
            ax.barh(X + 0.60, Graph[:, 4], height=0.15, color='k', label="GSSA-ARD-SA")
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(True)

            # Customizations
            plt.yticks(X + 0.25, ['1', '2', '3', '4', '5'], fontname="Arial", fontsize=12,
                       fontweight='bold', color='k')
            plt.ylabel('Kfold', fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.xlabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.xticks(fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.tight_layout()
            plt.legend(loc='center left', bbox_to_anchor=(0.95, 0.5))
            path = "./Results/%s_mod_bar.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def Line_PlotTesults():
    eval = np.load('Evaluates.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1 Score',
             'MCC', 'FOR', 'PT', 'CSI', 'BA', 'FM', 'BM', 'MK', 'LR+', 'LR-', 'DOR', 'Prevalence']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 17, 18]
    Algorithm = ['NGO', 'SOA', 'Proposed']
    colors = ['violet', 'crimson', 'k']
    Kfold = [1, 2, 3, 4, 5]

    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.15, 0.6, 0.6])
            X = np.arange(len(Kfold))
            ax.barh(X + 0.00, Graph[:, 0], height=0.15, color='royalblue', label="CWO-ARD-SA")
            ax.barh(X + 0.15, Graph[:, 1], height=0.15, color='violet', label="TOT-ARD-SA")
            ax.barh(X + 0.30, Graph[:, 2], height=0.15, color='palegreen', label="WOA-ARD-SA")
            ax.barh(X + 0.45, Graph[:, 3], height=0.15, color='crimson', label="SSA-ARD-SA")
            ax.barh(X + 0.60, Graph[:, 4], height=0.15, color='k', label="GSSA-ARD-SA")
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(True)

            # Customizations
            plt.yticks(X + 0.25, ['1', '2', '3', '4', '5'], fontname="Arial", fontsize=12,
                       fontweight='bold', color='k')
            plt.ylabel('Kfold', fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.xlabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=12, fontweight='bold', color='k')
            plt.xticks(fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
            plt.tight_layout()
            plt.legend(loc='center left', bbox_to_anchor=(0.95, 0.5))
            path = "./Results/%s_Alg_bar.png" % (Terms[Graph_Terms[j]])
            plt.savefig(path)
            plt.show()


def Plot_Confusion():
    Actual = np.load('Actual.npy', allow_pickle=True)
    Predict = np.load('Predict.npy', allow_pickle=True)
    class_2 = ['Jute', 'Maize', 'Rice', 'Sugarcane', 'Wheat']
    fig, ax = plt.subplots(figsize=(10, 8))
    confusion_matrix = metrics.confusion_matrix(Actual.argmax(axis=1), Predict.argmax(axis=1))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=class_2)
    cm_display.plot(ax=ax)
    path = "./Results/Confusion.png"
    plt.title("Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(path)
    plt.show()


def plotConvResults_Pest():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fit.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'CWO-ARD-SA', 'TOT-ARD-SA', 'WOA-ARD-SA', 'SSA-ARD-SA', 'GSSA-ARD-SA']

    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Conv_Graph = np.zeros((len(Algorithm) - 1, len(Terms)))
    for j in range(len(Algorithm) - 1):  # for 5 algms
        Conv_Graph[j, :] = Statistical(Fitness[0, j, :])

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- Statistical Analysis  ',
          '--------------------------------------------------')
    print(Table)

    fig = plt.figure()
    fig.canvas.manager.set_window_title('Convergence Curve')
    length = np.arange(Fitness.shape[2])
    Conv_Graph = Fitness[0]
    plt.plot(length, Conv_Graph[0, :], color='#FF69B4', linewidth=3, marker='*', markerfacecolor='red',
             markersize=12, label='CWO-ARD-SA')
    plt.plot(length, Conv_Graph[1, :], color='#7D26CD', linewidth=3, marker='*', markerfacecolor='#00FFFF',
             markersize=12, label='TOT-ARD-SA')
    plt.plot(length, Conv_Graph[2, :], color='#FF00FF', linewidth=3, marker='*', markerfacecolor='blue',
             markersize=12, label='WOA-ARD-SA')
    plt.plot(length, Conv_Graph[3, :], color='#43CD80', linewidth=3, marker='*', markerfacecolor='magenta',
             markersize=12, label='SSA-ARD-SA')
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
             markersize=12, label='GSSA-ARD-SA')
    plt.xlabel('No. of Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    plt.savefig("./Results/Conv_pest.png")
    plt.show()


def Plot_ROC_Curve_Pest():
    cls = ['LSTM', 'Faster R-CNN', 'GRU', 'RD-SA', 'GSSA-ARD-SA']
    Actual = np.load('Pest_Target.npy', allow_pickle=True)
    lenper = round(Actual.shape[0] * 0.75)
    Actual = Actual[lenper:, :]
    fig = plt.figure(facecolor='#F9F9F9')
    fig.canvas.manager.set_window_title('ROC Curve')
    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])
    ax.set_facecolor("#F9F9F9")
    colors = cycle(["blue", "darkorange", "limegreen", "deeppink", "black"])
    for i, color in zip(range(len(cls)), colors):  # For all classifiers
        Predicted = np.load('Score.npy', allow_pickle=True)[i]
        false_positive_rate, true_positive_rate, _ = roc_curve(Actual.ravel(), Predicted.ravel())
        roc_auc = roc_auc_score(Actual.ravel(), Predicted.ravel())
        roc_auc = roc_auc * 100

        plt.plot(
            false_positive_rate,
            true_positive_rate,
            color=color,
            lw=2,
            label=f'{cls[i]} (AUC = {roc_auc:.2f} %)',
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Accuracy')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    path = "./Results/ROC_pest.png"
    plt.savefig(path)
    plt.show()


if __name__ == '__main__':
    plotConvResults()
    plotConvResults_Pest()
    Plot_ROC_Curve_Pest()
    Line_PlotTesults()
    Plots_Results()
    Plot_ROC_Curve()
    Table()
