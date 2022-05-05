import matplotlib.pyplot as plt
from sklearn import metrics


def roc_plot(data, save_pth):
    """
    data(dict):
    {modelA:{'true_label':[],
            'proba'      :[]},
    modelB:{'true_label' :[],
            'proba'      :[]},
    modelC:{'true_label' :[],
            'proba'      :[]},
    ......
    }
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for key in data:
        true_label = data[key]['true']
        proba      = data[key]['proba']
        fpr[key], tpr[key], _ = metrics.roc_curve(true_label, proba, pos_label=1)
        roc_auc[key] = metrics.auc(fpr[key], tpr[key])

    lw = 2
    color_list = ['#2B92E4', '#D9716F', '#8DA077', '#D4805C', '#87C8CB', '#8D8DAA', '#FDB35D']

    plt.rc('font', family='Times New Roman', weight='bold', size=18)
    fig, ax = plt.subplots(figsize=(9, 8))
    for i, key in enumerate(data):
        # color = plt.cm.viridis(i / 10.)
        color = color_list[i]
        plt.plot(fpr[key], tpr[key], color=color, lw=lw, label=f'ROC curve of {key} (area = {roc_auc[key]:0.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(save_pth)
    plt.show()


