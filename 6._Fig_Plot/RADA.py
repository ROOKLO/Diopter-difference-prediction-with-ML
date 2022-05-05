import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.font_manager import FontProperties


def polar_plot(scores, save_pth, text_height=1.05):
    """
    scores(Dataframe):
    ------------------------------------------------------
                ACC    AUC     Sensitivity     Specificity
    ------------------------------------------------------
    model A
    model B
    model C
    model D
    -------------------------------------------------------
    """
    num_model     = scores.shape[0]
    num_criterion = scores.shape[1]
    N = num_model * num_criterion

    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)                               # x
    width = 2*np.pi / N

    plt.rc('font', family='Times New Roman', weight='bold', size=25)

    fig = plt.figure(figsize=(15, 9))
    ax = plt.subplot(111, projection='polar')
    ax.set_xticks(theta)
    ax.set_xticklabels([])

    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])

    # colors = ['#E2AB7F', '#C05850', '#3A353F', '#505668']


    for i in range(num_criterion):
        radii = scores.iloc[:, i]
        model_names = scores.index.to_list()
        criterion = radii.name
        colors = plt.cm.viridis(np.repeat(i, num_model) / 5.)
        theta_curr = theta[i*num_model:(i+1)*num_model]
        ax.bar(theta_curr+np.pi/N, radii, width=width, bottom=0.0, color=colors, alpha=1, label=criterion, )

        for _, info_ in enumerate(zip(theta_curr, model_names)):
            if  0 <= info_[0] < np.pi/2 :
                ax.text(info_[0]+np.pi/N, text_height, info_[1], ha='center', va='center', rotation=((info_[0]+width*0.5)/(2*np.pi))*360)
            elif np.pi/2 <= info_[0] < np.pi:
                ax.text(info_[0]+np.pi/N, text_height, info_[1], ha='center', va='center', rotation=((info_[0]+width*0.5+np.pi)/(2*np.pi))*360)
            elif np.pi <= info_[0] < 3*np.pi/2:
                ax.text(info_[0]+np.pi/N, text_height, info_[1], ha='center', va='center', rotation=((info_[0]+width*0.5+np.pi)/(2*np.pi))*360)
            elif 3*np.pi/2 <= info_[0] < 2*np.pi:
                ax.text(info_[0]+np.pi/N, text_height, info_[1], ha='center', va='center', rotation=((info_[0]+width*0.5)/(2*np.pi))*360)
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.savefig(save_pth)
    plt.show()

