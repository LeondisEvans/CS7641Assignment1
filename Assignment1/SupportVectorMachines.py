from sklearn import svm
from sklearn.pipeline import Pipeline
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import chartfunctions as chart
from sklearn import preprocessing


def plot_validation_curve(experiments:list,basepath:str,parameters :np.ndarray):
    scaler = preprocessing.StandardScaler()
    savename =  "SVM_validation_curve"

    max_iter = 20000

    kernels = ['linear','poly']
    index = 0
    fig, axes = plt.subplots(len(experiments), len(parameters) * len(kernels), figsize=(10, 5))

    for e in experiments:

        cur_axe = 0
        for k in kernels:

            title =  e["datatitle"] + " SVM \n" + "Kernel:{k} max_iter:{max_iter}\n" 
            title = title.format(k = k, max_iter =max_iter)
            for p in parameters:        
                chart.plot_validation_curve(svm.SVC(kernel= k,random_state=1, decision_function_shape = 'ovr',max_iter=max_iter),p[0],title,scaler.fit_transform(e["x"]),e["y"].values.ravel(),axes[index][cur_axe],p[1])
                cur_axe += 1
        index += 1

    plt.tight_layout()
    plt.savefig(basepath + savename )
    plt.clf()


def plot_learning_curve(experiments:list,runs,basepath:str):
    
    scaler = preprocessing.StandardScaler()
    index = 0
    total_runs = len(runs)
    kernels = ['linear','poly']

    max_iter = 20000
    for e in experiments:
    
        cur_axe = 0
        
        savename = e["datatitle"]  +" SVM_Learning_Curve"

        fig, axes = plt.subplots(3, total_runs * len(kernels) , figsize=(20, 15))

        for k in kernels:
            for r in runs:

                title =  e["datatitle"] + " Learning Curves  SVM Kernel:{k} shrinking:{md} max_iter:{max_iter}\n"
                title = title.format(k = k,md = r["shrinking"], max_iter =max_iter)

                dt = svm.SVC(kernel= k, decision_function_shape = 'ovr',max_iter=max_iter,shrinking= r["shrinking"],random_state=1)


                chart.plot_learning_curve(
                    dt,
                    title,
                    e["x"],
                    e["y"].values.ravel(),
                    axes=axes[:, cur_axe],
                #   ylim=(1.4, 2.02),
                    ylim=(0.7, 1.01),
                    cv=5,
                    n_jobs=4,
                    scoring="accuracy"
                )      
                cur_axe += 1

        plt.subplots_adjust(wspace = .2,hspace = .4)
        plt.savefig(basepath + savename)
        plt.clf()

