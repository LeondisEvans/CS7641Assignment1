from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import chartfunctions as chart


def plot_validation_curve(experiments:list,basepath:str,parameters :np.ndarray):
    
    savename =  "KNearestNeighbors_validation_curve"

    index = 0
    fig, axes = plt.subplots(len(experiments), 1, figsize=(10, 5))

    for e in experiments:

        title =  e["datatitle"] + " KNearestNeighbors \n"   
        cur_axe = 0
        for p in parameters:
        
            chart.plot_validation_curve(KNeighborsClassifier(),p[0],title,e["x"],e["y"].values.ravel(),axes[index],p[1])
            cur_axe += 1
        index += 1

    plt.tight_layout()
    plt.savefig(basepath + savename )
    plt.clf()

def plot_learning_curve(experiments:list,runs,basepath:str):
    
    
    index = 0
    total_runs = len(runs)
    

    for e in experiments:

        savename = e["datatitle"]  +" KNearestNeighbors_Learning_Curve"

        fig, axes = plt.subplots(3, total_runs , figsize=(20, 15))
        cur_axe = 0
        for r in runs:

            title =  e["datatitle"] + " Learning Curves  KNearestNeighbors \n # of neighbors:{md}"
            title = title.format(md = r["n_neighbors"])

            dt = KNeighborsClassifier(n_neighbors= r["n_neighbors"])


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
