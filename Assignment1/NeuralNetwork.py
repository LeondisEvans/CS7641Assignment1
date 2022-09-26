import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import preprocessing
import chartfunctions as chart
import os
import time
from sklearn.neural_network import MLPClassifier



def plot_validation_curve(experiments:list,basepath:str,parameters :np.ndarray):

    savename =  "NeuralNetwork_MLP_validation_curve"

    index = 0
    fig, axes = plt.subplots(len(experiments), len(parameters), figsize=(10, 5))


    for e in experiments:

        title =  e["datatitle"] + " NeuralNetwork MLP \n"   
        cur_axe = 0
        for p in parameters:
        
            chart.plot_validation_curve(MLPClassifier(hidden_layer_sizes=(5,5,5),random_state=1,max_iter= 5000),p[0],title,e["x"],e["y"].values.ravel(),axes[index][cur_axe],p[1])
            cur_axe += 1
        index += 1

    plt.tight_layout()
    plt.savefig(basepath + savename )
    plt.clf()

def plot_learning_curve(experiments:list,runs,basepath:str):
    
    

    index = 0
    total_runs = len(runs)
    

    for e in experiments:

        savename = e["datatitle"]  +" NeuralNetwork_Learning_Curve"

        fig, axes = plt.subplots(3, total_runs , figsize=(20, 15))
        cur_axe = 0
        for r in runs:

            title =  e["datatitle"] + " Learning Curves  NeuralNetwork \n learning_rate_init:{md} \n max_iter:{ml}"
            title = title.format(md = r["learning_rate_init"], ml = r["max_iter"])

            dt = MLPClassifier(hidden_layer_sizes=(5,5,5),learning_rate_init= r["learning_rate_init"],max_iter = r["max_iter"],random_state=1)


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
