import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
import chartfunctions as chart
import os
import time

#sp =pd.read_csv("./data/sample_submission.csv")
#est_set = pd.read_csv("./data/testing_set.csv")
#training_set = pd.read_csv("./data/training_set.csv")h
#test_set['Loan_Status'] = sp.loc[:,['Loan_Status']]
#data = pd.concat([test_set,training_set])


def plot_validation_curve(experiments:list,basepath:str,parameters :np.ndarray):

    fig, axes = plt.subplots(len(experiments),len(parameters), figsize=(10, 5))
    index = 0;
    for e in experiments:

        title =  e["datatitle"]
        savename =  e["datatitle"] + "_Validation_curve_DecisionTree"
        cur_axe = 0

        for p in parameters:
        
            chart.plot_validation_curve(DecisionTreeClassifier(random_state=1),p[0],title,e["x"],e["y"].values.ravel(),axes[index][cur_axe],p[1])
            cur_axe += 1
        index +=1

    plt.tight_layout()
    plt.savefig(basepath + "DecisionTree_Validation_Curve" )
    plt.clf()

def plot_learning_curve(experiments:list,runs,basepath:str):
    
    

    index = 0
    total_runs = len(runs)
    

    for e in experiments:

        savename = e["datatitle"]  +" DecisionTree_Learning_Curve"

        fig, axes = plt.subplots(3, total_runs , figsize=(20, 15))
        cur_axe = 0
        for r in runs:

            title =  e["datatitle"] + " Learning Curves  DecisionTree \n maxdepth:{md} \n min_samples_leaf:{ml}"
            title = title.format(md = r["max_depth"], ml = r["min_samples_leaf"])

            dt = DecisionTreeClassifier(max_depth= r["max_depth"],min_samples_leaf = r["min_samples_leaf"],random_state=1)

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





def Execute(experimentName:str, data_x:pd.DataFrame,data_y:pd.DataFrame,basepath:str,features:list, encoder:preprocessing.LabelEncoder,runs):
    
    cur_axe = 0
    total_runs = len(runs)

    for r in runs:
        pass

        x_train, x_test,y_train,y_test = train_test_split(data_x,data_y, random_state= 1)

    
        if not os.path.exists(basepath):
            os.makedirs(basepath)
        
        dt = DecisionTreeClassifier(max_depth= r["max_depth"],min_samples_leaf = r["min_samples_leaf"],random_state=0)

        t_fit_start = time.time()
        dt.fit(x_train,y_train)
        fit_wall_time = time.time() - t_fit_start

        print("fit wall time: " + str(fit_wall_time))
        print("fit score :" + str(dt.score(x_test,y_test)))
        scores = cross_val_score(dt, x_train, y_train, cv=5)
        print("cross scores :" + str(scores))

        figure, axis = plt.subplots()

        axis.set_title(experimentName)
        #class_names= label_cols
        tree.plot_tree(dt,ax = axis, feature_names = features)
        #tree.plot_tree(dt,ax = axis, class_names = encoder.classes_, feature_names = features)

        inputtxt = "maxdepth{md}__minsamplesleaf{msl}_DecisionTree"
        inputtxt = inputtxt.format(md = r["max_depth"], msl = r["min_samples_leaf"])

        plt.savefig(basepath + inputtxt,dpi=1200)
        plt.clf()


