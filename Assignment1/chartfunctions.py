import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit
from pandas.plotting import table

def chart_data(data :pd.DataFrame,features:list,y_col: str, savepath : str ):


    data_cols = list(data)
    
    for f in features: 

        f_cols = []

        f_cols.append(y_col)

        [f_cols.append(col) for col in data_cols if f in col ]

        table_data =data.loc[:,f_cols]
        #df.set_index('Loan_ID',inplace =True)
        generateBarChart = True
        #generateBarChart = False

        colCount = table_data.nunique()
        for fc in f_cols:

           
            if colCount[fc] > 2:
                        
                print(fc)
                generateBarChart = False
                break

        if generateBarChart :
            ax = table_data.groupby(y_col).sum().plot(kind ='bar')

            #Add the counts on top the bars
            for p in ax.patches:
                ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))


            plt.savefig(savepath + f,dpi=1200)
        else:
            fig, ax = plt.subplots()

            for fc in f_cols:
                if fc != y_col:
                    feature_data = table_data.loc[:,[fc,y_col]]

                    negative = feature_data[feature_data[y_col] == 0].loc[:,[fc]]
                    negative_y = np.arange(1, len(negative) + 1, 1)
                    positive = feature_data[feature_data[y_col] == 1].loc[:,[fc]]
                    positive_y = np.arange(1, len(positive) + 1, 1)

                    ax.scatter(negative_y,negative,label = fc + ": - :"+ y_col,alpha=0.3, edgecolors='none')
                    ax.scatter(positive_y,positive,label = fc + ": + :"+ y_col,alpha=0.3, edgecolors='none')
                    ax.set_xlabel("data points")
                    ax.set_ylabel(f)
                    ax.legend()

            #table_data.sort_values(f_cols, inplace = True)

            #fig, ax = plt.subplots(1,2) 
            
            #ax[0].set_axis_off()
            #ax[1].set_axis_off()
            
            #ax[0].set_title(f)
            #ax[1].set_title(f)
            
            #table(ax[0],table_data[table_data[y_col] >= 1])
            #table(ax[1],table_data[table_data[y_col] < 1])




            #table = ax.table(cellText = table_data.values, rowLabels = table_data.index, colLabels = table_data.columns, cellLoc = 'center', rowLoc = 'center', loc = 'upper left')
            #ax.set_title(h)
            #table.scale(1.5, 1.5)
            #plt.show()
            plt.savefig(savepath + f ,bbox_inches="tight")
        plt.clf()


 # code used and modifed from Sckikit 
#https://scikit-learn.org/stable/modules/cross_validation.html
#https://scikit-learn.org/stable/modules/model_evaluation.html
def plot_learning_curve(estimator,title,X,y,axes=None,ylim=None,cv=None,n_jobs=None,scoring=None,train_sizes=np.linspace(0.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title, pad = 20.0)

    
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

# code used and modifed from Sckikit 
#https://scikit-learn.org/stable/modules/cross_validation.html
#https://scikit-learn.org/stable/modules/model_evaluation.html
def plot_validation_curve(estimator,param_name,title,X,y,axes,parameter_range:np.ndarray):

    # Calculate accuracy on training and test set using the
    # gamma parameter with 5-fold cross validation
    train_score, test_score = validation_curve(estimator, X, y,
                                       param_name = param_name,
                                       param_range = parameter_range,
                                        cv = 5, scoring = "accuracy")
 
    # Calculating mean and standard deviation of training score
    mean_train_score = np.mean(train_score, axis = 1)
    std_train_score = np.std(train_score, axis = 1)
 
    # Calculating mean and standard deviation of testing score
    mean_test_score = np.mean(test_score, axis = 1)
    std_test_score = np.std(test_score, axis = 1)
 
    # Plot mean accuracy scores for training and testing scores
    axes.plot(parameter_range, mean_train_score,
     label = "Training Score", color = 'b')
    
    axes.plot(parameter_range, mean_test_score,
   label = "Cross Validation Score", color = 'g')
 
    # Creating the plot
    axes.set_title(title + "Validation Curve")
    axes.set_xlabel(param_name)
    axes.set_ylabel("Accuracy")
    axes.legend(loc = 'best')
    #plt.show()

    return plt
