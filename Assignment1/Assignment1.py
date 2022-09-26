#from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.utils import resample
import chartfunctions as chart
import DecisionTreeTest as dt_test
import NeuralNetwork as nn_test
import AdaBoost as ada_boost_test
import SupportVectorMachines as svm_test
import KNearestNeighbors as knn_test
import os


#lblBinary = preprocessing.LabelBinarizer()


# code referenced from elitedatascience
#https://elitedatascience.com/imbalanced-classes
def balance_data(data:pd.DataFrame,y_col:str):

    df_negative = data[data[y_col]== 0]
    df_positive = data[data[y_col]== 1]

   # print(len(df_negative))
   # print(len(df_positive))
   # print(data[y_col].value_counts())

    df_minority = pd.DataFrame()
    df_upsampled = pd.DataFrame()
    sample_count = 0
    
    if len(df_negative) > len(df_positive):
        df_upsampled = df_negative
        df_minority = df_positive
        sample_count = len(df_negative)
    else:
        df_upsampled = df_positive
        df_minority = df_negative
        sample_count = len(df_positive)
            
    df_minority_upsampled = resample(df_minority, 
                                 replace=True,
                                 n_samples=sample_count,
                                 random_state=1)

    df_upsampled = pd.concat([df_upsampled, df_minority_upsampled])

    #print( df_upsampled[y_col].value_counts())

    return df_upsampled;


def generate_loan_data(filepath:str = "./data/credit/dataset.csv",basepath:str = "./Experiments/Data/Loan/", generatecharts:bool = True):

  
    lblencoder = preprocessing.LabelEncoder() 
    scaler = preprocessing.StandardScaler()

    exclude_cols = ['Loan_ID']
    label_cols = ['N','Y']
    y_label = ['Loan_Status']

    dummy_features_ignore = ['CoapplicantIncome','ApplicantIncome','LoanAmount','Loan_Amount_Term']

    df = pd.read_csv(filepath)
    df = df.dropna()
    df.drop(columns = exclude_cols,inplace =True)

    df[y_label[0]] = lblencoder.fit_transform(df[y_label[0]])
    #df['Married'] = lblencoder.fit_transform(df['Married'])


    features = list(df)[:-1]

    dummy_features =  [x for x in features if x not in dummy_features_ignore ]


    data = pd.get_dummies(df, columns= dummy_features)

    data = balance_data(data,y_label[0]);

    if generatecharts:
        #basepath = basepath.format(experimentName = experimentname)
        imagepath = basepath 
        
        createFolder(imagepath)

        chart.chart_data(data,features,y_label[0],imagepath)

    data_x = data.drop(columns= y_label)

    feature_list = list(data_x)

   # data_x = scaler.fit_transform(data_x)

    data_y = data.loc[:,y_label]


    return data_x, data_y,lblencoder, feature_list

def generate_stroke_data(filepath:str = "./data/stroke/healthcare-dataset-stroke-data.csv",basepath:str = "./Experiments/Data/Stroke/", generatecharts:bool = True):

    lblencoder = preprocessing.LabelEncoder()
    scaler = preprocessing.StandardScaler()
  
    exclude_cols = ['id']
    y_label = ['stroke']

    dummy_features_ignore = ["age","avg_glucose_level","bmi"]

    df = pd.read_csv(filepath)
    df = df.dropna()
    df.drop(columns = exclude_cols,inplace =True)

    df[y_label[0]] = lblencoder.fit_transform(df[y_label[0]])


    features = list(df)[:-1]

    dummy_features =  [x for x in features if x not in dummy_features_ignore ]

    data = pd.get_dummies(df, columns= dummy_features)

    data = balance_data(data,y_label[0]);

    if generatecharts:
        #basepath = basepath.format(experimentName = experimentname)
        imagepath = basepath 
        
        createFolder(imagepath)

        chart.chart_data(data,features,y_label[0],imagepath)

    data_x = data.drop(columns= y_label)

    feature_list = list(data_x)

    #data_x = scaler.fit_transform(data_x)
    
    data_y = data.loc[:,y_label]

    return data_x, data_y,lblencoder,feature_list



def createFolder(path):

    if not os.path.exists(path):
        os.makedirs(path)




def run_DT_experiments(experiments:list):
    
    
    basepath = "./Experiments/DecisionTree/"
    
    createFolder(basepath)

    runs = [ 
        { 'max_depth': 100,'min_samples_leaf' :100 },
        { 'max_depth': 100,'min_samples_leaf' :25 },
    ]

    parameters = [["min_samples_leaf",np.arange(5, 100, 5)],["max_depth",np.arange(1, 100, 1)]]
    
    dt_test.plot_validation_curve(experiments,basepath,parameters)
    dt_test.plot_learning_curve(experiments,runs,basepath)

    for e in experiments:

        experimentbasepath = basepath + e["datatitle"]  + "/"

        dt_test.Execute(e["datatitle"],e["x"],e["y"],experimentbasepath,e["features"],e["labelencoder"],runs)

def run_neuralnetwork_experiments(experiments:list):

    basepath = "./Experiments/NeuralNetwork/"
    
    createFolder(basepath)

    runs = [ 
        { 'learning_rate_init': .001,'max_iter' :5000 },
        { 'learning_rate_init': .005,'max_iter' :5000 },
    ]

    parameters = [["learning_rate_init",np.arange(.001, .005,.001)],["max_iter",np.arange(1000, 10000, 1000)]]
    
    nn_test.plot_validation_curve(experiments,basepath,parameters)
    nn_test.plot_learning_curve(experiments,runs,basepath)

def run_ada_boost(experiments:list):

    basepath = "./Experiments/AdaBoost/"
    
    createFolder(basepath)

    runs = [ 
        { 'learning_rate': 1.0, "n_estimators" : 50},
        { 'learning_rate': 1.0, "n_estimators" : 100}
    ]

    parameters = [["learning_rate",np.arange(1, 5,1)],["n_estimators",np.arange(50, 200,10)]]
    
    ada_boost_test.plot_validation_curve(experiments,basepath,parameters)
    ada_boost_test.plot_learning_curve(experiments,runs,basepath)

def run_svm(experiments:list):

    basepath = "./Experiments/SupportVectorMachines/"
    
    createFolder(basepath)
    
    runs = [ 
        { 'shrinking': False},
        { 'shrinking': True}
    ]

    parameters = [["shrinking",[False,True]]]
    svm_test.plot_validation_curve(experiments,basepath,parameters)
    svm_test.plot_learning_curve(experiments,runs,basepath)

def run_k_nearest_neighbor(experiments:list):

    basepath = "./Experiments/KNearestNeighbor/"
    
    createFolder(basepath)

    runs = [ 
        { 'n_neighbors': 10},
        { 'n_neighbors': 2}
    ]

    parameters = [["n_neighbors",np.arange(2, 20,2)]]
    knn_test.plot_validation_curve(experiments,basepath,parameters)
    knn_test.plot_learning_curve(experiments,runs,basepath)


makechart = False

loan_data_x, loan_data_y, loan_lbl_encoder, loan_feature_list = generate_loan_data(generatecharts=makechart)
stroke_data_x, stroke_data_y,stroke_lbl_encoder, stroke_feature_list= generate_stroke_data(generatecharts=makechart)

experiment_data = [ {"datatitle":"Loan", "x":loan_data_x, "y":loan_data_y, "features":loan_feature_list,"labelencoder":loan_lbl_encoder}, {"datatitle":"Stroke","x":stroke_data_x,"y":stroke_data_y,"features":stroke_feature_list,"labelencoder":stroke_lbl_encoder}]

run_DT_experiments(experiment_data)
run_ada_boost(experiment_data)
run_neuralnetwork_experiments(experiment_data)
run_k_nearest_neighbor(experiment_data)
run_svm(experiment_data)