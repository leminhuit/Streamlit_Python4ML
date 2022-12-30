from re import I
import streamlit as st
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
import os

# Function to run the model with Train Test Split only
def TrainTestRunOnly():
    dataYArray = np.ravel(dataY)
    X_train, X_test, y_train, y_test = train_test_split(dataX, dataYArray, test_size = trainTestRatio, random_state=42)
    sc_x = StandardScaler()
    X_train = sc_x.fit_transform(X_train) 
    X_test = sc_x.transform(X_test)
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)

    preScore = precision_score(y_test, predictions)
    recScore = recall_score(y_test, predictions)
    f1Score = f1_score(y_test, predictions)
    logLossScore = log_loss(y_test, predictions)

    plt.figure(figsize=(8, 4))
    ax1 = plt.subplot()
    ax1.bar(np.arange(1) - 0.21, [preScore], 0.4, label='Precision', color='maroon')
    plt.xticks(np.arange(1), [str(trainTestRatio)])
    plt.ylabel("Precision Score", color='maroon')
    ax2 = ax1.twinx()
    ax2.bar(np.arange(1) + 0.21, [recScore], 0.4, label='Recall', color='green')
    plt.ylabel('Recall Score', color='green')
    plt.title("EVALUATION METRIC with Precision and Recall")

    plt.savefig('chart.png')
    with open('model.pkl','wb') as f:
        pickle.dump(model, f)

    ax3 = plt.subplot()
    ax3.bar(np.arange(1) + 0.21, [f1Score], 0.4, label='F1 Score', color='red')
    plt.xticks(np.arange(1), [str(trainTestRatio)])
    plt.ylabel('F1 Score', color='red')
    ax4 = ax3.twinx()
    ax4.bar(np.arange(1) + 0.21, [logLossScore], 0.4, label='Log Loss Score', color='blue')
    plt.ylabel('Log Loss Score', color='blue')
    plt.title("EVALUATION METRIC with F1 Score and Log Loss")
    
    plt.savefig('chart2.png')
    with open('model.pkl','wb') as f:
        pickle.dump(model, f)

    img = cv2.imread('chart.png')
    if img is not None: 
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img2 = cv2.imread('chart2.png')
    if img2 is not None: 
        st.image(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

# Funciton to run the cross-validation/K-Fold version
def KFoldScore(X, y, k):
    kf = KFold(n_splits=int(k))
    outputPre = 0; preArray = []; 
    outputRec = 0; recArray = []; 
    outputF1 = 0; f1Array = []; 
    outputLogLoss = 0; logLossArray = []; 
    folds = [str(fold) for fold in range(1, k+1)]
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        y_train_array = np.ravel(y_train)
        y_test_array = np.ravel(y_test)
        sc_x = StandardScaler()
        X_train = sc_x.fit_transform(X_train) 
        X_test = sc_x.transform(X_test)
        logreg = model.fit(X_train, y_train_array)
        y_pred = logreg.predict(X_test)
        outputPre += precision_score(y_pred, y_test_array)
        preArray.append(round(outputPre, 2))
        outputRec += recall_score(y_pred, y_test_array)
        recArray.append(round(outputRec, 2))
        outputF1 += f1_score(y_pred, y_test_array)
        f1Array.append(round(outputF1, 2))
        outputLogLoss += log_loss(y_pred, y_test_array)
        logLossArray.append(round(outputLogLoss, 2))
        with open('model.pkl','wb') as f:
            pickle.dump(model, f)

    outputPre = outputPre/k
    outputRec = outputRec/k
    outputF1 = outputF1/k
    outputLogLoss = outputLogLoss/k

    plt.figure(figsize=(8, 4))
    ax1 = plt.subplot()
    ax1.bar(np.arange(len(folds)) - 0.21, preArray, 0.4, label='Precision Score', color='maroon')
    plt.xticks(np.arange(len(folds)), folds)
    plt.xlabel("Folds", color='blue')
    plt.ylabel("Precision Score", color='maroon')
    ax2 = ax1.twinx()
    ax2.bar(np.arange(len(folds)) + 0.21, recArray, 0.4, label='Recall Score', color='green')
    plt.ylabel('Recall Score', color='green')
    plt.title("EVALUATION METRIC K-Fold of Precision and Recall")
    plt.savefig('chart3.png')

    img3 = cv2.imread('chart3.png')
    if img3 is not None: 
        st.image(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB))

    ax3 = plt.subplot()
    ax3.bar(np.arange(len(folds)) - 0.21, f1Array, 0.4, label='F1 Score', color='blue')
    plt.xticks(np.arange(len(folds)), folds)
    plt.xlabel("Folds", color='yellow')
    plt.ylabel("Precision Score", color='blue')
    ax4 = ax3.twinx()
    ax4.bar(np.arange(len(folds)) + 0.21, logLossArray, 0.4, label='Log Loss Score', color='orange')
    plt.ylabel('Recall Score', color='orange')
    plt.title("EVALUATION METRIC K-Fold of F1 Score and Log Loss")
    plt.savefig('chart4.png')

    img4 = cv2.imread('chart4.png')
    if img4 is not None: 
        st.image(cv2.cvtColor(img4, cv2.COLOR_BGR2RGB))

    st.write("Precision Score with K-Fold", outputPre)
    st.write("Recall Score with K-Fold", outputRec)
    st.write("F1 Score with K-Fold", outputF1)
    st.write("Log Loss with K-Fold", outputLogLoss)    

st.markdown(
"""
# Linear Regression Using Streamlit
"""
)

st.markdown(
    """
    ## 1. Import your dataset
    """
)

# Importing csv file to start running the model
uploaded_file = st.file_uploader("Import your file...");
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    path = 'data/' + uploaded_file.name
    with open('data/' + uploaded_file.name, 'wb') as f:
        f.write(bytes_data)
    data = pd.read_csv(uploaded_file, sep = ';', header = 0)

    st.write(data)

    st.markdown(
    """
    ## 2. Input feature
    """
    )

    # Choosing the features you want to use to train your model with
    numberOfFeatures = len(data.columns) - 1
    cols = st.columns(numberOfFeatures)
    option = []
    # dataX = pd.DataFrame()
    for i, x in enumerate(cols):
        option.append(x.checkbox(data.columns[i], value=True))
    check = []
    for i in range(len(option)):
        if option[i] == True:
            check.append(data.columns[i])

    # Showing the features that you chose
    df = pd.DataFrame()
    for i in check:
        df[i] = data[i]
    st.write(df)

    # Automatically assume the last column will be te Y
    colsName = data.columns.values.tolist()
    outputName = colsName[-1]

    st.markdown(
        """
        ### Output: 
        """,
    )   
    st.write(outputName)

    # Preprocessing the data, label encoding the string/words type of data
    dataX = df
    dataY = data.iloc[:,-1:]
    le = preprocessing.LabelEncoder()

    for i in check:
        if (dataX[i].dtype == object):
            textLabel = np.unique(dataX[i].values)
            le.fit(textLabel)
            result = le.transform(dataX[i])
            dataX[i] = result

    st.markdown(
        """
        ## 3. Train/test split ratio and K-Fold Testing Method
        """
    )

    # Choosing your type of training route: K-Fold or simple Train Test Split
    kFold = st.checkbox("K-Fold", 0)

    trainTestRatio = st.number_input("Input your train ratio by choosing a number : ", min_value=0.1, max_value= 0.9, step=0.1)

    if kFold:
        numberOfK = st.number_input("Input your K-Fold amount: ", min_value=2, step=1)

    model = LogisticRegression(random_state=0)

    if st.button("Run Model"):
        if kFold == 0:
            TrainTestRunOnly()
        else:
            if kFold == 1:
                st.write(KFoldScore(dataX, dataY, numberOfK))

else:
    st.markdown(
    """
    ### Please import the data so that you can select your feature and run the model
    """
)