# Import thư viện cần thiết
from re import I
import streamlit as st
import pandas as pd
import cv2
import numpy as np
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss, accuracy_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

st.markdown(
"""
# Grid Search
"""
)

st.markdown(
    """
    ## 1. Import your own dataset
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

    # Automatically assume the last column will be the Y
    colsName = data.columns.values.tolist()
    outputName = colsName[-1]

    st.write("Output: ", outputName)   

    # Preprocessing the data, label encoding the string/words type of data
    dataX = df
    dataY = data.iloc[:,-1:]
    le = preprocessing.LabelEncoder()
    sc = StandardScaler()

    for i in check:
        if (dataX[i].dtype == object):
            textLabel = np.unique(dataX[i].values)
            le.fit(textLabel)
            result = le.transform(dataX[i]) 
            dataX[i] = result
    dataX = sc.fit_transform(dataX)

    button = st.button("Start finding best paramaters")
    if button:
        # Load the models needed
        parameters = {'kernel':('linear', 'rbf', 'poly', 'sigmoid'), 'C':[1, 10]}
        svc = svm.SVC()
        clf = GridSearchCV(svc, parameters, cv=4)
        clf.fit(dataX, dataY)

        st.write(" Results from Grid Search " )
        st.write("\n The best estimator across ALL searched params:\n",clf.best_estimator_)
        st.write("\n The best score across ALL searched params:\n",clf.best_score_)
        st.write("\n The best parameters across ALL searched params:\n",clf.best_params_)