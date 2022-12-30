from re import I
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
import os

# Function to run the model with Train Test Split only
def TrainTestRunOnly():
    X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size = trainTestRatio, random_state=42)
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    st.write("Model prediction with train test split result : ")
    st.write('mean_squared_error : ', mean_squared_error(y_test, predictions))
    st.write('mean_absolute_error : ', mean_absolute_error(y_test, predictions))

# Funciton to run the cross-validation/K-Fold version
def KFoldScore(X, y, k):
    kf = KFold(n_splits=int(k))
    output = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        reg = model.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        output += mean_absolute_error(y_pred, y_test)

    output = output/k
    st.write("Result by running K-Fold with " + str(k) + " is:", output)
    

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
    data = pd.read_csv(uploaded_file)
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

    model = LinearRegression()

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
    ### Or tick the Use Wine checkbox to use the Wine Dataset
    """
)

    
