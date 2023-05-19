import numpy as np
import pandas as pd
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Web App Title
st.markdown('''
# **The EDA App & Classifiers**
---
''')

# Upload CSV data
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader(
        "Upload your input CSV file", type=["csv"])


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # st.write(df)

    # Identify potential categorical variables based on data type
    categorical_columns = []
    for column in df.columns:
        if df[column].dtype == 'object':  # or df[column].dtype == np.object
            categorical_columns.append(column)

    # Convert categorical variables to discrete numerical representation
    for column in categorical_columns:
        df[column] = df[column].astype('category').cat.codes

    st.write(df)

    # Assuming the last column is the target variable
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    classifier_name = st.sidebar.selectbox(
        'Select classifier',
        ('KNN', 'SVM', 'Random Forest'))

    def add_parameter_ui(clf_name):
        params = dict()
        if clf_name == 'SVM':
            C = st.sidebar.slider('C', 0.01, 10.0)
            params['C'] = C
        elif clf_name == 'KNN':
            K = st.sidebar.slider('K', 1, 15)
            params['K'] = K
        else:
            max_depth = st.sidebar.slider('max_depth', 2, 15)
            params['max_depth'] = max_depth
            n_estimators = st.sidebar.slider('n_estimators', 1, 100)
            params['n_estimators'] = n_estimators
        return params

    params = add_parameter_ui(classifier_name)

    def get_classifier(clf_name, params):
        clf = None
        if clf_name == 'SVM':
            clf = SVC(C=params['C'])
        elif clf_name == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=params['K'])
        else:
            clf = clf = RandomForestClassifier(
                n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=1234)
        return clf

    clf = get_classifier(classifier_name, params)

    #### CLASSIFICATION ####

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    st.write(f'Classifier = {classifier_name}')
    st.write(f'Accuracy =', acc)

    #### PLOT DATASET ####
# Project the data onto the 2 primary principal components
    pca = PCA(2)
    X_projected = pca.fit_transform(X)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    fig = plt.figure()
    plt.scatter(x1, x2, c=y, alpha=0.8)

    plt.colorbar()

    # plt.show()
    st.pyplot(fig)

    # data = pd.read_csv(uploaded_file)
    pr = ProfileReport(df, explorative=True)
    st.header('**Input DataFrame**')
    st.write(df)
    st.write('---')
    st.header('**Pandas Profiling Report**')
    st_profile_report(pr)

    # # Pandas Profiling Report
    # if uploaded_file is not None:
    #     def load_csv():
    #         csv = pd.read_csv(uploaded_file)
    #         return csv
    #     data = load_csv()
    #     pr = ProfileReport(data, explorative=True)
    #     st.header('**Input DataFrame**')
    #     st.write(data)
    #     st.write('---')
    #     st.header('**Pandas Profiling Report**')
    #     st_profile_report(pr)
    # else:
    #     st.info('Awaiting for CSV file to be uploaded.')
    #     if st.button('Press to use Example Dataset'):
    #         # Example data
    #         @st.cache
    #         def load_data():
    #             a = pd.DataFrame(
    #                 np.random.rand(100, 5),
    #                 columns=['a', 'b', 'c', 'd', 'e']
    #             )
    #             return a
    #         df = load_data()
    #         pr = ProfileReport(df, explorative=True)
    #         st.header('**Input DataFrame**')
    #         st.write(df)
    #         st.write('---')
    #         st.header('**Pandas Profiling Report**')
    #         st_profile_report(pr)
