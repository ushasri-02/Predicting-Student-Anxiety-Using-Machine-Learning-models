import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

st.title("Student Anxiety Prediction Using Machine Learning Models")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset")
    st.write(df.head())  

    numeric_cols = df.select_dtypes(include=[np.number]).columns
   
    imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    correlation_matrix = df.corr()


    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='Blues', ax=ax)
    st.pyplot(fig)

    df1 = df[['GAD_T', 'SWL_T', 'SPIN_T']]
    df2 = df[['Age', 'Hours']]

    pc1 = PCA(n_components=2)
    x1 = pc1.fit_transform(df1)

    pc2 = PCA(n_components=2)
    x2 = pc2.fit_transform(df2)

    x = np.hstack((x1, x2))

    WCSS = []
    for i in range(1, 12):
        model = KMeans(n_clusters=i, init='k-means++', n_init=10)
        model.fit(x)
        WCSS.append(model.inertia_)

    fig, ax = plt.subplots()
    ax.plot(range(1, 12), WCSS, marker='o', color='green')
    ax.set_xticks(np.arange(1, 12))
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("WCSS")
    st.pyplot(fig)

    model = KMeans(n_clusters=3, init="k-means++", max_iter=300, n_init=10, random_state=0)
    y_clusters = model.fit_predict(x)
    df['Label'] = y_clusters

    X = df.drop(columns=['Label'])
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=44)

    xgb_clf = XGBClassifier(n_estimators=1000, max_depth=8, random_state=44, use_label_encoder=False, eval_metric='mlogloss')
    xgb_clf.fit(X_train, y_train)
    y_pred = xgb_clf.predict(X_test)

    st.write("### XGBoost Classifier Results")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    st.text(classification_report(y_test, y_pred))

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title("Confusion Matrix - XGBoost")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.write("### Predict Anxiety Level for a New Student")

    age = st.number_input("Age", min_value=10, max_value=100, value=20)
    hours = st.number_input("Study Hours per Day", min_value=0.0, max_value=24.0, value=4.0)
    gad_t = st.number_input("GAD Score", min_value=0, max_value=100, value=50)
    swl_t = st.number_input("SWL Score", min_value=0, max_value=100, value=50)
    spin_t = st.number_input("SPIN Score", min_value=0, max_value=100, value=50)

    if st.button("Predict Anxiety Level"):
        user_df = pd.DataFrame({
            'Age': [age],
            'Hours': [hours],
            'GAD_T': [gad_t],
            'SWL_T': [swl_t],
            'SPIN_T': [spin_t]
        })

        user_df[numeric_cols] = imputer.transform(user_df[numeric_cols])
        user_df = user_df[X_train.columns]
        predicted_label = xgb_clf.predict(user_df)[0]


        if predicted_label == 0:
            predicted_anxiety = "Low"
        elif predicted_label == 1:
            predicted_anxiety = "Moderate"
        else:
            predicted_anxiety = "High"

        st.write("### Predicted Anxiety Level:", predicted_anxiety)
