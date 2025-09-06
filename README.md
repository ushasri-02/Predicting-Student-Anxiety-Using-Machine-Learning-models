# Predicting-Student-Anxiety-Using-Machine-Learning-models
This project is a Streamlit web application that predicts student anxiety levels using clustering and machine learning models. It allows users to upload a dataset, explore correlations, perform clustering, and predict anxiety levels for new student data.

1.Features
->Upload CSV dataset and preview data
->Handle missing values with imputation
->Visualize correlation matrix using heatmaps
->Dimensionality reduction with PCA
->Cluster students into groups using KMeans
->Determine optimal number of clusters with WCSS (Elbow Method)
->Train an XGBoost Classifier to predict anxiety levels
->Evaluate model with accuracy, classification report, and confusion matrix
->Interactive inputs to predict anxiety level for a new student

2.Tech Stack
Frontend & UI: Streamlit
Data Handling: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Machine Learning: Scikit-learn, XGBoost

3.How It Works
->Upload a CSV file with student data (Age, Study Hours, GAD, SWL, SPIN scores).
->The app performs PCA and KMeans clustering to group students by anxiety patterns.
->An XGBoost classifier is trained on the clustered dataset.
->The trained model predicts whether a new student has Low, Moderate, or High anxiety based on their inputs.

4.Usage
# Clone repository
git clone https://github.com/your-username/student-anxiety-prediction.git
cd student-anxiety-prediction
# Install dependencies
pip install -r requirements.txt
# Run the app
streamlit run predicting_Student_Anxiety_Using_MLMODELS.py


5. Example Prediction
Input:
Age = 20
Study Hours = 4
GAD = 50
SWL = 50
SPIN = 50
Output:
Predicted Anxiety Level: Moderate
