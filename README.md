# Customer-Churn-Analysis

# 📖 Introduction
Customer churn is a critical challenge for businesses, impacting revenue and customer retention. This project provides an end-to-end churn analysis solution, including data processing, visualization, and predictive modeling to identify at-risk customers and provide actionable insights.

The project focuses on building a complete data pipeline, integrating SQL Server for ETL, Power BI for dashboards, and Machine Learning (Random Forest) for churn prediction.

# 🎯 Project Objectives

📊 Visualize & Analyze Customer Data across demographics, geography, payment info, and services
🔍 Identify Churner Profiles and key factors contributing to customer churn
📈 Predict Future Churners using machine learning (Random Forest)
🚀 Provide Actionable Insights to improve customer retention strategies 

# 📂 Project Structure

📁 Churn-Analysis-Project/
│── 📜 README.md               # Project documentation  
│── 📁 data/                   # Raw and processed data files  
│── 📁 sql/                    # SQL scripts for ETL, data exploration, and views  
│── 📁 powerbi/                # Power BI reports and dashboard files  
│── 📁 machine_learning/       # Python code for churn prediction model  
│── 📜 churn_analysis.ipynb    # Jupyter Notebook with ML model implementation  
│── 📜 LICENSE                 # Open-source license (MIT recommended) 

# 📊 Step 1: ETL Process in SQL Server
Tools Used: SQL Server, SSMS (SQL Server Management Studio)

Create Database:

CREATE DATABASE db_Churn;

Import Customer Data (CSV) into SQL Server Staging Table
Perform Data Exploration (Checking unique values, missing data, etc.)
Clean Data & Insert into Production Table (prod_Churn)
Create Views for Power BI (vw_ChurnData, vw_JoinData)

# 📊 Step 2: Power BI Dashboard
Tools Used: Power BI

Visualizations & Insights:
Summary Page: Total Customers, Churn Rate, New Joiners
Demographic Analysis: Gender & Age Group vs. Churn
Account Information: Payment Methods, Contract Types
Geographic Churn Trends
Service Usage Impact on Churn
Data Transformations:
Created calculated columns for Churn Status, Age Groups, and Tenure Groups
Unpivoted service columns for better visualization

# 🤖 Step 3: Churn Prediction Model
Tools Used: Python (Pandas, NumPy, Scikit-learn), Jupyter Notebook

Data Preprocessing:

Removed unnecessary columns (Customer_ID, Churn_Reason)
Label encoded categorical variables
Split data into training (80%) and testing (20%)
Train Random Forest Classifier:

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


Evaluate Model Performance:

from sklearn.metrics import classification_report, confusion_matrix

y_pred = rf_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

Predict Future Churners:

Loaded new customers (who haven’t churned yet)
Applied the trained model to predict potential churners
Saved results to Predictions.csv

# 📌 How to Use This Project?
🔧 Prerequisites
Ensure you have the following installed:

Python (Anaconda Recommended): Install Here
SQL Server & SSMS: Download Here
Power BI Desktop: Download Here
💻 Setup Instructions
1️⃣ Clone the Repository

git clone https://github.com/omkarmagare27/churn-analysis.git
cd churn-analysis

2️⃣ Install Required Dependencies

pip install -r requirements.txt

3️⃣ Run SQL ETL Pipeline

Open SQL Server Management Studio (SSMS)
Run scripts in /sql folder to create database, load data, and generate views

4️⃣ Explore Power BI Dashboard

Open powerbi/Churn_Dashboard.pbix in Power BI
Analyze customer churn trends & insights

5️⃣ Train & Test Machine Learning Model

cd machine_learning
jupyter notebook churn_analysis.ipynb
Train the Random Forest model
Evaluate performance & analyze feature importance

6️⃣ Predict Future Churners

python churn_prediction.py
Generates a Predictions.csv file with potential churners

# 📈 Results & Key Insights
Churn Rate Analysis: Contract type & payment method significantly impact churn
Customer Profile Trends: Younger customers & shorter tenure have higher churn risk
ML Model Performance: 85% accuracy in predicting churn
Business Impact: Helps telecom companies implement targeted retention strategies

# 🛠️ Technologies Used
✅ SQL Server – Data storage, ETL pipeline
✅ Power BI – Interactive dashboards & insights
✅ Python (Scikit-Learn) – Machine learning model for churn prediction
✅ Pandas & NumPy – Data preprocessing
✅ Matplotlib & Seaborn – Data visualization

# 📌 Future Improvements
🔹 Deploy as a web app using Streamlit
🔹 Implement Deep Learning (Neural Networks) for improved accuracy
🔹 Add real-time churn monitoring via automated dashboards

# 🔗 Connect With Me
If you found this project useful, feel free to connect!

📩 Email: omkarrajeshmagare@gmail.com
💼 LinkedIn: linkedin.com/in/omkarrajeshmagare
🚀 GitHub: github.com/omkarmagare27
