import streamlit as st
import pandas as pd
import joblib
kmeans=joblib.load("kmeans_model.pkl")
scalar=joblib.load("scalar.pkl")
st.title("Custom Segmentation App")
st.write("Enter customer details to predict the segment.")


age=st.number_input("Age",min_value=18,max_value=120,value=30)
income=st.number_input("Income",min_value=0,max_value=200000,value=40000)
total_spending=st.number_input("Total Spending(sum of purchases)",min_value=0,max_value=200000,value=1100)
num_web_purchases=st.number_input("Number of Web Purchases",min_value=0,max_value=200000,value=12)
num_store_purchases=st.number_input("Number of Store Purchases",min_value=0,max_value=2000,value=10)
num_web_visits=st.number_input("Number of Web Visits per month",min_value=0,max_value=2000,value=12)
recency=st.number_input("Recency(days since last purchase)",min_value=0,max_value=365,value=23)

input_data=pd.DataFrame({
    "Age":[age],
    "Income":[income],
     "Total_Spending": [total_spending],
    "NumWebPurchases": [num_web_purchases],
    "NumStorePurchases": [num_store_purchases],
    "NumWebVisitsMonth": [num_web_visits],
    "Recency": [recency]
})
input_scaled=scalar.transform(input_data)
if st.button("Predict Segment"):
     cluster=kmeans.predict(input_scaled)[0]
     st.success(f"The cluster is: {cluster}")
st.write(

             "Cluster 0 : High Budget,Web Visitors\n"
             "Cluster 1 :  Low Income, Low Spending\n"
             "Cluster 2 :  High Budget, Store-Focused\n"
             "Cluster 3 : Medium Income, Web Explorers\n"
            "cluster 4 :  Inactive Customers"
            )