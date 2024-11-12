import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def dashboard_page():
    st.title("Dashboard")

    #Load data
    data = pd.read_csv("data/traindata.csv")

    st.header("Data Overview")
    st.write("Here is a quick summary of the dataset")
    st.dataframe(data.head())

    st.subheader("Churn Count")
    churn_count = data['Churn'].value_counts()
    st.bar_chart(churn_count)


    #Plot Correlation
    st.subheader("Correlation Heatmap")
    numeric_data = data.select_dtypes(include=['number'])
    corr = numeric_data.corr()
    plt.figure(figsize=(10,6))
    sns.heatmap(corr, annot=True, cmap = "coolwarm")
    st.pyplot(plt)

    