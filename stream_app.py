import streamlit as st
from agent import ChatbotAgent
import pandas as pd

# Streamlit UI for file upload
st.title("CSV File Uploader")
uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=["csv","pdf","xlsx"])

dataframes = {}
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith('.csv'):
            df_name = uploaded_file.name.split('.')[0]
            dataframes[df_name] = uploaded_file
            st.write(f"DataFrame: {df_name}")
            df = pd.read_csv(uploaded_file)
            st.write(df.head())
            #print(dataframes)
            #st.dataframe(dataframes[df_name].head())

    # Chatbot interaction
    st.title("Chatbot Interaction")
    user_question = st.text_input("Ask a question about the dataframes:")

    if user_question:
        agent = ChatbotAgent(df)
        response = agent.answer_question(user_question)
        st.write("Chatbot Response:")
        st.write(response)
