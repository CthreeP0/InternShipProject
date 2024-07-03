import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from transformers import pipeline

# Initialize the Hugging Face summarization and QA pipelines
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")

# Function to read and summarize PDF files
def summarize_pdf(file):
    pdf_reader = PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split text into manageable chunks for summarization
    max_chunk_size = 1000
    text_chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

    # Use the Hugging Face summarization pipeline
    summaries = summarizer(text_chunks, max_length=130, min_length=30, do_sample=False)
    summary = " ".join([summary['summary_text'] for summary in summaries])
    return summary

# Function to handle CSV and Excel files
def handle_csv_excel(file, file_type):
    if file_type == 'csv':
        df = pd.read_csv(file)
    elif file_type == 'excel':
        df = pd.read_excel(file)
    
    st.write(df.head())

    # Example operation on the dataframe
    summary = f"The file has {df.shape[0]} rows and {df.shape[1]} columns."
    return summary

# Function to perform question answering on text
def qa_text(text, question):
    result = qa_pipeline(question=question, context=text)
    return result['answer']

# Function to read PDF and perform question answering
def qa_pdf(file, question):
    pdf_reader = PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()

    return qa_text(text, question)

# Streamlit file uploader
st.title("File Analysis App")
uploaded_file = st.file_uploader("Upload a PDF, CSV, or Excel file", type=["pdf", "csv", "xlsx"])

if uploaded_file is not None:
    file_type = uploaded_file.type
    if file_type == "application/pdf":
        st.write("PDF File Uploaded.")

        # Additional options for PDF handling
        option = st.selectbox(
            'Choose an operation',
            ('Summarize PDF', 'Question-Answer PDF')
        )

        if option == 'Summarize PDF':
            st.write("Summarizing PDF file...")
            summary = summarize_pdf(uploaded_file)
            st.write("Summary:")
            st.write(summary)

        elif option == 'Question-Answer PDF':
            question = st.text_input("Enter your question:")
            if question:
                st.write("Performing question answering on PDF...")
                answer = qa_pdf(uploaded_file, question)
                st.write("Answer:")
                st.write(answer)

    elif file_type == "text/csv":
        st.write("Handling CSV file...")
        summary = handle_csv_excel(uploaded_file, 'csv')
        st.write("Summary:")
        st.write(summary)

    elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        st.write("Handling Excel file...")
        summary = handle_csv_excel(uploaded_file, 'excel')
        st.write("Summary:")
        st.write(summary)

    else:
        st.write("Unsupported file type.")
