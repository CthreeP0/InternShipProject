
import pandas as pd
import streamlit as st
import pdfplumber
from langchain_openai import AzureChatOpenAI
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import StrOutputParser

from utils import initialize_llm, create_prompt_template, create_chain, create_chain_pdf

# Initialize Azure LLM
llm = initialize_llm()

# Streamlit UI for file upload
st.title("File Uploader")
uploaded_files = st.file_uploader("Upload files", accept_multiple_files=True, type=["csv", "pdf", "xlsx"])



def convert_pdf_to_dataframe(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        first_page = pdf.pages[0]
        table = first_page.extract_table()
        df = pd.DataFrame(table[1:], columns=table[0])
    return df

# Don't forget to install pdfplumber
# pip install pdfplumber
dataframes = {}
pdf_files = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        df_name = uploaded_file.name.split('.')[0]
        if uploaded_file.name.endswith('.csv'):
            dataframes[df_name] = pd.read_csv(uploaded_file)
            st.write(f"DataFrame: {df_name}")
            st.dataframe(dataframes[df_name].head())
        elif uploaded_file.name.endswith('.xlsx'):
            dataframes[df_name] = pd.read_excel(uploaded_file)
            st.write(f"DataFrame: {df_name}")
            st.dataframe(dataframes[df_name].head())
        elif uploaded_file.name.endswith('.pdf'):
            pdf_files.append(uploaded_file)
            st.write(f"File Uploaded: {df_name}")
        #elif uploaded_file.name.endswith('.pdf'):
            # Assuming you have a function to handle PDF conversion to dataframe
            # dataframes[df_name] = convert_pdf_to_dataframe(uploaded_file) 



# Convert the dataframes to markdown for the prompt
df_template = """```python
{df_name}.head().to_markdown()
>>> {df_head}
```"""
df_context = "\n\n".join(
    df_template.format(df_head=_df.head().to_markdown(), df_name=df_name)
    for df_name, _df in dataframes.items()
)



# Define system prompt
system_prompt = f"""You have access to a number of pandas dataframes. \
Here is a sample of rows from each dataframe and the python code that was used to generate the sample:

{df_context}

Given a user question about the dataframes, write the Python code to answer it. \
Don't assume you have access to any libraries other than built-in Python ones and pandas. \
Make sure to refer only to the variables mentioned above."""

# Define system prompt
system_prompt_pdf = """You have uploaded PDF files. \
Given a user question about the PDF files, provide a summary or answer as required."""


# Define chain
prompt_template = create_prompt_template(system_prompt)
chain = create_chain(llm, dataframes, prompt_template)

# Define pdf chain
#prompt_template_pdf = create_prompt_template(system_prompt_pdf)
#chain_pdf = create_chain_pdf(llm, pdf_files, prompt_template_pdf)



# Chatbot interaction
st.title("Chatbot Interaction")
user_question = st.text_input("Ask a question about the dataframes or PDF files:")

if user_question:
    if any(dataframes):
        result_csv = chain.invoke({"question": user_question})
        st.write("Debugging Result (CSV):")
        st.write(result_csv)  # This will help us see the structure of `result_csv`
        
        if isinstance(result_csv, dict) and 'response' in result_csv:
            st.write("Chatbot Response (CSV):")
            st.write(result_csv['response'])
        else:
            st.write("Unexpected result format (CSV). Here is the raw result:")
            st.write(result_csv)

    #if pdf_files:
       # result_pdf = chain_pdf.invoke({"question": user_question})
       # st.write("Debugging Result (PDF):")
        #st.write(result_pdf)  # This will help us see the structure of `result_pdf`
        
       # if isinstance(result_pdf, dict) and 'response' in result_pdf:
       #     st.write("Chatbot Response (PDF):")
       #     st.write(result_pdf['response'])
       # else:
       #     st.write("Unexpected result format (PDF). Here is the raw result:")
        #    st.write(result_pdf)