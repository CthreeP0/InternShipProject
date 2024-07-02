
import pandas as pd
import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter

from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import StrOutputParser

from utils import initialize_llm, create_prompt_template, create_chain

# Initialize Azure LLM
llm = initialize_llm()

# Streamlit UI for file upload
st.title("CSV File Uploader")
uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=["csv", "pdf", "xlsx"])

import pdfplumber
import pandas as pd

def convert_pdf_to_dataframe(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        first_page = pdf.pages[0]
        table = first_page.extract_table()
        df = pd.DataFrame(table[1:], columns=table[0])
    return df

# Don't forget to install pdfplumber
# pip install pdfplumber
dataframes = {}
if uploaded_files:
    for uploaded_file in uploaded_files:
        df_name = uploaded_file.name.split('.')[0]
        if uploaded_file.name.endswith('.csv'):
            dataframes[df_name] = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            dataframes[df_name] = pd.read_excel(uploaded_file)
        #elif uploaded_file.name.endswith('.pdf'):
            # Assuming you have a function to handle PDF conversion to dataframe
            # dataframes[df_name] = convert_pdf_to_dataframe(uploaded_file) 

        st.write(f"DataFrame: {df_name}")
        st.dataframe(dataframes[df_name].head())


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

# Define chain
prompt_template = create_prompt_template(system_prompt)
chain = create_chain(llm, dataframes, prompt_template)

# Chatbot interaction
st.title("Chatbot Interaction")
user_question = st.text_input("Ask a question about the dataframes:")

if user_question:
    result = chain.invoke({"question": user_question})
    st.write("Debugging Result:")
    st.write(result)  # This will help us see the structure of `result`
    
    if isinstance(result, dict) and 'response' in result:
        st.write("Chatbot Response:")
        st.write(result['response'])
    else:
        st.write("Unexpected result format. Here is the raw result:")
        st.write(result)
