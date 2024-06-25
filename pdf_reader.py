import os
import streamlit as st
from dotenv import load_dotenv
import PyPDF2
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain_openai import AzureChatOpenAI

# Load environment variables from .env file if you have one
load_dotenv()

# Set the OpenAI API key and endpoint for Azure OpenAI Service
os.environ["OPENAI_API_KEY"] = "9cb47c3891a34c2e8c1ec63d8965bf2e"
os.environ["OPENAI_API_ENDPOINT"] = "https://ptsg-5cvm-oai01.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"

# Enable tracing for LangChain
os.environ["LANGCHAIN_TRACING_V2"] = "True"

st.title("PDF Summarizer with Azure OpenAI")

# File uploader widget
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Read the PDF file with PyPDF2
    pdf_reader = PyPDF2.PdfFileReader(uploaded_file)
    docs = []
    for page_num in range(pdf_reader.numPages):
        page = pdf_reader.getPage(page_num)
        text = page.extract_text()
        # Create a dictionary with the required structure
        doc = {
            "text": text,  # Provide the extracted text
            "metadata": {"page_number": page_num + 1}  # Optional metadata
        }
        docs.append(doc)

    # Define prompt
    prompt_template = """Write a 500 word summary of the following:
    "{text}"
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    # Define LLM
    llm = AzureChatOpenAI(
        deployment_name="gpt-35-turbo-16k",
        azure_endpoint=os.getenv("OPENAI_API_ENDPOINT"),
        openai_api_type="azure",
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0
    )

    # Define LLM chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

    # Invoke the chain and print the result
    result = stuff_chain.invoke(docs)
    st.write(result["output_text"])
