
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

def initialize_llm():
    return AzureChatOpenAI(
        deployment_name="gpt-35-turbo-16k",
        azure_endpoint=st.secrets['OPENAI_API_ENDPOINT'],
        openai_api_type="azure",
        openai_api_version="2023-07-01-preview"
    )

def create_prompt_template(system_prompt):
    return ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])

def create_chain(llm, dataframes, prompt_template):
    tool = PythonAstREPLTool(locals=dataframes)
    llm_with_tool = llm.bind_tools(tools=[tool], tool_choice=tool.name)
    parser = JsonOutputKeyToolsParser(key_name=tool.name, first_tool_only=True)
    
    def _get_chat_history(x: dict) -> list:
        ai_msg = x["ai_msg"]
        tool_call_id = x["ai_msg"].additional_kwargs["tool_calls"][0]["id"]
        tool_msg = ToolMessage(tool_call_id=tool_call_id, content=str(x["tool_output"]))
        return [ai_msg, tool_msg]

    return (
        RunnablePassthrough.assign(ai_msg=prompt_template | llm_with_tool)
        .assign(tool_output= itemgetter("ai_msg") | parser | tool)
        .assign(chat_history=_get_chat_history)
        .assign(response=prompt_template | llm | StrOutputParser())
        .pick(["tool_output", "response"])
    )
