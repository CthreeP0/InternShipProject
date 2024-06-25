from langchain_experimental.tools import PythonAstREPLTool
from langchain_openai import AzureChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
import os
from langchain_core.prompts import format_document
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st
from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser

from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.tools import PythonAstREPLTool
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from operator import itemgetter

from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough


llm = AzureChatOpenAI(
            deployment_name = "gpt-35-turbo-16k",
            azure_endpoint= st.secrets['OPENAI_API_ENDPOINT'],
            openai_api_type="azure",
            openai_api_version = "2023-07-01-preview"
        )



data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45],
  "fat" : ["tired","hungry","Energetic"]

}

data2 = {
    "x" : [1,2] ,
    "y" : ['a','b']
}

#df1 = pd.DataFrame(data, index = ["day1", "day2", "day3"])
df1 = pd.DataFrame(data)
df2 = pd.DataFrame(data2)
print(df1) 
print(df2)

#Chain
ai_msg = llm.invoke(
    "I have a pandas DataFrame 'df1' with columns 'calories' and 'duration'. Write code to compute the correlation between the two columns. Return Markdown for a Python code snippet and nothing else."
)
print(ai_msg.content)

correlation = df1['calories'].corr(df1['duration'])
correlation


tool = PythonAstREPLTool(locals={"df": df1})

llm_with_tools = llm.bind_tools([tool], tool_choice=tool.name)
llm_with_tools.invoke(
    "I have a dataframe 'df1' and want to know the correlation between the 'calories' and 'duration' columns"
)


parser = JsonOutputKeyToolsParser(key_name=tool.name, first_tool_only=True)
(llm_with_tools | parser).invoke(
    "I have a dataframe 'df1' and want to know the correlation between the 'calories' and 'duration' columns"
)

parser = JsonOutputKeyToolsParser(key_name=tool.name, first_tool_only=True)
(llm_with_tools | parser).invoke(
    "I have a dataframe 'df1' and want to know the correlation between the 'calories' and 'duration' columns"
)




system = f"""You have access to a pandas dataframe `df`. \
Here is the output of `df.head().to_markdown()`:
{df1.head().to_markdown()}

Given a user question, write the Python code to answer it. \
Don't assume you have access to any libraries other than built-in Python ones and pandas.
Respond directly to the question once you have enough information to answer it."""
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system,
        ),
        ("human", "{question}"),
        # This MessagesPlaceholder allows us to optionally append an arbitrary number of messages
        # at the end of the prompt using the 'chat_history' arg.
        MessagesPlaceholder("chat_history", optional=True),
    ]
)


def _get_chat_history(x: dict) -> list:
    """Parse the chain output up to this point into a list of chat history messages to insert in the prompt."""
    ai_msg = x["ai_msg"]
    tool_call_id = x["ai_msg"].additional_kwargs["tool_calls"][0]["id"]
    tool_msg = ToolMessage(tool_call_id=tool_call_id, content=str(x["tool_output"]))
    return [ai_msg, tool_msg]


chain = (
    RunnablePassthrough.assign(ai_msg=prompt | llm_with_tools)
    .assign(tool_output=itemgetter("ai_msg") | parser | tool)
    .assign(chat_history=_get_chat_history)
    .assign(response=prompt | llm | StrOutputParser())
    .pick(["tool_output", "response"])
)

chain.invoke({"question": "What's the correlation between calories and duration"})



df_1 = df1
df_2 = df2
df_3 = df3

tool = PythonAstREPLTool(locals={"df_1": df_1, "df_2": df_2, "df_3":df_3})
llm_with_tool = llm.bind_tools(tools=[tool], tool_choice=tool.name)
df_template = """```python
{df_name}.head().to_markdown()
>>> {df_head}
```"""
df_context = "\n\n".join(
    df_template.format(df_head=_df.head().to_markdown(), df_name=df_name)
    for _df, df_name in [(df_1, "df_1"), (df_2, "df_2"),(df_3, "df_3")]
)

system = f"""You have access to a number of pandas dataframes. \
Here is a sample of rows from each dataframe and the python code that was used to generate the sample:

{df_context}

Given a user question about the dataframes, write the Python code to answer it. \
Don't assume you have access to any libraries other than built-in Python ones and pandas. \
Make sure to refer only to the variables mentioned above."""
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])

chain = prompt | llm_with_tool | parser | tool
chain.invoke(
    {
        #"question": "Calculate the difference between calories and x. Next return the name with the highest difference and the condition of the its resigned column."
        #"question": "List the names",
        "question": "Which name from the column name has the highest difference between calories and x."
        #"question": "Calculate the difference between x and calories ?"
    }
)