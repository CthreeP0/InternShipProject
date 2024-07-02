import pandas as pd
import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import ToolMessage


class ChatbotAgent:
    def __init__(self, dataframes):
        self.dataframes = {name: pd.read_csv(df) for name, df in dataframes.items()}
        self.llm = AzureChatOpenAI(
            deployment_name="gpt-35-turbo-16k",
            azure_endpoint=st.secrets['OPENAI_API_ENDPOINT'],
            openai_api_type="azure",
            openai_api_version="2023-07-01-preview"
        )
        self.tool = PythonAstREPLTool(locals=self.dataframes)
        self.llm_with_tool = self.llm.bind_tools(tools=[self.tool], tool_choice=self.tool.name)

    def answer_question(self, question):
        df_context = "\n\n".join(
            f"```python\n{df_name}.head().to_markdown()\n>>> {_df.head().to_markdown()}\n```"
            for df_name, _df in self.dataframes.items()
        )

        system_prompt = f"""You have access to a number of pandas dataframes. \
Here is a sample of rows from each dataframe and the python code that was used to generate the sample:

{df_context}

Given a user question about the dataframes, write the Python code to answer it. \
Don't assume you have access to any libraries other than built-in Python ones and pandas. \
Make sure to refer only to the variables mentioned above."""

        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", question)])
        parser = JsonOutputKeyToolsParser(key_name=self.tool.name, first_tool_only=True)

        def _get_chat_history(x):
            ai_msg = x["ai_msg"]
            tool_call_id = x["ai_msg"].additional_kwargs["tool_calls"][0]["id"]
            tool_msg = ToolMessage(tool_call_id=tool_call_id, content=str(x["tool_output"]))
            return [ai_msg, tool_msg]

        chain = (
            RunnablePassthrough.assign(ai_msg=prompt | self.llm_with_tool)
            .assign(tool_output=lambda x: x["ai_msg"] | parser | self.tool)
            .assign(chat_history=_get_chat_history)
            .assign(response=prompt | self.llm | StrOutputParser())
            .pick(["tool_output", "response"])
        )

        result = chain.invoke({"question": question})

        if isinstance(result, dict) and 'response' in result:
            return result['response']
        else:
            return "Unexpected result format. Unable to generate a response."
