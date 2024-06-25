import os
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI

# Replace deprecated import
from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser

# Set OpenAI API key and endpoint (if not using environment variables)
OPENAI_API_KEY = "9cb47c3891a34c2e8c1ec63d8965bf2e"
OPENAI_API_ENDPOINT = "https://ptsg-5cvm-oai01.openai.azure.com/"
OPENAI_API_VERSION = "2023-07-01-preview"

# Initialize the OpenAI client
openai_client = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo-0613",
    openai_api_key=OPENAI_API_KEY,  # Pass the API key directly
    api_base=OPENAI_API_ENDPOINT,
    api_version=OPENAI_API_VERSION
)

# Create the CSV agent
agent = create_csv_agent(
    openai_client,
    "titanic.csv",
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

# Run the agent with a query
response = agent.run("how many rows are there?")
print(response)
