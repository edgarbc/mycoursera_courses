
from langchain.agents import initialize_agent, AgentType, load_tools

# Set your API Key from OpenAI
openai_api_key = ''

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0, openai_api_key=openai_api_key)

# Define the tools
tools = load_tools(["llm-math"], llm=llm)

# Define the agent
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Run the agent
agent.run("What is 10 multiplied by 50?")
