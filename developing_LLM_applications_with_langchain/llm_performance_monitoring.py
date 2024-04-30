import time
from langchain import BaseCallbackHandler

# Set your API Key from OpenAI
openai_api_key = '<OPENAI_API_TOKEN>'

# Complete the PerformanceMonitoringCallback class to return the token and time
class PerformanceMonitoringCallback(BaseCallbackHandler):
  def on_llm_new_token(self, token: str, **kwargs) -> None:
    print(f"Token: {repr(token)} generated at time: {time.time()}")

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key=openai_api_key, temperature=0, streaming=True)
prompt_template = "Describe the process of photosynthesis."
chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))

# Call the chain with the callback
output = chain.run({}, callbacks=[PerformanceMonitoringCallback()])
print("Final Output:", output)
