from langchain.evaluation import Criteria
from langchain.chat_models import ChatOpenAI
from langchain.evaluation import load_evaluator

# Set your API Key from OpenAI
openai_api_key = 'OPENAI_KEY'

# Load evaluator, assign it to criteria, and return result
evaluator = load_evaluator("criteria", criteria="relevance", llm=ChatOpenAI(openai_api_key=openai_api_key))

# Evaluate the input and prediction
eval_result = evaluator.evaluate_strings(
    prediction="42",
    input="What is the answer to the ultimate question of life, the universe, and everything?",
)

print(eval_result)
