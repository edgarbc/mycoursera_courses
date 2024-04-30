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


custom_criteria = {"simplicity": "Does the language use brevity?", 
		"bias": "Does the language stay free of human bias?", 
		"clarity": "Is the writing easy to understand?",
		"truthfulness": "Is the writing honest and factual?"}

evaluator = load_evaluator("criteria", criteria=custom_criteria, 
			llm=ChatOpenAI(open_api_key= open_api_key)

eval_result = evluator.evaluate_strings (
	input = "What is the best Italian restaurantin New York City?", 
	prediction = "That is a subjective statement and I can not answer that.")

print(eval_result)
