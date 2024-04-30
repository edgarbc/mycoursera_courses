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


###

# Now using a RAG application and chains to do evaluation

embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
docstorage = Chroma.from_documents(docs, embedding)
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", openai_api_key=openai_api_key)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docstorage.as_retriever(), input_key="question")

# Generate the model responses using the RetrievalQA chain and question_set
predictions = qa.apply(question_set)

# Define the evaluation chain
eval_chain = QAEvalChain.from_llm(llm)

# Evaluate the ground truth against the answers that are returned
results = eval_chain.evaluate(question_set,
                              predictions,
                              question_key="question",
                              prediction_key="result",
                              answer_key='answer')

for i, q in enumerate(question_set):
    print(f"Question {i+1}: {q['question']}")
    print(f"Expected Answer: {q['answer']}")
    print(f"Model Prediction: {predictions[i]['result']}\n")
    
print(results)
