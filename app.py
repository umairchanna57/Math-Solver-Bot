from flask import Flask, request, jsonify, render_template
from langchain.llms import HuggingFaceEndpoint
from langchain.tools import Tool
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType
from langchain.utilities import WikipediaAPIWrapper

app = Flask(__name__)

api_key = 'your Hugging Face API key'
llm = HuggingFaceEndpoint(
    huggingfacehub_api_token=api_key,
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.8,
    max_length=150
)


wikipedia = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia.run,
    description="A useful tool for searching the Internet to find information on world events, issues, dates, years, etc. Worth using for general topics. Use precise questions."
)


problem_chain = LLMMathChain.from_llm(llm=llm)
math_tool = Tool.from_function(
    name="Calculator",
    func=problem_chain.run,
    description="Useful for when you need to answer questions about math. This tool is only for math questions and nothing else. Only input math expressions."
)


word_problem_template = """You are a reasoning agent tasked with solving the user's logic-based questions. Logically arrive at the solution and be factual. In your answers, YOUR FINAL ANSWER SHOULD BE IN DETAILS NOT ONLY ANSWER clearly detail the steps involved and give the final answer. Provide the response in bullet points, and ensure all explanations are included in the final answer.
Question: {input}
Answer:"""

math_assistant_prompt = PromptTemplate(
    input_variables=["input"],
    template=word_problem_template
)
word_problem_chain = LLMChain(llm=llm, prompt=math_assistant_prompt)
word_problem_tool = Tool.from_function(
    name="Reasoning Tool",
    func=word_problem_chain.run,
    description="Useful for when you need to answer logic-based/reasoning questions."
)


agent = initialize_agent(
    tools=[wikipedia_tool, math_tool, word_problem_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/solve', methods=['POST'])
def solve():
    input_data = request.json
    response = agent.invoke({"input": input_data['question']})
    answer = response['output']
    return answer

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
