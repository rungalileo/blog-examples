from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_community.chat_models import ChatPerplexity
from langchain.chains import LLMChain
from langchain import PromptTemplate

from dotenv import load_dotenv

load_dotenv()

# Initialize tools
tavily_tool = TavilySearchResults(max_results=1)

# You can create the tool to pass to an agent
python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

# perplexity tool
perplexity_llm = ChatPerplexity(
    temperature=0, model="llama-3.1-sonar-small-128k-online"
)

perplexity_prompt = PromptTemplate(
    template="""Given the following overall question `{input}`.

    Perform the task by understanding the problem, extracting variables, and being smart
    and efficient. Write a detailed response that address the task.
    When confronted with choices, make a decision yourself with reasoning.
    """,
    input_variables=["input"],
)

llm_chain = LLMChain(llm=perplexity_llm, prompt=perplexity_prompt)

perplexity_tool = Tool(
    name="Reason",
    func=llm_chain.run,
    description="Reason about task via existing information or understanding. Make decisions / selections from options.",
)
