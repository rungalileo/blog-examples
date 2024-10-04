from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from langgraph.prebuilt import create_react_agent
from promptquality import EvaluateRun
import promptquality as pq
from tools import tavily_tool, repl_tool, perplexity_tool

load_dotenv()

# Create an LLM instance
llm = ChatOpenAI(model="gpt-4o-mini")

# setup galileo callback
run_name = "stock-analysis"
metrics = [pq.Scorers.context_adherence_plus]
evaluate_run = EvaluateRun(run_name="my_run", project_name="agent-exp", scorers=metrics)


pq.login("console.demo.rungalileo.io")

input = {
    "messages": [
        (
            "user",
            "Should we invest in Tesla given the current situation of EV?",
        )
    ]
}

agent = create_react_agent(llm, [tavily_tool])
output = agent.invoke(input)
agent_wf = evaluate_run.add_agent_workflow(input=input, output=output, duration_ns=100)
evaluate_run.finish()
