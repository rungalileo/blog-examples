from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from promptquality import EvaluateRun
import promptquality as pq
from tools import tavily_tool, repl_tool, perplexity_tool

load_dotenv()

# Create an LLM instance
llm = ChatOpenAI(model="gpt-4o-mini")

# setup galileo callback
run_name = "stock-analysis-1"
metrics = [pq.Scorers.context_adherence_plus]
evaluate_run = EvaluateRun(run_name=run_name, project_name="agent-exp", scorers=metrics)


pq.login("console.demo.rungalileo.io")

input = {
    "messages": [
        HumanMessage("Should we invest in Tesla given the current situation of EV?")
    ]
}

agent = create_react_agent(llm, [tavily_tool])
output = agent.invoke(input)

print("\n\n\n\n START")
print(output)
print("\n\n\n\n END")


def convert_langchain_output_to_messages_list(data):
    # A mapping from message class to role
    role_map = {"HumanMessage": "user", "AIMessage": "assistant", "ToolMessage": "tool"}

    # Function to get the role from a message object
    def get_role(message):
        message_class = message.__class__.__name__
        return role_map.get(message_class, "unknown")

    # Extract the messages
    messages = data.get("messages", [])
    converted_messages = []

    for message in messages:
        role = get_role(message)
        converted_messages.append({"role": role, "content": message.content})

    return converted_messages


agent_wf = evaluate_run.add_agent_workflow(
    input=[
        {
            "role": "user",
            "content": "Should we invest in Tesla given the current situation of EV?",
        }
    ],
    output=convert_langchain_output_to_messages_list(output),
    duration_ns=100,
)
evaluate_run.finish()
