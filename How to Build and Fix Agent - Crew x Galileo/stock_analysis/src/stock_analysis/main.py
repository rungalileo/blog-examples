import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from stock_analysis.crew import StockAnalysisCrew
from promptquality import Scorers

all_metrics = [
    Scorers.latency,
    Scorers.context_adherence,
]

project_name = "agent-crew-galileo"
run_name = "stock-analysis"

evaluate_handler = pq.GalileoPromptCallback(
    project_name=project_name, run_name=run_name, scorers=all_metrics
)


inputs = {
    "query": "What is the best running shoe for beginner",
    "company_stock": "AMZN",
}
StockAnalysisCrew().crew().kickoff(inputs=inputs)

pq.login("console.demo.rungalileo.io")


run()
evaluate_handler.finish()
