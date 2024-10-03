import sys, os
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from stock_analysis.tools.calculator_tool import CalculatorTool
from stock_analysis.tools.sec_tools import SEC10KTool, SEC10QTool

from crewai_tools import WebsiteSearchTool, ScrapeWebsiteTool, TXTSearchTool
from langchain_community.chat_models import ChatOpenAI


# from stock_analysis.crew import StockAnalysisCrew
import promptquality as pq
from promptquality import Scorers

all_metrics = [
    Scorers.context_adherence_plus,
]

project_name = "agent-crew-galileo"
run_name = "stock-analysis"

evaluate_handler = pq.GalileoPromptCallback(
    project_name=project_name, run_name=run_name, scorers=all_metrics
)

llm = ChatOpenAI(model="gpt-4o-mini")


@CrewBase
class StockAnalysisCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def financial_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["financial_analyst"],
            verbose=True,
            llm=llm,
            tools=[
                ScrapeWebsiteTool(),
                WebsiteSearchTool(),
                CalculatorTool(),
                SEC10QTool("AMZN"),
                SEC10KTool("AMZN"),
            ],
        )

    @task
    def financial_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["financial_analysis"],
            agent=self.financial_agent(),
            callback=evaluate_handler,
        )

    @agent
    def research_analyst_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["research_analyst"],
            verbose=True,
            llm=llm,
            tools=[
                ScrapeWebsiteTool(),
                # WebsiteSearchTool(),
                SEC10QTool("AMZN"),
                SEC10KTool("AMZN"),
            ],
        )

    @task
    def research(self) -> Task:
        return Task(
            config=self.tasks_config["research"],
            agent=self.research_analyst_agent(),
            callback=evaluate_handler,
        )

    @agent
    def financial_analyst_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["financial_analyst"],
            verbose=True,
            llm=llm,
            tools=[
                ScrapeWebsiteTool(),
                WebsiteSearchTool(),
                CalculatorTool(),
                SEC10QTool(),
                SEC10KTool(),
            ],
        )

    @task
    def financial_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["financial_analysis"],
            agent=self.financial_analyst_agent(),
            callback=evaluate_handler,
        )

    @task
    def filings_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["filings_analysis"],
            agent=self.financial_analyst_agent(),
            callback=evaluate_handler,
        )

    @agent
    def investment_advisor_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["investment_advisor"],
            verbose=True,
            llm=llm,
            tools=[
                ScrapeWebsiteTool(),
                WebsiteSearchTool(),
                CalculatorTool(),
            ],
        )

    @task
    def recommend(self) -> Task:
        return Task(
            config=self.tasks_config["recommend"],
            agent=self.investment_advisor_agent(),
            callback=evaluate_handler,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Stock Analysis"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )


pq.login("console.demo.rungalileo.io")

inputs = {
    "query": "how is the amazon AI revenue growth",
    "company_stock": "AMZN",
}
StockAnalysisCrew().crew().kickoff(inputs=inputs)

evaluate_handler.finish()
