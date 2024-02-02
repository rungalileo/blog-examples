
import sys, os, random

from tqdm import tqdm

# tqdm.pandas()

import pandas as pd
from pinecone import Pinecone

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import Pinecone as langchain_pinecone

from dotenv import load_dotenv

load_dotenv("../.env")

import promptquality as pq
from promptquality import Scorers
from promptquality import SupportedModels

project_name = "pinecone-webinar-feb2"
run_name = "default"
index_name = "webinar-default-config"
model_name="gpt-3.5-turbo-1106"
questions_per_conversation = 5
temperature = 0.1
k = 4

metrics = [
    Scorers.context_adherence,
    Scorers.latency,
    Scorers.pii,
    Scorers.toxicity,
    Scorers.tone,
    # Uncertainty, BLEU and ROUGE are automatically included
]

#Custom scorer for response length
def executor(row) -> float:
    return len(row.response)

def aggregator(scores, indices) -> dict:
    return {'Response Length': sum(scores)/len(scores)}

length_scorer = pq.CustomScorer(name='Response Length', executor=executor, aggregator=aggregator)
metrics.append(length_scorer)

pq.login("console.demo.rungalileo.io")

df = pd.read_csv("../data/bigbasket_garnier.csv")
df.head()

df["questions"] = df["questions"].apply(eval)
questions = df.explode("questions")["questions"].tolist()
random.Random(0).shuffle(questions)
# split questions into chunks of 5
questions = [questions[i : i + questions_per_conversation] for i in range(0, len(questions), questions_per_conversation)]

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
embeddings = OpenAIEmbeddings()

index = pc.Index(index_name)
vectorstore = langchain_pinecone(index, embeddings.embed_query, "text")
retriever = vectorstore.as_retriever(search_kwargs={"k": k})  # https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/vectorstores.py#L553

llm = ChatOpenAI(model_name=model_name, temperature=temperature)

galileo_handler = pq.GalileoPromptCallback(project_name=project_name, run_name=run_name, scorers=metrics)

print("Ready to chat!")
for question_chunk in tqdm(questions):
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
    
    for q in question_chunk:
        print("Question: ", q)
        print(qa.run(q, callbacks=[galileo_handler]))
        print("\n\n")
galileo_handler.finish()