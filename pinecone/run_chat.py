
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

project_name = "feb7"
# index_name = "garnier-4000-200"
index_name = "garnier-200-50"
llm_model_name="gpt-3.5-turbo-1106"
emb_model_name = "text-embedding-3-small"
questions_per_conversation = 5
temperature = 0.1
k = 4
run_name = f"{index_name}-k-{k}"

metrics = [
    Scorers.latency,
    Scorers.pii,
    Scorers.toxicity,
    Scorers.tone,
    Scorers.correctness,
    #rag metrics below
    Scorers.context_adherence,
    Scorers.completeness_gpt,
    Scorers.chunk_attribution_utilization_gpt,
    # Uncertainty, BLEU and ROUGE are automatically included
]

#Custom scorer for response length
def executor(row) -> float:
    return len(row.response)

def aggregator(scores, indices) -> dict:
    return {'Response Length': sum(scores)/len(scores)}

length_scorer = pq.CustomScorer(name='Response Length', executor=executor, aggregator=aggregator)
metrics.append(length_scorer)
galileo_handler = pq.GalileoPromptCallback(project_name=project_name, run_name=run_name, scorers=metrics)

pq.login("console.dev.rungalileo.io")

# Prepare questions for the conversation
df = pd.read_csv("../data/bigbasket_garnier.csv")
df["questions"] = df["questions"].apply(eval)
questions = df.explode("questions")["questions"].tolist()
random.Random(0).shuffle(questions)
# split questions into chunks of 5
questions = [questions[i : i + questions_per_conversation] for i in range(0, len(questions), questions_per_conversation)]

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(index_name)
embeddings = OpenAIEmbeddings(model=emb_model_name)
vectorstore = langchain_pinecone(index, embeddings.embed_query, "text")
retriever = vectorstore.as_retriever(search_kwargs={"k": k})  # https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/vectorstores.py#L553
llm = ChatOpenAI(model_name=llm_model_name, temperature=temperature)

print("Ready to chat!")
for question_chunk in tqdm(questions):
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
    
    for q in question_chunk:
        print("Question: ", q)
        print(qa.run(q, callbacks=[galileo_handler]))
        print("\n\n")
galileo_handler.finish()