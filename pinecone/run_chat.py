
import random
from dotenv import load_dotenv

import pandas as pd
import promptquality as pq
from tqdm import tqdm

from common import get_indexing_configuration
from metrics import all_metrics
from qa_chain import get_qa_chain

load_dotenv("../.env")

#fixed variables
project_name = "feb12-qa"
temperature = 0.1
# questions_per_conversation = 5

indexing_config, qa_config = 1, 1
# indexing_config, qa_config = 2, 1
# indexing_config, qa_config = 3, 1
# indexing_config, qa_config = 4, 1
# indexing_config, qa_config = 5, 1
# indexing_config, qa_config = 5, 2
# indexing_config, qa_config = 5, 3


_, embeddings, emb_model_name, dimension, index_name = get_indexing_configuration(indexing_config)


if qa_config == 1:
    llm_model_name, llm_identifier, k = "gpt-3.5-turbo-1106", "3.5-1106", 20
if qa_config == 2:
    llm_model_name, llm_identifier, k = "gpt-3.5-turbo-1106", "3.5-1106", 15
elif qa_config == 3:
    llm_model_name, llm_identifier, k = "gpt-3.5-turbo-0125", "3.5-0125", 15

pq.login("console.staging.rungalileo.io")

# Prepare questions for the conversation
df = pd.read_csv("../data/bigbasket_beauty.csv")
df["questions"] = df["questions"].apply(eval)
questions = df.explode("questions")["questions"].tolist()
random.Random(0).shuffle(questions)
# split questions into chunks of 5
# questions = [questions[i : i + questions_per_conversation] for i in range(0, len(questions), questions_per_conversation)]
questions = questions[:100] # selecting only first 100 turns

qa = get_qa_chain(embeddings, index_name, k, llm_model_name, temperature)
run_name = f"{index_name}-{llm_identifier}-k{k}"
index_name_tag = pq.RunTag(key="Index config", value=index_name, tag_type=pq.TagType.RAG)
encoder_model_name_tag = pq.RunTag(key="Encoder", value=emb_model_name, tag_type=pq.TagType.RAG)
llm_model_name_tag = pq.RunTag(key="LLM", value=llm_model_name, tag_type=pq.TagType.RAG)
dimension_tag = pq.RunTag(key="Dimension", value=str(dimension), tag_type=pq.TagType.RAG)
topk_tag = pq.RunTag(key="Top k", value=str(k), tag_type=pq.TagType.RAG)

evaluate_handler = pq.GalileoPromptCallback(project_name=project_name, run_name=run_name, scorers=all_metrics, run_tags=[encoder_model_name_tag, llm_model_name_tag, index_name_tag, dimension_tag, topk_tag])

print("Ready to ask!")
for i, q in enumerate(tqdm(questions)):
    print(f"Question {i}: ", q)
    print(qa.invoke(q, config=dict(callbacks=[evaluate_handler])))
    print("\n\n")

evaluate_handler.finish()