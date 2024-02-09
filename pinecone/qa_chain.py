import os

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain_community.vectorstores import Pinecone as langchain_pinecone
from pinecone import Pinecone

def get_qa_chain(embeddings, index_name, k, llm_model_name, temperature):
    # setup retriever
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(index_name)
    vectorstore = langchain_pinecone(index, embeddings.embed_query, "text")
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})  # https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/vectorstores.py#L553

    # setup prompt
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the question based only on the provided context."
            ),
            ("human", "Context: '{context}' \n\n Question: '{question}'"),
        ]
    )

    # setup llm
    llm = ChatOpenAI(model_name=llm_model_name, temperature=temperature)

    # helper function to format docs
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    # setup chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain