import os

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain_community.vectorstores import Pinecone as langchain_pinecone
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from pinecone import Pinecone

def get_qa_chain(embeddings, index_name, emb_k, rerank_k, llm_model_name, temperature):
    # setup retriever
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(index_name)
    vectorstore = langchain_pinecone(index, embeddings.embed_query, "text")
    compressor = CohereRerank(model="rerank-english-v2.0", top_n=rerank_k)
    retriever = vectorstore.as_retriever(search_kwargs={"k": emb_k})  # https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/vectorstores.py#L553
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )

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
    llm = ChatOpenAI(model_name=llm_model_name, temperature=temperature, tiktoken_model_name="cl100k_base")

    # helper function to format docs
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    # setup chain
    rag_chain = (
        {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain