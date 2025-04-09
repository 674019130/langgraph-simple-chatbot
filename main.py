from init_env import _set_env

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_doc_from_web(urls: list[str]) -> list[Document]:
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return docs


def splict_docs(docs: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlad=50
    )
    text_splitter.split_documents(docs)


def add_to_vector_store(docs: list[Document]):
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(),
        collection_name="rag_chroma",
    )

    return vectorstore.as_retriever()


if __name__ == "__main__":
    _set_env("OPENAI_API_KEY")
    urls = [
        "https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag"
    ]
    docs = load_doc_from_web(urls)
    splict_docs(docs)
    add_to_vector_store(docs)
