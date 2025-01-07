import os
from dotenv import load_dotenv

# Importing necessary classes and functions from langchain and related libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variable from a .env file
load_dotenv()

def setup_qa_system(file_path):
    """
    Set up a Question-Answering (QA) system using a PDF document.

    Args:
        file_path (str): The path to the PDF file to be used as the knowledge base.

    Returns:
        RetrievalQA: An instance of of the QA chain ready to answer questions.
    """
    # Initialize the PDF loader with the provided file path
    loader = PyPDFLoader(file_path)

    # Load and split the PDF into individual pages or sections
    docs = loader.load_and_split()

    # Initialize a text splitted to divide documents into managable chucnks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # Split the loaded documents into smaller text chunks
    chunks = text_splitter.split_documents(docs)

    # Initialize the embeddings model to convert text chunks into vector representations
    embeddings = OpenAIEmbeddings()

    # Create a FAISS vector store from the document chunks and their embeddings
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Convert the vector store into a retriever object for fetching relevant documents
    retriever = vector_store.as_retriever()

    # Initialize the language model (LLM) with specified parameters
    llm = ChatOpenAI(temperature=0, model_name='gpt-4o')

    # Create a RetrievalQA chain that uses the LLM and retriever to answer questions
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    return qa_chain


if __name__ == "__main__":
    # Set up the QA system using the specified PDF file
    qa_chain = setup_qa_system('new.pdf')

    # Start an interactive loop to accept user questions
    while True:
        question = input('\nAsk a question: ')

        # Exit the loop if the user types 'exit'
        if question.lower() == 'exit':
            break
        
        # Invoke the QA chain with the user's question
        answer = qa_chain.invoke(question)

        # Print the answer returned by the QA system
        print('Answer: ')
        print(answer['result'])
        
        
