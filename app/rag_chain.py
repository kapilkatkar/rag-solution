from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import shutil

# Load API key from .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env")

os.environ["OPENAI_API_KEY"] = openai_api_key  

# Load and split .txt documents
def load_documents():
    print("Loading documents...")
    loader = TextLoader("app/documents/doc1.txt", encoding="utf-8")
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    
    return chunks

# Create vector store (No need to manually persist)
def get_vectorstore(docs):
    print("Creating vector store...")
    embeddings = OpenAIEmbeddings()
    persist_directory = "app/chroma_db"
    
    # If directory exists, remove it to reset the Chroma DB
    if os.path.exists(persist_directory):
        print(f"Removing existing Chroma DB directory: {persist_directory}")
        shutil.rmtree(persist_directory)
    
    # Create a new vector store (Chroma will automatically persist the data now)
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    print(f"Vector store created with {len(docs)} documents.")
    
    return vectorstore

# Build the RAG chain
def get_rag_chain():
    print("Building RAG chain...")
    docs = load_documents()  # Load and process the documents
    vectorstore = get_vectorstore(docs)  # Create the vector store
    retriever = vectorstore.as_retriever()  # Set up the retriever
    
    # Ensure correct model name
    llm = ChatOpenAI(model="gpt-4o", temperature=0)  # Fixed the model name
    
    # Create the RetrievalQA chain
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    
    return rag_chain

# Define the dynamic prompt that asks the LLM to behave like the person in the document
def get_persona_query(query, context):
    """Create a prompt that instructs the LLM to behave as the person described in the context"""
    prompt_template = """
    You are Kapil Katkar, a front-end developer with experience in React, JavaScript, and other web technologies.
    Respond to the following query exactly as Kapil Katkar would, using only the information provided in the context below.
    Always answer as Kapil Katkar, and do not refer to yourself as an AI or a bot.

    User Query: {query}

    Context (Kapil's Details):
    {context}

    Instructions:
    - Respond only with information from the context above.
    - Always act and respond as Kapil Katkar.
    - Do not break character. Never say "I am an AI" or "I don't know".
    - Provide responses based on your personal experiences as described in the context.
    """
    
    prompt = prompt_template.format(context=context, query=query)
    return prompt

# Test the RAG chain with a query and persona behavior
def test_rag_chain(query):

    # Run the RAG chain to get relevant context (retrieved from the document)
    qa_chain = get_rag_chain()
    
    # Get the context (retrieved documents) that matches the persona in the query
    result = qa_chain.run(query)
    
    # Extract the relevant context for persona behavior (this is from the source documents)
    context = " ".join([doc.page_content for doc in result["source_documents"]])
    
    # Create the custom prompt with the context from the document
    custom_prompt = get_persona_query(query, context)
  
    # Use the LLM to respond to the query with the custom persona prompt
    llm = ChatOpenAI(model="gpt-4o", temperature=0)  # Ensure model is correctly set
    response = llm.invoke(custom_prompt)
    return response

# Example usage of the updated function
if __name__ == "__main__":
    query = "What is your experience with React?"
    result = test_rag_chain(query)
    print(f"Final Answer: {result}")
