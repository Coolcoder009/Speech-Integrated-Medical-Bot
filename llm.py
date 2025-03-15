import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HUGGINGFACE_REPO_ID = "01-ai/Yi-34B"

# Function to load the LLM
def load_llm():
    print(HUGGINGFACE_REPO_ID)
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        task="text-generation",
        temperature=0.7,
        model_kwargs={
            "token": HF_TOKEN,
            "max_length": 512
        }
    )

# Define custom prompt template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don’t know the answer, just say that you don’t know. Don’t try to make up an answer. 
Provide the answer in a way that the user understands.
Don’t provide anything outside the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

# Function to set custom prompt
def set_custom_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

# Load FAISS vector database
DB_FAISS_PATH = "vectorstores/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=False,  # No need to return source documents for chat display
    chain_type_kwargs={'prompt': set_custom_prompt()}
)

# Function to get response
def get_response(query):
    response = qa_chain.invoke({'query': query})
    return response["result"]  # Extract the final result
