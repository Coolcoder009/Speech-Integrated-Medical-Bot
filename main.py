from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, tempfile
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import whisper
import google.generativeai as genai  # Google Gemini API

# Load environment variables
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set!")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set!")


# Initialize Google Gemini API
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")  # Using Gemini-Pro model

HUGGINGFACE_REPO_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"

# Function to get Tamil response using Google Gemini API
def get_tamil_response(query: str) -> str:
    try:
        prompt = f"""
        நீங்கள் ஒரு நுண்ணறிவுச் செயலி.
        கீழே உள்ள கேள்விக்கான விரிவான, தெளிவான, மற்றும் பயனுள்ள பதிலை தமிழில் 3-4 வரிகளில் வழங்கவும்.
        பதில் பொருத்தமான தகவல்களுடன் இருக்க வேண்டும், ஆனால் தேவையற்ற தகவல்கள் சேர்க்க வேண்டாம்.

        கேள்வி: {query}
        பதில் (தமிழில், 3-4 வரிகள், தெளிவாக மற்றும் முழுமையாக):
        """
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print("Error in Gemini API:", e)
        return "மன்னிக்கவும், பதில் உருவாக்குவதில் சிக்கல் ஏற்பட்டது."


print("HF_TOKEN loaded. Starting model initialization...")


def load_llm():
    return HuggingFaceEndpoint(
        repo_id=HUGGINGFACE_REPO_ID,
        task="text-generation",
        temperature=0.7,
        model_kwargs={
            "token": HF_TOKEN,
            "max_length": 512
        }
    )



CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don’t know the answer, just say that you don’t know. Don’t try to make up an answer.
Provide the answer in a way that the user understands.
Don’t provide anything outside the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""


def set_custom_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])


DB_FAISS_PATH = "vectorstores/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Global initialization of the English LLM and QA chain
llm = load_llm()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=False,
    chain_type_kwargs={'prompt': set_custom_prompt()}
)


def get_response(query: str) -> str:
    response = qa_chain.invoke({'query': query})
    return response["result"]


# Load Whisper model for transcription (using CPU)
print("Loading Whisper model...")
whisper_model = whisper.load_model("large", device="cpu")
print("Whisper model loaded.")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        if not request.message:
            return JSONResponse(status_code=400, content={"reply": "No message provided."})
        print("Received text message:", request.message)

        # Check if the input is Tamil or English
        if any("\u0B80" <= char <= "\u0BFF" for char in request.message):  # Tamil Unicode range
            reply = get_tamil_response(request.message)
        else:
            reply = get_response(request.message)

        print("Generated reply:", reply)
        return {"reply": reply}
    except Exception as e:
        print("Error in /chat endpoint:", e)
        return JSONResponse(status_code=500, content={"reply": "Sorry, an error occurred: " + str(e)})


@app.post("/audio")
async def audio(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        result = whisper_model.transcribe(tmp_path)
        transcription = result["text"].strip()
        os.remove(tmp_path)

        # If detected language is Tamil (ISO code "ta"), use Gemini API
        if result["language"] == "ta":
            llm_reply = get_tamil_response(transcription)
        else:
            llm_reply = get_response(transcription)

        return {"transcription": transcription, "reply": llm_reply}
    except Exception as e:
        print("Error in /audio endpoint:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)


