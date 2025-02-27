from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import shutil
from flask import Flask, render_template, jsonify, request
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import HuggingFaceHub
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

app = Flask(__name__)

# Load environment variables
load_dotenv()
HUGGINGFACE_API_KEY = os.environ.get('HUGGINGFACE_API_KEY')
os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Ensure ChromaDB is reset if needed
persist_directory = "db"
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

# Initialize ChromaDB
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Load a better text-generation model
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
    model_kwargs={"temperature": 0.7, "max_length": 512}
)

# Conversation memory
memory = ConversationSummaryMemory(llm=llm, memory_key="history", return_messages=True)

# Conversational retrieval chain
qa = ConversationalRetrievalChain.from_llm(
    llm, retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 8}), memory=memory
)

@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)

    if input.lower() == "clear":
        os.system("rm -rf repo")

    result = qa.invoke({"question": input})
    print(result['answer'])
    return jsonify({"response": result["answer"]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
