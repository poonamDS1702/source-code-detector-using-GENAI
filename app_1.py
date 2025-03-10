from langchain.vectorstores import Chroma
from src.helper import load_embedding
from dotenv import load_dotenv
import os
from src.helper import repo_ingestion
from flask import Flask, render_template, jsonify, request
#from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "sentence-transformers"], check=True)
#subprocess.run(["pip", "install", "sentence-transformers"], check=True)
subprocess.run(["pip", "install", "chromadb"], check=True)


app = Flask(__name__)


load_dotenv()

HUGGINGFACE_API_KEY = os.environ.get('HUGGINGFACE_API_KEY')
os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY


embeddings = load_embedding()
persist_directory = "db"
# Now we can load the persisted database from disk, and use it as normal.
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)



from langchain.chat_models import HuggingFaceHub
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
import os

# Set Hugging Face API Key
os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY

from transformers import pipeline
from langchain.llms import HuggingFacePipeline

text_generator = pipeline("text-generation", model="gpt2")
llm = HuggingFacePipeline(pipeline=text_generator)

# Load a Hugging Face chat model
#llm = HuggingFaceHub(
#   repo_id="sentence-transformers/all-MiniLM-L6-v2", huggingfacehub_api_token=HUGGINGFACE_API_KEY,# Replace with your preferred HF model
#   model_kwargs={"temperature": 0.7, "max_length": 512}
#)

# Conversation memory
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)

# Conversational retrieval chain
qa = ConversationalRetrievalChain.from_llm(
    llm, retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 8}), memory=memory
)




@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')



@app.route('/chatbot', methods=["GET", "POST"])
def gitRepo():

    if request.method == 'POST':
        user_input = request.form['question']
        repo_ingestion(user_input)
        os.system("python store_index.py")

    return jsonify({"response": str(user_input) })



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)

    if input == "clear":
        os.system("rm -rf repo")

    result = qa(input)
    print(result['answer'])
    return str(result["answer"])





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)