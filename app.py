from flask import Flask, render_template, request, jsonify,redirect
from langchain_core.messages import HumanMessage
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
import os
import json

from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
app = Flask(__name__)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = ""


GROQ_API_KEY = ""
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Load initial documents
loader = TextLoader(r"enter your requirment file path")
loader = TextLoader(r"C:\Users\KIIT\Documents\GitHub\mlsa_hack\chatbot\chatbot\map_description.txt")
documents = loader.load()

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
text_splitter = CharacterTextSplitter(
    separator="/n",
    chunk_size=1000,
    chunk_overlap=200
)

doc_chunks = text_splitter.split_documents(documents)
vectorstore = FAISS.from_documents(doc_chunks, embeddings)


# Initialize LLM and memory
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.9
)
retriever = vectorstore.as_retriever()
memory = ConversationBufferMemory(
    llm=llm,
    output_key="answer",
    memory_key="chat_history",
    return_messages=True
)

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    chain_type="map_reduce",
    memory=memory,
    verbose=True
)

def load_json_data():
    """Load data from update.json and update the vector store."""
    try:
        with open('update.json', 'r') as f:
            json_data = json.load(f)
            # Assuming json_data is a list of text documents
            new_documents = [HumanMessage(content=item) for item in json_data]
            new_doc_chunks = text_splitter.split_documents(new_documents)
            vectorstore.add_documents(new_doc_chunks, embeddings)
            print("Vector store updated with new documents from update.json.")
    except Exception as e:
        print(f"Error loading JSON data: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    prompt = "You are a helpful bot in apocalypse help user to navigate safaly dont put them in High-Density Areas always provide route to safe zone. user input= "  
    full_input = prompt + user_input
   
    # Invoke the model and get a response
    output = chain.invoke({"question": full_input})
    bot_response = output.get("answer")

    return jsonify({"response": bot_response})

@app.route("/update", methods=['POST'])
def update():
        url = "https://api.mlsakiit.com/survivors"
        response = requests.get(url)

        if response.status_code == 200:
            json_data = response.json()  # Step 2: Parse the JSON data
        else:
            return jsonify({"error": "Failed to fetch data"}), response.status_code

        
        documents = []
        for item in json_data:
            # Create a document string from the survivor data
            document_content = f"Survivor ID: {item['survivor_id']}, District: {item['district']}, Latitude: {item['lat']}, Longitude: {item['lon']}"
            # Create a Document object
            documents.append(Document(page_content=document_content))

        # Split documents into chunks
        doc_chunks = text_splitter.split_documents(documents)

        # Step 4: Update the vector store
        vectorstore.add_documents(doc_chunks)
        return "updated"

if __name__ == '__main__':
    app.run(debug=True)
