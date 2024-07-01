from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings


app = Flask(__name__)

load_dotenv()

# Load environment variables
groq_api_key = os.getenv('GROQ_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

# Debug: Check if environment variables are loaded
print(f"GROQ_API_KEY: {groq_api_key}")
print(f"GOOGLE_API_KEY: {google_api_key}")

# Initialize session state variables
def initialize_session_state():
    if not hasattr(initialize_session_state, 'vectors'):
        try:
            print("Initializing session state...")
            initialize_session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            initialize_session_state.loader = PyPDFDirectoryLoader("./pdfFiles")  # Data Ingestion
            initialize_session_state.docs = initialize_session_state.loader.load()  # Document Loading
            initialize_session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=1000)  # Chunk Creation
            initialize_session_state.final_documents = initialize_session_state.text_splitter.split_documents(initialize_session_state.docs[:20])  # Splitting
            initialize_session_state.vectors = FAISS.from_documents(initialize_session_state.final_documents, initialize_session_state.embeddings)  # Vector creation
            print("Session state initialized successfully.")
        except Exception as e:
            print(f"Error initializing session state: {e}")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma-7b-it")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
        {context}
    <context>
    Questions:{input}
    """
)

def retrieve_documents(input_prompt):
    try:
        initialize_session_state()

        # Create document chain and retrieval chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = initialize_session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Measure response time
        start = time.process_time()
        response = retrieval_chain.invoke({'input': input_prompt})
        response_time = time.process_time() - start

        # Check if no answer found
        if response.get('answer', '') == 'No input provided':
            return {
                'response_time': response_time,
                'answer': f'For more information, visit http://vcet.edu.in',
                'context': []
            }

        # Return normal response
        return {
            'response_time': response_time,
            'answer': response.get('answer', 'No answer found.'),
            'context': [doc.page_content for doc in response.get("context", [])]
        }
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return {
            'response_time': 0,
            'answer': 'Error retrieving documents.',
            'context': []
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_prompt = data.get('message', '')

    # Define greeting keywords and responses
    greetings = {
        "Hi": "Hello! How can I assist you today?",
        "hi": "Hello! How can I assist you today?",
        "Hello": "Hi there! What can I do for you?",
        "hello": "Hi there! What can I do for you?",
        "thanks": "You're welcome! Is there anything else I can help with?",
        "Thanks": "You're welcome! Is there anything else I can help with?",
        "thank you": "You're welcome! Is there anything else I can help with?",
        "Thank you": "You're welcome! Is there anything else I can help with?",
        "goodbye": "Goodbye! Have a great day!",
        "Goodbye": "Goodbye! Have a great day!",
        "Bye": "Goodbye! Have a great day!",
        "bye": "Goodbye! Have a great day!"
    }

    # Check for greetings
    for greeting in greetings:
        if greeting in input_prompt:
            return jsonify({'answer': greetings[greeting]})

    # Process input if not a greeting
    if input_prompt:
        response = retrieve_documents(input_prompt)
        return jsonify({
            'answer': response['answer'],
            'response_time': response['response_time'],
            'context': response['context']
        })

    return jsonify({'answer': 'No input provided'})

if __name__ == "__main__":
    app.run(debug=True)

