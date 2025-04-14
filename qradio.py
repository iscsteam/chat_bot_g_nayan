import os
from dotenv import load_dotenv
import json
import time
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from datetime import datetime
from typing import List, Dict, Any, Tuple
import gradio as gr

# --- Configure Logging ---
log_dir = "/app/logs"

# Create a specific logger for G-Nayan-Chatbot
logger = logging.getLogger("G-Nayan-Chatbot")
logger.setLevel(logging.DEBUG)

# Create a file handler
log_file_path = f"{log_dir}/G_nayan_chatbot.log"
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

# Create a stream handler
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set the formatter for both handlers
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Load environment variables
load_dotenv()
api_key_value = os.getenv("api_key_1")

if api_key_value:
    os.environ["GROQ_API_KEY"] = api_key_value
    print("GROQ_API_KEY environment variable set successfully.")
else:
    print("Error: 'api_key_1' environment variable not found.")

# Initialize embedding model
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Load FAISS Vector Store
faiss_save_path = "./fasis_db"
vector_store_FAISS = FAISS.load_local(faiss_save_path, embed_model, allow_dangerous_deserialization=True)

# Initialize LLM for query rewriting
for_query = ChatGroq(model_name="llama3-8b-8192", temperature=0.0, max_tokens=100)

def llm_query_rewrite(query):
    """Rewrite user queries using LLM for better retrieval performance."""
    prompt = f"""
    Rewrite the following search query in a well-structured, concise format.
    Directly return **only** 2-3 query variations, with each query on a new line.
    No explanations, no numbering, no extra words.

    Input: {query}
    Output:
    """

    try:
        response = for_query.invoke(prompt)
        if hasattr(response, "content") and isinstance(response.content, str):
            queries = response.content.strip().split("\n")
            return [q.strip() for q in queries if q.strip()]
        return [query]
    except Exception as e:
        print(f"LLM Query Rewrite Error: {str(e)}")
        return [query]

def similarity_search(query, k):
    """Perform similarity search against the vector database."""
    vector_store_FAISS = FAISS.load_local(faiss_save_path, embed_model, allow_dangerous_deserialization=True)
    cleaned_queries = llm_query_rewrite(query)
    query_embedding = embed_model.embed_query(cleaned_queries[0])
    retrieved_docs = vector_store_FAISS.similarity_search_by_vector(query_embedding, k)
    return retrieved_docs

# Initialize Groq LLM Model for RAG
chat_model_rag = ChatGroq(model_name="llama3-8b-8192", temperature=0, max_tokens=4000)

def rag_pipeline(user_input):
    """Process user queries through the RAG pipeline."""
    results = similarity_search(user_input, k=5)
    context_text = "\n\n".join([doc.page_content for doc in results])

    system_prompt_template = PromptTemplate.from_template(
        "Summarize the following documents relevant to the query and give full description of the query without truncation:\n"
        "{context}\n\n"
        "Your response should include a full, detailed description of the query without truncation step by step. "
        "Provide the summary in detailed format with 500 to 600 words, ensuring:\n"
        "- No hallucinations.\n"
        "- No repetition from the referenced document.\n"
        "- Maintain factual accuracy.\n"
        "- If there is no similarity between query and generation, just return 'non'."
    )

    system_message = SystemMessage(content=system_prompt_template.format(context=context_text))
    response = chat_model_rag.invoke([
        system_message,
        HumanMessage(content=user_input)
    ])

    if hasattr(response, 'content'):
        return response.content
    elif isinstance(response, dict) and "content" in response:
        return response["content"]
    else:
        return str(response)

def chat_bot(user_input):
    """Handle general conversation with the user."""
    user_input = user_input.strip()
    chat_model = ChatGroq(model_name="llama3-8b-8192", temperature=0.1, max_tokens=1000)

    system_prompt = """
    [Your existing system prompt here]
    """

    response = chat_model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ])

    return response.content

# Define tools and initialize agent
chatbot_tool = Tool(
    name="ChatBot",
    func=chat_bot,
    description="Use for general conversational queries only."
)

rag_tool = Tool(
    name="RAG Retrieval",
    func=rag_pipeline,
    description="Use for medical and technical questions about diabetic retinopathy."
)

# Initialize Agent
Agent_llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.1, max_tokens=4000)
memory = ConversationBufferMemory(memory_key="chat_history", output_key="output", return_messages=True)
agent_system_message = """You are G-Nayan, a specialized chatbot focused on diabetic retinopathy.

For ANY information or medical queries ABOUT DIABETIC RETINOPATHY, you MUST use the RAG Retrieval tool.
The RAG tool should be your default choice for diabetic retinopathy questions.

Only use the ChatBot tool for greetings and general conversation that doesn't require any specific information.

IMPORTANT: When using the RAG Retrieval tool, NEVER modify or summarize its output.
Return the COMPLETE RAG output WITHOUT any "Final Answer" or additional commentary.
The RAG output is already optimized and should be presented to the user exactly as received.

For medical topics about diabetes eye care, information about G-Nayan, diabetes eye health, or any diabetic retinopathy related questions,
ALWAYS use the RAG Retrieval tool and NEVER attempt to answer these questions yourself.

You CANNOT answer questions about non-diabetic retinopathy medical conditions. For any question outside your scope,
respond with: "I'm specialized in diabetic retinopathy topics. I'd be happy to help with questions about eye health
for diabetic patients, G-Nayan technology, or general conversation. What would you like to know about diabetic retinopathy?"""
agent = initialize_agent(
    llm=Agent_llm,
    tools=[chatbot_tool, rag_tool],
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    return_intermediate_steps=True,
    handle_parsing_errors=True,
    memory=memory,
    system_message=agent_system_message
)

# FastAPI setup
app = FastAPI(title="G-Nayan Chatbot API")

class UserInput(BaseModel):
    user_input: str

def process_agent_response(response_dict):
    """Process agent response to ensure RAG outputs are returned correctly."""
    if "intermediate_steps" in response_dict:
        for step in response_dict["intermediate_steps"]:
            if isinstance(step, (list, tuple)) and len(step) >= 2:
                action = step[0]
                result = step[1]
                if hasattr(action, "tool") and action.tool == "RAG Retrieval":
                    return result

    if "output" in response_dict:
        return response_dict["output"]

    return "I apologize, but I couldn't generate a proper response. Please rephrase your question."
def is_information_query(query):
    """Determine if the query is seeking information (rather than just conversation)."""
    info_patterns = [
        'what is', 'how does', 'explain', 'describe', 'tell me about', 'what are',
        'causes', 'symptoms', 'treatment', 'diagnosis', 'prevention', 'stages',
        'why', 'when', 'where', 'who', 'which', 'can you', 'how to', 'how can'
    ]

    query_lower = query.lower()

    for pattern in info_patterns:
        if pattern in query_lower:
            return True
def is_within_scope(query):
    """Determine if the query is within the scope of diabetic retinopathy and related topics."""
    scope_keywords = [
        'diabetic', 'retinopathy', 'g-nayan', 'gnayan', 'eye', 'vision', 'diabetes',
        'iscs', 'retina', 'fundus', 'ophthalmology', 'macular edema', 'blindness',
        'eye screening', 'fundus camera', 'diabetic eye', 'blood sugar', 'glucose'
    ]

    query_lower = query.lower()

    for keyword in scope_keywords:
        if keyword in query_lower:
            return True

    return False
def get_out_of_scope_response():
    """Return a polite response for queries outside the scope."""
    responses = [
        "I'm specialized in diabetic retinopathy topics. I'd be happy to help with questions about eye health for diabetic patients, G-Nayan technology, or general conversation. What would you like to know about diabetic retinopathy?",

        "I'm focused on diabetic retinopathy and related eye conditions. While I can't address that specific topic, I'd be glad to discuss diabetic eye care, prevention, or treatment options. Would you like information about any of these areas?",

        "I'm designed to assist with diabetic retinopathy information. I can't provide details on that topic, but I'm happy to discuss early detection, symptoms, or management of diabetic eye conditions. What aspect of diabetic retinopathy interests you?",

        "I'm limited to discussions on diabetic retinopathy and general chatbot interactions. I'd be pleased to help with questions related to diabetic eye health, G-Nayan technology, or how retinopathy affects vision. Would you like to explore any of these topics?"
    ]

    import random
    return random.choice(responses)
@app.post("/chat/")
def chat_endpoint(user_input_data: UserInput):
    """Process user input and generate appropriate responses."""
    user_input = user_input_data.user_input.strip()
    logger.info(f"User Input: {user_input}")

    # Quick responses for simple interactions
    if user_input.lower() in ["hi", "hello", "hey", "hai"]:
        response = "Hello! I am your AI Assistant G-Nayan. How can I assist you today?"
        logger.info(f"Response: {response}")
        return JSONResponse({"G-Nayan": response})

    if user_input.lower() in ["exit", "quit", "thank you", "bye", "tq"]:
        response = "Goodbye! Feel free to return if you need assistance."
        logger.info(f"Response: {response}")
        return JSONResponse({"G-Nayan": response})

    try:
        if is_information_query(user_input) and not is_within_scope(user_input):
            response = get_out_of_scope_response()
            logger.info(f"Response (Out of Scope): {response}")
            return JSONResponse({"G-Nayan": response})

        if is_information_query(user_input) and is_within_scope(user_input):
            rag_response = rag_pipeline(user_input)
            response = f"Based on my knowledge: {rag_response}"
            logger.info(f"Response (RAG): {response[:100]}...")
            return JSONResponse({"G-Nayan": response})

        response = agent.invoke(user_input)
        processed_response = process_agent_response(response)
        logger.info(f"Response (Agent): {processed_response[:100]}...")
        return JSONResponse({"G-Nayan": processed_response})

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"response": "I apologize for the technical difficulties. Please try again."}
        )

def chat_interface(message: str, history: List[Tuple[str, str]]) -> str:
    """Handle chat interactions through Gradio interface."""
    try:
        # Create user input data
        user_input_data = UserInput(user_input=message)
        # Call the FastAPI endpoint
        response = chat_endpoint(user_input_data)

        # Handle JSONResponse properly
        if isinstance(response, JSONResponse):
            response_data = json.loads(response.body.decode())
            return response_data.get("G-Nayan", "I apologize, but I couldn't process your request.")
        elif isinstance(response, dict):
            return response.get("G-Nayan", "I apologize, but I couldn't process your request.")
        else:
            return str(response)

    except Exception as e:
        logger.error(f"Gradio interface error: {str(e)}")
        return "I apologize, but I encountered an error. Please try again."

import time

def create_gradio_interface():
    """Create and configure the Gradio chat interface with streaming responses."""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        chatbot = gr.Chatbot(
            label="G-Nayan Chatbot",
            height=400,
            show_label=True,
            avatar_images=(None, "https://api.dicebear.com/7.x/identicon/svg?seed=RetinaBot"),
            bubble_full_width=False,
        )
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Type your message here...",
                show_label=False,
                lines=2,
                scale=9,
                container=False,
            )
            submit = gr.Button("Send", scale=1, variant="primary")

        with gr.Accordion("Example Questions", open=False):
            gr.Examples(
                examples=[
                    "What is diabetic retinopathy?",
                    "What are the symptoms of diabetic retinopathy?",
                    "How is diabetic retinopathy diagnosed?",
                    "What treatments are available for diabetic retinopathy?",
                    "How can I prevent diabetic retinopathy?",
                ],
                inputs=msg,
            )

        with gr.Row():
            clear = gr.Button("Clear Chat")

        def respond(message, chat_history):
            """Stream response with typing effect."""
            if message.strip() == "":
                return chat_history
            
            # Add user message immediately
            chat_history.append((message, ""))
            yield chat_history
            
            try:
                # Get full response using chat interface
                full_response = chat_interface(message, chat_history[:-1])
                formatted_response = f"G-Nayan: {full_response}"
                
                # Stream the response character by character
                for i in range(1, len(formatted_response) + 1):
                    partial_response = formatted_response[:i]
                    chat_history[-1] = (message, partial_response)
                    time.sleep(0.01)  # Adjust typing speed here (lower = faster)
                    yield chat_history
                    
            except Exception as e:
                logger.error(f"Error in chat response: {str(e)}")
                error_msg = "G-Nayan: I apologize, but I encountered an error. Please try again."
                chat_history[-1] = (message, error_msg)
                yield chat_history

        # Set up event handlers for streaming
        msg.submit(respond, [msg, chatbot], [chatbot]).then(
            lambda: "", None, msg
        )
        submit.click(respond, [msg, chatbot], [chatbot]).then(
            lambda: "", None, msg
        )
        clear.click(lambda: [], None, chatbot)

    return demo

if __name__ == "__main__":
    import uvicorn
    from fastapi.middleware.cors import CORSMiddleware
    import threading

    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Create Gradio interface
    demo = create_gradio_interface()

    # Define server functions
    def run_fastapi():
        uvicorn.run(app, host="0.0.0.0", port=8000)

    def run_gradio():
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True
        )

    # Start servers in separate threads
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    gradio_thread = threading.Thread(target=run_gradio, daemon=True)

    try:
        fastapi_thread.start()
        gradio_thread.start()

        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down servers...")