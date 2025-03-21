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

# Load environment variables
load_dotenv()
api_key_value = os.getenv("api_key_1")

if api_key_value:
    os.environ["GROQ_API_KEY"] = api_key_value
    print("GROQ_API_KEY environment variable set successfully.")
else:
    print("Error: 'api_key_1' environment variable not found. Please make sure it's set in your .env file or environment.")

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

        # Ensure we extract text correctly
        if hasattr(response, "content") and isinstance(response.content, str):
            queries = response.content.strip().split("\n")
            return [q.strip() for q in queries if q.strip()]  # Clean spaces and blank lines

        return [query]  # Fallback to original query
    except Exception as e:
        print(f"LLM Query Rewrite Error: {str(e)}")
        return [query]  # Fallback to original query

def similarity_search(query, k):
    """Perform similarity search against the vector database."""
    # Load FAISS index
    vector_store_FAISS = FAISS.load_local(faiss_save_path, embed_model, allow_dangerous_deserialization=True)

    # Rewrite the query for better retrieval
    cleaned_queries = llm_query_rewrite(query)

    # Convert the first rewritten query into an embedding
    query_embedding = embed_model.embed_query(cleaned_queries[0])

    # Perform similarity search using the query vector
    retrieved_docs = vector_store_FAISS.similarity_search_by_vector(query_embedding, k)

    return retrieved_docs

# Initialize Groq LLM Model for RAG with increased max_tokens
chat_model_rag = ChatGroq(model_name="llama3-8b-8192", temperature=0, max_tokens=4000)

def rag_pipeline(user_input):
    """Process user queries through the RAG pipeline to generate comprehensive responses."""
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

    # Use invoke instead of stream to get complete response at once
    response = chat_model_rag.invoke([
        system_message,
        HumanMessage(content=user_input)
    ])

    # Return complete content
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
    1. You are G-Nayan, an AI chatbot specializing in general conversations, RAG-integrated chatbot functionalities, and discussions related to diabetic retinopathy.
       - You engage in general conversation greetings.
       - You cannot answer questions unrelated to diabetic retinopathy.
       - You explain your role as an AI chatbot with a RAG pipeline integration.
       - You provide accurate and concise responses related to diabetic retinopathy.

    **You must strictly follow the below guidelines.**

    2. **Behavioral Constraints:** Important Rules to follow
       - Never explicitly state that you are an AI unless asked.
       - Do not go out of your role.
       - Do not repeatedly mention: *"As a specialist in diabetic retinopathy."*
       - If asked about unrelated or inappropriate topics, respond with:
         *"I am limited to discussions on diabetic retinopathy and general chatbot interactions."*
       - Always prioritize clear, factual, and concise responses.
       - Provide detailed, comprehensive answers with sufficient information.

    3. **Medical Queries:**
       - Provide reliable and relevant information about diabetic retinopathy.
       - Avoid offering medical diagnoses or personalized treatment recommendations.
       - If a user asks unrelated or inappropriate medical questions, respond with:
         *"I can only assist with general discussions and diabetic retinopathy-related topics."*

    4. **Conversation Flow:**
       - Maintain professionalism while keeping the interaction engaging.
       - Avoid redundant or unnecessary explanations.
       - Ensure responses are informative yet easy to understand.
       - Provide thorough, detailed responses rather than short answers.

    Follow these rules strictly to ensure consistency and user satisfaction.
    """

    response = chat_model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ])

    return response.content

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

# Define tools for Langchain
chatbot_tool = Tool(
    name="ChatBot",
    func=chat_bot,
    description=(
        "Use this ability ONLY for general conversational queries like greetings and normal discussions. "
        "If the user greets you or asks about your capabilities, respond using this function. "
        "This ability is NOT meant for medical or technical questions about diabetic retinopathy. "
        "DO NOT use this for any medical queries or questions about diabetic retinopathy - use RAG Retrieval instead. "
    )
)
rag_tool = Tool(
    name="RAG Retrieval",
    func=rag_pipeline,
    description=(
        "Use this tool for ANY questions related to G-Nayan, Diabetic Retinopathy, or Retinopathy. "
        "This is your PRIMARY tool for medical questions including causes, symptoms, stages, diagnosis, and treatment options "
        "of diabetic retinopathy. Also use this for questions about food recommendations, lifestyle changes, "
        "the company iSCS, or any medical specialists for diabetic retinopathy management. "
        "ALWAYS use this tool over ChatBot when the query involves any medical or technical information. "
        "IMPORTANT: Return the full response from this tool WITHOUT summarizing or shortening it."
    )
)

# Initialize LLM for the agent with increased max_tokens
Agent_llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.1, max_tokens=4000)
memory = ConversationBufferMemory(memory_key="chat_history", output_key="output", return_messages=True)

# System message for the agent that prioritizes RAG
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
for diabetic patients, G-Nayan technology, or general conversation. What would you like to know about diabetic retinopathy?"
"""

# Create an agent with improved configuration
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

# Custom wrapper function to ensure RAG responses are returned properly
def process_agent_response(response_dict):
    """Process agent response to ensure RAG outputs are returned correctly."""
    # First check for RAG tool usage in intermediate steps
    if "intermediate_steps" in response_dict:
        for step in response_dict["intermediate_steps"]:
            if isinstance(step, (list, tuple)) and len(step) >= 2:
                action = step[0]
                result = step[1]

                # If RAG tool was used, return its complete output
                if hasattr(action, "tool") and action.tool == "RAG Retrieval":
                    return result

    # If no RAG tool was used, return the final output
    if "output" in response_dict:
        return response_dict["output"]

    # Fallback if we can't find anything useful
    return "I apologize, but I couldn't generate a proper response. Could you please rephrase your question?"

# Initialize FastAPI
app = FastAPI(title="G-Nayan Chatbot API", description="An AI chatbot for general conversations and diabetic retinopathy discussions.", version="0.1")

class UserInput(BaseModel):
    user_input: str

@app.post("/chat/")
def chat_endpoint(user_input_data: UserInput):
    """Process user input and generate appropriate responses."""
    user_input = user_input_data.user_input.strip()

    # Quick responses for simple greetings and goodbyes
    if user_input.lower() in ["hi", "hello", "hey", "hai"]:
        return JSONResponse({"response": "Hello! I am your AI Assistant G-Nayan. How can I assist you today with diabetic retinopathy information or general conversation?"})

    if user_input.lower() in ["exit", "quit", "thank you", "bye", "tq"]:
        return JSONResponse({"response": "Goodbye! Just ping me 'hi' if you need any help with diabetic retinopathy. You can learn more at https://www.iscstech.com"})

    try:
        # Check if query is an information query but outside of scope
        if is_information_query(user_input) and not is_within_scope(user_input):
            return JSONResponse({"response": get_out_of_scope_response()})

        # Directly use RAG for in-scope information queries
        if is_information_query(user_input) and is_within_scope(user_input):
            rag_response = rag_pipeline(user_input)
            return JSONResponse({"response": rag_response})

        # Use agent for general conversation and non-information queries
        response = agent.invoke(user_input)

        # Process the agent's response to ensure RAG outputs are returned correctly
        processed_response = process_agent_response(response)
        return JSONResponse({"response": processed_response})

    except Exception as e:
        print(f"Error in processing: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"response": "I apologize for the technical difficulties. Please try again with a different question."}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)