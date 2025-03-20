import os 
from dotenv import load_dotenv
import os
import getpass
import json
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from fastapi.responses import JSONResponse
# FastAPI imports
from fastapi import FastAPI, HTTPException, Request,Body
from pydantic import BaseModel
import time
load_dotenv()
api_key_value = os.getenv("api_key_1") # Get the value of 'api_key_1' from environment

if api_key_value: # Check if api_key_value is not None (meaning 'api_key_1' was found)
    os.environ["GROQ_API_KEY"] = api_key_value  # Correctly set GROQ_API_KEY environment variable
    print("GROQ_API_KEY environment variable set successfully.")
else:
    print("Error: 'api_key_1' environment variable not found. Please make sure it's set in your .env file or environment.")
# ✅ Initialize FastAPI
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# ✅ Load FAISS Vector Store
faiss_save_path = "./fasis_db"
vector_store_FAISS = FAISS.load_local(faiss_save_path, embed_model, allow_dangerous_deserialization=True)
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

        # ✅ Ensure we extract text correctly
        if hasattr(response, "content") and isinstance(response.content, str):
            queries = response.content.strip().split("\n")
            return [q.strip() for q in queries if q.strip()]  # Clean spaces and blank lines

        return [query]  # Fallback to original query
    except Exception as e:
        print(f"LLM Query Rewrite Error: {str(e)}")
        return [query]  # Fallback to original query



for_query = ChatGroq(model_name="llama3-8b-8192", temperature=0.0, max_tokens=50)
def llm_query_rewrite(query):
    prompt = f"""
    Rewrite the following search query in a well-structured, concise format.
    Directly return **only** 2-3 query variations, with each query on a new line.
    No explanations, no numbering, no extra words.

    Input: {query}
    Output:
    """
    try:
        response = for_query.invoke(prompt)

        # ✅ Ensure we extract text correctly
        if hasattr(response, "content") and isinstance(response.content, str):
            queries = response.content.strip().split("\n")
            return [q.strip() for q in queries if q.strip()]  # Clean spaces and blank lines

        return [query]  # Fallback to original query
    except Exception as e:
        print(f"LLM Query Rewrite Error: {str(e)}")
        return [query]  # Fallback to original query

def similarity_search(query, k):
    # Load FAISS index
    vector_store_FAISS = FAISS.load_local(faiss_save_path, embed_model,allow_dangerous_deserialization=True)

    # Rewrite the query for better retrieval
    cleaned_queries = llm_query_rewrite(query)
    # print(cleaned_queries)
    # Convert the first rewritten query into an embedding
    query_embedding = embed_model.embed_query(cleaned_queries[0])

    # Perform similarity search using the query vector
    retrieved_docs = vector_store_FAISS.similarity_search_by_vector(query_embedding, k)

    return retrieved_docs

# ✅ Initialize Groq LLM Model
chat_model_rag = ChatGroq(model_name="llama3-8b-8192", temperature=0, max_tokens=2000)

def rag_pipeline(user_input):
    results = similarity_search(user_input, k=3)
    context_text = "\n\n".join([doc.page_content for doc in results])

    system_prompt_template = PromptTemplate.from_template(
        "Summarize the following documents relevant to the query and give full description of the query without truncate:\n"
        "{context}\n\n"
        "Your response should include a full, detailed description of the query without truncation step by step"
        "Provide the summary in detailed formate more than 500 to 600 words , ensuring:\n"
        "- No hallucinations.\n"
        "- No repetition from the referenced document.\n"
        "- Maintain factual accuracy.\n"
        "- If there is no similarity between query and generation, just return 'non'."
    )
    system_message = SystemMessage(content=system_prompt_template.format(context=context_text))
    response = chat_model_rag.stream([
        system_message,
        HumanMessage(content=user_input)
    ])
    final_response=""
    for chunk in response:
        if hasattr(chunk, 'content'):
            print(chunk.content, end='', flush=True)
            final_response += chunk.content
            time.sleep(0.05)  # Typing effect
        elif isinstance(chunk, dict) and "content" in chunk:
            print(chunk["content"], end='', flush=True)
            final_response += chunk["content"]
            time.sleep(0.05)
    return final_response


def chat_bot(user_input):
    user_input = user_input.strip()
    chat_model = ChatGroq(model_name="llama3-8b-8192", temperature=0.1, max_tokens=250)
    system_prompt = """
 1.   You are G-Nayan, an AI chatbot specializing in general conversations, RAG-integrated chatbot functionalities, and discussions related to diabetic retinopathy.
   - You engage in general conversation greetings.
   - You cannot answer questions unrelated to diabetic retinopathy.
   - You explain your role as an AI chatbot with a RAG pipeline integration.
   - You provide accurate and concise responses related to diabetic retinopathy.
**You must strictly follow the below guidelines.**
2. **Behavioral Constraints:** Important Rules to follow 
   - Never explicitly state that you are an AI unless asked.
    -do not go out of your role.
   - Do not repeatedly mention: *"As a specialist in diabetic retinopathy."*
   - If asked about unrelated or inappropriate topics, respond with: any other topics except diabetic retinopathy
     *"I am limited to discussions on diabetic retinopathy and general chatbot interactions."*
   - Always prioritize clear, factual, and concise responses.
3. **Medical Queries:**
   - Provide reliable and relevant information about diabetic retinopathy.
   - Avoid offering medical diagnoses or personalized treatment recommendations.
   - If a user asks unrelated or inappropriate medical questions, respond with:
     *"I can only assist with general discussions and diabetic retinopathy-related topics."*

4. **Conversation Flow:**
   - Maintain professionalism while keeping the interaction engaging.
   - Avoid redundant or unnecessary explanations.
   - Ensure responses are informative yet easy to understand.

Follow these rules strictly to ensure consistency and user satisfaction.

    """

    response = chat_model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ])
   
    return f"this is the reponse please use this {response.content}"


# ✅ Define tools for Langchain
chatbot_tool = Tool(
    name="ChatBot",
    func=chat_bot,
   description=(
        "Use this ability for general conversational queries like greetings and normal discussions. "
        "If the user greets you, respond accordingly using the chat_bot function. "
        "This ability is for handling casual conversation and answering non-medical questions. "
        "For a diabetic retinopathy report analysis, redirect to the system prompt for reports. "
        "Do not use this ability for medical queries; those should be handled by RAG. "
    )
)

rag_tool = Tool(
    name="RAG Retrieval",
    func=rag_pipeline,
    description=(
    "Use this tool when the user seeks document-based information from the RAG pipeline. "
    "It is ideal for answering queries related to 'G-Nayan', 'Diabetic Retinopathy', 'Retinopathy','and the Company whic made it iSCS '"
    "its causes, symptoms, stages, diagnosis, and treatment options. "
    "Additionally, use this tool when the user inquires about food recommendations, lifestyle changes, "
    "or relevant medical specialists for diabetic retinopathy management."
))


Agent_llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.1, max_tokens=2000)
memory = ConversationBufferMemory(memory_key="chat_history", output_key="output")

# ✅ Create an agent with updated configuration
agent = initialize_agent(
    llm=Agent_llm,
    tools=[chatbot_tool,rag_tool],
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=2,  # Limit the number of thinking iterations
    early_stopping_method="generate",
    return_intermediate_steps=True,
    return_only_outputs=True,  #,  # Ensure it doesn't get stuck
    handle_parsing_errors=True,  # Handle errors gracefully
    memory=memory
)
app = FastAPI(title="G-Nayan Chatbot API", description="An AI chatbot for general conversations and diabetic retinopathy discussions.", version="0.1")


class UserInput(BaseModel):
    user_input: str

@app.post("/chat/")
def chat_endpoint(user_input_data: UserInput):
    user_input = user_input_data.user_input.strip().lower()

    if user_input in ["hi", "hello", "hey", "hai"]:
        return JSONResponse({"response": "Hello! Hi am your AI Assistance G-Nayan ...How can I assist you?"})

    if user_input in ["exit", "quit", "thank you", "bye","tq"]:
        return JSONResponse({"response": "Goodbye! Just ping me 'hi' if you need any help and you know diabetic retinopathy follow the link: https://www.iscstech.com"})

    try:
        response = agent.invoke(user_input)
        return JSONResponse({"response": response['output']})
    # return JSONResponse({"raw_response": response}) # Return the 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot Error: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) # You can change host and port if needed    