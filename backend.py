# -----------------------------------------------------------------------------------
#                               Import and Init statements
# -----------------------------------------------------------------------------------

import os 
from dotenv import load_dotenv
load_dotenv()
import certifi

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUEST_CA_BUNDLE'] = certifi.where()

from typing import TypedDict, Annotated
import operator, uuid , psycopg
from psycopg.rows import dict_row
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage,AnyMessage

from langchain_groq import ChatGroq
from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights

# -----------------------------------------------------------------------------------
#                                   Init Statements
# -----------------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)

# ────────────────────Database Url Setup────────────────────
def get_database_url():
    """
        Modify the Database Url inorder to avoid the issue
        Returns:
            database_url in str 
    """
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("Database url is missing.. Please add your render PostGreSQL external database")

    if "sslmode=" not in database_url:
        seperator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{seperator}sslmode=require"

    return database_url

# -----------------------------------------------------------------------------------
#                                   Graph Statements
# -----------------------------------------------------------------------------------

# ────────────────────State for langgraph────────────────────
class TravelState(TypedDict):
    messages : Annotated[list[AnyMessage] , operator.add]
    user_query : str 
    flight_results : str 
    hotel_results : str 
    itinerary : str 
    llm_calls : int 

# ────────────────────────Flight Agent──────────────────────── 
def flight_agent(state : TravelState):
    """
        Getting all the flight information for the trip
        it will call the search_flights function
    """
    query = state['user_query']
    flight_data = search_flights(query)

    return {
        "flight_results" : flight_data,
        "messages" : [
            AIMessage(content="Flight results fetched.")
        ],
        "llm_calls" : state.get("llm_calls" , 0) + 1 
    }

# ────────────────────Hotel Agent────────────────────
def hotel_agent(state : TravelState):
    """
        Getting all the hotel information for the travel
        it will call teh tavily_search function
    """
    query = f"Best hotels for {state['user_query']}"
    hotel_results = tavily_search(query)

    return {
        "hotel_results" : hotel_results,
        "messages" : [
            AIMessage(content = "Hotel information fetched.")
        ],
        "llm_calls" : state.get("llm_calls" , 0) + 1 
    }

# ────────────────────Itenary Agent────────────────────
def itinerary_agent(state : TravelState):
    """
        Creating the itinerary for the particular 
        travel plan based on the other information in the state
    """
    prompt = f"""
    Create a complete travel itinerary.

    User Query:
    {state['user_query']}

    Flight Results:
    {state['flight_results']}

    Hotel Results:
    {state['hotel_results']}

    Make the itinerary practival, budget-aware and easy to follow 
    """

    response = llm.invoke([
        SystemMessage(content = "You are an expert travel planner."),
        HumanMessage(content = prompt)
    ])

    return {
        "itinerary" : response.content,
        "messages" : [response],
        "llm_calls" : state.get("llm_calls" , 0) + 1  
    }


# ────────────────────Final Response Agent────────────────────
def final_agent(state : TravelState):
    """
    
    """
    final_prompt = f"""
    Generate the final travel response for the user.

    User Request:
    {state['user_query']}

    Flights:
    {state['flight_results']}

    Hotels:
    {state['hotel_results']}

    Itinerary:
    {state['itinerary']}

    Format the final answer beautifully using these sections:

    1. Trip Summary
    2. Flight Information
    3. Hotel Suggestions
    4. Day by Dat Itinerary
    5. Estimated Budget
    6. Final Recommendations

    Important:
    - Be clear and practical.
    - Mention that live flight API may not provide tickets prices if pricing is unavailable
    - Keep the response useful for real travel planning.
    """

    response = llm.invoke([
        SystemMessage(content = "You are a professional AI Travel booking assistant."),
        HumanMessage(content = final_prompt)
    ])

    return {
        "messages" : [response],
        "llm_calls" : state.get("llm_calls" , 0) + 1 
    }


# ────────────────────Build Graph────────────────────

graph = StateGraph(TravelState)
graph.add_node("flight_agent" , flight_agent)
graph.add_node("hotel_agent" , hotel_agent)
graph.add_node("itinerary_agent" , itinerary_agent)
graph.add_node("final_agent" , final_agent)

graph.add_edge(START , "flight_agent")
graph.add_edge("flight_agent" , "hotel_agent")
graph.add_edge("hotel_agent","itinerary_agent")
graph.add_edge("itinerary_agent","final_agent")
graph.add_edge("final_agent" , END)

# ────────────────────PostGreSQL Checkpointer────────────────────
DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)

checkpointer = PostgresSaver(conn=_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)


# -----------------------------------------------------------------------------------
#                                   Function for FastAPI
# -----------------------------------------------------------------------------------

def run_travel_agent(user_input : str , thread_id : str | None = None):
    """
        Will take the user query and thread id and invoke the 
        travel_graph inorder to get the proper iternary for user
        Args:
            user_input(str) : User Question
            thread_id(str) : id of user inorder to save the database on the id 
    """
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable" : {
            "thread_id" : thread_id
        }
    }

    result = travel_graph.invoke(
        {
            "messages" : [
                HumanMessage(content = user_input)
            ],
            "user_query" : user_input,
            "flight_results" : "",
            "hotel_results" : "",
            "itinerary" : "",
            "llm_calls" : 0
        },
        config = config 
    )

    final_answer = result['messages'][-1].content

    return {
        "thread_id" : thread_id,
        "answer" : final_answer,
        "flight_results" : result.get("flight_results" , ""),
        "hotel_results" : result.get("hotel_results" , ""),
        "itinerary" : result.get("itinerary" , ""),
        "llm_calls" : result.get("llm_calls" , 0)
    }