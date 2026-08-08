# -----------------------------------------------------------------------------------
#                               Import and Init statements
# -----------------------------------------------------------------------------------

import os 
from dotenv import load_dotenv
load_dotenv()
import certifi
import asyncio

os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUEST_CA_BUNDLE'] = certifi.where()

from typing import TypedDict, Annotated
import operator, uuid , psycopg
from psycopg.rows import dict_row
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage,AnyMessage

from langchain_groq import ChatGroq
# from tools.tavily_tool import tavily_search
# from tools.flight_tool import search_flights
from mcp_client import tavily_mcp_search , aviation_mcp_call

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

FLIGHT_AGENT_PROMPT = """
    You are a travel flight expert.

    User Query:
    {query}

    Airport Information:
    {airport_data}

    Airline Information:
    {airline_data}

    Generate:
    
    1. Likely departure airport 
    2. Likely arrival airport 
    3. Airlines serving this route
    4. Typical flight duration
    5. Estimated airfare range
    6. Peak season pricing warning
    7. Booking Advice

    Return concise travel guidance.
    """

def flight_agent(state : TravelState):
    """
        Getting all the flight information for the trip
        it will call the search_flights function
    """
    print("\nINSIDE FLIGHT AGENT---\n")
    query = state['user_query']

    try:
        airports = asyncio.run(
            aviation_mcp_call("list_airports")
        )

        airlines = asyncio.run(
            aviation_mcp_call("list_airlines")
        )

        print("\nAIRPORTS :- " , airports)
        print("\nAIRLINES :- " , airlines)

        prompt = FLIGHT_AGENT_PROMPT.format(query=query , airport_data = str(airports)[:3000] , airline_data = str(airlines)[:3000])

        response = llm.invoke([
            SystemMessage(content="You are an expert travel flight planner."),
            HumanMessage(content = prompt)
        ])

        flight_data = response.content 

    except Exception as e:
        flight_data = f"Flight information unavailable : {str(e)}"

    return {
        "flight_results" : flight_data,
        "messages" : [
            AIMessage(content="Flight recommendations generated")
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
    hotel_results = asyncio.run(tavily_mcp_search(query))

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
    You are an expert AI Travel Planner.

    Create a visually attractive travel plan using GitHub Markdown.

    USER REQUEST
    -------------
    {state["user_query"]}

    FLIGHT RESULTS
    --------------
    {state["flight_results"]}

    HOTEL RESULTS
    -------------
    {state["hotel_results"]}

    ITINERARY
    ---------
    {state["itinerary"]}

    IMPORTANT FORMATTING RULES

    Use emojis in every section.

    Use proper Markdown headings (#, ##, ###).

    Use tables wherever appropriate.

    Use bullet points instead of long paragraphs.

    Use horizontal dividers (---) between sections.

    Highlight important information using **bold**.

    Use blockquotes (>) for travel tips.

    Use checklists where useful.

    Keep everything easy to scan.

    Never output one huge paragraph.

    The response should look like a premium travel website.

    Follow this structure exactly:

    # 🌍 Trip Summary

    - Destination
    - Duration
    - Best Time
    - Travel Style
    - Quick Overview

    ---

    # ✈️ Flight Information

    Create a table.

    | Airline | Route | Duration | Stops | Price |
    |---------|--------|----------|-------|-------|

    If price is unavailable write:

    "Live flight pricing is currently unavailable."

    ---

    # 🏨 Recommended Hotels

    Create a table.

    | Hotel | City | Price/Night | Rating | Why Stay Here |
    |--------|------|-------------|---------|---------------|

    ---

    # 📅 Day-by-Day Itinerary

    For every day use this format:

    ## Day 1️⃣

    Morning ☀️

    Afternoon 🌇

    Evening 🌃

    Food 🍽️

    Transport 🚕

    Budget 💰

    Repeat for every day.

    ---

    # 💸 Estimated Budget

    Create a table.

    | Category | Estimated Cost |
    |----------|----------------|

    Include

    - Flights
    - Hotels
    - Food
    - Transport
    - Attractions
    - Shopping
    - Total

    ---

    # 🎒 Packing Checklist

    Use markdown checkboxes.

    Example

    - [ ] Passport
    - [ ] Visa
    - [ ] Charger

    ---

    # 💡 Travel Tips

    Give 5-10 useful tips.

    Use emojis.

    ---

    # ⚠️ Important Notes

    Mention:

    - Flight prices may be unavailable from live APIs.
    - Verify visa requirements.
    - Check weather before departure.

    ---

    # ❤️ Final Recommendation

    Finish with an encouraging closing message.
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