# from tools.tavily_tool import tavily_search

# res = tavily_search("Best Hotels in India")
# print(res)

# from tools.flight_tool import search_flights
# res = search_flights("Plan a 7 days Japan trip from usa")
# print(res)

from backend import run_travel_agent
res = run_travel_agent("plan a complete 7 days India trip from Bangladesh including flights, hotels and sightseeing under 2 lakhs." , "test_user_1")
print(res['answer'])

