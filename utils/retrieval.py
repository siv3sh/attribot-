from models.embeddings import HREmbedder

def hr_rag_query(query, source=None, filters=None):
    """
    Enhanced HR RAG query function with filtering capabilities
    
    Args:
        query: The search query string
        source: Optional source filter (e.g., 'employee_lifecycle')
        filters: Optional dictionary of additional filters
        
    Returns:
        Dictionary containing documents, metadatas, and sources
    """
    embedder = HREmbedder()
    
    # Build query arguments
    query_args = {
        "query_texts": [query],
        "n_results": 5,  # Default number of results
        "include": ["metadatas", "documents"]
    }

    # Build where clause for filtering
    where_clause = {}
    
    # Add source filter if provided
    if source:
        where_clause["source"] = source
        
    # Add additional filters if provided
    if filters:
        if 'department' in filters:
            where_clause["department"] = filters['department']
        if 'min_risk_score' in filters:
            where_clause["risk_score"] = {"$gte": filters['min_risk_score']}
        if 'confidentiality' in filters:
            where_clause["confidentiality"] = filters['confidentiality']
    
    if where_clause:
        query_args["where"] = where_clause

    try:
        # Perform vector similarity query
        results = embedder.collection.query(**query_args)
        
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        return {
            "documents": documents,
            "metadatas": metadatas,
            "sources": [m.get("source", "N/A") for m in metadatas]
        }
        
    except Exception as e:
        print(f"Error in hr_rag_query: {str(e)}")
        return {
            "documents": [],
            "metadatas": [],
            "sources": []
        }


# from models.embeddings import HREmbedder
# from langchain.agents import Tool, AgentExecutor
# from langchain.agents import initialize_agent

# class HRAgenticRAG:
#     def __init__(self):
#         self.embedder = HREmbedder()
#         self.tools = self._setup_tools()
#         self.agent = self._initialize_agent()
    
#     def _setup_tools(self):
#         return [
#             Tool(
#                 name="Basic_HR_Query",
#                 func=self._basic_rag_query,
#                 description="Standard document retrieval"
#             ),
#             Tool(
#                 name="Comparative_Analysis",
#                 func=self._comparative_analysis,
#                 description="Compare metrics across departments/periods"
#             ),
#             Tool(
#                 name="Attrition_Risk_Calculator",
#                 func=self._calculate_attrition_risk,
#                 description="Predict attrition probability"
#             )
#         ]
    
#     def _initialize_agent(self):
#         return initialize_agent(
#             tools=self.tools,
#             llm=get_gemini_model(),  # Your existing LLM
#             agent="structured-chat-react",
#             verbose=True
#         )

#     def _basic_rag_query(self, query, source=None, filters=None):
#         """Your existing RAG function as a tool"""
#         # ... your current hr_rag_query implementation ...
    
#     def _comparative_analysis(self, query):
#         """Agent-powered comparative analysis"""
#         # Example: "Compare attrition between Sales and Marketing"
#         departments = extract_entities(query)  # Implement entity extraction
#         results = []
        
#         for dept in departments:
#             docs = self._basic_rag_query(
#                 query,
#                 filters={"department": dept}
#             )
#             results.append((dept, analyze_sentiment(docs)))  # Custom analysis
            
#         return format_comparison(results)

#     def query(self, query, source=None, filters=None):
#         """Enhanced entry point that routes queries appropriately"""
#         if is_complex_query(query):  # Implement complexity detection
#             return self.agent.run(
#                 f"HR Question: {query}\n\n"
#                 f"Source Filter: {source}\n"
#                 f"Additional Filters: {filters}"
#             )
#         return self._basic_rag_query(query, source, filters)