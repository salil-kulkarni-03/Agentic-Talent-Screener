import os
import json
from typing import TypedDict, List, Dict, Any
from groq import Groq
from langgraph.graph import StateGraph, END

# Define Agent State
class AgentState(TypedDict):
    messages: List[Dict[str, str]]  # Conversational history
    query: str                      # Current user query
    candidates: List[Dict[str, Any]] # Condensed database context
    is_in_scope: bool               # Flag for safety filter
    reply: str                      # Final agent response

# Initialize Groq client
def _get_groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set.")
    return Groq(api_key=api_key)

# Node 1: Guardrail Classifier (Input Validation)
def classify_input(state: AgentState) -> Dict[str, Any]:
    client = _get_groq_client()
    query = state["query"]
    
    prompt = f"""You are a security guardrail classifier for an AI Recruiter application.
Analyze the user's input and determine if it is relevant to the recruitment project, candidate database, job description, skills matching, resumes, or hiring decisions.

Allowed Topics:
- Questions about specific candidates (e.g., "What are Salil's skills?", "Tell me about candidates with Python experience").
- Database summaries (e.g., "How many candidates do we have?", "Who has the highest score?").
- Comparison of candidates or JDs.

Forbidden Topics:
- General knowledge (e.g., "What is the capital of France?", "Who is Einstein?").
- Coding requests unrelated to this project (e.g., "Write a bubble sort in Java").
- Conversational chat unrelated to recruiting.

Reply with exactly a single JSON object with a single boolean key "in_scope":
{{
  "in_scope": true or false
}}

User Input: "{query}"
JSON Response:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a precise classifier. Return ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content.strip())
        is_in_scope = bool(result.get("in_scope", False))
    except Exception:
        # Fallback to in-scope in case of API failure, the generation prompt also has safety checks
        is_in_scope = True
        
    return {"is_in_scope": is_in_scope}

# Node 2: Refusal Generator (Out-of-Scope Response)
def generate_refusal(state: AgentState) -> Dict[str, Any]:
    return {"reply": "I can only help with questions related to the candidates and this recruitment project."}

# Node 3: Database Context Injector (Context Preparation)
def inject_database(state: AgentState) -> Dict[str, Any]:
    # This node receives the active candidates list passed during graph invocation.
    # It ensures only relevant structured fields are sent, keeping tokens low.
    raw_candidates = state.get("candidates") or []
    condensed = []
    
    for i, c in enumerate(raw_candidates):
        # Extract experience summaries
        exp_list = [e.get("summary", "") for e in c.get("experience", [])]
        # Extract education summaries
        edu_list = [e.get("summary", "") for e in c.get("education", [])]
        
        # Build a condensed representation
        condensed.append({
            "rank": c.get("rank") or (i + 1),
            "name": c.get("name", "Unknown"),
            "hire_probability": c.get("hire_probability") or c.get("score"),
            "decision": c.get("decision", "Unknown"),
            "composite_score": c.get("composite_score"),
            "semantic_score": c.get("semantic_score") or c.get("semanticMatch"),
            "keyword_score": c.get("keyword_score"),
            "experience_score": c.get("experience_score"),
            "candidate_strength": c.get("candidate_strength") or c.get("mlStrength"),
            "skills_matched": c.get("skills_matched", []),
            "skills_missing": c.get("skills_missing", []),
            "experience": exp_list,
            "education": edu_list
        })
        
    return {"candidates": condensed}

# Node 4: Contextual Answer Generator (Factual Response Synthesis)
def generate_answer(state: AgentState) -> Dict[str, Any]:
    client = _get_groq_client()
    query = state["query"]
    candidates_json = json.dumps(state["candidates"], indent=2)
    history = state.get("messages") or []
    
    # Construct conversation context
    messages = [
        {
            "role": "system",
            "content": f"""You are the AI Recruiter Database Assistant. Your sole job is to answer questions about the candidate database.
You are provided with the active database of candidates below in JSON format.

Active Candidate Database:
```json
{candidates_json}
```

Rules:
1. Rely ONLY on the candidate data provided in the JSON above.
2. If the user asks for details not present (e.g. email, phone, or specific experience not listed), reply: "Not mentioned in resume" or "No information available in the database."
3. Do NOT hallucinate or guess any values (such as project details, scores, or names).
4. Keep your responses clear, professional, and well-structured (use bullet points or lists if comparing candidates).
"""
        }
    ]
    
    # Append conversation history
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
        
    # Append current query
    messages.append({"role": "user", "content": query})
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.1
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        reply = f"Error generating answer: {e}"
        
    return {"reply": reply}

# Define the State Machine Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("classify", classify_input)
workflow.add_node("refuse", generate_refusal)
workflow.add_node("inject", inject_database)
workflow.add_node("generate", generate_answer)

# Set Entry Point
workflow.set_entry_point("classify")

# Routing logic
def route_query(state: AgentState):
    if state["is_in_scope"]:
        return "inject"
    return "refuse"

# Add Conditional Edges
workflow.add_conditional_edges(
    "classify",
    route_query,
    {
        "inject": "inject",
        "refuse": "refuse"
    }
)

# Add standard transitions
workflow.add_edge("inject", "generate")
workflow.add_edge("generate", END)
workflow.add_edge("refuse", END)

# Compile the Graph App
chat_agent = workflow.compile()
