# main.py - FastAPI Gateway (Corrected Parsing Logic)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import asyncio
from typing import Dict, Optional, List, Any
import logging
import time

app = FastAPI(title="ADK Agent Gateway", version="1.1.0")

# --- Request/Response Models ---
class AgentRequest(BaseModel):
    agent_name: str
    message: str
    user_id: str
    session_id: Optional[str] = None

class AgentResponse(BaseModel):
    response: str
    session_id: str
    agent_name: str
    status: str

# --- Agent Registry ---
AGENT_REGISTRY = {
    "greeting_agent": "http://localhost:8000",
    "jarvis_agent": "http://localhost:8000",
    "calc_agent": "http://localhost:8000",
    # Add more agents as you deploy them
}

# --- Main Endpoint ---
@app.post("/run_agent", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    """Main gateway endpoint - routes requests to appropriate ADK agents"""
    if request.agent_name not in AGENT_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Agent '{request.agent_name}' not found")
    
    agent_url = AGENT_REGISTRY[request.agent_name]
    
    try:
        # Create session first, then run the agent
        session_id = await create_session(agent_url, request.user_id, request.agent_name)
        
        response_text = await run_agent_session(
            agent_url, 
            request.message,
            request.user_id,
            request.agent_name,
            session_id
        )
        
        return AgentResponse(
            response=response_text,
            session_id=session_id,
            agent_name=request.agent_name,
            status="success"
        )
        
    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP Error running agent {request.agent_name}: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Error from ADK server: {e.response.text}")
    except Exception as e:
        logging.error(f"Error running agent {request.agent_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Helper Functions ---
async def create_session(agent_url: str, user_id: str, app_name: str) -> str:
    """Create a new session with the ADK agent"""
    session_id = f"session-{int(time.time())}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{agent_url}/apps/{app_name}/users/{user_id}/sessions/{session_id}",
            headers={"Content-Type": "application/json"},
            json={}
        )
        response.raise_for_status()
        return session_id

async def run_agent_session(agent_url: str, message: str, user_id: str, app_name: str, session_id: str) -> str:
    """Run the agent with a message using the simple /run endpoint"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{agent_url}/run",
            headers={"Content-Type": "application/json"},
            json={
                "app_name": app_name,
                "user_id": user_id,
                "session_id": session_id,
                "new_message": {
                    "role": "user",
                    "parts": [{"text": message}]
                }
            }
        )
        response.raise_for_status()

        # --- CORRECTED PARSING LOGIC ---
        # The /run endpoint returns a list of all events in the agent's turn.
        # We need to find the LAST event from the 'model' to get the final answer.
        events: List[Dict[str, Any]] = response.json()
        
        final_response = "Agent did not provide a final text response."
        
        # Iterate through all events to find the last valid text response from the model.
        for event in events:
            content = event.get("content")
            if (
                content and 
                isinstance(content, dict) and
                content.get("role") == "model" and 
                "parts" in content
            ):
                parts = content.get("parts", [])
                if parts and isinstance(parts[0], dict) and "text" in parts[0]:
                    # This is a valid text response from the model.
                    # We keep overwriting until we get the last one in the list.
                    final_response = parts[0]["text"]
        
        return final_response
    
# --- Utility Endpoints ---
@app.get("/health")
async def health_check():
    return {"status": "healthy", "agents": list(AGENT_REGISTRY.keys())}

@app.get("/agents")
async def list_agents():
    """List all available agents"""
    return {"agents": list(AGENT_REGISTRY.keys())}

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
