import os
import json
import logging
import requests
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import litellm
from litellm import completion
from fastapi.middleware.cors import CORSMiddleware
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active connections
active_connections = []

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/models")
async def get_models():
    """Get available models."""
    try:
        models = {
            "openai": ["gpt-4", "gpt-3.5-turbo"],
            "anthropic": ["claude-2.1", "claude-instant"],
            "openrouter": ["openai/gpt-4", "anthropic/claude-2.1"]
        }
        return models
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return {"error": str(e)}

async def handle_task(task: str, config: dict) -> dict:
    """Handle a task request
    
    Args:
        task: The task description
        config: Configuration dictionary
        
    Returns:
        dict: Response containing task results
    """
    try:
        # Extract configuration
        provider = config.get("provider", "anthropic")
        model = config.get("model", "anthropic/claude-2.1")
        research_only = config.get("research_only", True)
        cowboy_mode = config.get("cowboy_mode", False)
        hil = config.get("hil", False)
        web_research = config.get("web_research_enabled", False)
        
        # Log configuration
        logger.info(f"Task: {task}")
        logger.info(f"Config: {json.dumps(config, indent=2)}")
        
        # Prepare model parameters
        model_params = {
            "model": model,
            "messages": [{"role": "user", "content": task}],
            "temperature": 0.7,
            "max_tokens": 4096
        }
        
        try:
            # Call LLM using litellm
            response = completion(**model_params)
            
            # Extract response
            result = response.choices[0].message.content
            
            # Format research results
            research_results = {
                "status": "success",
                "research_notes": [],
                "key_facts": {},
                "response": result
            }
            
            # Parse response into sections
            sections = result.split('\n\n')
            for section in sections:
                if section.lower().startswith('research notes:'):
                    notes = section[len('research notes:'):].strip().split('\n')
                    research_results['research_notes'].extend([note.strip('- ') for note in notes if note.strip()])
                elif section.lower().startswith('key facts:'):
                    facts = section[len('key facts:'):].strip().split('\n')
                    for fact in facts:
                        if ':' in fact:
                            key, value = fact.strip('- ').split(':', 1)
                            research_results['key_facts'][key.strip()] = value.strip()
            
            return research_results
                
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
            
    except Exception as e:
        logger.error(f"Error in handle_task: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    logger.info("Client connected")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                
                if message.get("type") == "task":
                    # Handle task request
                    result = await handle_task(
                        message.get("content"),
                        message.get("config", {})
                    )
                    
                    # Send results back
                    if result.get("status") == "success":
                        if result.get("research_notes"):
                            await websocket.send_json({
                                "type": "research",
                                "content": "### Research Notes\n" + "\n".join([f"- {note}" for note in result["research_notes"]])
                            })
                            
                        if result.get("key_facts"):
                            await websocket.send_json({
                                "type": "research",
                                "content": "### Key Facts\n" + "\n".join([f"- **{key}**: {value}" for key, value in result["key_facts"].items()])
                            })
                            
                        await websocket.send_json({
                            "type": "success",
                            "content": "✅ Research completed successfully!"
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "content": result.get("error", "Unknown error")
                        })
                else:
                    # Echo other messages back
                    await websocket.send_json({
                        "type": "response",
                        "data": message
                    })
                    
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received: {data}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
                
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        active_connections.remove(websocket)
        logger.info("Client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 