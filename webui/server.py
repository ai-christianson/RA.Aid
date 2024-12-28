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

async def handle_task(websocket: WebSocket, task: str, config: dict):
    """Handle a task request"""
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
        
        # Send status update
        await websocket.send_json({
            "type": "status",
            "content": "🔍 Starting Research Phase..."
        })
        
        try:
            # Call LLM using litellm
            response = completion(**model_params)
            
            # Extract response
            result = response.choices[0].message.content
            
            # Format research results
            research_results = {
                "success": True,
                "research_notes": [],
                "key_facts": {}
            }
            
            # Parse response into sections
            sections = result.split('\n\n')
            for section in sections:
                if section.lower().startswith('research notes:'):
                    notes = section.replace('Research Notes:', '', flags=re.IGNORECASE).strip().split('\n')
                    research_results['research_notes'].extend([note.strip('- ') for note in notes if note.strip()])
                elif section.lower().startswith('key facts:'):
                    facts = section.replace('Key Facts:', '', flags=re.IGNORECASE).strip().split('\n')
                    for fact in facts:
                        if ':' in fact:
                            key, value = fact.strip('- ').split(':', 1)
                            research_results['key_facts'][key.strip()] = value.strip()
            
            # Send research results
            if research_results['research_notes']:
                await websocket.send_json({
                    "type": "research",
                    "content": "### Research Notes\n" + "\n".join([f"- {note}" for note in research_results['research_notes']])
                })
                
            if research_results['key_facts']:
                await websocket.send_json({
                    "type": "research",
                    "content": "### Key Facts\n" + "\n".join([f"- **{key}**: {value}" for key, value in research_results['key_facts'].items()])
                })
            
            await websocket.send_json({
                "type": "success",
                "content": "✅ Research completed successfully!"
            })
                
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            await websocket.send_json({
                "type": "error",
                "content": f"Error: {str(e)}"
            })
            
    except Exception as e:
        logger.error(f"Error in handle_task: {e}")
        await websocket.send_json({
            "type": "error",
            "content": f"Error: {str(e)}"
        })

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
                    await handle_task(
                        websocket,
                        message.get("content"),
                        message.get("config", {})
                    )
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