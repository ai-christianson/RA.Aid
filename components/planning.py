import streamlit as st
from ra_aid.agent_utils import run_planning_agent
from ra_aid.llm import initialize_llm
from ra_aid.logger import logger
from components.memory import _global_memory
from typing import Dict, Any

def planning_component(task: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Handle the planning stage of RA.Aid."""
    try:
        # Validate required config fields
        required_fields = ["provider", "model", "hil"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required configuration field: {field}")

        # Initialize model
        model = initialize_llm(config["provider"], config["model"])
        
        # Update global memory configuration
        _global_memory['config'] = config.copy()
        
        st.write("📋 Creating Implementation Plan...")
        
        # Run planning agent
        results = run_planning_agent(
            task,
            model,
            expert_enabled=True,
            hil=config["hil"],
            config=config
        )
        
        # Ensure results is a dictionary
        if not isinstance(results, dict):
            raise ValueError("Planning agent returned invalid results format")
        
        # Display planning results
        if results.get("plan"):
            st.markdown("### Implementation Plan")
            st.markdown(results["plan"])
        
        if results.get("tasks"):
            st.markdown("### Tasks")
            for task_item in results["tasks"]:
                st.markdown(f"- {task_item}")
        
        # Ensure success field is present
        if "success" not in results:
            results["success"] = True
            
        return results

    except ValueError as e:
        logger.error(f"Planning Configuration Error: {str(e)}")
        st.error(f"Planning Configuration Error: {str(e)}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Planning Error: {str(e)}")
        st.error(f"Planning Error: {str(e)}")
        return {"success": False, "error": str(e)}
