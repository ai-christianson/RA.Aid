import streamlit as st
from ra_aid.agent_utils import run_research_agent
from ra_aid.llm import initialize_llm
from components.memory import _global_memory
from ra_aid.logger import logger
from typing import Dict, Any

def research_component(task: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Handle the research stage of RA.Aid."""
    try:
        # Validate required config fields
        required_fields = ["provider", "model", "research_only", "hil"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required configuration field: {field}")

        # Initialize model
        model = initialize_llm(config["provider"], config["model"])
        
        # Update global memory configuration
        _global_memory['config'] = config.copy()
        
        # Add status message
        st.write("🔍 Starting Research Phase...")
        
        # Run research agent
        results = run_research_agent(
            task,
            model,
            expert_enabled=True,
            research_only=config["research_only"],
            hil=config["hil"],
            web_research_enabled=config.get("web_research_enabled", False),
            config=config
        )
        
        # Debug logging
        logger.debug(f"Research agent results type: {type(results)}")
        logger.debug(f"Research agent results: {results}")
        
        # Ensure results is a dictionary
        if not isinstance(results, dict):
            raise ValueError("Research agent returned invalid results format")

        # Update global memory with research results
        _global_memory['related_files'] = results.get("related_files", {})
        _global_memory['implementation_requested'] = False
        
        # Display research results
        if results.get("research_notes"):
            st.markdown("### Research Notes")
            st.markdown(results["research_notes"])
            
        if results.get("key_facts"):
            st.markdown("### Key Facts")
            st.markdown(results["key_facts"])
            
        if results.get("related_files"):
            st.markdown("### Related Files")
            for file in results["related_files"]:
                st.code(file)
        
        # Ensure success field is present
        if "success" not in results:
            results["success"] = True
            
        return results

    except ValueError as e:
        logger.error(f"Research Configuration Error: {str(e)}")
        st.error(f"Research Configuration Error: {str(e)}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Research Error: {str(e)}")
        st.error(f"Research Error: {str(e)}")
        return {"success": False, "error": str(e)}
