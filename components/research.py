import streamlit as st
from ra_aid.agent_utils import run_research_agent
from ra_aid.llm import initialize_llm
from components.memory import _global_memory
from ra_aid.logger import logger

def research_component(task: str, config: dict) -> dict:
    """Handle the research stage of RA.Aid."""
    try:
        # Initialize model
        model = initialize_llm(config["provider"], config["model"])
        
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
        
        return results

    except Exception as e:
        logger.error(f"Research Error: {str(e)}")
        st.error(f"Research Error: {str(e)}")
        return {"success": False, "error": str(e)}
