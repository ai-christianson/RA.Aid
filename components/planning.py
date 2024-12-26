import streamlit as st
from ra_aid.agent_utils import run_planning_agent
from ra_aid.llm import initialize_llm
from ra_aid.logger import logger

def planning_component(task: str, config: dict) -> dict:
    """Handle the planning stage of RA.Aid."""
    try:
        # Initialize model
        model = initialize_llm(config["provider"], config["model"])
        
        st.write("📋 Creating Implementation Plan...")
        
        # Run planning agent
        results = run_planning_agent(
            task,
            model,
            expert_enabled=True,
            hil=config["hil"],
            config=config
        )
        
        # Display planning results
        if results.get("plan"):
            st.markdown("### Implementation Plan")
            st.markdown(results["plan"])
        
        if results.get("tasks"):
            st.markdown("### Tasks")
            for task_item in results["tasks"]:
                st.markdown(f"- {task_item}")
        
        return results

    except Exception as e:
        logger.error(f"Planning Error: {str(e)}")
        st.error(f"Planning Error: {str(e)}")
        return {"success": False, "error": str(e)}
