import streamlit as st
from ra_aid.agent_utils import run_task_implementation_agent
from ra_aid.llm import initialize_llm
from ra_aid.logger import logger

def implementation_component(task: str, research_results: dict, planning_results: dict, config: dict) -> dict:
    """Handle the implementation stage of RA.Aid."""
    try:
        # Initialize model
        model = initialize_llm(config["provider"], config["model"])
        
        st.write("🛠️ Starting Implementation...")
        
        tasks = planning_results.get("tasks", [])
        results = {"success": True, "implemented_tasks": []}
        
        # Create a progress bar
        progress_bar = st.progress(0)
        task_count = len(tasks)
        
        # Implement each task
        for idx, task_spec in enumerate(tasks):
            st.markdown(f"**Implementing task {idx + 1}/{task_count}:**")
            st.markdown(f"_{task_spec}_")
            
            task_result = run_task_implementation_agent(
                base_task=task,
                tasks=tasks,
                task=task_spec,
                plan=planning_results.get("plan", ""),
                related_files=research_results.get("related_files", []),
                model=model,
                expert_enabled=True,
                config=config
            )
            
            results["implemented_tasks"].append(task_result)
            
            # Update progress
            progress_bar.progress((idx + 1) / task_count)
            
            if task_result.get("success"):
                st.success(f"Task completed: {task_spec}")
            else:
                st.error(f"Task failed: {task_spec}")
                st.error(task_result.get("error", "Unknown error"))
                results["success"] = False
                results["error"] = task_result.get("error", "Unknown error")
                break  # Stop processing tasks after the first failure
        
        return results

    except Exception as e:
        logger.error(f"Implementation Error: {str(e)}")
        st.error(f"Implementation Error: {str(e)}")
        return {"success": False, "error": str(e)}
