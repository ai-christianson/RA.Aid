import streamlit as st
from ra_aid.agent_utils import run_task_implementation_agent
from ra_aid.llm import initialize_llm
from ra_aid.logger import logger
from components.memory import _global_memory
from typing import Dict, Any

def implementation_component(task: str, research_results: Dict[str, Any], planning_results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Handle the implementation stage of RA.Aid."""
    try:
        # Validate required config fields
        required_fields = ["provider", "model", "hil"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required configuration field: {field}")

        # Validate research and planning results
        if not isinstance(research_results, dict) or not isinstance(planning_results, dict):
            raise ValueError("Invalid research or planning results format")

        if not planning_results.get("tasks"):
            raise ValueError("No tasks found in planning results")

        # Initialize model
        model = initialize_llm(config["provider"], config["model"])
        
        # Update global memory configuration
        _global_memory['config'] = config.copy()
        
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
            
            # Validate task result
            if not isinstance(task_result, dict):
                raise ValueError(f"Invalid task result format for task: {task_spec}")
            
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

    except ValueError as e:
        logger.error(f"Implementation Configuration Error: {str(e)}")
        st.error(f"Implementation Configuration Error: {str(e)}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Implementation Error: {str(e)}")
        st.error(f"Implementation Error: {str(e)}")
        return {"success": False, "error": str(e)}
