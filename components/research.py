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
        raw_results = run_research_agent(
            task,
            model,
            expert_enabled=True,
            research_only=config["research_only"],
            hil=config["hil"],
            web_research_enabled=config.get("web_research_enabled", False),
            config=config
        )
        
        # Debug logging
        logger.debug(f"Research agent raw results: {raw_results}")
        
        # Format results
        if raw_results is None:
            raise ValueError("Research agent returned no results")
            
        # Parse research notes and key facts from the raw results
        results = {
            "success": True,
            "research_notes": [],
            "key_facts": {},
            "related_files": _global_memory.get('related_files', {})
        }
        
        # Extract research notes and key facts from raw results
        if isinstance(raw_results, str):
            # Split the results into sections
            sections = raw_results.split('\n\n')
            for section in sections:
                if section.startswith('Research Notes:'):
                    notes = section.replace('Research Notes:', '').strip().split('\n')
                    results['research_notes'].extend([note.strip('- ') for note in notes if note.strip()])
                elif section.startswith('Key Facts:'):
                    facts = section.replace('Key Facts:', '').strip().split('\n')
                    for fact in facts:
                        if ':' in fact:
                            key, value = fact.strip('- ').split(':', 1)
                            results['key_facts'][key.strip()] = value.strip()
        
        # Update global memory with research results
        _global_memory['research_notes'] = results['research_notes']
        _global_memory['key_facts'] = results['key_facts']
        _global_memory['implementation_requested'] = False
        
        # Display research results
        if results['research_notes']:
            st.markdown("### Research Notes")
            for note in results['research_notes']:
                st.markdown(f"- {note}")
            
        if results['key_facts']:
            st.markdown("### Key Facts")
            for key, value in results['key_facts'].items():
                st.markdown(f"- **{key}**: {value}")
            
        if results['related_files']:
            st.markdown("### Related Files")
            for file in results['related_files']:
                st.code(file)
        
        return results

    except ValueError as e:
        logger.error(f"Research Configuration Error: {str(e)}")
        st.error(f"Research Configuration Error: {str(e)}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Research Error: {str(e)}")
        st.error(f"Research Error: {str(e)}")
        return {"success": False, "error": str(e)}
