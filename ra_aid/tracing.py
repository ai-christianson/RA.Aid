"""Tracing utilities for LangSmith integration."""

import functools
from typing import Optional, Callable, Any
import time

from langsmith import Client
from ra_aid.langsmith_config import LangSmithConfig

def with_tracing(run_name: str):
    """Decorator to enable LangSmith tracing for agent runs.
    
    Args:
        run_name: Name of the run in LangSmith
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, config: Optional[dict] = None, **kwargs):
            if not config:
                config = {}
            
            langsmith_config = config.get("langsmith", LangSmithConfig())
            if not langsmith_config.is_configured():
                return func(*args, config=config, **kwargs)
            
            client = Client(
                api_key=langsmith_config.api_key
            )
            
            try:
                run = client.create_run(
                    project_name=langsmith_config.project_name,
                    name=run_name,
                    run_type="chain",
                    inputs={"args": str(args), "kwargs": str(kwargs)}
                )
                
                if run is None:
                    return func(*args, config=config, **kwargs)
                
                start_time = time.time()
                result = func(*args, config=config, **kwargs)
                end_time = time.time()
                
                client.update_run(
                    run_id=run.id,
                    outputs={"result": str(result)},
                    end_time=end_time,
                    error=None
                )
                return result
            except Exception as e:
                # If we have a valid run, try to update it with the error
                if 'run' in locals() and run is not None:
                    try:
                        end_time = time.time()
                        client.update_run(
                            run_id=run.id,
                            outputs={},
                            end_time=end_time,
                            error=str(e)
                        )
                    except Exception:
                        pass  # Ignore errors when updating the run
                raise  # Re-raise the original exception
                
        return wrapper
    return decorator
