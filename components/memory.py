# Global memory store
_global_memory = {}

def initialize_memory():
    """Initialize global memory with default values."""
    _global_memory.clear()
    _global_memory['related_files'] = {}
    _global_memory['implementation_requested'] = False
    _global_memory['config'] = {}
    _global_memory['research_notes'] = []
    _global_memory['plans'] = []
    _global_memory['tasks'] = {}
    _global_memory['key_facts'] = {}
    _global_memory['key_snippets'] = {}
    
def get_memory():
    """Retrieve the current state of global memory."""
    return _global_memory
