from langgraph.graph.graph import CompiledGraph
from typing import TYPE_CHECKING

# Unfortunately need this to avoid Circular Imports
if TYPE_CHECKING:
    from ra_aid.agents.ciayn_agent import CiaynAgent

    RAgents = CompiledGraph | CiaynAgent
else:
    RAgents = CompiledGraph
