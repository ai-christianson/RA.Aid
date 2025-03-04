"""
Repository package for database access abstractions.

This package contains repository implementations for various models,
following the repository pattern for data access abstraction.
"""

from ra_aid.database.repositories.human_input_repository import (
    HumanInputRepository, 
    HumanInputRepositoryManager, 
    get_human_input_repository
)
from ra_aid.database.repositories.key_fact_repository import (
    KeyFactRepository, 
    KeyFactRepositoryManager, 
    get_key_fact_repository
)
from ra_aid.database.repositories.key_snippet_repository import (
    KeySnippetRepository, 
    KeySnippetRepositoryManager, 
    get_key_snippet_repository
)
from ra_aid.database.repositories.related_files_repository import (
    RelatedFilesRepository,
    RelatedFilesRepositoryManager,
    get_related_files_repository
)
from ra_aid.database.repositories.research_note_repository import (
    ResearchNoteRepository,
    ResearchNoteRepositoryManager,
    get_research_note_repository
)

__all__ = [
    'HumanInputRepository', 
    'HumanInputRepositoryManager', 
    'get_human_input_repository',
    'KeyFactRepository', 
    'KeyFactRepositoryManager', 
    'get_key_fact_repository',
    'KeySnippetRepository', 
    'KeySnippetRepositoryManager', 
    'get_key_snippet_repository',
    'RelatedFilesRepository',
    'RelatedFilesRepositoryManager',
    'get_related_files_repository',
    'ResearchNoteRepository',
    'ResearchNoteRepositoryManager',
    'get_research_note_repository',
]