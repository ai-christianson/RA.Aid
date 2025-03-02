"""
Backward compatibility layer for the prompts module.

This module imports and re-exports all prompt constants from the prompts package.
It is maintained for backward compatibility with existing code.

DEPRECATED: Import directly from ra_aid.prompts package modules instead.
"""

import warnings

warnings.warn(
    "ra_aid.prompts module is deprecated. Import from ra_aid.prompts package instead.",
    DeprecationWarning,
    stacklevel=2,
)

from ra_aid.prompts import *