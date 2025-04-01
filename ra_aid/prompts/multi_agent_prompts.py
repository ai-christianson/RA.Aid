"""
Multi-Agent Communication Protocol Prompts

This module contains the prompts and JSON structures for multi-agent communication
between the RA-Aid system and other agent systems. It defines the protocol for request/response handling.
"""

# JSON Schema for multi-agent communication
MULTI_AGENT_REQUEST_SCHEMA = {
    "type": "object",
    "required": ["task_id", "status", "questions", "request_timestamp"],
    "properties": {
        "task_id": {"type": "string"},
        "status": {"type": "string", "enum": ["pending", "answered", "partial", "error"]},
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["question_id", "text", "context"],
                "properties": {
                    "question_id": {"type": "string"},
                    "text": {"type": "string"},
                    "context": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["file"],
                            "properties": {
                                "file": {"type": "string"},
                                "section": {"type": "string"},
                                "line_start": {"type": "integer"},
                                "line_end": {"type": "integer"}
                            }
                        }
                    },
                    "avoid_tokens": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
        },
        "desired_output": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["explanation", "code", "summary", "list", "comparison", "filepath"]
                },
                "details": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        },
        "request_timestamp": {"type": "string", "format": "date-time"}
    }
}

# Comprehensive prompt for handling multi-agent communication
MULTI_AGENT_QUERY_HANDLER_PROMPT = '''
You are an agent that handles both creation and processing of multi-agent communication queries. Follow these guidelines:

1. Creating Requests:
   - Generate requests in `/agent_docs/multi_agent/{task_id}.json` using this structure:
```json
{
    "task_id": "unique_id",
    "status": "pending",
    "questions": [
        {
            "question_id": "q1",
            "text": "Concise question.",
            "context": [
                {"file": "path/file1.md", "section": "Optional"},
                {"file": "path/file2.py", "line_start": 10, "line_end": 25}
            ],
            "avoid_tokens": ["unnecessary", "words"]
        }
    ],
    "desired_output": {
        "type": "explanation|code|summary|list|comparison|filepath",
        "details": ["brief", "example"]
    },
    "request_timestamp": "YYYY-MM-DDTHH:MM:SSZ"
}
```

2. Request Creation Guidelines:
   - Generate unique task_id and question_id values
   - Write concise, token-efficient questions
   - Specify precise context (files, sections, line numbers)
   - Use avoid_tokens to exclude unnecessary words
   - Set clear desired_output type and details

3. Processing Requests:
   - Monitor `/agent_docs/multi_agent/` for requests
   - Validate against MULTI_AGENT_REQUEST_SCHEMA
   - Extract questions and context requirements
   - Follow section markers if specified
   - Respect avoid_tokens list

4. Response Generation:
   - Create responses matching the requested type:
     * explanation: Clear, concise explanations
     * code: Working code examples with necessary context
     * summary: Brief, focused summaries
     * list: Structured, enumerated points
     * comparison: Clear contrasts and similarities
     * filepath: References to relevant files

5. Response Structure:
   - Update the original JSON file in-place
   - For each question, add:
```json
{
    "answer": {
        "response_text": "Direct answer or reference to response file",
        "code_example_filepath": "path/to/example.py",  // If code is involved
        "related_files": ["path/to/relevant/files"]     // Any supporting files
    }
}
```

6. Large Response Handling:
   - Create separate files for lengthy content
   - Use format: `/agent_docs/multi_agent/{task_id}_{question_id}_response.{ext}`
   - Reference these files in response_text
   - Include all supporting files in related_files

7. Implementation Steps:
   - For creating requests:
     * Generate unique IDs
     * Write efficient questions
     * Specify context
     * Set desired output type
   - For processing requests:
     * Read and validate query
     * Gather required context
     * Generate appropriate response
     * Update JSON with response
     * Create supporting files if needed
     * Set status to "answered"
     * Add response_timestamp

8. Error Handling:
   - If implementation fails:
     * Set status to "error"
     * Provide clear error description
     * Suggest potential fixes if possible
   - If partial implementation:
     * Set status to "partial"
     * Complete what's possible
     * Explain what couldn't be done

9. Quality Checks:
   - Verify all file paths exist
   - Validate code examples run
   - Ensure responses match desired_output type
   - Check all context is properly referenced
   - Confirm JSON schema compliance
'''

# Response implementation schema
MULTI_AGENT_IMPLEMENTATION_SCHEMA = {
    "type": "object",
    "required": ["implementation_status", "response"],
    "properties": {
        "implementation_status": {
            "type": "string",
            "enum": ["complete", "partial", "error"]
        },
        "response": {
            "type": "object",
            "required": ["answer"],
            "properties": {
                "answer": {
                    "type": "object",
                    "required": ["response_text"],
                    "properties": {
                        "response_text": {"type": "string"},
                        "code_example_filepath": {"type": "string"},
                        "related_files": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    }
                }
            }
        },
        "error_details": {
            "type": "object",
            "properties": {
                "error_type": {"type": "string"},
                "error_message": {"type": "string"},
                "suggested_fixes": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
    }
} 