"""
Pydantic models for ra_aid database entities.

This module defines Pydantic models that correspond to Peewee ORM models,
providing validation, serialization, and deserialization capabilities.
"""

import datetime
import json
from typing import Dict, Any, Optional

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator


class SessionModel(BaseModel):
    """
    Pydantic model representing a Session.
    
    This model corresponds to the Session Peewee ORM model and provides
    validation and serialization capabilities. It handles the conversion
    between JSON-encoded strings and Python dictionaries for the machine_info field.
    
    Attributes:
        id: Unique identifier for the session
        created_at: When the session record was created
        updated_at: When the session record was last updated
        start_time: When the program session started
        command_line: Command line arguments used to start the program
        program_version: Version of the program
        machine_info: Dictionary containing machine-specific metadata
        display_name: Display name for the session (derived from human input or command line)
    """
    id: Optional[int] = None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    start_time: datetime.datetime
    command_line: Optional[str] = None
    program_version: Optional[str] = None
    machine_info: Optional[Dict[str, Any]] = None
    display_name: Optional[str] = None
    
    # Configure the model to work with ORM objects
    model_config = ConfigDict(from_attributes=True)
    
    @field_validator("machine_info", mode="before")
    @classmethod
    def parse_machine_info(cls, value: Any) -> Optional[Dict[str, Any]]:
        """
        Parse the machine_info field from a JSON string to a dictionary.
        
        Args:
            value: The value to parse, can be a string, dict, or None
            
        Returns:
            Optional[Dict[str, Any]]: The parsed dictionary or None
            
        Raises:
            ValueError: If the JSON string is invalid
        """
        if value is None:
            return None
        
        if isinstance(value, dict):
            return value
            
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in machine_info: {e}")
        
        raise ValueError(f"Unexpected type for machine_info: {type(value)}")
    
    @field_serializer("machine_info")
    def serialize_machine_info(self, machine_info: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        Serialize the machine_info dictionary to a JSON string for storage.
        
        Args:
            machine_info: Dictionary to serialize
            
        Returns:
            Optional[str]: JSON-encoded string or None
        """
        if machine_info is None:
            return None
        
        return json.dumps(machine_info)


class HumanInputModel(BaseModel):
    """
    Pydantic model representing a HumanInput.
    
    This model corresponds to the HumanInput Peewee ORM model and provides
    validation and serialization capabilities.
    
    Attributes:
        id: Unique identifier for the human input
        created_at: When the record was created
        updated_at: When the record was last updated
        content: The text content of the input
        source: The source of the input ('cli', 'chat', or 'hil')
        session_id: Optional reference to the associated session
    """
    id: Optional[int] = None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    content: str
    source: str
    session_id: Optional[int] = None
    
    # Configure the model to work with ORM objects
    model_config = ConfigDict(from_attributes=True)


class KeyFactModel(BaseModel):
    """
    Pydantic model representing a KeyFact.
    
    This model corresponds to the KeyFact Peewee ORM model and provides
    validation and serialization capabilities.
    
    Attributes:
        id: Unique identifier for the key fact
        created_at: When the record was created
        updated_at: When the record was last updated
        content: The text content of the key fact
        human_input_id: Optional reference to the associated human input
        session_id: Optional reference to the associated session
    """
    id: Optional[int] = None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    content: str
    human_input_id: Optional[int] = None
    session_id: Optional[int] = None
    
    # Configure the model to work with ORM objects
    model_config = ConfigDict(from_attributes=True)


class KeySnippetModel(BaseModel):
    """
    Pydantic model representing a KeySnippet.
    
    This model corresponds to the KeySnippet Peewee ORM model and provides
    validation and serialization capabilities.
    
    Attributes:
        id: Unique identifier for the key snippet
        created_at: When the record was created
        updated_at: When the record was last updated
        filepath: Path to the source file
        line_number: Line number where the snippet starts
        snippet: The source code snippet text
        description: Optional description of the significance
        human_input_id: Optional reference to the associated human input
        session_id: Optional reference to the associated session
    """
    id: Optional[int] = None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    filepath: str
    line_number: int
    snippet: str
    description: Optional[str] = None
    human_input_id: Optional[int] = None
    session_id: Optional[int] = None
    
    # Configure the model to work with ORM objects
    model_config = ConfigDict(from_attributes=True)


class ResearchNoteModel(BaseModel):
    """
    Pydantic model representing a ResearchNote.
    
    This model corresponds to the ResearchNote Peewee ORM model and provides
    validation and serialization capabilities.
    
    Attributes:
        id: Unique identifier for the research note
        created_at: When the record was created
        updated_at: When the record was last updated
        content: The text content of the research note
        human_input_id: Optional reference to the associated human input
        session_id: Optional reference to the associated session
    """
    id: Optional[int] = None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    content: str
    human_input_id: Optional[int] = None
    session_id: Optional[int] = None
    
    # Configure the model to work with ORM objects
    model_config = ConfigDict(from_attributes=True)


class TrajectoryModel(BaseModel):
    """
    Pydantic model representing a Trajectory.
    
    This model corresponds to the Trajectory Peewee ORM model and provides
    validation and serialization capabilities. It handles the conversion
    between JSON-encoded strings and Python dictionaries for the tool_parameters,
    tool_result, and step_data fields.
    
    Attributes:
        id: Unique identifier for the trajectory
        created_at: When the record was created
        updated_at: When the record was last updated
        human_input_id: Optional reference to the associated human input
        tool_name: Name of the tool that was executed
        tool_parameters: Dictionary containing the parameters passed to the tool
        tool_result: Dictionary containing the result returned by the tool
        step_data: Dictionary containing UI rendering data
        record_type: Type of trajectory record
        current_cost: Optional cost of the last LLM message
        input_tokens: Optional input/prompt token usage
        output_tokens: Optional output/completion token usage
        is_error: Flag indicating if this record represents an error
        error_message: The error message if is_error is True
        error_type: The type/class of the error if is_error is True
        error_details: Additional error details if is_error is True
        session_id: Optional reference to the associated session
    """
    id: Optional[int] = None
    created_at: datetime.datetime
    updated_at: datetime.datetime
    human_input_id: Optional[int] = None
    tool_name: Optional[str] = None
    tool_parameters: Optional[Dict[str, Any]] = None
    tool_result: Optional[Any] = None
    step_data: Optional[Dict[str, Any]] = None
    record_type: Optional[str] = None
    current_cost: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    is_error: bool = False
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    error_details: Optional[str] = None
    session_id: Optional[int] = None
    
    # Configure the model to work with ORM objects
    model_config = ConfigDict(from_attributes=True)
    
    @field_validator("tool_parameters", mode="before")
    @classmethod
    def parse_tool_parameters(cls, value: Any) -> Optional[Dict[str, Any]]:
        """
        Parse the tool_parameters field from a JSON string to a dictionary.
        
        Args:
            value: The value to parse, can be a string, dict, or None
            
        Returns:
            Optional[Dict[str, Any]]: The parsed dictionary or None
            
        Raises:
            ValueError: If the JSON string is invalid
        """
        if value is None:
            return None
        
        if isinstance(value, dict):
            return value
            
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in tool_parameters: {e}")
        
        raise ValueError(f"Unexpected type for tool_parameters: {type(value)}")
    
    @field_validator("tool_result", mode="before")
    @classmethod
    def parse_tool_result(cls, value: Any) -> Optional[Any]:
        """
        Parse the tool_result field from a JSON string to a Python object.
        
        Args:
            value: The value to parse, can be a string, dict, list, or None
            
        Returns:
            Optional[Any]: The parsed object or None
            
        Raises:
            ValueError: If the JSON string is invalid
        """
        if value is None:
            return None
        
        if not isinstance(value, str):
            return value
            
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in tool_result: {e}")
    
    @field_validator("step_data", mode="before")
    @classmethod
    def parse_step_data(cls, value: Any) -> Optional[Dict[str, Any]]:
        """
        Parse the step_data field from a JSON string to a dictionary.
        
        Args:
            value: The value to parse, can be a string, dict, or None
            
        Returns:
            Optional[Dict[str, Any]]: The parsed dictionary or None
            
        Raises:
            ValueError: If the JSON string is invalid
        """
        if value is None:
            return None
        
        if isinstance(value, dict):
            return value
            
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in step_data: {e}")
        
        raise ValueError(f"Unexpected type for step_data: {type(value)}")
    
    @field_serializer("tool_parameters")
    def serialize_tool_parameters(self, tool_parameters: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        Serialize the tool_parameters dictionary to a JSON string for storage.
        
        Args:
            tool_parameters: Dictionary to serialize
            
        Returns:
            Optional[str]: JSON-encoded string or None
        """
        if tool_parameters is None:
            return None
        
        return json.dumps(tool_parameters)
    
    @field_serializer("tool_result")
    def serialize_tool_result(self, tool_result: Optional[Any]) -> Optional[str]:
        """
        Serialize the tool_result object to a JSON string for storage.
        
        Args:
            tool_result: Object to serialize
            
        Returns:
            Optional[str]: JSON-encoded string or None
        """
        if tool_result is None:
            return None
        
        return json.dumps(tool_result)
    
    @field_serializer("step_data")
    def serialize_step_data(self, step_data: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        Serialize the step_data dictionary to a JSON string for storage.
        
        Args:
            step_data: Dictionary to serialize
            
        Returns:
            Optional[str]: JSON-encoded string or None
        """
        if step_data is None:
            return None
        
        return json.dumps(step_data)
