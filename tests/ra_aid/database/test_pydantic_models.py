"""
Tests for the Pydantic models in ra_aid.database.pydantic_models
"""

import datetime
import json
import pytest

from ra_aid.database.models import Session
from ra_aid.database.pydantic_models import SessionModel


class TestSessionModel:
    """Tests for the SessionModel Pydantic model"""
    
    def test_from_peewee_model(self):
        """Test conversion from a Peewee model instance"""
        # Create a Peewee Session instance
        now = datetime.datetime.now()
        metadata = {"os": "Linux", "cpu_cores": 8, "memory_gb": 16}
        session = Session(
            id=1,
            created_at=now,
            updated_at=now,
            start_time=now,
            command_line="ra-aid run",
            program_version="1.0.0",
            machine_info=json.dumps(metadata)
        )
        
        # Convert to Pydantic model
        session_model = SessionModel.model_validate(session, from_attributes=True)
        
        # Verify fields
        assert session_model.id == 1
        assert session_model.created_at == now
        assert session_model.updated_at == now
        assert session_model.start_time == now
        assert session_model.command_line == "ra-aid run"
        assert session_model.program_version == "1.0.0"
        assert session_model.machine_info == metadata
    
    def test_with_dict_machine_info(self):
        """Test creating a model with a dict for machine_info"""
        # Create directly with a dict for machine_info
        now = datetime.datetime.now()
        metadata = {"os": "Windows", "cpu_cores": 4, "memory_gb": 8}
        
        session_model = SessionModel(
            id=2,
            created_at=now,
            updated_at=now,
            start_time=now,
            command_line="ra-aid --debug",
            program_version="1.0.1",
            machine_info=metadata
        )
        
        # Verify fields
        assert session_model.id == 2
        assert session_model.machine_info == metadata
    
    def test_with_none_machine_info(self):
        """Test creating a model with None for machine_info"""
        now = datetime.datetime.now()
        
        session_model = SessionModel(
            id=3,
            created_at=now,
            updated_at=now,
            start_time=now,
            command_line="ra-aid",
            program_version="1.0.0",
            machine_info=None
        )
        
        assert session_model.id == 3
        assert session_model.machine_info is None
    
    def test_invalid_json_machine_info(self):
        """Test error handling for invalid JSON in machine_info"""
        now = datetime.datetime.now()
        
        # Invalid JSON string should raise ValueError
        with pytest.raises(ValueError):
            SessionModel(
                id=4,
                created_at=now,
                updated_at=now,
                start_time=now,
                command_line="ra-aid",
                program_version="1.0.0",
                machine_info="{invalid json}"
            )
    
    def test_unexpected_type_machine_info(self):
        """Test error handling for unexpected type in machine_info"""
        now = datetime.datetime.now()
        
        # Integer type should raise ValueError
        with pytest.raises(ValueError):
            SessionModel(
                id=5,
                created_at=now,
                updated_at=now,
                start_time=now,
                command_line="ra-aid",
                program_version="1.0.0",
                machine_info=123  # Not a dict or string
            )