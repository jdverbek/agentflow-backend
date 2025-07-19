import pytest
from unittest.mock import patch
from app.tasks import run_orchestrator

@patch('app.tasks.Crew.kickoff')
@patch('app.tasks.Task.execute')
def test_run(mock_execute, mock_kickoff):
    mock_kickoff.return_value = "Mock deliverable"
    mock_execute.return_value = "Mock verification: All items pass"
    
    out = run_orchestrator("Test task")
    assert "deliverable" in out
    assert "verification" in out
    assert "All items pass" in out["verification"]

