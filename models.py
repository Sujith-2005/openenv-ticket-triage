from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class Ticket(BaseModel):
    id: str
    subject: str
    body: str

class RoutingDecision(BaseModel):
    category: str
    priority: str
    department: str

class State(BaseModel):
    task_id: str
    inbox: List[Ticket]
    routed_tickets: Dict[str, RoutingDecision]
    replies: Dict[str, str]
    step_count: int
    max_steps: int

class Action(BaseModel):
    action_type: str = Field(..., description="Must be one of: 'read', 'route', 'ask_customer', 'submit'")
    ticket_id: Optional[str] = Field(None, description="The ID of the ticket to act upon ('read', 'route', 'ask_customer')")
    category: Optional[str] = Field(None, description="'billing', 'technical', 'sales', 'general'. Required for 'route'.")
    priority: Optional[str] = Field(None, description="'low', 'medium', 'high'. Required for 'route'.")
    department: Optional[str] = Field(None, description="'finance', 'engineering', 'sales', 'support'. Required for 'route'.")
    question: Optional[str] = Field(None, description="The question to ask the customer. Required for 'ask_customer'.")

class Observation(BaseModel):
    feedback: str
    inbox_summary: List[Dict[str, str]]
    done: bool

class Reward(BaseModel):
    value: float

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]
