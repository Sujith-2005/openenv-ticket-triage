from models import Ticket
from typing import Dict, Any

TASKS = {
    "task_1_easy": {
        "difficulty": "easy",
        "description": "Route a single clear technical support ticket.",
        "tickets": [
            Ticket(
                id="t1",
                subject="Internet down",
                body="My internet is down and my router has a red blinking light."
            )
        ],
        "expected": {
            "t1": {"category": "technical", "priority": "high", "department": "support"}
        },
        "customer_replies": {}
    },
    "task_2_medium": {
        "difficulty": "medium",
        "description": "Route a batch of 3 tickets to appropriate departments.",
        "tickets": [
            Ticket(
                id="t1",
                subject="Update credit card",
                body="I need to update my credit card on file before my next billing cycle."
            ),
            Ticket(
                id="t2",
                subject="Annual plan discount?",
                body="Can I get a discount if I switch to an annual plan instead of monthly?"
            ),
            Ticket(
                id="t3",
                subject="App crashing on startup",
                body="The iOS app crashed and lost all my work. I can't even open it now."
            )
        ],
        "expected": {
            "t1": {"category": "billing", "priority": "medium", "department": "finance"},
            "t2": {"category": "sales", "priority": "low", "department": "sales"},
            "t3": {"category": "technical", "priority": "high", "department": "engineering"}
        },
        "customer_replies": {}
    },
    "task_3_hard": {
        "difficulty": "hard",
        "description": "Triage 3 tickets, including one ambiguous ticket that requires asking the customer for more info before routing.",
        "tickets": [
            Ticket(
                id="t1",
                subject="Invite users?",
                body="How do I invite more users to my workspace?"
            ),
            Ticket(
                id="t2",
                subject="Double charge!",
                body="My bill is wrong, I was charged twice this month!"
            ),
            Ticket(
                id="t3",
                subject="Button doesn't work",
                body="It doesn't work when I click the button."
            )
        ],
        "expected": {
            "t1": {"category": "general", "priority": "low", "department": "support"},
            "t2": {"category": "billing", "priority": "high", "department": "finance"},
            "t3": {"category": "technical", "priority": "high", "department": "engineering"}
        },
        "customer_replies": {
            "t3": "The submit button on the checkout page gives a 500 internal server error. I can't buy anything."
        }
    }
}

def grade_submission(task_id: str, routed_tickets: Dict[str, Any]) -> float:
    """
    Grades the final routed_tickets dictionary against the expected routing.
    Returns a score strictly between 0.0 and 1.0.
    """
    if task_id not in TASKS:
        return 0.01
    
    expected = TASKS[task_id]["expected"]
    if not expected:
        return 0.99
    
    total_tickets = len(expected)
    score_per_ticket = 1.0 / total_tickets
    total_score = 0.0
    
    for t_id, exp in expected.items():
        if t_id in routed_tickets:
            decision = routed_tickets[t_id]
            # Award points for correct category, priority, and department
            correct_fields = 0
            if getattr(decision, 'category', decision.get('category')) == exp["category"]:
                correct_fields += 1
            if getattr(decision, 'priority', decision.get('priority')) == exp["priority"]:
                correct_fields += 1
            if getattr(decision, 'department', decision.get('department')) == exp["department"]:
                correct_fields += 1
            
            total_score += (correct_fields / 3.0) * score_per_ticket
            
    final_score = round(total_score, 2)
    return max(0.01, min(0.99, final_score))
