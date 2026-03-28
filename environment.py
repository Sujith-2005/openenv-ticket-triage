from models import Action, Observation, Reward, State, Ticket, RoutingDecision, StepResponse
from tasks import TASKS, grade_submission
import copy

class SupportTriageEnv:
    def __init__(self, task_id="task_1_easy", max_steps=15):
        self.task_id = task_id
        if self.task_id not in TASKS:
             self.task_id = "task_1_easy"
        
        task_data = TASKS[self.task_id]
        self.original_tickets = task_data["tickets"]
        self.customer_replies = task_data.get("customer_replies", {})
        self.max_steps = max_steps
        self.reset()
        
    def reset(self) -> Observation:
        self.inbox = [Ticket(**t.model_dump()) for t in self.original_tickets]
        self.routed_tickets = {}
        self.replies = {}
        self.step_count = 0
        self.done = False
        return self._get_observation("Environment reset.")
        
    def _get_observation(self, feedback: str) -> Observation:
        inbox_summary = [{"id": t.id, "subject": t.subject} for t in self.inbox]
        return Observation(
            feedback=feedback,
            inbox_summary=inbox_summary,
            done=self.done
        )
        
    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        if self.done:
            return self._get_observation("Episode already finished."), 0.0, True, {}
            
        self.step_count += 1
        reward = 0.0
        feedback = ""
        
        if action.action_type == "submit":
            self.done = True
            feedback = "Submitted final routing decisions."
            
        elif action.action_type == "read":
            ticket = next((t for t in self.inbox if t.id == action.ticket_id), None)
            if ticket:
                reply = self.replies.get(ticket.id, "")
                extra = f"\n\nCustomer Reply: {reply}" if reply else ""
                feedback = f"Ticket [{ticket.id}] Subject: {ticket.subject}\nBody: {ticket.body}{extra}"
                reward = 0.01
            else:
                feedback = f"Error: Ticket {action.ticket_id} not found in inbox."
                reward = -0.05
                
        elif action.action_type == "ask_customer":
            ticket = next((t for t in self.inbox if t.id == action.ticket_id), None)
            if not ticket:
                feedback = f"Error: Ticket {action.ticket_id} not found."
                reward = -0.05
            elif not action.question:
                feedback = "Error: Must provide a question for ask_customer."
                reward = -0.05
            else:
                if ticket.id in self.customer_replies:
                    reply = self.customer_replies[ticket.id]
                    self.replies[ticket.id] = reply
                    feedback = f"Customer replied to ticket {ticket.id}: '{reply}'"
                    reward = 0.1
                else:
                    feedback = f"Customer replied to ticket {ticket.id}: 'I think my initial message had all the details.'"
                    reward = -0.05
                    
        elif action.action_type == "route":
            ticket = next((t for t in self.inbox if t.id == action.ticket_id), None)
            if not ticket:
                feedback = f"Error: Ticket {action.ticket_id} not found."
                reward = -0.05
            elif not action.category or not action.priority or not action.department:
                feedback = "Error: Route action requires category, priority, and department."
                reward = -0.05
            else:
                decision = RoutingDecision(
                    category=action.category,
                    priority=action.priority,
                    department=action.department
                )
                self.routed_tickets[ticket.id] = decision
                self.inbox = [t for t in self.inbox if t.id != ticket.id]
                feedback = f"Successfully routed ticket {ticket.id}."
                reward = 0.2
        else:
            feedback = f"Error: Unknown action type '{action.action_type}'"
            reward = -0.05
            
        if not self.inbox or self.step_count >= self.max_steps:
            self.done = True
            
        obs = self._get_observation(feedback)
        
        info = {}
        if self.done:
            info["score"] = self.get_score()
            
        return obs, reward, self.done, info
        
    def get_state(self) -> State:
        return State(
            task_id=self.task_id,
            inbox=self.inbox,
            routed_tickets=self.routed_tickets,
            replies=self.replies,
            step_count=self.step_count,
            max_steps=self.max_steps
        )
        
    def get_score(self) -> float:
        return grade_submission(self.task_id, {k: v.model_dump() for k, v in self.routed_tickets.items()})
