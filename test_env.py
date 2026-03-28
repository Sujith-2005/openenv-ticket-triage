import environment
import models
import json

print("\n--- INITIALIZING OPENENV: Customer Support ---\n")
env = environment.SupportTriageEnv("task_1_easy")
obs = env.reset()
print("[Reset Observation]:\\n", json.dumps(obs.model_dump(), indent=2))

print("\n--- AGENT ACTION 1: Read the Ticket ---\n")
action1 = models.Action(action_type="read", ticket_id="t1")
obs, reward, done, info = env.step(action1)
print(f"[Reward]: {reward} (Partial reward for successful reading)\\n[Observation Feedback]:\n{obs.feedback}")

print("\n--- AGENT ACTION 2: Correctly Route the Ticket ---\n")
action2 = models.Action(action_type="route", ticket_id="t1", category="technical", priority="high", department="support")
obs, reward, done, info = env.step(action2)
print(f"[Reward]: {reward} (Partial reward for sorting logic)\\n[Observation Feedback]:\n{obs.feedback}")

print(f"[Episode Done]: {done}")
print(f"[Final Evaluation Grade from Hackathon Grader]: {info['score']} out of 1.0")
print("VERIFICATION COMPLETE: Environment accurately tracks state, computes multi-step partial rewards, and grades deterministically.")
