# OpenEnv: Customer Support Ticket Triage

## Environment Description
**Customer Support Ticket Triage** is a complete, real-world OpenEnv environment where an AI agent acts as a customer support specialist. It must read incoming tickets, ask customers for more information when tickets are ambiguous, and route them to the correct department with the proper category and priority.

This is not a toy problem—it simulates real helpdesk routing logic often handled by humans or automated systems. The environment uses the standard `step() / reset() / state()` API through FastAPI.

## Action Space
The agent can perform the following actions via `POST /step`:
- `read`: Read the full content of a ticket by providing `ticket_id`.
- `ask_customer`: Ask the customer for more information by providing `ticket_id` and `question`.
- `route`: Route a ticket using `ticket_id`, `category`, `priority`, and `department`.
  - Categories: `billing`, `technical`, `sales`, `general`
  - Priorities: `low`, `medium`, `high`
  - Departments: `finance`, `engineering`, `sales`, `support`
- `submit`: Finish the episode once triage is complete.

## Observation Space
Returns JSON containing:
- `feedback`: String containing the result of the last action (e.g. ticket content, customer reply, or success/error message).
- `inbox_summary`: List of tickets currently pending.
- `done`: Boolean indicating episode completion.

## Rewards
The environment provides dense, meaningful rewards pointing towards partial progress:
- **+0.2** for successfully routing a ticket.
- **+0.1** for asking a necessary clarifying question.
- **+0.01** for reading a ticket (to encourage exploration).
- **-0.05** for invalid actions, bad routing attempts, or unnecessary questions.

## Setup Instructions

### 1. Docker (Recommended for HF Spaces)
```bash
docker build -t openenv-triage .
docker run -p 7860:7860 -e OPENAI_API_KEY=\"your_api_key_here\" openenv-triage
```

### 2. Local Python Setup
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

## Running the Baseline

The baseline uses the OpenAI API client (`gpt-4o-mini`) to infer against the environment tasks.
Make sure you have your open API key set in your terminal:
```bash
export OPENAI_API_KEY=\"sk-your-key\"
```

Run via endpoint:
`POST http://localhost:7860/baseline`

Or run locally via script:
```bash
python baseline.py
```

### Baseline Scores (Typical)
- **task_1_easy** (1 ticket): `1.0`
- **task_2_medium** (3 tickets): `1.0`
- **task_3_hard** (3 tickets + 1 ambiguous requiring question): `1.0`

## Endpoints

This environment fully complies with OpenEnv specs and standard HF Space requirements.
- `POST /reset` - Resets and returns initial observation.
- `POST /step` - Takes Pydantic `Action`, returns `StepResponse`.
- `GET /state` - Dumps full state dictionary.
- `GET /tasks` - Lists tasks (`task_1_easy`, `task_2_medium`, `task_3_hard`) and action schema.
- `GET /grader` - Evaluates the run (0.0 - 1.0).
- `POST /baseline` - Triggers the background inference script baseline.
"# openenv-ticket-triage" 
