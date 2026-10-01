# AI Laboratory — Agents (Goal-Based Agent)

**Student:** Pranay Ghumatkar | **Roll No:** 2024A3PS0328G
**Worksheet:** [agents_lab.pdf](agents_lab.pdf)

A **goal-based agent** for warehouse navigation. Unlike a reflex agent, it holds
an explicit goal and selects actions that move it towards it; the decision
component is an A* search over the grid.

## Files

| File | Purpose |
|---|---|
| `warehouse_agent.py` | `Warehouse` environment and `GoalBasedAgent` (state, goal test, actions, A* planner) |
| `ANSWERS.md` | Task 1–3 answers, block diagram, prompt, results and reflection |
| `results.txt` | Program output |
| `agents_lab.pdf` | Worksheet |

## How to run

```bash
python warehouse_agent.py  # prints results.txt
```

Requires only the Python standard library.

See [ANSWERS.md](ANSWERS.md) for the full answers.
