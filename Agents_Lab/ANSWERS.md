# CS-407 AI Laboratory — Agents (Goal-Based Agent)

**Student:** Pranay Ghumatkar
**Roll No:** 2024A3PS0328G

All numbers quoted below are produced by `warehouse_agent.py` and recorded in
`results.txt`.

---

## Task 1 — Understanding the problem

1. **What is the environment?** A static warehouse grid (`7 × 21`) of free cells
   `.`, obstacles `#`, a start `S` at `(1,1)` and a goal `G` at `(1,19)`. It is
   fully observable and deterministic.
2. **What is the goal of the agent?** To reach the dispatch cell `G` — i.e. to
   make the proposition `At(Vehicle, G)` true.
3. **What actions are available?** `Up`, `Down`, `Left`, `Right`, each changing
   the position by one grid square and allowed only into a free cell.
4. **What information must the agent maintain to choose its next action?** Its
   current position is a sufficient state (the environment is static), and the
   goal cell. For a non-reflex decision it also keeps a *plan* — the remaining
   action sequence — and the search frontier/visited set while planning.
5. **Why is this a goal-based agent rather than a simple reflex agent?** A reflex
   agent maps the current percept straight to an action with no notion of
   purpose. This agent has an explicit goal and *selects actions that move it
   towards that objective*: it evaluates candidate actions by whether they reduce
   the (estimated) distance to the goal. The path is chosen globally, not
   reflexively.

> **Think About It — what if the warehouse doubles in size?** The same A*
> strategy still applies (it is optimal and complete on any finite grid), but the
> search space grows roughly four-fold, so planning takes longer and memory for
> the frontier/visited set grows. With a much larger or dynamically changing
> warehouse one would want a stronger heuristic, hierarchical planning, or
> replanning rather than one global search.

---

## Task 2 — Designing the agent

**Components:** environment, current state, goal, available actions and the
decision-making component. The full block diagram is:

```
Sensors --> State --> Goal test --> (yes) STOP
                        | no
                        v
                  A* Planner (f = g + h, Manhattan)
                        |
                        v
                 Action selection --> Actuators --> Environment
```

| Component | Design |
|---|---|
| Environment | grid of strings; `is_free(node)` decides passability |
| State | current position `(row, col)` |
| Goal | the `G` cell; goal test is `state == goal` |
| Actions | four moves, filtered by `is_free` |
| Decision component | A* search minimising `g(n) + h(n)`, Manhattan `h` |

---

## Task 3 — Prompt engineering and the generated program

**Prompt used:**

> Write a well-documented Python program implementing a goal-based agent for a
> warehouse navigation problem. The warehouse is a 2-D grid where `S` is the
> start, `G` is the goal, `#` are obstacles and `.` is free space. The agent can
> move Up, Down, Left or Right by one square, and must determine a collision-free
> path from S to G, avoiding all obstacles. Print either the path found or a
> suitable message if no path exists, and explain the search algorithm chosen and
> why it is appropriate. Represent the agent explicitly as a goal-based agent
> with a state, a goal test, available actions and a decision component.

**Results** (in `results.txt`):

| Quantity | Value |
|---|---|
| Goal reached | **True** |
| Path found | True |
| Path length | **20** moves |
| States expanded | 22 |
| First moves | `Right Right Right Right Down Right Right Up Right ...` |

The solution path drawn on the map:

```
#####################
#S****#************G#
#.##.***##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################
```

**Answers to the questions:**

1. **Did the LLM generate a working program on the first attempt?** It generated
   a working *search*, but the first version was written as a bare function
   rather than as a goal-based agent (no explicit state/goal-test/actions
   structure), so it did not match the specification.
2. **How can the prompt be improved?** By stating the *agent architecture*
   explicitly — state, goal test, available actions and a decision component —
   rather than only "write a program that finds a path". Specifying the structure
   is what makes the output an agent.
3. **What search algorithm did the LLM choose?** A* with the Manhattan distance
   heuristic (some drafts offered BFS; A* is the better default here).
4. **Why do you think it selected this algorithm?** Because the problem is a
   shortest-path task on a grid with a cheap, admissible heuristic, which is the
   canonical use case for A*; the analogous BFS would be correct but expand many
   more states.

---

## Testing

The agent was checked, independently of the map rendering, that:

- the agent **reaches the goal** and its final state equals `G`;
- the **path is valid** (consecutive cells adjacent and free);
- the **path is shortest** — length equals an independent BFS;
- **actions respect obstacles** (no `Up`/`Left` at the start cell);
- a **walled-off goal** is reported as unreachable, not looped on;
- the agent **executes action by action**, ending at the goal.

All of these checks pass.

---

## Reflection — how the LLM was used and how its output was validated

**Workflow.** Specify → design → prompt → implement → test → reflect. I wrote
down the environment, the agent's goal, its actions and the information it
maintains (Task 1), and drew the block diagram (Task 2) before prompting.

**Validation.** (1) Read the code against the design — does it actually contain a
*state*, a *goal test*, an *action set* and a *decision component*, or is it just
a pathfinding function with agent framing bolted on? (2) Tested with known-answer
cases — the real map, a walled-off goal, and the path length against an
independent BFS. (3) Checked the rendered solution against the grid to confirm no
move crosses an obstacle.

**Issue found and corrected.** The first LLM draft was a correct shortest-path
search but **not a goal-based agent** — no explicit state, goal test or action
set — so it did not satisfy the specification even though it "worked". The fix
was to restructure it into a `GoalBasedAgent` with `percept`, `goal_reached`,
`available_actions`, `plan` and `execute`, and to state the architecture in the
prompt. A program that produces the right path can still be the wrong kind of
program for the task.

**LLM strengths and limits.** Good at the mechanics of grid search — parsing the
map, the heap frontier, Manhattan distance and rendering the path. Not good at
knowing that the architectural *framing* was part of the requirement.
