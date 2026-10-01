# CS-407 AI Laboratory — Search (A* and BFS)

**Student:** Pranay Ghumatkar
**Roll No:** 2024A3PS0328G

All numbers quoted below are produced by `search_agent.py` and recorded in
`results.txt`.

---

## Task 0 — The search problem `P = (S, A, T, s0, G, c)`

| Component | Specification |
|---|---|
| **State S** | A grid cell `(row, col)` that is not an obstacle. |
| **Actions A** | `{Up, Down, Left, Right}` — move one cell. |
| **Transition T** | `(r,c) → (r+dr, c+dc)` for the chosen action. |
| **Initial state s0** | The cell marked `S` = `(1,1)`. |
| **Goal G** | The cell marked `G` = `(1,19)`. |
| **Cost c** | Every move costs `1`, so path cost = number of moves. |

**(a) What is necessary to specify a state?** Only the agent's position — the
environment is static and fully known, so the agent's location is a sufficient
state description (it is a *Markovian* state).

**(b) What makes an action invalid?** Moving outside the grid or into an
obstacle `#`. Each movement must land on a free cell (`.` , `S` or `G`).

**(c) Is this a deterministic search problem?** Yes. Each action maps a state to
exactly one successor and every action has a known cost, so there is no
uncertainty.

**(d) What constitutes a solution?** A sequence of actions from `S` to `G` such
that the concatenated transitions land exactly on `G`, with no move leaving an
obstacle or the grid.

> **Think About It — why formulate before implementing?** Without a clear state
> definition, action set and cost, the code has nothing to be measured against;
> "it found a path" would be unfalsifiable.

---

## Task 1 — Agent design (written before prompting)

1. **State representation:** a tuple `(row, col)`.
2. **Warehouse representation:** a list of strings; `grid[r][c] != '#'` decides
   freedom.
3. **Valid actions:** the four moves filtered by `is_free` (in bounds, not `#`).
4. **Goal recognition:** `node == goal`.
5. **Frontier:** a priority queue keyed on `f(n) = g(n) + h(n)` for A* (a FIFO
   queue for BFS).
6. **Path reconstruction:** a `came_from` dictionary walked back from the goal.

**Reported on termination:** whether a solution was found, the path, the path
length, and the number of states expanded.

---

## Task 2 — Prompt used with the LLM

> I am implementing a simple goal-based search agent in Python. The environment
> is a grid represented by an ASCII map. The agent starts at S and must reach G;
> `#` are obstacles and `.` are free cells. The agent can move up, down, left or
> right, every movement costs 1. Implement A* search using Manhattan distance as
> the heuristic `h(n) = |x−xG| + |y−yG|`. The program should represent grid
> positions as states, maintain a frontier, calculate g(n), h(n) and f(n), avoid
> repeatedly expanding the same state, reconstruct the path when the goal is
> reached, and report the path, its length and the number of states expanded.

---

## Task 3 — Testing the generated program

| Test | Map | Path found? | Path length | States expanded |
|---|---|---|---|---|
| **1 original warehouse** | winding corridor | Yes | **40** | 63 |
| **2 trivial** | goal adjacent (`#SG##`) | Yes | **1** | 1 |
| **3 no solution** | goal walled off | **No** (no infinite loop) | — | — |
| **4 alternative paths** | several shortest paths | Yes | **4** (shortest) | — |

Test 3 is the important one: the program *reports failure* rather than looping.
Test 4 checks that the returned path is a genuine shortest path, not merely *a*
path.

> **Think About It.** A plausible path is not a validated algorithm; testing must
> include cases whose correct behaviour is known in advance.

---

## Task 4 — Where each concept appears

| Concept | Where in the code |
|---|---|
| State | `(row, col)` tuples |
| Action | the four `(dr, dc)` offsets in `neighbors` |
| Transition | `nb = (r+dr, c+dc)` |
| Goal test | `if node == goal` in `astar` / `bfs` |
| `g(n)` | `tentative = g + 1` (cost so far) |
| `h(n)` | `heuristic(nb, goal)` |
| `f(n)` | `tentative + heuristic(nb, goal)` — the priority |
| Frontier | `heapq` heap `frontier` (A*), `deque` (BFS) |
| Visited states | `best_g` / `came_from` dictionaries |
| Path reconstruction | `reconstruct(came_from, node)` |

**(a) Frontier data structure:** a binary heap (`heapq`) for A*, a FIFO `deque`
for BFS.

**(b) Selecting the next state:** `heapq.heappop` returns the smallest `f(n)`.

**(c) Where the heuristic is calculated:** in the `heuristic` argument, inside
the priority pushed onto the heap.

**(d) Is `f(n) = g(n) + h(n)` explicit?** Yes — `tentative + heuristic(nb, goal)`.

**(e) Preventing repeated exploration:** a state is only pushed when it improves
its best-known `g`; `best_g`/`came_from` act as the visited set.

---

## Task 5 — A* vs BFS

| Measure | BFS | A* (Manhattan) |
|---|---|---|
| Solution found | True | True |
| Path length | 40 | 40 |
| States expanded | 63 | 63 |

**(a)** Both found a solution. **(b)** Both found paths of length 40. **(c)** On
*this* map they expanded the same number of states (63). **(d)** A* expands fewer
states when there is real branching to prune; here the warehouse is essentially
a single winding corridor, so almost every free cell is on the search frontier
either way — the heuristic has nothing to cut. On an **open map** with many
branches the difference is stark:

| Measure (open map) | BFS | A* (Manhattan) |
|---|---|---|
| Path length | 13 | 13 |
| States expanded | **49** | **13** |

> **Think About It.** A* is not better "because it is smarter" — it is better
> because good `h` information focuses the search. With `h ≡ 0` it becomes
> uniform-cost search, i.e. BFS.

---

## Task 6 — Heuristic investigation

| Heuristic | Found | Path length | States expanded |
|---|---|---|---|
| Manhattan (admissible) | True | 40 | 63 |
| `h = 0` | True | 40 | 63 |
| Euclidean (admissible, weaker) | True | 40 | 63 |
| Manhattan × 2 (inadmissible) | True | 40 | **67** |

**Why Manhattan is appropriate:** the robot moves only horizontally and
vertically one cell at a time, so the minimum number of moves to reach the goal
is exactly the sum of the absolute row and column differences. Manhattan is
therefore an **admissible** heuristic: `h(n) ≤ h*(n)` — it never overestimates
the true remaining cost, so A* still returns an optimal path.

- **`h = 0`** removes the informant; A* degrades to uniform-cost search and
  behaves like BFS.
- **Euclidean** is smaller than Manhattan here, hence still admissible but a
  weaker lower bound — it gives the heuristic less to cut with.
- **Manhattan × 2** is **inadmissible**: it can overestimate the remaining cost,
  so A* explores more states (67 vs 63) and in general may return a suboptimal
  path. Too aggressive a heuristic damages A*.

> **Think About It — admissibility.** An *over*-optimistic heuristic (too small)
> stays safe but uninformative; an *aggressive* heuristic (too large) loses the
> optimality guarantee. The experiments show both directions.

---

## Task 7 — Evaluating the LLM-generated agent

1. **Correct immediately:** the A* skeleton — heap frontier, `g`/`h`/`f`, row/column
   moves, and path reconstruction.
2. **Bugs/design problems:** the no-solution case needed an explicit "not found"
   return rather than falling off the end of the loop, and goal-testing on the
   heap *push* instead of the *pop* can return a suboptimal path.
3. **How I found them:** by running the deliberately-hard tests (walled goal,
   alternative paths), not by reading the code.
4. **Unfamiliar terminology/data structures:** using a heap with a tie-breaking
   counter to keep entries comparable was worth reading up on.
5. **Modifications:** I factorised the heuristic into a parameter so the Task 6
   variants could be swapped in, and made the result a dictionary.
6. **Most useful tests:** the no-solution test and the alternative-paths test.
7. **Could I have trusted it without testing?** No — the corridor map hides
   bugs, so the constructed edge cases are what give confidence.
8. **What I understand better now:** why admissibility matters — the ×2
   experiment made the optimality guarantee concrete.

---

## Final Reflection

1. **Why formulate the search problem first.** It fixes what a state, an action
   and the cost are, which is precisely what the algorithm manipulates; without
   it, "the program works" cannot be defined.
2. **In what sense A* is "informed".** Its choice of which state to expand next
   uses `h(n)`, an estimate of the cost to the goal — information about the goal,
   not just about what has been seen. Blind search uses only `g`.
3. **Why the heuristic matters.** It is the difference between exploring a
   corridor exhaustively (63) and heading almost straight to the goal (13); and
   if it is inadmissible it can cost optimality as well as time.
4. **What the LLM contributed.** The mechanical implementation of the algorithm
   from a precise specification, quickly and in readable form.
5. **What could go wrong without testing.** A wrong heuristic or a mis-placed
   goal test still returns *a* path, so an engineer could ship a suboptimal or
   subtly incorrect agent believing it correct. The failure would only surface in
   edge cases — exactly the ones testing must supply.

---

## Reflection — how the LLM was used and how its output was validated

**Workflow.** Specify → design → prompt → implement → test → reflect. I wrote the
tuple `(S, A, T, s0, G, c)`, the agent design (state, frontier, path
reconstruction) and the reporting requirements before prompting.

**Validation.** (1) Read the code against the spec — where `g`, `h`, `f`, the
frontier and the visited set live, and where the goal is tested. (2) Ran four
known-answer cases: the real map, a one-step map, a walled-off goal, and a map
with several shortest paths. (3) Checked structural properties — consecutive path
cells adjacent and free, A* and BFS agreeing on length, and `h(start) ≤ true
cost` (admissibility).

**Issues found and corrected.** My first write-up claimed "A* expands fewer
states because the heuristic focuses the search"; the measurement contradicted it
— on the corridor-like warehouse both expand 63. I corrected the claim and added
an **open map** test (BFS 49 vs A* 13) so the difference is demonstrated, not
asserted. The inadmissible Manhattan × 2 also expanded *more* states (67), the
opposite of the naive intuition. A goal test on the wrong side of the heap (push
instead of pop) can return a non-optimal path, which is why it is tested on pop.

**LLM strengths and limits.** Good at the A* skeleton — the heap frontier,
`g`/`h`/`f` bookkeeping, the moves and path reconstruction. Not good at knowing
whether the *experiment* supports the claim; the state counts had to be measured,
and one of my own written conclusions had to be retracted.
