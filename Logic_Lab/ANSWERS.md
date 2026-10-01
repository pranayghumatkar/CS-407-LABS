# CS-407 AI Laboratory — Logical Planning (Logical Reasoning for Planning)

**Student:** Pranay Ghumatkar
**Roll No:** 2024A3PS0328G

All numbers quoted below are produced by `planner.py` and recorded in `results.txt`.

---

## Task 0 — Understand the planning problem

`(I, A, G)` with the warehouse robot:

- **Initial state** `I = { At(Robot,A), At(Package,A) }`
- **Goal** `G = { At(Package,C) }`
- **Actions** `A` — `Move(X,Y)`, `PickUp(Package,L)`, `Drop(Package,L)`:

| Action | Preconditions | Effects (add / delete) |
|---|---|---|
| `Move(X,Y)` | `At(Robot,X)` | add `At(Robot,Y)`; delete `At(Robot,X)` |
| `PickUp(Package,L)` | `At(Robot,L)`, `At(Package,L)` | add `Holding(Package)`; delete `At(Package,L)` |
| `Drop(Package,L)` | `At(Robot,L)`, `Holding(Package)` | add `At(Package,L)`; delete `Holding(Package)` |

**Which actions are initially applicable?**

- `PickUp(Package,A)` — **applicable**: both preconditions (`At(Robot,A)` and
  `At(Package,A)`) hold in `I`.
- `Drop(Package,C)` — **not applicable**: `At(Robot,C)` is false and
  `Holding(Package)` is false. An action is applicable only if *all* its
  preconditions are satisfied; the fact that it appears in the action list is
  irrelevant.

> **Think About It.** Applicability is a logical entailment, `S ⊨ Preconditions(a)`,
> not a syntactic check on the action list.

---

## Task 1 — Plan constructed by hand

Sequence: `PickUp(Package,A) → Move(A,B) → Move(B,C) → Drop(Package,C)`.

| State | Facts |
|---|---|
| S0 | `At(Robot,A), At(Package,A)` |
| S1 (after `PickUp(Package,A)`) | `At(Robot,A), Holding(Package)` |
| S2 (after `Move(A,B)`) | `At(Robot,B), Holding(Package)` |
| S3 (after `Move(B,C)`) | `At(Robot,C), Holding(Package)` |
| S4 (after `Drop(Package,C)`) | `At(Package,C), At(Robot,C)` ⊨ G |

Note the worksheet's *illustrative* sequence picks up at **B**, but in the actual
problem the package starts at **A**, so the pickup must happen before any move.
The independent replay in `planner.validate_plan()` confirms the plan is valid.

---

## Task 2 — The prompt used with the LLM

> I want to implement a simple planning agent in Python. Represent a state as a
> set of logical propositions. Each action should contain a name, positive
> preconditions, negative preconditions, positive effects and negative effects.
> An action is applicable if all of its preconditions are satisfied by the
> current state. When an action is applied, remove its negative effects from the
> state and add its positive effects. Use breadth-first search to find a sequence
> of actions that achieves a specified goal. The program should detect when no
> plan exists, print the resulting sequence of actions, and print the states
> reached after each action.

Field identification in the generated code: preconditions → `Action.applicable`;
effects → `Action.apply` (delete then add); goal → `plan_bfs`'s `goal <= state`
test; BFS → the `deque` frontier. I changed the draft's dict-of-sets state to a
`frozenset` so states are hashable and can be deduplicated, and I added
`validate_plan()` to replay a plan independently.

---

## Task 3 — Testing the generated planner

| Test | Initial state | Goal | Plan found? | Plan | Valid? |
|---|---|---|---|---|---|
| **A** original | `{At(Robot,A), At(Package,A)}` | `{At(Package,C)}` | Yes | `PickUp(A) → Move(A,B) → Move(B,C) → Drop(C)` (length 4) | Yes |
| **B** impossible (no `PickUp`) | same | `{At(Package,C)}` | **No** — reports "no plan found" | — | — |
| **C** irrelevant action (moves only) | same | `{At(Package,C)}` | **No** | — | — |
| **C′** same actions | same | `{At(Robot,C)}` | Yes | `Move(A,B) → Move(B,C)` | Yes |

Test A expanded **7** states; Test B expanded **3**. Test C confirms the planner
does *not* treat "robot at C" as "package at C": with only `Move` actions, the
package goal is unreachable while the robot goal is reachable.

---

## Task 4 — Logic and search

- **Logic** determines what is *possible*: `S ⊨ Preconditions(a)` decides
  applicability, and `S' = Apply(S,a)` is the logical state transition.
- **Search** determines what to *try*: BFS explores alternative action sequences.

Completed description: **Current state → Check action preconditions (logic) →
[applicable?] → Generate successor state (logic) → Search over alternatives
(BFS) → Goal? → output plan.** Logic decides the edges; search decides the order
to traverse them.

---

## Task 5 — Can the LLM verify its own plan?

The LLM's explanation is a *generated* claim; the Python replay in
`validate_plan()` is an *independent* computation. When they disagree, the
executed transitions must be trusted, because they can be checked against the
declared preconditions and effects. A generated explanation is not the same as
an independent verification.

---

## Reflection Questions

1. **Why specify preconditions and effects before prompting?** They are the
   specification: without them the "planner" has no defined semantics, and any
   output is unfalsifiable. With them, applicability and state change become
   checkable properties.
2. **An error from not checking preconditions** — e.g. applying
   `PickUp(Package,B)` while the robot is still at A. The state would gain
   `Holding(Package)` without the robot ever being near the package; the plan
   would look plausible but be physically impossible.
3. **Why "looks reasonable" ≠ valid.** A plan is valid only if every action's
   preconditions hold in the state where it is executed. Reasonableness is a
   human judgement about the printed action names; validity is a check against
   the transitions.
4. **What the LLM contributed.** The BFS frontier, the action dataclass, and the
   successor-generation loop — the mechanical skeleton.
5. **What I verified independently.** That `Plan → States` actually satisfies the
   goal (Test A), that the impossible problem returns "no plan" (Test B), and
   that irrelevant actions do not substitute for the real goal (Test C).
6. **Where logic is used.** In the applicability test `S ⊨ Preconditions(a)` and
   in computing the effects (`S' = Apply(S,a)`).
7. **Relation to search.** Planning *is* search over the state space, where the
   allowed edges are exactly those licensed by logic. Hence *Logic + Search =
   Planning*.

---

## Optional extension — Prolog (Tasks 6–8)

`planner.pl` encodes `connected/2` facts and the rule
`can_move(X,Y) :- connected(X,Y).`

- `?- can_move(a,b).` → **true**
- `?- can_move(b,c).` → **true**
- `?- can_move(a,c).` → **false** (there is no `connected(a,c)` fact)

**Task 7.** For the plan `Move(a,b), Move(b,c)`, `valid_move(a,b)` and
`valid_move(b,c)` succeed, while `valid_move(a,c)` fails. If the Python planner
proposed `Move(a,c)`, Prolog rejects it — an independent check of a candidate
action against the warehouse knowledge.

**Task 8.** `wet_road.` (fact), `slippery :- wet_road.`, `reduce_speed :- slippery.`
Query `?- reduce_speed.` → **true**. Reasoning: `WetRoad ⇒ Slippery ⇒ ReduceSpeed`.

**Prolog reflection.**

1. A **fact** asserts something unconditionally (`connected(a,b).`); a **rule**
   derives something from a condition (`can_move(X,Y) :- connected(X,Y).`).
2. A query `?- q.` asks whether `q` is entailed by the facts and rules — Prolog
   tries to prove it by inference.
3. Prolog gives an *independent* verifier: the plan is generated by Python but
   checked against a separate logical description, so a bug in the planner
   cannot silently validate itself.
4. When a plan was produced with LLM help, an independent verifier is valuable
   precisely because the LLM may produce a confident but unsupported answer.

> SWI-Prolog was not available in the environment, so the queries and their
> logical answers are recorded above.

---

## Reflection — how the LLM was used and how its output was validated

**Workflow.** Specify → design → prompt → implement → test → reflect. Before
prompting I wrote down `(I, A, G)`, every action's preconditions and effects, and
the manual plan (Tasks 0–1). That specification is what I tested against.

**Validation.** (1) Read the code against the spec — the applicability test, the
delete-then-add effect order, and the goal test. (2) Tested with known-answer
cases: a solvable problem, an impossible one, and a trap where the robot reaching
the goal location is *not* the goal. (3) Replayed each returned plan
independently, re-checking every precondition.

**Issues found and corrected.** My first "hand" plan was assembled by filtering
the action list, which returned domain order (`Move` before `PickUp`) and failed a
precondition — the *order* of actions is part of the plan. The worksheet's
example also picks up at B, but the package starts at A. And applicability is a
logical test, not membership of the action list: a planner that only checked
membership would apply `Drop(Package,C)` from the initial state.

**LLM strengths and limits.** Good at the mechanics — the action dataclass, the
`deque` BFS frontier and successor generation. Not good at knowing whether a
candidate plan is *valid*, which is why an independent verifier (the Python replay
and, optionally, Prolog) matters: a generated explanation is not an independent
verification.
