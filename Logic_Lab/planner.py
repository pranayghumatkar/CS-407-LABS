"""
CS-407 AI Laboratory - Logical Planning
Logic + Search = Planning

A simple planner for the warehouse robot.  A state is a set of logical
propositions.  Each action has positive/negative preconditions and
positive/negative effects.  An action is applicable iff all positive
preconditions hold and no negative precondition holds.  Breadth-first search
finds a sequence of actions that reaches the goal.

Student : Pranay Ghumatkar
Roll No : 2024A3PS0328G
"""

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    """A planning action with logical preconditions and effects."""
    name: str
    positive_preconditions: frozenset
    negative_preconditions: frozenset
    add_effects: frozenset
    delete_effects: frozenset

    def applicable(self, state):
        """S |= Preconditions(a): positives present, negatives absent."""
        return (self.positive_preconditions <= state
                and not (self.negative_preconditions & state))

    def apply(self, state):
        """Remove negative effects, then add positive effects."""
        return (state - self.delete_effects) | self.add_effects


# ---------------------------------------------------------------------------
# The warehouse domain (Task 0)
# ---------------------------------------------------------------------------
LOCATIONS = ("A", "B", "C")
CONNECTED = (("A", "B"), ("B", "A"), ("B", "C"), ("C", "B"))

INITIAL_STATE = frozenset({"At(Robot,A)", "At(Package,A)"})
GOAL = frozenset({"At(Package,C)"})


def build_domain():
    actions = []
    for x, y in CONNECTED:
        actions.append(Action(
            f"Move({x},{y})",
            frozenset({f"At(Robot,{x})"}), frozenset(),
            frozenset({f"At(Robot,{y})"}), frozenset({f"At(Robot,{x})"})))
    for loc in LOCATIONS:
        actions.append(Action(
            f"PickUp(Package,{loc})",
            frozenset({f"At(Robot,{loc})", f"At(Package,{loc})"}), frozenset(),
            frozenset({"Holding(Package)"}), frozenset({f"At(Package,{loc})"})))
        actions.append(Action(
            f"Drop(Package,{loc})",
            frozenset({f"At(Robot,{loc})", "Holding(Package)"}), frozenset(),
            frozenset({f"At(Package,{loc})"}), frozenset({"Holding(Package)"})))
    return actions


# ---------------------------------------------------------------------------
# Planning = search over applicable actions (BFS)
# ---------------------------------------------------------------------------
def plan_bfs(initial, goal, actions):
    """Breadth-first search for an action sequence reaching `goal`.

    Returns (plan, expanded_states).  `plan` is None when no plan exists.
    """
    frontier = deque([(initial, [])])
    visited = {initial}
    expanded = 0
    while frontier:
        state, plan = frontier.popleft()
        if goal <= state:
            return plan, expanded
        expanded += 1
        for action in actions:
            if action.applicable(state):
                successor = action.apply(state)
                if successor not in visited:
                    visited.add(successor)
                    frontier.append((successor, plan + [action]))
    return None, expanded


def validate_plan(initial, goal, actions, plan):
    """Independently replay a plan, checking every precondition (Task 3)."""
    by_name = {a.name: a for a in actions}
    state = initial
    trace = [("S0", frozenset(state))]
    for i, action in enumerate(plan, 1):
        if action.name not in by_name:
            return False, trace, f"action {action.name!r} not in domain"
        if not action.applicable(state):
            return False, trace, f"precondition of {action.name} not satisfied"
        state = action.apply(state)
        trace.append((f"S{i} (after {action.name})", frozenset(state)))
    return goal <= state, trace, None


def show(name, state):
    return f"{name}: " + ", ".join(sorted(state))


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------
def main():
    actions = build_domain()
    print("=" * 70)
    print("LOGICAL PLANNING - warehouse robot  (Logic + Search = Planning)")
    print("Pranay Ghumatkar  |  2024A3PS0328G")
    print("=" * 70)

    print("\n--- Task 0: the planning problem (I, A, G) ---")
    print(f"    Initial state I : {sorted(INITIAL_STATE)}")
    print(f"    Goal G          : {sorted(GOAL)}")
    print(f"    Actions A       : {len(actions)} actions")
    for a in actions:
        pre = sorted(a.positive_preconditions)
        print(f"      {a.name:<22} pre {pre}")

    print("\n--- Task 1: plan constructed by hand ---")
    by_name = {a.name: a for a in actions}
    hand = [by_name[n] for n in
            ("PickUp(Package,A)", "Move(A,B)", "Move(B,C)", "Drop(Package,C)")]
    valid, trace, error = validate_plan(INITIAL_STATE, GOAL, actions, hand)
    for name, st in trace:
        print("    " + show(name, st))
    print(f"    hand plan valid : {valid}")

    print("\n--- Task 2/3: BFS planner on the original problem (Test A) ---")
    plan, expanded = plan_bfs(INITIAL_STATE, GOAL, actions)
    print(f"    plan found      : {plan is not None}")
    print(f"    plan            : {' -> '.join(a.name for a in plan)}")
    print(f"    plan length     : {len(plan)}")
    print(f"    states expanded : {expanded}")
    valid, trace, error = validate_plan(INITIAL_STATE, GOAL, actions, plan)
    print(f"    plan valid      : {valid}")

    print("\n--- Test B: impossible problem (remove PickUp) ---")
    no_pickup = [a for a in actions if not a.name.startswith("PickUp")]
    plan_b, expanded_b = plan_bfs(INITIAL_STATE, GOAL, no_pickup)
    print(f"    plan found      : {plan_b is not None}  (expected: no plan found)")
    print(f"    states expanded : {expanded_b}")

    print("\n--- Test C: irrelevant moves do not move the package ---")
    moves_only = [a for a in actions if a.name.startswith("Move")]
    plan_c1, _ = plan_bfs(INITIAL_STATE, frozenset({"At(Package,C)"}), moves_only)
    plan_c2, _ = plan_bfs(INITIAL_STATE, frozenset({"At(Robot,C)"}), moves_only)
    print(f"    goal At(Package,C) with moves only : {plan_c1 is not None}  (expected: no)")
    print(f"    goal At(Robot,C)   with moves only : {plan_c2 is not None}  (expected: yes)")
    print("    the robot reaching C is NOT treated as the package reaching C")


if __name__ == "__main__":
    main()
