"""
CS-407 AI Laboratory - Agents
Constructing a Goal-Based Agent using a Large Language Model

A goal-based warehouse vehicle.  Unlike a simple reflex agent, it holds an
explicit goal and selects actions that move it towards that goal.  The
decision-making component is an A* search over the grid.

Student : Pranay Ghumatkar
Roll No : 2024A3PS0328G
"""

import heapq
import itertools

# ---------------------------------------------------------------------------
# The environment: an ASCII warehouse map (S start, G goal, # obstacle)
# ---------------------------------------------------------------------------
WAREHOUSE = """\
#####################
#S....#............G#
#.##....##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################"""

ACTIONS = ((-1, 0, "Up"), (1, 0, "Down"), (0, -1, "Left"), (0, 1, "Right"))


class Warehouse:
    """The environment: knows which cells are free."""

    def __init__(self, text):
        self.grid = [row for row in text.splitlines() if row]
        self.start = self.goal = None
        for r, row in enumerate(self.grid):
            for c, ch in enumerate(row):
                if ch == "S":
                    self.start = (r, c)
                elif ch == "G":
                    self.goal = (r, c)
        if self.start is None or self.goal is None:
            raise ValueError("map must contain S and G")

    def is_free(self, node):
        r, c = node
        return (0 <= r < len(self.grid) and 0 <= c < len(self.grid[0])
                and self.grid[r][c] != "#")

    def render(self, path=None):
        path = set(path or [])
        out = []
        for r, row in enumerate(self.grid):
            out.append("".join(
                "*" if (r, c) in path and ch == "." else ch
                for c, ch in enumerate(row)))
        return "\n".join(out)


class GoalBasedAgent:
    """A goal-based agent: sensors -> state -> goal test -> planner -> action."""

    def __init__(self, environment):
        self.env = environment
        self.goal = environment.goal
        self.state = environment.start

    def percept(self, environment):
        """The agent's current position (its percept)."""
        return self.state

    def goal_reached(self, state):
        return state == self.goal

    def available_actions(self, state):
        """Valid actions: moves into a free cell."""
        return [(dr, dc, name) for dr, dc, name in ACTIONS
                if self.env.is_free((state[0] + dr, state[1] + dc))]

    def heuristic(self, node):
        return abs(node[0] - self.goal[0]) + abs(node[1] - self.goal[1])

    def plan(self):
        """A* search: choose the action sequence that reaches the goal."""
        counter = itertools.count()
        frontier = [(self.heuristic(self.state), 0, next(counter), self.state)]
        came_from = {self.state: None}
        best_g = {self.state: 0}
        expanded = 0
        while frontier:
            _, g, _, node = heapq.heappop(frontier)
            if self.goal_reached(node):
                return self._path(came_from, node), expanded
            expanded += 1
            for dr, dc, _ in self.available_actions(node):
                nb = (node[0] + dr, node[1] + dc)
                tentative = g + 1
                if nb not in best_g or tentative < best_g[nb]:
                    best_g[nb] = tentative
                    came_from[nb] = node
                    heapq.heappush(frontier,
                                   (tentative + self.heuristic(nb),
                                    tentative, next(counter), nb))
        return None, expanded

    @staticmethod
    def _path(came_from, node):
        path = []
        while node is not None:
            path.append(node)
            node = came_from[node]
        return path[::-1]

    def execute(self):
        """Run the agent: plan, then follow the plan to the goal."""
        path, expanded = self.plan()
        if path is None:
            return {"found": False, "path": None, "length": None,
                    "expanded": expanded}
        for node in path[1:]:
            self.state = node                 # apply each action
        return {"found": True, "path": path, "length": len(path) - 1,
                "expanded": expanded}


def main():
    print("=" * 70)
    print("AGENTS LAB - goal-based warehouse vehicle (A* planning)")
    print("Pranay Ghumatkar  |  2024A3PS0328G")
    print("=" * 70)

    env = Warehouse(WAREHOUSE)
    agent = GoalBasedAgent(env)

    print("\n--- Environment ---")
    print(f"    grid {len(env.grid)}x{len(env.grid[0])}   start {env.start}   goal {env.goal}")
    print(env.render())

    print("\n--- Goal-based agent run ---")
    result = agent.execute()
    print(f"    goal reached    : {agent.goal_reached(agent.state)}")
    print(f"    path found      : {result['found']}")
    print(f"    path length     : {result['length']}")
    print(f"    states expanded : {result['expanded']}")

    print("\n--- Solution path drawn on the map ---")
    print(env.render(result["path"]))

    print("\n--- Action sequence (first 12 moves) ---")
    names = []
    for a, b in zip(result["path"], result["path"][1:]):
        dr, dc = b[0] - a[0], b[1] - a[1]
        names.append({(-1, 0): "Up", (1, 0): "Down",
                      (0, -1): "Left", (0, 1): "Right"}[(dr, dc)])
    print("    " + " ".join(names[:12]) + " ...")


if __name__ == "__main__":
    main()
