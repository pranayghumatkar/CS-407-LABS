"""
CS-407 AI Laboratory - Search
Search = (S, A, T, s0, G, c)   with   f(n) = g(n) + h(n)

A goal-based warehouse robot agent.  Implements A* and BFS on an ASCII grid,
with a heuristic investigation (Task 6).

Student : Pranay Ghumatkar
Roll No : 2024A3PS0328G
"""

import heapq
import itertools
import math
from collections import deque

# ---------------------------------------------------------------------------
# Maps (Task 4)
# ---------------------------------------------------------------------------
WAREHOUSE = """\
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.........#.#
#.###########.#.#
#.............#G#
#################"""

TRIVIAL = """\
#####
#SG##
#####"""

NO_SOLUTION = """\
#######
#S....#
###.###
#...#G#
#######"""

ALTERNATIVE_PATHS = """\
#####
#S..#
#.#.#
#..G#
#####"""

OPEN_GRID = """\
################
#S............G#
#..............#
#..............#
#..............#
################"""


def parse_map(text):
    grid = [row for row in text.splitlines() if row]
    start = goal = None
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == "S":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)
    if start is None or goal is None:
        raise ValueError("map must contain S and G")
    return grid, start, goal


def free(grid, node):
    r, c = node
    return 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] != "#"


def neighbors(grid, node):
    r, c = node
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nb = (r + dr, c + dc)
        if free(grid, nb):
            yield nb


# ---------------------------------------------------------------------------
# Heuristics (Task 6)
# ---------------------------------------------------------------------------
def manhattan(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])


def zero(node, goal):
    return 0


def euclidean(node, goal):
    return math.hypot(node[0] - goal[0], node[1] - goal[1])


def manhattan_x2(node, goal):
    return 2 * manhattan(node, goal)


HEURISTICS = {
    "manhattan": manhattan,
    "zero (h=0)": zero,
    "euclidean": euclidean,
    "manhattan x2": manhattan_x2,
}


def reconstruct(came_from, node):
    path = []
    while node is not None:
        path.append(node)
        node = came_from[node]
    return path[::-1]


def render(grid, path=None):
    """Show the grid with the solution path drawn as '*'."""
    path = set(path or [])
    out = []
    for r, row in enumerate(grid):
        line = []
        for c, ch in enumerate(row):
            if (r, c) in path and ch == ".":
                line.append("*")
            else:
                line.append(ch)
        out.append("".join(line))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# A* and BFS (Tasks 2 & 5)
# ---------------------------------------------------------------------------
def astar(grid, start, goal, heuristic=manhattan):
    """A*: priority is f(n) = g(n) + h(n)."""
    counter = itertools.count()
    frontier = [(heuristic(start, goal), 0, next(counter), start)]
    came_from = {start: None}
    best_g = {start: 0}
    expanded = 0
    while frontier:
        _, g, _, node = heapq.heappop(frontier)
        if node == goal:
            path = reconstruct(came_from, node)
            return {"found": True, "path": path, "length": len(path) - 1,
                    "expanded": expanded, "cost": g}
        expanded += 1
        for nb in neighbors(grid, node):
            tentative = g + 1                       # every move costs 1
            if nb not in best_g or tentative < best_g[nb]:
                best_g[nb] = tentative
                came_from[nb] = node
                heapq.heappush(frontier,
                               (tentative + heuristic(nb, goal),
                                tentative, next(counter), nb))
    return {"found": False, "path": None, "length": None,
            "expanded": expanded, "cost": None}


def bfs(grid, start, goal):
    """Blind breadth-first search (uniform cost 1)."""
    frontier = deque([start])
    came_from = {start: None}
    expanded = 0
    while frontier:
        node = frontier.popleft()
        if node == goal:
            path = reconstruct(came_from, node)
            return {"found": True, "path": path, "length": len(path) - 1,
                    "expanded": expanded, "cost": len(path) - 1}
        expanded += 1
        for nb in neighbors(grid, node):
            if nb not in came_from:
                came_from[nb] = node
                frontier.append(nb)
    return {"found": False, "path": None, "length": None,
            "expanded": expanded, "cost": None}


def main():
    print("=" * 70)
    print("SEARCH LAB - warehouse robot  (A* and BFS)")
    print("Pranay Ghumatkar  |  2024A3PS0328G")
    print("=" * 70)

    grid, start, goal = parse_map(WAREHOUSE)
    print(f"\n    start {start}   goal {goal}   grid {len(grid)}x{len(grid[0])}")

    print("\n--- Test 1: original warehouse (A*, Manhattan) ---")
    res = astar(grid, start, goal, manhattan)
    print(f"    path found      : {res['found']}")
    print(f"    path length     : {res['length']}")
    print(f"    states expanded : {res['expanded']}")
    print(render(grid, res["path"]))

    print("\n--- Test 2: trivial case (goal adjacent to start) ---")
    g2, s2, t2 = parse_map(TRIVIAL)
    r2 = astar(g2, s2, t2, manhattan)
    print(f"    path found      : {r2['found']}   length {r2['length']}   expanded {r2['expanded']}")

    print("\n--- Test 3: no solution (goal enclosed by walls) ---")
    g3, s3, t3 = parse_map(NO_SOLUTION)
    r3 = astar(g3, s3, t3, manhattan)
    b3 = bfs(g3, s3, t3)
    print(f"    A*  path found  : {r3['found']}  (expected False, no infinite loop)")
    print(f"    BFS path found  : {b3['found']}")

    print("\n--- Test 4: alternative paths (multiple shortest paths) ---")
    g4, s4, t4 = parse_map(ALTERNATIVE_PATHS)
    r4 = astar(g4, s4, t4, manhattan)
    b4 = bfs(g4, s4, t4)
    print(f"    A*  length {r4['length']}   BFS length {b4['length']}")
    print(f"    both shortest   : {r4['length'] == b4['length']}")

    bres = bfs(grid, start, goal)
    print("\n--- Task 5: BFS vs A* on the warehouse ---")
    print(f"    {'measure':<20}{'BFS':>8}{'A* (manhattan)':>18}")
    print("    " + "-" * 44)
    print(f"    {'solution found':<20}{str(bres['found']):>8}{str(res['found']):>18}")
    print(f"    {'path length':<20}{bres['length']:>8}{res['length']:>18}")
    print(f"    {'states expanded':<20}{bres['expanded']:>8}{res['expanded']:>18}")
    print(
        "    Both find a shortest path, and here both expand the same number of\n"
        "    states: the warehouse is essentially a single winding corridor, so\n"
        "    there is little branching for the heuristic to prune.  A*'s advantage\n"
        "    appears when the map has many alternative branches."
    )

    print("\n--- Task 5b: A* vs BFS on an open map (many branches) ---")
    go, so, to = parse_map(OPEN_GRID)
    ro = astar(go, so, to, manhattan)
    bo = bfs(go, so, to)
    print(f"    {'measure':<20}{'BFS':>8}{'A* (manhattan)':>18}")
    print("    " + "-" * 44)
    print(f"    {'path length':<20}{bo['length']:>8}{ro['length']:>18}")
    print(f"    {'states expanded':<20}{bo['expanded']:>8}{ro['expanded']:>18}")
    print("    Same-length path, but A* expands far fewer states here.")

    print("\n--- Task 6: heuristic investigation on the warehouse ---")
    print(f"    {'heuristic':<18}{'found':>8}{'length':>10}{'expanded':>12}")
    for name, h in HEURISTICS.items():
        r = astar(grid, start, goal, h)
        print(f"    {name:<18}{str(r['found']):>8}{r['length']:>10}{r['expanded']:>12}")
    print(
        "    h=0 removes the heuristic (A* degrades to uniform-cost search);\n"
        "    euclidean is a weaker admissible lower bound; manhattan x2 is\n"
        "    inadmissible (too aggressive) and expands more states."
    )


if __name__ == "__main__":
    main()
