# AI Laboratory — Search (A* and BFS)

**Student:** Pranay Ghumatkar | **Roll No:** 2024A3PS0328G
**Worksheet:** [search_lab_ex.pdf](search_lab_ex.pdf)

Warehouse robot navigation as a search problem `P = (S, A, T, s0, G, c)`, with
**A\*** (`f(n) = g(n) + h(n)`, Manhattan) and **BFS** on an ASCII grid, plus a
heuristic investigation.

## Files

| File | Purpose |
|---|---|
| `search_agent.py` | Grid parsing, A*, BFS and the heuristic variants (the program) |
| `ANSWERS.md` | Problem formulation, design, prompt, test results, BFS/A* comparison, heuristic study and reflection |
| `results.txt` | Program output |
| `search_lab_ex.pdf` | Worksheet |

## How to run

```bash
python search_agent.py     # prints results.txt
```

Requires only the Python standard library.

See [ANSWERS.md](ANSWERS.md) for the full answers.
