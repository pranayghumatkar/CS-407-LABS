# AI Laboratory — Logical Planning

**Student:** Pranay Ghumatkar | **Roll No:** 2024A3PS0328G
**Worksheet:** [logic_lab_ex.pdf](logic_lab_ex.pdf)

*Logic + Search = Planning.* A state is a set of logical propositions; each
action has positive/negative preconditions and effects; an action is applicable
iff `S ⊨ Preconditions(a)`, and applying it gives `S' = Apply(S,a)`. Breadth-first
search finds a plan. `planner.pl` is an independent Prolog verifier.

## Files

| File | Purpose |
|---|---|
| `planner.py` | Action model, BFS planner and independent plan replay (the program) |
| `planner.pl` | Prolog knowledge base used to verify proposed moves (optional extension) |
| `ANSWERS.md` | Specification, manual plan, prompt, test results, answers and reflection |
| `results.txt` | Program output |
| `logic_lab_ex.pdf` | Worksheet |

## How to run

```bash
python planner.py          # prints results.txt
```

Requires only the Python standard library. SWI-Prolog is optional
(`swipl planner.pl`).

See [ANSWERS.md](ANSWERS.md) for the full answers.
