# AI Laboratory — Neural Models

**Student:** Pranay Ghumatkar | **Roll No:** 2024A3PS0328G
**Worksheet:** [neur_models_lab_ex.pdf](neur_models_lab_ex.pdf)

The worksheet asks to submit **one notebook or Python file plus a short report**,
so everything is in a single program. It builds a 2→2→1 PyTorch network for XOR
and shows why a **nonlinear hidden layer** is necessary, how the **backward pass**
is the chain rule, why **symmetric initialisation** stalls learning, and how the
**output layer must match the task**.

## Files

| File | Purpose |
|---|---|
| `neural_models_lab.py` | XOR + gradient inspection, symmetry, activation comparison, three-class softmax, finite-difference check (single program) |
| `ANSWERS.md` | Tasks 1–5, Think-About-Its and Reflection Questions 1–7 |
| `results.txt` | Program output |
| `neur_models_lab_ex.pdf` | Worksheet |

## How to run

```bash
pip install torch        # CPU-only is enough
python neural_models_lab.py
```

Developed with Python 3.14 and `torch 2.9.1+cpu`.

See [ANSWERS.md](ANSWERS.md) for the full answers.
