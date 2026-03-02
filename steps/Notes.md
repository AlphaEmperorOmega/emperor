# Notes

---

## Init (step 0)

```python
log_never_halt: [0.0]
# shape: (1, 1)
# log(1.0) = certain not halted yet
accumulated_hidden: [0.0, 0.0]
# shape: (1, 1, 2)
```

---

## Step 1 — gate sees first output, produces

`log P(continue)=log(0.7)=-0.36, log P(halt)=log(0.3)=-1.20`

```python
# log_never_halt[..., None]: [[0.0]] shape: (1, 1, 1)   →  prob: [[1.00]]
# current_log_gates: [[-0.36, -1.20]] shape: (1, 1, 2)  →  prob: [[0.70, 0.30]]

log_halt = [[0.0]] + [[-0.36, -1.20]] = [[-0.36, -1.20]]
# shape: (1, 1, 2)
log_never_halt = log_halt[..., 0] = [[-0.36]]
# shape: (1, 1)
# log(0.7)
halting_probability = log_halt[..., 1].exp() = [[0.30]]
# shape: (1, 1)
# P(halt at step 1) = 1.0 × 0.3
```

---

## Step 2 — gate sees second output, produces

`log P(continue)=log(0.6)=-0.51, log P(halt)=log(0.4)=-0.92`

```python
# log_never_halt[..., None]: [[-0.36]] shape: (1, 1, 1)  →  prob: [[0.70]]
# current_log_gates: [[-0.51, -0.92]] shape: (1, 1, 2)  →  prob: [[0.60, 0.40]]

log_halt = [[-0.36]] + [[-0.51, -0.92]] = [[-0.87, -1.28]]
# shape: (1, 1, 2)
log_never_halt = log_halt[..., 0] = [[-0.87]]
# shape: (1, 1)
# log(0.7 × 0.6) = log(0.42)
halting_probability = log_halt[..., 1].exp() = [[0.28]]
# shape: (1, 1)
# P(halt at step 2) = 0.7 × 0.4
```

---

## Step 3 — gate sees third output, produces

`log P(continue)=log(0.4)=-0.92, log P(halt)=log(0.6)=-0.51`

```python
# log_never_halt[..., None]: [[-0.87]] shape: (1, 1, 1)  →  prob: [[0.42]]
# current_log_gates: [[-0.92, -0.51]] shape: (1, 1, 2)   →  prob: [[0.40, 0.60]]

log_halt = [[-0.87]] + [[-0.92, -0.51]] = [[-1.79, -1.38]]
# shape: (1, 1, 2)
log_never_halt = log_halt[..., 0] = [[-1.79]]
# shape: (1, 1)
# log(0.42 × 0.4) = log(0.17)
halting_probability = log_halt[..., 1].exp() = [[0.25]]
# shape: (1, 1)
# P(halt at step 3) = 0.42 × 0.6
```
