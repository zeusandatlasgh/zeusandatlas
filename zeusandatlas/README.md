# zeusandatlas

A small Python package built for CPE 486/586 Homework 0 (Fall 2026, UAH).

It provides a `differential` subpackage with a `discrete` module implementing
`diff(t, x)`, which computes the discrete derivative of a timeseries using:

v(t_k) = (x(t_k) - x(t_k-1)) / (t_k - t_k-1)

## Installation

```bash
uv add zeusandatlas
```

## Usage

```python
from zeusandatlas.differential.discrete import diff

t = [0, 0.1, 0.3, 0.4, 0.55, 0.67, 0.71]
x = [23.1, 22.5, 23.5, 21.88, 22.5, 23.5, 24.88]

v = diff(t, x)
print(v)
```
