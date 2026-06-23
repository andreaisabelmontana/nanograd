# nanograd

[![CI](https://github.com/andreaisabelmontana/nanograd/actions/workflows/ci.yml/badge.svg)](https://github.com/andreaisabelmontana/nanograd/actions/workflows/ci.yml)
![license: MIT](https://img.shields.io/badge/license-MIT-blue)
![c++](https://img.shields.io/badge/C%2B%2B-17-00599C)

A reverse-mode automatic-differentiation engine and a small neural network — written from scratch in ~200 lines of header-only C++17, no dependencies. It's the machinery behind PyTorch's `loss.backward()`, reduced to its essence so you can read the whole thing.

A `Value` is a node in a computation graph; every `+`, `*`, `tanh` records how to send gradients back to its inputs; `backward()` walks the graph in reverse-topological order and fills in every derivative. On top of that sits an MLP that learns XOR with plain SGD.

## Build & run

```bash
cmake -B build && cmake --build build
./build/nanograd          # trains XOR, prints the loss curve and learned outputs
ctest --test-dir build    # runs the tests
```

Or without CMake:

```bash
c++ -std=c++17 -O2 -Iinclude src/main.cpp -o nanograd && ./nanograd
```

Example output:

```
epoch   0   loss 4.5xxxxx
epoch 250   loss 0.0xxxxx
epoch 500   loss 0.00xxxx

learned XOR:
  [0, 0] -> -0.97   (target -1)
  [0, 1] ->  0.96   (target  1)
  [1, 0] ->  0.96   (target  1)
  [1, 1] -> -0.95   (target -1)
```

## How it works

```cpp
Value a = 2.0, b = -3.0;
Value loss = (a * b + a).tanh();
loss.backward();          // fills a.grad() and b.grad()
```

- **`value.hpp`** — the engine. Each operation creates a new graph node holding its result and a closure that pushes the output's gradient to its parents (the chain rule, locally). `backward()` does one topological sort, seeds the output grad with 1, and applies those closures in reverse. Repeated use of the same `Value` accumulates gradients correctly.
- **`mlp.hpp`** — `Neuron` → `Layer` → `MLP`, all expressed in `Value`s. Parameters are persistent leaf nodes; each forward pass builds a fresh graph over them, so a training step is: forward → `backward()` → nudge each parameter by `-lr * grad`.

## Tests

`tests/test_nanograd.cpp` checks three things, gated in CI:

1. **Forward composition** evaluates to the expected value.
2. **Gradients match finite differences** — the analytic grads from `backward()` agree with numerical `(f(x+h) - f(x-h)) / 2h` to 1e-4.
3. **XOR converges** — training drives the loss below 0.1 and all four predictions get the right sign.

## Structure

```
include/nanograd/
  value.hpp   reverse-mode autograd engine (Value, Node, backward)
  mlp.hpp     Neuron / Layer / MLP on top of Value
src/main.cpp  XOR training demo
tests/        finite-difference grad checks + XOR convergence
```

## License

MIT — see [LICENSE](LICENSE).
