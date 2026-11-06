# Ariadne Neural Set

[![macOS Status](https://github.com/ariadne-cps/neural-set/actions/workflows/macos.yml/badge.svg)](https://github.com/ariadne-cps/neural-set/actions/workflows/macos.yml)
[![Ubuntu Status](https://github.com/ariadne-cps/neural-set/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/ariadne-cps/neural-set/actions/workflows/ubuntu.yml)
[![Windows Status](https://github.com/ariadne-cps/neural-set/actions/workflows/win.yml/badge.svg)](https://github.com/ariadne-cps/neural-set/actions/workflows/win.yml)
[![Sanitizers Status](https://github.com/ariadne-cps/neural-set/actions/workflows/sanitizers.yml/badge.svg)](https://github.com/ariadne-cps/neural-set/actions/workflows/sanitizers.yml)
[![Coverage Status](https://github.com/ariadne-cps/neural-set/actions/workflows/coverage.yml/badge.svg)](https://github.com/ariadne-cps/neural-set/actions/workflows/coverage.yml)
[![codecov](https://codecov.io/gh/ariadne-cps/neural-set/branch/main/graph/badge.svg)](https://codecov.io/gh/ariadne-cps/neural-set)

Ariadne Neural Set provides rigorous bounds, validated Taylor-model evaluation
and constrained image-set representations for immutable ANN snapshots.

## Dependencies

```text
neural-set
  +-- ann           (network snapshots and native evaluation)
  |    +-- threading
  +-- set           (validated constrained image sets)
       +-- linear-programming
       +-- constraint-solving
            +-- nonlinear-programming ... function, numeric
```

The direct production dependencies are `ann` and `set`. Shared Build and
Testing submodule revisions are checked by CMake.

## Validated operations

`NeuralValidatedEvaluator` owns an immutable `NeuralParameterSnapshot`.
It supports these operations, using the existing Ariadne types:

- `bounds(FloatDPExactBox)` -> `FloatDPUpperBox`.
- `bounds(FloatDPBoundsVector)` and `bounds(FloatDPBallVector)` ->
  `FloatDPBoundsVector` (outward-rounded bounds, including ball radii).
- `taylor(Vector<ValidatedTaylorModelDP>)` -> the same vector type,
  preserving shared parameter dependence.
- `constrained_taylor(...)` -> one `ValidatedConstrainedImageSet`, with
  Taylor input remainders and an implicit ReLU complementarity graph.
- `branched_taylor(..., max_branches)` -> `List<ValidatedConstrainedImageSet>`,
  an optional explicit ReLU-activation partition.
- `neural_validated_bounds(snapshot, input_box)` is a convenience wrapper.

All five ANN activations (Identity, ReLU, Tanh, Sigmoid, Softplus) are
supported. Frozen binary64 parameters are interpreted as exact dyadic
real coefficients by validated operations; results are mathematical
inclusions, not bitwise representations of approximate CPU inference.
Constructing a constrained image set does not invoke a solver; optional
refinement via `reduce()` may be expensive.

```cpp
NeuralValidatedEvaluator evaluator(snapshot);
auto outputs = evaluator.bounds(input_box);
auto direct = evaluator.taylor(taylor_inputs);
auto implicit_relu = evaluator.constrained_taylor(taylor_inputs);
auto branches = evaluator.branched_taylor(taylor_inputs, 1024u);
```

## Build and test

```sh
git clone --recurse-submodules https://github.com/ariadne-cps/neural-set.git
cd neural-set
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug \\
  -DBUILD_PYTHON_BINDINGS=ON -DWARNINGS_AS_ERRORS=ON
cmake --build build --target tests
ctest --test-dir build --output-on-failure
```

The C++ umbrella header is `ariadne-neural-set.hpp`; the installed library
is `ariadne-neural-set`, the interface target is `neural_set`, and the
standalone Python module is `pyariadne_neural_set`.
Standalone C++ and Python examples are in [tutorial/](tutorial/README.md).
See the [design journal](doc/neural_set_design.md) for mathematical contracts,
implementation details and the migration history.
