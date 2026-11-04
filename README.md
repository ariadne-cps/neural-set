# Ariadne Neurosymbolic

Mathematical integration of trained neural networks with Ariadne Symbolic,
Function and validated Numeric facilities. This repository complements
[Neural](https://github.com/ariadne-cps/neural), which retains ownership of
network topology, inference, differentiation, training and persistence.

## Dependency architecture

```text
neurosymbolic
  +-- neural                (frozen snapshots, training and CPU evaluation)
  |     +-- threading
  +-- symbolic              (expression DAGs and differentiation)
        +-- function        (mathematical function abstractions)
              +-- geometry -> algebra -> interval -> numeric
```

The checked-in Git submodules pin dependency commits and enforce shared
Configuration/Testing commits. No downstream Numeric/Function/Symbolic
dependencies are added to Neural.

## Implemented operations

**Rigorous interval bound propagation** on finite closed boxes:

```cpp
NeurosymbolicValidatedEvaluator evaluator(snapshot);
auto result = evaluator.bounds({{-1.0, 1.0}});
```

Uses Ariadne `FloatDPBounds` with outward-rounded affine operations and
validated activation enclosures for identity, ReLU, tanh, sigmoid and
softplus. The stored IEEE-754 binary64 weights and biases are converted
to *exact mathematical constants*, never reinterpreted as decimal source
literals. Intermediate infinities are rejected. The result is a
mathematical inclusion, not a guarantee of tightness or of the exact
floating-point execution order of Neural's approximate CPU engine.

**Symbolic expansion** creates native Ariadne `Expression<Real>` trees:

```cpp
RealVariable x("x");
auto expressions = neurosymbolic_expand(snapshot, {x});
auto derivative = Ariadne::derivative(expressions.at(0), x);
```

Supports every current dense activation. ReLU becomes `max(0,x)`,
sigmoid becomes `1/(1+exp(-x))`, and softplus becomes `log(1+exp(x))`.
Analytic derivatives at activation kinks are not silently assumed;
expression DAGs may grow substantially. Symbolic expansion is currently a
C++ API; Python initially exposes the validated evaluator and existing
Symbolic dependency types.

## Build and tests

```sh
git clone https://github.com/ariadne-cps/neurosymbolic.git
cd neurosymbolic
git submodule update --init --recursive
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug \
    -DBUILD_PYTHON_BINDINGS=ON -DWARNINGS_AS_ERRORS=ON
cmake --build build --target tests
ctest --test-dir build --output-on-failure
```

The five CI workflows follow the same Ubuntu, macOS, Windows, sanitizer and
Codecov patterns used by Neural. The Codecov workflow checks
`secrets.CODECOV_TOKEN` on macOS before attempting its upload; this secret
must be configured for the new repository.

## Scope and limitations

- The validated evaluator does not implement Jacobian intervals, affine or
  Taylor enclosures, certificates, or proof search.
- Floating-point sampling can falsify a bad enclosure but cannot prove
  mathematical inclusion. Correctness depends on the directed-rounding
  guarantees and supported elementary functions in Numeric.
- Python symbolic expression expansion and full Function object adapters
  remain separate roadmap increments; they are not claimed implemented.
- `reasoning` may consume this repository in future, but this module does
  not import the reasoning engine.

See [doc/neurosymbolic_design.md](doc/neurosymbolic_design.md) for the
implementation journal, numeric semantics and remaining acceptance criteria.

License: GPL-3.0-or-later.
