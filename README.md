# Ariadne Neurosymbolic

Mathematical integration of trained neural networks with Ariadne Expression Set,
Symbolic, Function and validated Numeric/Set facilities. This repository complements
[Neural](https://github.com/ariadne-cps/neural), which retains ownership of
network topology, inference, differentiation, training and persistence.
Although independently maintained and usable, this repository is also part of the [Ariadne repository structure](https://github.com/ariadne-cps/ariadne/blob/main/doc/structure.md).

## Dependency architecture

```text
neurosymbolic
  +-- neural                (frozen snapshots, training and CPU evaluation)
  |     +-- threading
  +-- expression-set        (symbolic descriptions of sets)
        +-- symbolic        (expressions and differentiation)
        |     +-- function  (Taylor models and mathematical functions)
        +-- set             (validated constrained images)
              +-- linear-programming
              +-- constraint-solving
                    +-- nonlinear-programming
```

The checked-in Git submodules pin dependency commits and enforce shared
Configuration/Testing commits. No downstream Numeric/Function/Symbolic/Set
dependencies are added to Neural.

## Implemented operations

**Rigorous interval bound propagation** on finite closed boxes:

```cpp
NeurosymbolicValidatedEvaluator evaluator(snapshot);
FloatDPExactBox domain({
    FloatDPExactInterval(FloatDP(ExactDouble(-1.0),dp), FloatDP(ExactDouble(1.0),dp))
});
FloatDPUpperBox result = evaluator.bounds(domain);
```

The evaluator also accepts `FloatDPBoundsVector` and `FloatDPBallVector`,
both returning `FloatDPBoundsVector`. Ball radii are rigorously enclosed,
and all three APIs reuse the same outward-rounded layer evaluator:

```cpp
FloatDPBoundsVector intervals({FloatDPBounds(ExactDouble(-1.0),ExactDouble(1.0),dp)});
FloatDPBallVector balls({FloatDPBall(intervals[0])});
auto interval_outputs = evaluator.bounds(intervals);
auto ball_outputs = evaluator.bounds(balls);
```

The existing API accepts `FloatDPExactBox` and returns `FloatDPUpperBox`; both use the `Interval` and `Box` types
from Ariadne Interval/Geometry, without repository-specific interval or box
wrappers. Internally it uses Numeric's `FloatDPBounds` and invokes Numeric's
existing outward-rounded `tanh`, `sigmoid`, and `softplus` operations
directly (plus `max` for ReLU). The stored IEEE-754 binary64 weights and biases are converted
to *exact mathematical constants*, never reinterpreted as decimal source
literals. Intermediate infinities are rejected. The result is a
mathematical inclusion, not a guarantee of tightness or of the exact
floating-point execution order of Neural's approximate CPU engine.

**Validated Taylor-model propagation** preserves dependence on a shared
unit-box parameter domain. No neural-specific Taylor types are introduced:

```cpp
const Sweeper<FloatDP> swp=ThresholdSweeper<FloatDP>(dp,1e-12);
auto x=ValidatedTaylorModelDP::coordinate(1u,0u,swp);
auto outputs=evaluator.taylor(Vector<ValidatedTaylorModelDP>{x});
```

Identity, ReLU, Tanh, Sigmoid and Softplus are propagated using Function's
validated Taylor operations. ReLU uses the existing validated `max`
enclosure, rather than splitting into activation-constrained branches yet.
The parameter-domain variables are maintained through the network. The
input Taylor models must share the same parameter dimension; nonfinite
enclosures are rejected. Both C++ and Python expose Taylor-model propagation. The Python adapter
accepts a list of existing `ValidatedTaylorModelDP` objects rather than
introducing another vector type.

**Implicit constrained ReLU graph (default)** uses one
`ValidatedConstrainedImageSet`, without enumerating activation patterns:

```cpp
auto graph=evaluator.constrained_taylor(
    Vector<ValidatedTaylorModelDP>{x});
```

The extended parameter domain contains the original Taylor coordinates,
bounded uncertain neural inputs, and one auxiliary output for every ReLU.
Constraints encode the Taylor input remainder `-e <= x-p(s) <= e`
and ReLU exactly using `y>=0`, `y>=z` and `y*(y-z)==0`.
The output function projects the resulting graph to the neural outputs.
No solver is called automatically; the domain can be reduced later by
calling `graph.reduce()` if desired. A single constrained set does
not imply an easy convex feasibility problem: complementarity constraints
may still require substantial branching by the solver.

**Explicit activation branching (alternative)** is also available:

```cpp
auto pieces=evaluator.branched_taylor(
    Vector<ValidatedTaylorModelDP>{x},1024u);
```

It returns a list of constrained images, with a configurable bound on
the number of pieces. It preserves the existing Taylor uncertainty
semantics and may contain unpruned infeasible branches.

**Symbolic expansion** creates native Ariadne `Expression<Real>` trees:

```cpp
RealVariable x("x");
auto expressions = neurosymbolic_expand(snapshot, {x});
auto derivative = Ariadne::derivative(expressions.at(0), x);
```

Symbolic expansion supports Identity, ReLU, Tanh, Sigmoid and Softplus.
The two newer operations are provided directly by Symbolic as
`Expression<Real>` compositions of existing elementary operators, with
equivalent C++/Python tests in Symbolic. Numeric's existing correctly
rounded implementations are reused in the validated interval evaluator.
Analytic derivatives at activation kinks are not silently assumed;
expression DAGs may grow substantially. Symbolic network expansion is available in C++ and Python, reusing existing
Symbolic `RealVariable` and `RealExpression` bindings.

## Tutorials — C++ and Python

The matching tutorials in [`tutorial/`](tutorial/README.md) exercise the
main APIs: Box, Bounds, Ball, Taylor propagation, implicit and explicit
ReLU constraints, symbolic expansion and differentiation, and all five
activations. The standalone C++ and Python examples are also run in CI;
their scenarios are intentionally identical.

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

- The validated evaluator does not yet implement Jacobian intervals, constrained
  ReLU splitting, certificates, or proof search. Taylor propagation currently
  uses the existing natural Function calculus rather than adaptive branching.
- Expression Set provides symbolic set descriptions, and Set provides validated
  constrained images. Their solver dependencies are intended for optional
  refinement of neural enclosures, not for every interval evaluation.
- Floating-point sampling can falsify a bad enclosure but cannot prove
  mathematical inclusion. Correctness depends on the directed-rounding
  guarantees and supported elementary functions in Numeric.
- Python binding parity covers the implemented Neurosymbolic evaluation and
  expansion methods; no duplicate Numeric, Function, Set or Symbolic types
  are introduced.
- `reasoning` may consume this repository in future, but this module does
  not import the reasoning engine.

See [doc/neurosymbolic_design.md](doc/neurosymbolic_design.md) for the
implementation journal, numeric semantics and remaining acceptance criteria.

License: GPL-3.0-or-later.
