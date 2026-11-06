# Neural Set — design and verification journal

**Historical baseline:** `main@953d1a8464b2b09c2fa479b64e97317a25891e24` (2026-11-05).
**Current repository:** `ariadne-cps/neural-set`; see the 2026-11-06 migration entry.
This document separates the current mathematical contract, implementation,
known limitations, validation plan, and historical decisions. Source code and
tests remain authoritative; this is not a record of superseded CI failures.

## 1. Purpose and architectural boundaries

Neural Set is the validated mathematical adapter for immutable snapshots
from [ANN](https://github.com/ariadne-cps/ann). ANN owns training, native
CPU inference, network topology and serialization. Neural Set provides
rigorous bounds, validated Taylor evaluation and constrained image-set
representations. It does not symbolically expand networks.

**Direct production dependencies:** `ann` and `set`; Function, Numeric
and Geometry are obtained through Set. Build and Testing remain infrastructure.
ANN stays independent of Set, Function and Symbolic. The prior symbolic
expansion implementation and its tests were removed from this repository
in the 2026-11-06 rename; eventual ownership is in Reasoning, where
expansion must remain an optional proof-oriented operation, never the
ordinary evaluation path of System/Evolution.

**Invariants and decisions:**

- Reuse Ariadne Numeric, Function, Symbolic, and Set types. Do not implement
  duplicate balls, bounds, Taylor models, or constrained image sets.
- Never relabel approximate CPU arithmetic as validated. Validated paths
  must preserve outward enclosures and reject nonfinite intermediates.
- Constructing a constrained set must not invoke a solver automatically.
  Feasibility/refinement is a separate, optional operation.
- No dependencies on `evolution` or `reasoning`. Compose with those modules
  downstream using existing Function/Set operations.
- Preserve mathematical behaviour and test coverage rather than adjusting
  tests to force a green regression.

## 2. Mathematical semantics

The reference map is a dense feed-forward network whose layer activations
are **Identity, ReLU, Tanh, Sigmoid, and Softplus**. Stored IEEE-754 binary64
weights and biases denote their **exact dyadic real values**: the stored
binary64 approximation of a decimal literal is not the exact decimal rational.
This is real-function semantics, not bit-for-bit equivalence with Neural's
floating-point CPU execution order.

The three validated input families are finite closed intervals, balls,
and Taylor models. Natural interval arithmetic may lose dependencies;
Taylor arithmetic retains polynomial dependencies and validated remainder
bounds, but no path promises optimal tightness.

For `d` Taylor inputs on a common `n`-dimensional unit parameter box,
the intended enclosure is

```text
s in [-1,+1]^n
x_i(s) in p_i(s) + [-e_i,+e_i]     (i = 0,...,d-1)
```

Here `p_i` is a Taylor polynomial and `e_i` a nonnegative validated error
bound. The error interval is an **enclosure**, not a claim that all errors
are independently realizable. Replacing the true correlated error set by
independently bounded auxiliary inputs is an intentional outer
approximation. In particular, retaining inputs within their Taylor bounds
is not equivalent to proving a unique exact input value for every `s`.

Validated evaluation rejects incompatible dimensions and nonfinite
intermediate enclosures. No derivative is claimed at a ReLU kink.

## 3. Current public interfaces

| Operation | Input | Output | Meaning |
| --- | --- | --- | --- |
| `bounds` | `FloatDPExactBox` | `FloatDPUpperBox` | Validated interval enclosure |
| `bounds` | `FloatDPBoundsVector` | `FloatDPBoundsVector` | Validated interval enclosure |
| `bounds` | `FloatDPBallVector` | `FloatDPBoundsVector` | Rigorous ball-to-bounds, then interval propagation |
| `taylor` | `Vector<ValidatedTaylorModelDP>` | Same vector type | Direct validated Taylor propagation |
| `constrained_taylor` | Taylor-model vector | **One** `ValidatedConstrainedImageSet` | Default implicit constraint representation |
| `branched_taylor` | Taylor-model vector, branch cap | `List<ValidatedConstrainedImageSet>` | Optional explicit ReLU branches |

`Neural SetValidatedEvaluator` consumes an immutable
`NeuralParameterSnapshot`. The three `bounds` overloads,
`neural_validated_bounds`, direct Taylor propagation, both
constrained Taylor strategies have Python
bindings. Taylor arguments cross the Python boundary as lists of native
`ValidatedTaylorModelDP` objects; output vectors and branch lists are
returned as Python lists. Native Function and Set objects are
reused rather than re-declared by Neural Set.

### 3.1 Bounds and direct Taylor evaluation

All `bounds` overloads share a `FloatDPBounds` evaluator; a ball's full
radius is retained during conversion. Affine operations and the Numeric
implementations of `max`, `tanh`, `sigmoid`, and `softplus` are used
instead of local approximate approximations.

`taylor(input)` directly propagates existing validated Taylor models
through affine layers and all five activations. It uses Function's
validated `max` for ReLU; Sigmoid uses `rec(1+exp(-z))`, and Softplus
uses `log(1+exp(z))`. The input models must have a common parameter
dimension. Neither evaluator invokes a solver.

### 3.2 Implicit constraints — the default

`constrained_taylor(input)` returns **one**
`ValidatedConstrainedImageSet`, with no enumeration of activation
patterns. Its current parameter ordering is:

```text
(s_0,...,s_{n-1} | x_0,...,x_{d-1} | y_0,...,y_{r-1})
 original Taylor s    input auxiliaries     ReLU auxiliaries
```

Here `r` is the total number of ReLU neurons (not merely ambiguous
ones). The construction has `n+d+r` parameters. The original parameters
range over the unit box; input and ReLU auxiliary domains use validated
outward interval bounds. Every input adds the constraint

```text
-e_i <= x_i - p_i(s) <= e_i.
```

Each ReLU with preactivation `z` and auxiliary output `y` adds

```text
y >= 0
y >= z
y * (y - z) == 0
```

At the mathematical real-algebraic level, these three conditions encode
`y=max(0,z)` with no explicit disjunction. Smooth activations are direct
functions of the same parameters. The current implementation adds
`d + 3*r` constraints (one bounded input relation per input, three per
ReLU), without calling `ConstraintSolver`.

**Output image versus joint input/output graph.** The returned set's
`function()` maps hidden parameters to *neural outputs only*; the
`domain()` and `constraints()` preserve the hidden inputs. The joint
image `(x,N(x))` can be constructed by **existing Function/Set
composition**, not a new Neural Set wrapper:

```cpp
auto output_set = evaluator.constrained_taylor(inputs);
SizeType n = inputs[0].argument_size();
SizeType d = inputs.size();
SizeType m = output_set.number_of_parameters();

ValidatedVectorMultivariateFunction input_projection(d, m);
for (SizeType i = 0; i < d; ++i)
    input_projection.set(i,
        ValidatedScalarMultivariateFunction::coordinate(m, n + i));

ValidatedConstrainedImageSet joint(
    output_set.domain(),
    join(input_projection, output_set.function()),
    output_set.constraints());
```

This construction retains **the same** parameterization and all
constraints. `join` here concatenates **functions** sharing the same
arguments. It is *not* `join` of constrained image sets: the latter has
different preconditions, including no constraints on its input sets.
`image(set,h)` and `set.apply(h)` post-compose the existing output
mapping; they cannot retrieve hidden input coordinates from that mapping
alone. The parameter order shown is the current internal convention:
cross-repository consumers relying on it need explicit regression tests.

### 3.3 Explicit branches — optional

`branched_taylor(input,max_branches)` maintains Taylor models on the
original parameter box. A ReLU with a certified sign does not split; an
ambiguous ReLU creates active and inactive branches. Exceeding the
configured cap throws `std::length_error`, not a silent union/merge.

For a preactivation `z(s) in p(s)+[-e,e]`, the implemented **necessary**
guards are

```text
active:   p(s) >= -e     -> propagate the preactivation Taylor model
inactive: p(s) <= +e     -> propagate zero
```

They account for the Taylor remainder but can overlap, retain infeasible
activation patterns, and permit spurious output values on a branch.
Especially for `e>0`, they are **not** equivalent to the exact ReLU
complementarity relation of the implicit model. The result is a sound
outer representation intended for comparison and optional refinement;
it may grow exponentially. Neither mode invokes a solver automatically.
Separate method names are intentional because their existing result
types differ; no new variant or wrapper is needed.

### 3.4 Solver semantics

A `ValidatedConstrainedImageSet` is defined without solving its
constraints. In Set, `reduce()` invokes `ConstraintSolver` and tries
to shrink `reduced_domain()`; `domain()`, `function()`, and the
stored constraints remain unchanged.

**Caution:** `is_empty()` currently calls `reduce()` internally
(and does so through a `const` interface). Thus it can invoke the
solver and change the reduced domain; it is not a cheap read-only
query. Solver results may be inconclusive on nonconvex complementarity
constraints. Reduction is not a general feasibility/completeness
guarantee and must not be claimed to remove all infeasible branches.

## 4. Verification: evidence versus gaps

**Reference baseline:** all five workflows — Ubuntu, macOS, Windows,
Sanitizers, and Coverage — passed for `953d1a8`. This validates the
current regression suite in its CI environments, **not** the full
mathematical soundness/completeness of the model.

Existing regression suites exercise:

- Bounds and Ball paths, affine examples, invalid inputs, and all
  activations (`tests/test_validated.cpp` and Python tests).
- Direct Taylor affine dependency, derivatives, remainder enclosure,
  and all activations (`tests/test_taylor.cpp`).
- Implicit graph construction for small ReLU cases, the kink, nonzero
  Taylor remainder, and a two-ReLU example.
- Explicit branch coverage, stable signs, remainder-relaxed guards,
  and the branch cap.

**Open verification gaps:**

1. Negative semantic tests that exclude impossible input/activation/output
   assignments, beyond checking that known valid witnesses satisfy the
   constraints. Include deeper networks and correlated inputs.
2. Feasible, infeasible, and undecidable/inconclusive exercises of
   `reduce()` and `is_empty()`, with checks that no admissible witness
   is excluded by refinement.
3. Independent comparisons of implicit and explicit modes, accounting
   for their **different** set semantics when Taylor remainders are
   nonzero. Do not assert equality merely because sampled outputs match.
4. Reproducible performance/memory benchmarks for construction and
   optional solver reduction as depth, width, ReLU ambiguity, and Taylor
   remainder size change.
5. A downstream end-to-end controlled-dynamics example preserving the
   coupled state/control parameterization. This is not yet a
   Neural Set feature and must not force a dependency on Evolution.

## 5. Next work and acceptance gate

The next phase is **validation and benchmarking**, not adding new
Neural Set abstractions.

1. **Semantic oracles:** small deterministic affine/one-ReLU/multilayer
   networks, including boundary cases and nonzero Taylor remainders.
   Verify both valid witnesses and provably forbidden assignments,
   preferably against independent analytic or dyadic references.
   Numerical sampling alone cannot certify global inclusion.
2. **Solver characterization:** test `reduce()` and `is_empty()` on
   known satisfiable, unsatisfiable, and inconclusive cases. Record
   reductions, solver time, and any inability to decide; preserve
   admissible cases and investigate every discrepancy.
3. **Strategy benchmarks:** compare implicit versus explicit results
   under their actual semantics, with timing, memory, parameter count,
   constraint count, branch count, and bounding quality. Include
   networks in which many ReLUs have uncertain signs.
4. **Downstream composition after validation:** use Function's
   function-level `join` and Set's constrained-image constructors to
   build the coupled graph for a minimal controlled dynamical system.
   Integrate with `evolution` only after the relation has been
   independently validated; do not introduce composition into
   Neural Set without a demonstrated missing primitive.

**Exit criteria:** strong semantic inclusion/exclusion checks,
documented solver behaviour (including inconclusive outcomes), and
reproducible comparative benchmarks. Green CI or coverage alone is
insufficient.

## 6. Implementation history (condensed)

| Date | Milestone | Current significance |
| --- | --- | --- |
| 2026-11-04 | Standalone project, Numeric bounds and Symbolic expansion | Frozen ANN inputs, binary64-exact parameters, five activations |
| 2026-11-04 | Dependency consolidation | `set` provides Symbolic/Function and Set/solver facilities |
| 2026-11-04 | Ball/Bounds support and regression corrections | Reuse of native Numeric types; no duplicate interval wrappers |
| 2026-11-05 | Direct validated Taylor propagation | Native `taylor()` with shared parameters and validated remainders |
| 2026-11-05 | Explicit ReLU branching | `branched_taylor()` retained as a bounded, potentially loose alternative |
| 2026-11-05 | Implicit ReLU constraints as default | `constrained_taylor()` returns one constrained image with complementarity |
| 2026-11-05 | Baseline established | All five GitHub Actions workflows passed at `953d1a8` |
| 2026-11-05 | Matched standalone C++/Python tutorials | Expanded the same seven use cases in both languages and exposed missing Python adapter methods and symbolic expansion. Tutorials remain separate from CTest, pytest and coverage; mathematical engine unchanged. |

Superseded APIs, test repairs, and intermediate failed CI runs are
preserved in Git history, not repeated as active design requirements.

**Current source index:** `include/neural-set/validated.hpp`,
`include/neural-set/neural_set.hpp`, `src/validated.cpp`,
`src/taylor.cpp`, `tests/test_validated.cpp`,
`tests/test_taylor.cpp`, and Set's `include/set/function_set.hpp`.

**2026-11-06 rename and dependency migration:** Moved repository from
`ariadne-cps/neurosymbolic` to `ariadne-cps/neural-set`. Renamed its
public C++ and Python module interfaces and test/tutorial entry points.
Direct dependencies changed from `neural` and `expression-set` to
`ann` and `set`, preserving the previously pinned Set revision.
Removed the symbolic-expansion implementation, public header, C++ test
and Python binding from Neural Set. The earlier implementation can be
recovered from Git history for a separate move into Reasoning.
Validated bounds and Taylor/image-set algorithms were not intentionally
changed. Build, standalone install, C++/Python tests and CI results
must be checked after this commit.
