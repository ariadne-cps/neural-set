# Ariadne Neurosymbolic design and implementation journal

## Architecture and responsibilities

This repository owns **mathematical adapters around frozen Neural models**.
It depends directly on Neural and Expression Set. Expression Set includes
Symbolic (and Function) plus Set, which includes Linear Programming and
Constraint Solving (and transitively Nonlinear Programming). Neural itself must
remain independent of these mathematical components, retaining portable CPU
inference, training,
serializable binary64 snapshots and its own native backend selection.

The explicit backend boundary means the reserved `AriadneValidated` enum
in Neural is **not** activated there. Validated evaluation is a separate
`NeurosymbolicValidatedEvaluator` in this repository. No approximate fallback
is permitted or relabeled as validated.

## Canonical mathematical semantics

Inputs are vectors of closed finite binary64 endpoint intervals. Parameters
and biases from immutable Neural snapshots represent **exact** dyadic real
numbers identified by their IEEE-754 binary64 bit patterns. A coefficient
such as `0.1` represents the binary64 value, not exact decimal 1/10.

The reference mathematical map is a composition of dense affine layers
followed by identity, ReLU, tanh, sigmoid, or softplus activations.
Interval bound propagation uses Numeric's `FloatDPBounds` for all affine
terms and monotone validated elementary activations. Native `FloatDPBoundsVector`
and `FloatDPBallVector` inputs return `FloatDPBoundsVector`. A ball's complete
radius is enclosed through Numeric's directed-rounded conversion before
propagation. The existing `FloatDPExactBox` input and `FloatDPUpperBox`
output remain available, which
are composed from Ariadne's `Interval` and `Box` abstractions.
There is no custom interval or box. Existing `FloatDPBounds` operations
perform the arithmetic and Numeric's `tanh`, `sigmoid` and `softplus`
are invoked directly. Inputs are finite and ordered; operations abort on any nonfinite intermediate
bound, rather than advertising an invalid or incomplete certificate.

The approach is *sound but potentially loose*: dependency is forgotten
after every natural interval operation. A network can have correlated
intermediate neuron values, making output enclosures wider than necessary.
This first milestone does not promise the narrowest possible interval.

## Validated Taylor propagation (N3 initial)

`NeurosymbolicValidatedEvaluator::taylor(Vector<ValidatedTaylorModelDP>)`
propagates an existing vector of validated Taylor models directly through
the frozen network layers. All inputs must have the same number of independent
unit-box parameters. The affine operations use exact binary64 weights as
singleton `FloatDPBounds`, and the existing Function operations implement
Identity, ReLU (validated `max`), Tanh, Sigmoid
(`rec(1+exp(-x))`), and Softplus (`log(1+exp(x))`).
This maintains polynomial correlations and propagates each model's remainder
enclosure. It rejects nonfinite intermediate enclosures. It does not
construct a branch union or claim tightness at ReLU boundaries.
Python binding remains deferred until Function exposes native vectors
of validated Taylor models.

## Branch-constrained ReLU Taylor image (N3 second increment)

`NeurosymbolicValidatedEvaluator::constrained_taylor(input, max_branches)`
returns a `List<ValidatedConstrainedImageSet>`, each representing a validated
Taylor image over the original unit parameter box, restricted by a list
of validated parameter constraints. Ambiguous ReLU signs produce a pair of
branches; stable signs do not split. A ReLU active branch carries the original
preactivation Taylor model, while an inactive branch carries the zero model.
All downstream layers propagate each branch independently. No new public
numeric or set types are defined.

To obtain a sound *necessary* guard from preactivation `z(s) in p(s)+/-e`,
the active branch enforces `p(s)>=-e`, and the inactive branch enforces
`p(s)<=+e`. A temporary polynomial guard model is created by clobbering
the remainder only in a copy, never in the propagated output. The constraints
may overlap due to uncertainty, correctly retaining both possible activation
states. They are not exact activation regions when `e>0`. The current
implementation does not invoke a solver or prune infeasible branches,
and it rejects a branch count exceeding `max_branches` rather than
silently merging states. This function returns a *set image*, not a
first-class multivalued state-to-neural-output function; closed-loop
evolution must preserve the original parameterization to avoid losing
input-output correlation.

## Symbolic expansion

`neurosymbolic_expand(snapshot, variables)` builds ordinary
`Expression<Real>` objects, using exact `Real(ExactDouble(weight))`
coefficients. Every neuron produces a DAG node built from affine operations
and an activation expression. All five activations are supported:
Identity, ReLU, Tanh, Sigmoid and Softplus. The two latter functions were
added to Symbolic as `Expression<Real>` compositions of its elementary
operators, with tests for derivative, simplification, evaluation,
classification and Python bindings. The adapter now invokes
`sigmoid(value)` and `softplus(value)` directly rather than duplicating
their formulas. Numeric continues to provide the rigorous interval
implementations used by the validated evaluator.

Consequently Symbolic's already implemented `derivative`, `simplify`,
`substitute` and `make_function` can be used on resulting expressions
where defined. This does not automatically grant differentiability at ReLU
zero nor rigorous verification of properties. Expansion can be very large
on deep/wide networks. The native C++ model expansion API is initially
preferred over a bespoke Python binding to the large `Expression<Real>`
type hierarchy; it can be exposed in Python after its interoperability
and exception behavior have been validated.

## Constrained neural maps and Expression Set boundary

Neurosymbolic consumes Expression Set as its mathematical dependency, rather
than pinning Symbolic and Set separately. Native Expression<Real> expansion
remains supported; symbolic predicates and bounded expression-defined sets
are available for interoperability with validated constrained image sets.

Future neural-native propagation should accept existing FloatDPBall,
FloatDPBounds and validated Taylor function models, preserve parameter
dependencies, and return existing ValidatedConstrainedImageSet objects when
ReLU activation signs require branching. Each branch must retain its own
activation constraints and account rigorously for Taylor remainder errors.
Constraint Solving, Nonlinear Programming and Linear Programming are optional
refinement facilities: basic interval and Taylor evaluation must not depend
on invoking a solver. Evolution may consume these images later; Neurosymbolic
must not depend on Evolution or introduce duplicate constrained-Taylor types.

## Roadmap

1. **N0 — Standalone module and CI:** mirror the Ariadne CMake, public
   umbrella header, Python extension, submodule pinning, tests, tutorials,
   installation smoke tests, and five CI workflows.
2. **N1 — Validated bounds:** natural interval propagation, true Numeric
   directed rounding, activation coverage and model/parameter assumptions.
3. **N2 — Symbolic composition:** exact coefficient embedding into
   Symbolic Expression DAGs, differentiability and substitution tests.
4. **N3 — Stronger enclosure:** validated derivative bounds, dependency
   reduction (affine/Taylor propagation), and trustworthy cross-module
   independent enclosure oracles.
5. **N4 — Function integration:** explicit adapters to Function interfaces,
   interoperability with Reasoning without adding Reasoning dependencies.
6. **N5 — Larger-model performance:** avoid expression explosion and
   profile interval/CPU evaluations separately.

Do not introduce artificial version negotiation for the new repository's
model serialization; it consumes Neural's current snapshot contract.

## Implementation log

| Date (UTC) | Milestone | Status | Changes, evidence, limitations |
| --- | --- | --- | --- |
| 2026-11-05 | Clang -Wshadow fix in constrained Taylor tests | Source corrected; CI pending | Renamed a local NeuralParameterSnapshot test variable from `two` to `two_neuron_model` to avoid shadowing the existing Ariadne::two constant under Clang -Werror. Test scenarios, coverage, and production code unchanged. |
| 2026-11-05 | N3 constrained ReLU branching | Source and C++ tests added; CI pending | New constrained_taylor overload returning List<ValidatedConstrainedImageSet>. ReLU branches are guarded by remainder-relaxed parameter constraints, preserve the original Taylor models and branch through subsequent layers. Optional branch cap rejects exponential explosion. Tests cover branch coverage, Taylor error ambiguity, stable signs and cap validation. Constraint solver is not called. |
| 2026-11-05 | N3 initial: native validated Taylor-model propagation | Source and C++ regression tests added; CI pending | New evaluator.taylor overload for Vector<ValidatedTaylorModelDP>. Affine and all five activation functions reuse existing Function calculus and binary64 singleton bounds; tests cover affine derivatives, remainder propagation, dimension validation and pointwise certified inclusion. ReLU does not split into constrained branches. |
| 2026-11-04 | Fix Python invalid-box test setup | Test corrected; CI pending | Zero-dimensional FloatDPExactBox now uses the existing dimension constructor (`FloatDPExactBox(0)`) rather than an empty initializer list, which UniformArray rejects before validation. Kept reversed-endpoint rejection and added a nonempty wrong-dimension case; production Numeric and Neurosymbolic code unchanged. |
| 2026-11-04 | Symbolic expansion semantic regression tests | Source tests updated; CI pending | Replaced inconclusive lazy Real equality with certified FloatDPBounds containment and tightness checks. Replaced structural identity between mathematically equivalent but syntactically distinct sigmoid/softplus trees with multi-point rigorous evaluation comparisons, retaining all five activation cases and independent smooth derivative checks. Production symbolic expansion unchanged. |
| 2026-11-04 | N1 native ball and bounds propagation | Source, tests and Python bindings committed; CI pending | Added FloatDPBoundsVector and FloatDPBallVector overloads reusing the same FloatDPBounds propagation as the original FloatDPExactBox path. Ball conversion encloses the full radius. No custom interval types or solver calls. |
| 2026-11-04 | Adopt Expression Set as the mathematical dependency | Implemented in source; CI pending | Replaced the direct Symbolic gitlink with Expression Set, which carries Symbolic and Set (with Linear Programming, Constraint Solving, and Nonlinear Programming transitively). Updated CMake dependency consistency checks, Python binding linkage, documentation and the build graph. No new mathematical types. |
| 2026-11-04 | Repair existing validated Numeric API usage | Implemented in source; CI pending | Removed the conflicting local BoundsType alias and used FloatDPBounds directly. Constructed binary64 singleton intervals with ExactDouble and dp; changed C++ tests and tutorial to use the public FloatDP(ExactDouble(value),dp) constructor. |
| 2026-11-04 | Pin Symbolic activation evaluation test correction | Implemented; CI pending | Updated ss/sym to symbolic d17d699a, correcting a test that compared Real expression trees structurally instead of checking rigorous numeric enclosure of evaluated sigmoid(0) and softplus(0). The mathematical APIs and symbolic activation support are unchanged. |\n| 2026-11-04 | Restore all five symbolic neural activations | Implemented; CI pending | Pinned ss/sym to Symbolic commit 7115fe33, which exposes sigmoid and softplus Expression<Real> functions with equivalent C++ and Python tests. Neurosymbolic now invokes the Symbolic operations directly, supports all five activation types and checks the two expansions and their derivatives against Symbolic reference expressions. |
| 2026-11-04 | Reuse Ariadne Interval/Box and Numeric activations | Implemented; CI pending | Removed duplicate NeurosymbolicValidatedInterval and switched validated public signatures to FloatDPExactBox -> FloatDPUpperBox. Reused existing FloatDPBounds, make_interval and Numeric tanh/sigmoid/softplus. Symbolic expansion now rejects Sigmoid/Softplus where Expression<Real> lacks the operations instead of locally spelling them as formulas. Updated C++/Python tests, tutorials and README. Old CI was failing on compilation; this revision still needs CI verification. |
| 2026-11-04 | Repository naming normalization after GitHub rename | Implemented; CI pending | Renamed repository-specific headers, source files, tests, directory paths, public API names, CMake project/targets, find-package module and documentation consistently to neurosymbolic / Neurosymbolic / NEUROSYMBOLIC. Kept upstream Neural and Symbolic type names unchanged. Checked every tracked source/config/document for obsolete neuro-symbolic, neuro_symbolic and NeuroSymbolic spellings. Pre-existing CI failures on earlier commit remain to be diagnosed independently. |
| 2026-11-04 | N2 expression-kind enumeration correction | Implemented; CI pending | Used the actual Ariadne OperatorKind::NULLARY enumeration for a nonconstant symbolic-expression test; prevents an invalid enum identifier under compiler warnings-as-errors. |
| 2026-11-04 | N0/N1/N2 standalone bootstrap and first mathematical adapters | Implemented; CI pending | Created CMake, pinned Neural and Symbolic submodules, Configuration and Testing, Python bindings, C++ unit tests, tutorials, install smoke and five CI workflows. Added Numeric FloatDPBounds interval enclosure for five activations with strict bounds and explicit exact binary64 coefficients. Added C++ Symbolic Expression<Real> expansion and differentiation interoperability. Tests include analytic affine corners, sample inclusion, invalid inputs, symbolic coefficient and variable validation. The new repository has no Reasoning dependency; Neural remains unchanged. Final compiler/CI evidence is pending. |
