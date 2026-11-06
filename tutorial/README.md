# Neural Set standalone tutorial — C++ and Python parity

Both tutorials cover the **same six examples** and use the same frozen
networks and mathematical operations:

1. Rigorous box enclosure of a Tanh network.
2. Validated Bounds and Ball propagation, including ball radii.
3. Direct Taylor-model propagation via Function.
4. A single implicit `ValidatedConstrainedImageSet` for ReLU.
5. Explicit ReLU activation branches as an alternative.
6. Validated interval propagation for all five Neural activations.

The Python version uses the native existing Ariadne classes. NeuralSet
provides bindings for the adapter methods; it does
**not** create another Taylor-model, set, Python type.
Python lists are accepted for Taylor-model input vectors because Function
does not independently export `Vector<ValidatedTaylorModelDP>`.

Neither tutorial invokes the constraint solver implicitly. To experiment
with domain reduction, explicitly call `reduce()` on a constrained
image set; it can be expensive or inconclusive for complementarity constraints.

## Installed C++ and Python tutorials

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/install
cmake --build build
./build/tutorial_neural_set
python tutorial_neural_set.py
```

These are standalone examples. Run them explicitly using the commands
above; they are intentionally not registered as CTest or pytest tests.
Unit tests and coverage are maintained separately in `tests/` and
`python/tests/`.

These examples illustrate sound enclosures but do not constitute
independent formal proofs of all network behaviours. See
`../doc/neural_set_design.md` for the exact mathematical contracts.
