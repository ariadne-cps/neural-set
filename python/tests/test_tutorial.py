"""Exercise the installed Python-facing Neurosymbolic tutorial in CI.

Run the public tutorial itself rather than maintaining a separate fake example.
"""
from pathlib import Path
import runpy

import pytest

from pyariadne import (
    DoublePrecision,
    NeuralActivation,
    NeuralDenseLayerSpec,
    NeuralNetwork,
    NeuralParameterSnapshot,
    NeurosymbolicValidatedEvaluator,
    SweeperDP,
    ThresholdSweeperDP,
    ValidatedTaylorModelDP,
    RealVariable,
    neurosymbolic_expand,
)


def test_python_tutorial_runs():
    tutorial = Path(__file__).resolve().parents[2] / "tutorial" / "tutorial_neurosymbolic.py"
    runpy.run_path(str(tutorial), run_name="__main__")


def test_python_taylor_adapters_and_symbolic_expansion():
    model = NeuralParameterSnapshot(
        NeuralNetwork(1, [NeuralDenseLayerSpec(1, NeuralActivation.Relu)]),
        [1.0, 0.0],
    )
    evaluator = NeurosymbolicValidatedEvaluator(model)
    dp = DoublePrecision()
    sweeper = SweeperDP(ThresholdSweeperDP(dp, 1e-12))
    x = ValidatedTaylorModelDP.coordinate(1, 0, sweeper)

    taylor_output = evaluator.taylor([x])
    assert len(taylor_output) == 1
    assert taylor_output[0].argument_size() == 1

    implicit = evaluator.constrained_taylor([x])
    assert implicit.number_of_parameters() == 3
    assert implicit.number_of_constraints() == 4

    explicit = evaluator.branched_taylor([x], 2)
    assert len(explicit) == 2
    assert all(branch.number_of_constraints() == 1 for branch in explicit)

    symbol = RealVariable("x")
    expressions = neurosymbolic_expand(model, [symbol])
    assert len(expressions) == 1

    with pytest.raises(ValueError):
        evaluator.taylor([])
    with pytest.raises(ValueError):
        evaluator.constrained_taylor([])
    with pytest.raises(ValueError):
        evaluator.branched_taylor([])
