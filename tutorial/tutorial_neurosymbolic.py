# ***************************************************************************
# *            tutorial/tutorial_neurosymbolic.py
# *
# *  Copyright  2026  Luca Geretti
# *
# ***************************************************************************
#
# This file is part of Ariadne Neuro-Symbolic.
#
# Ariadne Neuro-Symbolic is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Ariadne Neuro-Symbolic is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Ariadne Neuro-Symbolic. If not, see <https://www.gnu.org/licenses/>.

from pyariadne import (
    NeuralActivation,
    NeuralDenseLayerSpec,
    NeuralNetwork,
    NeuralParameterSnapshot,
    NeuralValidatedEvaluator,
    NeuralValidatedInterval,
    neuro_symbolic_version,
)

model = NeuralNetwork(1, [
    NeuralDenseLayerSpec(1, NeuralActivation.Tanh)])
snapshot = NeuralParameterSnapshot(model, [2.0, 0.0])
enclosure = NeuralValidatedEvaluator(snapshot).bounds([
    NeuralValidatedInterval(-1.0, 1.0)])
print("Neuro-Symbolic", neuro_symbolic_version())
print("Validated output:", enclosure[0].lower, enclosure[0].upper)
