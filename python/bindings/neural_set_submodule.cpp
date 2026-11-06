/***************************************************************************
 *            python/bindings/neural_set_submodule.cpp
 *
 *  Copyright  2026  Luca Geretti
 *
 ****************************************************************************/

/*
 *  This file is part of Ariadne Neural Set.
 *
 *  Ariadne Neural Set is free software: you can redistribute it and/or
 *  modify it under the terms of the GNU General Public License as published
 *  by the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Ariadne Neural Set is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Ariadne Neural Set. If not, see <https://www.gnu.org/licenses/>.
 */

#include "pybind11.hpp"
#include "neural-set/neural_set.hpp"
#include "neural-set/validated.hpp"

#include <stdexcept>
#include <vector>

using namespace Ariadne;

namespace {

// Function already exposes the scalar ValidatedTaylorModelDP class to Python,
// but not Vector<ValidatedTaylorModelDP>. Cross this boundary using ordinary
// Python lists instead of defining a second, incompatible vector class.
Vector<ValidatedTaylorModelDP> taylor_models_from_python(pybind11::list const& items) {
    if (pybind11::len(items)==0)
        throw std::invalid_argument("At least one neural Taylor input is required");
    const auto first=items[0].cast<ValidatedTaylorModelDP>();
    Vector<ValidatedTaylorModelDP> result(static_cast<SizeType>(pybind11::len(items)),first);
    for (SizeType i=1u; i<result.size(); ++i)
        result[i]=items[i].cast<ValidatedTaylorModelDP>();
    return result;
}

pybind11::list taylor_models_to_python(Vector<ValidatedTaylorModelDP> const& models) {
    pybind11::list result;
    for (SizeType i=0u; i<models.size(); ++i)
        result.append(pybind11::cast(models[i]));
    return result;
}

pybind11::list constrained_sets_to_python(List<ValidatedConstrainedImageSet> const& sets) {
    pybind11::list result;
    for (auto const& set : sets)
        result.append(pybind11::cast(set));
    return result;
}

} // namespace

void neural_set_submodule(pybind11::module& module) {
    module.def("neural_set_version", &neural_set_version);

    pybind11::class_<NeuralValidatedEvaluator>(module, "NeuralValidatedEvaluator")
        .def(pybind11::init<NeuralParameterSnapshot>(),pybind11::arg("snapshot"))
        .def("bounds",pybind11::overload_cast<FloatDPExactBox const&>(
            &NeuralValidatedEvaluator::bounds,pybind11::const_),pybind11::arg("input_box"))
        .def("bounds",pybind11::overload_cast<FloatDPBoundsVector const&>(
            &NeuralValidatedEvaluator::bounds,pybind11::const_),pybind11::arg("input_bounds"))
        .def("bounds",pybind11::overload_cast<FloatDPBallVector const&>(
            &NeuralValidatedEvaluator::bounds,pybind11::const_),pybind11::arg("input_balls"))
        .def("taylor",[](NeuralValidatedEvaluator const& evaluator,
                           pybind11::list const& inputs) {
            return taylor_models_to_python(evaluator.taylor(taylor_models_from_python(inputs)));
        },pybind11::arg("inputs"))
        .def("constrained_taylor",[](NeuralValidatedEvaluator const& evaluator,
                                       pybind11::list const& inputs) {
            return evaluator.constrained_taylor(taylor_models_from_python(inputs));
        },pybind11::arg("inputs"))
        .def("branched_taylor",[](NeuralValidatedEvaluator const& evaluator,
                                    pybind11::list const& inputs, SizeType max_branches) {
            return constrained_sets_to_python(
                evaluator.branched_taylor(taylor_models_from_python(inputs),max_branches));
        },pybind11::arg("inputs"),pybind11::arg("max_branches")=1024u);

    module.def("neural_validated_bounds",
        &neural_validated_bounds,
        pybind11::arg("snapshot"),pybind11::arg("input_box"));


}
