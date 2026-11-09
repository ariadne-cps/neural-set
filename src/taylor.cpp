/***************************************************************************
 *            src/taylor.cpp
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

#include "neural-set/validated.hpp"
#include "numeric/builtin.hpp"
#include "function/taylor_function.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Ariadne {
namespace {

FloatDPBounds exact_taylor_parameter(double value) {
    // A stored binary64 coefficient denotes an exact dyadic real.
    return FloatDPBounds(ExactDouble(value),dp);
}

void check_finite_taylor(ValidatedTaylorModelDP const& model) {
    auto const enclosure=model.range();
    if (!std::isfinite(enclosure.lower_bound().raw().get_d()) ||
        !std::isfinite(enclosure.upper_bound().raw().get_d()))
        throw std::overflow_error("Nonfinite validated neural Taylor enclosure");
}

ValidatedTaylorModelDP activate_taylor(
    ValidatedTaylorModelDP const& value, NeuralActivation activation) {
    switch (activation) {
        case NeuralActivation::Relu:
            return max(value,value.create_constant(exact_taylor_parameter(0.0)));
        case NeuralActivation::Tanh: return tanh(value);
        case NeuralActivation::Sigmoid: return rec(1+exp(-value));
        case NeuralActivation::Softplus: return log(1+exp(value));
        // NeuralNetwork validates the immutable activation; the remaining
        // supported case is Identity.
        default: return value;
    }
}

} // namespace

Vector<ValidatedTaylorModelDP> NeuralValidatedEvaluator::taylor(
    Vector<ValidatedTaylorModelDP> const& input) const {
    auto const& network=snapshot_.network();
    if (input.size()!=network.input_dimension())
        throw std::invalid_argument("Wrong number of neural Taylor inputs");

    auto const parameter_dimension=input[0].argument_size();
    std::vector<ValidatedTaylorModelDP> state;
    state.reserve(input.size());
    for (SizeType i=0; i<input.size(); ++i) {
        if (input[i].argument_size()!=parameter_dimension)
            throw std::invalid_argument("Incompatible neural Taylor parameter dimensions");
        check_finite_taylor(input[i]);
        state.push_back(input[i]);
    }

    auto const& parameters=snapshot_.values();
    for (auto const& layer : network.layers()) {
        std::vector<ValidatedTaylorModelDP> output;
        output.reserve(layer.outputs);
        for (std::size_t o=0; o<layer.outputs; ++o) {
            auto sum=state[0].create_constant(
                exact_taylor_parameter(parameters[layer.biases_offset+o]));
            for (std::size_t i=0; i<layer.inputs; ++i) {
                sum+=state[i]*exact_taylor_parameter(
                    parameters[layer.weights_offset+o*layer.inputs+i]);
                check_finite_taylor(sum);
            }
            auto activated=activate_taylor(sum,layer.activation);
            check_finite_taylor(activated);
            output.push_back(std::move(activated));
        }
        state.swap(output);
    }

    Vector<ValidatedTaylorModelDP> result(state.size(),state[0]);
    for (SizeType i=0; i<state.size(); ++i)
        result[i]=state[i];
    return result;
}


namespace {

// Temporary algorithm state only; the public mathematical result is a list
// of existing ValidatedConstrainedImageSet objects.
struct TaylorActivationBranch {
    std::vector<ValidatedTaylorModelDP> state;
    std::vector<ValidatedTaylorModelDP> outputs;
    List<ValidatedConstraint> constraints;
};

ValidatedConstraint relu_parameter_guard(
    ValidatedTaylorModelDP const& preactivation,
    ExactBoxType const& parameter_domain,
    bool active) {
    // If the true preactivation z(s) lies in p(s) +/- e, its sign implies:
    //   z(s) >= 0  => p(s) >= -e
    //   z(s) <= 0  => p(s) <= +e.
    // Use p as a separate, exact polynomial guard; clobber() is deliberately
    // applied only to the guard's polynomial copy, never to a propagated map.
    auto polynomial=preactivation;
    auto const error=preactivation.error().raw().get_d();
    polynomial.clobber();
    ValidatedScalarMultivariateTaylorFunctionModelDP patch(parameter_domain,polynomial);
    ValidatedScalarMultivariateFunctionModelDP model(patch);
    ValidatedScalarMultivariateFunction function(model.raw_pointer()->_clone());
    ValidatedNumber const e(exact_taylor_parameter(error));
    return active ? ((-e)<=function) : (function<=e);
}

} // namespace

List<ValidatedConstrainedImageSet>
NeuralValidatedEvaluator::branched_taylor(
    Vector<ValidatedTaylorModelDP> const& input,
    SizeType max_branches) const {
    auto const& network=snapshot_.network();
    if (input.size()!=network.input_dimension())
        throw std::invalid_argument("Wrong number of neural Taylor inputs");
    if (max_branches==0u)
        throw std::invalid_argument("Neural Taylor branch limit must be positive");

    SizeType const parameter_dimension=input[0].argument_size();
    ExactBoxType const parameter_domain=ExactBoxType::unit_box(parameter_dimension);

    TaylorActivationBranch initial;
    initial.state.reserve(input.size());
    for (SizeType i=0u; i<input.size(); ++i) {
        if (input[i].argument_size()!=parameter_dimension)
            throw std::invalid_argument("Incompatible neural Taylor parameter dimensions");
        check_finite_taylor(input[i]);
        initial.state.push_back(input[i]);
    }
    std::vector<TaylorActivationBranch> branches;
    branches.push_back(std::move(initial));

    auto const& parameters=snapshot_.values();
    for (auto const& layer : network.layers()) {
        for (SizeType o=0u; o<layer.outputs; ++o) {
            std::vector<TaylorActivationBranch> advanced;
            for (auto& branch : branches) {
                auto pre=branch.state[0].create_constant(
                    exact_taylor_parameter(parameters[layer.biases_offset+o]));
                for (SizeType i=0u; i<layer.inputs; ++i) {
                    pre+=branch.state[i]*exact_taylor_parameter(
                        parameters[layer.weights_offset+o*layer.inputs+i]);
                    check_finite_taylor(pre);
                }

                if (layer.activation==NeuralActivation::Relu) {
                    auto const range=pre.range();
                    if (range.lower_bound().raw().get_d()>=0.0) {
                        branch.outputs.push_back(pre);
                    } else if (range.upper_bound().raw().get_d()<=0.0) {
                        branch.outputs.push_back(pre.create_zero());
                    } else {
                        // Two branches cover all realisations represented by
                        // the Taylor remainder; neither assumes its sign.
                        if (advanced.size()>=max_branches-1u)
                            throw std::length_error("Neural ReLU branch limit exceeded");
                        auto inactive=branch;
                        inactive.outputs.push_back(pre.create_zero());
                        inactive.constraints.append(
                            relu_parameter_guard(pre,parameter_domain,false));
                        advanced.push_back(std::move(inactive));
                        branch.outputs.push_back(pre);
                        branch.constraints.append(
                            relu_parameter_guard(pre,parameter_domain,true));
                    }
                } else {
                    branch.outputs.push_back(activate_taylor(pre,layer.activation));
                }

                check_finite_taylor(branch.outputs.back());
                if (advanced.size()>=max_branches)
                    throw std::length_error("Neural ReLU branch limit exceeded");
                advanced.push_back(std::move(branch));
            }
            branches.swap(advanced);
        }

        for (auto& branch : branches) {
            branch.state.swap(branch.outputs);
            branch.outputs.clear();
        }
    }

    List<ValidatedConstrainedImageSet> result;
    for (auto const& branch : branches) {
        Vector<ValidatedTaylorModelDP> models(branch.state.size(),branch.state[0]);
        for (SizeType i=0u; i<branch.state.size(); ++i)
            models[i]=branch.state[i];
        ValidatedVectorMultivariateTaylorFunctionModelDP patch(parameter_domain,models);
        ValidatedConstrainedImageSet image(
            parameter_domain,ValidatedVectorMultivariateFunctionModelDP(patch));
        for (auto const& constraint : branch.constraints)
            image.new_parameter_constraint(constraint);
        result.append(std::move(image));
    }
    return result;
}


namespace {

// Every outward-rounded interval is a box of exact binary64 endpoints.
ExactIntervalType exact_interval_from_bounds(FloatDPBounds const& bounds) {
    const double lo=bounds.lower_raw().get_d();
    const double hi=bounds.upper_raw().get_d();
    // Interval arithmetic on validated, ordered inputs preserves ordering.
    if (!std::isfinite(lo) || !std::isfinite(hi))
        throw std::overflow_error("Nonfinite neural graph enclosure");
    return ExactIntervalType(ExactDouble(lo),ExactDouble(hi));
}

FloatDPBounds bounds_from_taylor(ValidatedTaylorModelDP const& input) {
    check_finite_taylor(input);
    const auto range=input.range();
    return FloatDPBounds(
        ExactDouble(range.lower_bound().raw().get_d()),
        ExactDouble(range.upper_bound().raw().get_d()),dp);
}

ValidatedNumber exact_graph_parameter(double value) {
    return ValidatedNumber(exact_taylor_parameter(value));
}

ValidatedScalarMultivariateFunction polynomial_of_taylor(
    ValidatedTaylorModelDP const& model, SizeType full_parameter_count) {
    const SizeType n=model.argument_size();
    auto polynomial=model;
    polynomial.clobber(); // Only the polynomial copy loses the remainder.
    const ExactBoxType unit=ExactBoxType::unit_box(n);
    ValidatedScalarMultivariateTaylorFunctionModelDP patch(unit,polynomial);
    ValidatedScalarMultivariateFunctionModelDP fm(patch);
    ValidatedScalarMultivariateFunction scalar(fm.raw_pointer()->_clone());
    // The first n parameters are the original Taylor variables.
    ValidatedVectorMultivariateFunction projection(n,full_parameter_count);
    for (SizeType i=0u; i<n; ++i)
        projection.set(i,
            ValidatedScalarMultivariateFunction::coordinate(full_parameter_count,i));
    return compose(scalar,projection);
}

} // namespace

ValidatedConstrainedImageSet NeuralValidatedEvaluator::constrained_taylor(
    Vector<ValidatedTaylorModelDP> const& input) const {
    const auto& network=snapshot_.network();
    const SizeType input_count=input.size();
    if (input_count==0u || input_count!=network.input_dimension())
        throw std::invalid_argument("Wrong number of neural Taylor inputs");
    const SizeType n=input[0].argument_size();
    for (SizeType i=0u; i<input_count; ++i) {
        if (input[i].argument_size()!=n)
            throw std::invalid_argument("Incompatible neural Taylor parameter dimensions");
        check_finite_taylor(input[i]);
    }

    // Bound each auxiliary ReLU output using independent, directed-rounded
    // interval propagation. Constraints may later reduce these coarse bounds.
    std::vector<ExactIntervalType> auxiliary_intervals;
    auxiliary_intervals.reserve(input_count);
    std::vector<FloatDPBounds> interval_state;
    interval_state.reserve(input_count);
    for (SizeType i=0u; i<input_count; ++i) {
        auto const bounds=bounds_from_taylor(input[i]);
        auxiliary_intervals.push_back(exact_interval_from_bounds(bounds));
        interval_state.push_back(bounds);
    }
    auto const& coefficients=snapshot_.values();
    SizeType relu_count=0u;
    for (auto const& layer : network.layers()) {
        std::vector<FloatDPBounds> interval_output;
        interval_output.reserve(layer.outputs);
        for (SizeType j=0u; j<layer.outputs; ++j) {
            FloatDPBounds z=exact_taylor_parameter(coefficients[layer.biases_offset+j]);
            for (SizeType i=0u; i<layer.inputs; ++i) {
                z=z+exact_taylor_parameter(
                    coefficients[layer.weights_offset+j*layer.inputs+i])*interval_state[i];
                exact_interval_from_bounds(z);
            }
            FloatDPBounds y=z;
            switch (layer.activation) {
                case NeuralActivation::Relu:
                    y=max(z,exact_taylor_parameter(0.0));
                    auxiliary_intervals.push_back(exact_interval_from_bounds(y));
                    ++relu_count;
                    break;
                case NeuralActivation::Tanh: y=tanh(z); break;
                case NeuralActivation::Sigmoid: y=sigmoid(z); break;
                case NeuralActivation::Softplus: y=softplus(z); break;
                default: break; // Validated Identity activation.
            }
            exact_interval_from_bounds(y);
            interval_output.push_back(y);
        }
        interval_state.swap(interval_output);
    }

    const SizeType total_count=n+input_count+relu_count;
    ExactBoxType domain(total_count);
    for (SizeType i=0u; i<n; ++i)
        domain[i]=ExactIntervalType(-1,1);
    for (SizeType i=0u; i<auxiliary_intervals.size(); ++i)
        domain[n+i]=auxiliary_intervals[i];

    List<ValidatedConstraint> constraints;
    std::vector<ValidatedScalarMultivariateFunction> state;
    state.reserve(input_count);
    for (SizeType i=0u; i<input_count; ++i) {
        auto const x=ValidatedScalarMultivariateFunction::coordinate(
            total_count,n+i);
        const auto p=polynomial_of_taylor(input[i],total_count);
        const auto remainder=exact_graph_parameter(
            input[i].error().raw().get_d());
        // True x_i(s) belongs to p_i(s) +/- remainder.
        constraints.append(ValidatedConstraint(-remainder,x-p,remainder));
        state.push_back(x);
    }

    SizeType next_relu=n+input_count;
    for (auto const& layer : network.layers()) {
        std::vector<ValidatedScalarMultivariateFunction> next;
        next.reserve(layer.outputs);
        for (SizeType j=0u; j<layer.outputs; ++j) {
            auto z=ValidatedScalarMultivariateFunction::constant(
                total_count,exact_graph_parameter(coefficients[layer.biases_offset+j]));
            for (SizeType i=0u; i<layer.inputs; ++i)
                z=z+state[i]*exact_graph_parameter(
                    coefficients[layer.weights_offset+j*layer.inputs+i]);
            switch (layer.activation) {
                case NeuralActivation::Relu: {
                    auto const y=ValidatedScalarMultivariateFunction::coordinate(
                        total_count,next_relu++);
                    constraints.append(exact_graph_parameter(0.0)<=y);
                    constraints.append((z-y)<=exact_graph_parameter(0.0));
                    constraints.append(ValidatedConstraint(
                        exact_graph_parameter(0.0),y*(y-z),
                        exact_graph_parameter(0.0)));
                    next.push_back(y);
                } break;
                case NeuralActivation::Tanh:
                    next.push_back(tanh(z)); break;
                case NeuralActivation::Sigmoid:
                    next.push_back(rec(1+exp(-z))); break;
                case NeuralActivation::Softplus:
                    next.push_back(log(1+exp(z))); break;
                default: // Validated Identity activation.
                    next.push_back(z);
                    break;
            }
        }
        state.swap(next);
    }

    List<ValidatedScalarMultivariateFunction> outputs;
    for (auto const& fn : state) outputs.append(fn);
    ValidatedConstrainedImageSet result(
        domain,ValidatedVectorMultivariateFunction(outputs));
    for (auto const& constraint : constraints)
        result.new_parameter_constraint(constraint);
    return result;
}

} // namespace Ariadne
