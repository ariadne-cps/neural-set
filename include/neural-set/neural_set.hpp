/***************************************************************************
 *            neural_set.hpp
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

/** @file neural-set/neural_set.hpp
 *  @brief Version and module identity; mathematical adapters live separately.
 */
#ifndef ARIADNE_NEURAL_SET_HPP
#define ARIADNE_NEURAL_SET_HPP

namespace Ariadne {

/** @brief Return the installed NeuralSet module version. */
const char* neural_set_version() noexcept;

} // namespace Ariadne

#endif // ARIADNE_NEURAL_SET_HPP
