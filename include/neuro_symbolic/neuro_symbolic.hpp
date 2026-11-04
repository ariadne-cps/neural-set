/***************************************************************************
 *            include/neuro_symbolic/neuro_symbolic.hpp
 *
 *  Copyright  2026  Luca Geretti
 *
 ****************************************************************************/

/*
 *  This file is part of Ariadne Neuro-Symbolic.
 *
 *  Ariadne Neuro-Symbolic is free software: you can redistribute it and/or
 *  modify it under the terms of the GNU General Public License as published
 *  by the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Ariadne Neuro-Symbolic is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Ariadne Neuro-Symbolic. If not, see <https://www.gnu.org/licenses/>.
 */

/** @file neuro_symbolic/neuro_symbolic.hpp
 *  @brief Version and module identity; mathematical adapters live separately.
 */
#ifndef ARIADNE_NEURO_SYMBOLIC_HPP
#define ARIADNE_NEURO_SYMBOLIC_HPP

namespace Ariadne {

/** @brief Return the installed Neuro-Symbolic module version. */
char const* neuro_symbolic_version() noexcept;

} // namespace Ariadne

#endif // ARIADNE_NEURO_SYMBOLIC_HPP
