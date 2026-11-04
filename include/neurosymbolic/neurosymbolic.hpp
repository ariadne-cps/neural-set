/***************************************************************************
 *            include/neurosymbolic/neurosymbolic.hpp
 *
 *  Copyright  2026  Luca Geretti
 *
 ****************************************************************************/

/*
 *  This file is part of Ariadne Neurosymbolic.
 *
 *  Ariadne Neurosymbolic is free software: you can redistribute it and/or
 *  modify it under the terms of the GNU General Public License as published
 *  by the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Ariadne Neurosymbolic is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Ariadne Neurosymbolic. If not, see <https://www.gnu.org/licenses/>.
 */

/** @file neurosymbolic/neurosymbolic.hpp
 *  @brief Version and module identity; mathematical adapters live separately.
 */
#ifndef ARIADNE_NEUROSYMBOLIC_HPP
#define ARIADNE_NEUROSYMBOLIC_HPP

namespace Ariadne {

/** @brief Return the installed Neurosymbolic module version. */
char const* neurosymbolic_version() noexcept;

} // namespace Ariadne

#endif // ARIADNE_NEUROSYMBOLIC_HPP
