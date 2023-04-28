// Copyright (c) 2023 Matthew Murrian
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include "optila_expression_impl.h"

namespace optila {

template <typename ValueType, std::size_t NumRows, std::size_t NumCols>
using ConstantMatrix =
    Expression<Operation::ConstantMatrix<NumRows, NumCols>, Scalar<ValueType>>;

template <typename ValueType, std::size_t NumRows, std::size_t NumCols>
using ConstantDiagonal =
    Expression<Operation::ConstantDiagonal<NumRows, NumCols>,
               Scalar<ValueType>>;

template <typename ValueType, std::size_t NumRows, std::size_t NumCols>
inline constexpr auto IdentityMatrix =
    Expression<Operation::ConstantDiagonal<NumRows, NumCols>,
               Scalar<ValueType>>{Scalar<ValueType>{1}};

}  // namespace optila