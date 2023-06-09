// Copyright (c) 2023 Matthew Murrian
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <type_traits>

#include "optila_expression_impl.h"
#include "optila_operation_impl.h"

namespace optila {

// Element-wise Addition operation
template <typename Lhs, typename Rhs,
          typename = std::enable_if_t<details::is_expression_v<Lhs> &&
                                      details::is_expression_v<Rhs>>>
constexpr auto operator+(Lhs&& lhs, Rhs&& rhs) {
  if constexpr (details::is_scalar_v<Lhs> && details::is_scalar_v<Rhs>) {
    // Scalar addition operation
    return Expression<Operation::ScalarAddition, Lhs, Rhs>(
        std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
  } else {
    return Expression(Operation::Addition{}, std::forward<Lhs>(lhs),
                      std::forward<Rhs>(rhs));
  }
}

// Element-wise Subtraction operation
template <typename Lhs, typename Rhs,
          typename = std::enable_if_t<details::is_expression_v<Lhs> &&
                                      details::is_expression_v<Rhs>>>
constexpr auto operator-(Lhs&& lhs, Rhs&& rhs) {
  if constexpr (details::is_scalar_v<Lhs> && details::is_scalar_v<Rhs>) {
    // Scalar subtraction operation
    return Expression<Operation::ScalarSubtraction, Lhs, Rhs>(
        std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
  } else {
    return Expression(Operation::Subtraction{}, std::forward<Lhs>(lhs),
                      std::forward<Rhs>(rhs));
  }
}

template <typename Lhs, typename Rhs,
          typename = std::enable_if_t<details::is_expression_v<Lhs> &&
                                      details::is_expression_v<Rhs>>>
constexpr auto operator*(Lhs&& lhs, Rhs&& rhs) {
  if constexpr (details::is_matrix_v<Lhs> || details::is_matrix_v<Rhs>) {
    // Matrix multiplication operation
    return Expression(Operation::Multiplication{}, std::forward<Lhs>(lhs),
                      std::forward<Rhs>(rhs));
  } else if constexpr (details::is_scalar_v<Lhs> && details::is_scalar_v<Rhs>) {
    // Scalar multiplication operation
    return Expression(Operation::ScalarMultiplication{}, std::forward<Lhs>(lhs),
                      std::forward<Rhs>(rhs));
  }
}

// Matrix-Scalar or Scalar division operation
template <typename Lhs, typename Rhs,
          typename = std::enable_if_t<details::is_expression_v<Lhs> &&
                                      details::is_expression_v<Rhs>>>
constexpr auto operator/(Lhs&& lhs, Rhs&& rhs) {
  if constexpr (details::is_matrix_v<Lhs> && details::is_scalar_v<Rhs>) {
    // Matrix-Scalar division operation
    return Expression(Operation::MatrixScalarDivision{}, std::forward<Lhs>(lhs),
                      std::forward<Rhs>(rhs));
  } else {
    // Scalar division operation
    return Expression(Operation::ScalarDivision{}, std::forward<Lhs>(lhs),
                      std::forward<Rhs>(rhs));
  }
}

// Strict equality operation
template <typename Lhs, typename Rhs,
          typename = std::enable_if_t<details::is_expression_v<Lhs> &&
                                      details::is_expression_v<Rhs>>>
constexpr bool operator==(Lhs&& lhs, Rhs&& rhs) {
  return evaluate(Expression(Operation::StrictEquality{},
                             std::forward<Lhs>(lhs), std::forward<Rhs>(rhs)));
}

}  // namespace optila