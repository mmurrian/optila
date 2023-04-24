#pragma once

#include <type_traits>

#include "optila_expression_impl.h"
#include "optila_operation_impl.h"
#include "optila_optimizer_impl.h"

namespace optila {

// Transpose operation
template <typename Lhs>
constexpr auto transpose(Lhs&& mat) {
  return Expression(Operation::Transpose{}, std::forward<Lhs>(mat));
}

// Submatrix extraction operation
template <std::size_t StartRow, std::size_t StartCol, std::size_t NewRows,
          std::size_t NewCols, typename Lhs>
constexpr auto submatrix(Lhs&& mat) {
  return Expression(
      Operation::SubmatrixExtraction<StartRow, StartCol, NewRows, NewCols>{},
      std::forward<Lhs>(mat));
}

// Constant matrix
template <std::size_t NumRows, std::size_t NumCols, typename Lhs>
constexpr auto constant_matrix(Lhs&& scal) {
  return Expression(Operation::ConstantMatrix<NumRows, NumCols>{},
                    std::forward<Lhs>(scal));
}

// Constant diagonal matrix
template <std::size_t NumRows, std::size_t NumCols, typename Lhs>
constexpr auto constant_diagonal(Lhs&& scal) {
  return Expression(Operation::ConstantDiagonal<NumRows, NumCols>{},
                    std::forward<Lhs>(scal));
}

// Put a vector on the diagonal of a matrix
template <typename Lhs>
constexpr auto diagonal_from_vector(Lhs&& vec) {
  return Expression(Operation::DiagonalFromVector{}, std::forward<Lhs>(vec));
}

// Extract the diagonal of a matrix into a vector
template <typename Lhs>
constexpr auto diagonal_to_vector(Lhs&& mat) {
  return Expression(Operation::DiagonalToVector{}, std::forward<Lhs>(mat));
}

// Extract the diagonal of a matrix into a diagonal matrix
template <typename Lhs>
constexpr auto diagonal_matrix(Lhs&& mat) {
  return Expression(Operation::DiagonalMatrix{}, std::forward<Lhs>(mat));
}

// Dot product operation
template <typename Lhs, typename Rhs>
constexpr auto dot(Lhs&& lhs, Rhs&& rhs) {
  return Expression(Operation::DotProduct{}, std::forward<Lhs>(lhs),
                    std::forward<Rhs>(rhs));
}

// Vector norm operation
template <typename Lhs>
constexpr auto norm(Lhs&& vec) {
  auto dot_product_expr = Expression(
      Operation::DotProduct{}, std::forward<Lhs>(vec), std::forward<Lhs>(vec));
  return Expression(Operation::SquareRoot{}, std::move(dot_product_expr));
}

// Vector normalization operation
template <typename Lhs>
constexpr auto normalize(Lhs&& vec) {
  return vec / norm(vec);
}

// Static type cast operation
template <typename ToType, typename Lhs>
constexpr auto static_convert(Lhs&& mat) {
  using FromType = typename std::decay_t<Lhs>::value_type;
  return Expression(Operation::StaticConversion<FromType, ToType>{},
                    std::forward<Lhs>(mat));
}

template <typename Lhs>
constexpr decltype(auto) evaluate(Lhs&& expr) {
  // Accept l-value and r-value expressions but do not std::forward<Expr> to
  // the evaluator. The evaluator does not accept r-value expressions and will
  // not manage the lifetime of the expression.
  return Evaluator<std::decay_t<Lhs>, optimize_expression_t<Lhs>>(expr)
      .evaluate();
}

}  // namespace optila