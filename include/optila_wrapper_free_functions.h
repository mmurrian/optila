#pragma once

#include <type_traits>

#include "optila_expression_impl.h"
#include "optila_operation_impl.h"

namespace optila {

// Transpose operation
template <typename Lhs>
constexpr auto transpose(Lhs&& mat) {
  return Expression<Operation::Transpose, Lhs>(std::forward<Lhs>(mat));
}

// Submatrix extraction operation
template <std::size_t StartRow, std::size_t StartCol, std::size_t NewRows,
          std::size_t NewCols, typename Lhs>
constexpr auto submatrix(Lhs&& mat) {
  return Expression<
      Operation::SubmatrixExtraction<StartRow, StartCol, NewRows, NewCols>,
      Lhs>(std::forward<Lhs>(mat));
}

// Constant matrix
template <std::size_t NumRows, std::size_t NumCols, typename Lhs>
constexpr auto constant_matrix(Lhs&& scal) {
  return Expression<Operation::ConstantMatrix<NumRows, NumCols>, Lhs>(
      std::forward<Lhs>(scal));
}

// Constant diagonal matrix
template <std::size_t NumRows, std::size_t NumCols, typename Lhs>
constexpr auto constant_diagonal(Lhs&& scal) {
  return Expression<Operation::ConstantDiagonal<NumRows, NumCols>, Lhs>(
      std::forward<Lhs>(scal));
}

// Put a vector on the diagonal of a matrix
template <typename Lhs>
constexpr auto diagonal_from_vector(Lhs&& vec) {
  return Expression<Operation::DiagonalFromVector, Lhs>(std::forward<Lhs>(vec));
}

// Extract the diagonal of a matrix into a vector
template <typename Lhs>
constexpr auto diagonal_to_vector(Lhs&& mat) {
  return Expression<Operation::DiagonalToVector, Lhs>(std::forward<Lhs>(mat));
}

// Extract the diagonal of a matrix into a diagonal matrix
template <typename Lhs>
constexpr auto diagonal_matrix(Lhs&& mat) {
  return Expression<Operation::DiagonalMatrix, Lhs>(std::forward<Lhs>(mat));
}

// Dot product operation
template <typename Lhs, typename Rhs>
constexpr auto dot(Lhs&& lhs, Rhs&& rhs) {
  return Expression<Operation::DotProduct, Lhs, Rhs>(std::forward<Lhs>(lhs),
                                                     std::forward<Rhs>(rhs));
}

// Vector norm operation
template <typename Lhs>
constexpr auto norm(Lhs&& vec) {
  auto dot_product_expr = Expression<Operation::DotProduct, Lhs, Lhs>(
      std::forward<Lhs>(vec), std::forward<Lhs>(vec));
  return Expression<Operation::SquareRoot, decltype(dot_product_expr)>(
      std::move(dot_product_expr));
}

// Vector normalization operation
template <typename Lhs>
constexpr auto normalize(Lhs&& vec) {
  return Expression<Operation::Normalization, Lhs>(std::forward<Lhs>(vec));
}

// Static type cast operation
template <typename ToType, typename Lhs>
constexpr auto static_convert(Lhs&& mat) {
  using FromType = typename std::decay_t<Lhs>::value_type;
  return Expression<Operation::StaticConversion<FromType, ToType>, Lhs>(
      std::forward<Lhs>(mat));
}

// Matrix expression evaluation
template <typename Expr, StorageOrder Order = StorageOrder::RowMajor,
          typename = std::enable_if_t<details::is_matrix_v<Expr>>>
constexpr auto evaluate(const Expr& expr) {
  using ResultType = Matrix<typename Expr::value_type, Expr::num_rows_static(),
                            Expr::num_cols_static(), Order>;
  return static_cast<ResultType>(expr);
}

// Scalar expression evaluation
template <typename Expr,
          typename = std::enable_if_t<details::is_scalar_v<Expr>>>
constexpr decltype(auto) evaluate(const Expr& expr) {
  using ResultType = Scalar<typename Expr::value_type>;
  return ResultType(expr)();
}

}  // namespace optila