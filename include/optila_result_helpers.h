#pragma once

#include "details/optila_type_traits.h"
#include "optila_operation_impl.h"

namespace optila {

template <typename Op, typename Lhs = void, typename Rhs = void,
          typename Enable = void>
struct ResultValueType {
  using type = details::common_type_if_not_void_t<
      details::expr_value_type_if_not_void_t<Lhs>,
      details::expr_value_type_if_not_void_t<Rhs>, std::is_void_v<Rhs>>;
};

template <typename Op, typename Lhs = void, typename Rhs = void,
          typename Enable = void>
struct ResultNumRows;

template <typename Op, typename Lhs = void, typename Rhs = void,
          typename Enable = void>
struct ResultNumCols;

// === Binary matrix operations
template <typename Op, typename Lhs, typename Rhs>
struct ResultNumRows<
    Op, Lhs, Rhs,
    std::enable_if_t<details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs>>>
    : std::integral_constant<std::size_t,
                             std::decay_t<Lhs>::num_rows_static()> {
  static_assert(std::decay_t<Lhs>::num_rows_static() ==
                    std::decay_t<Rhs>::num_rows_static(),
                "Matrix dimensions must match");
};

template <typename Op, typename Lhs, typename Rhs>
struct ResultNumCols<
    Op, Lhs, Rhs,
    std::enable_if_t<details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs>>>
    : std::integral_constant<std::size_t,
                             std::decay_t<Lhs>::num_cols_static()> {
  static_assert(std::decay_t<Lhs>::num_cols_static() ==
                    std::decay_t<Rhs>::num_cols_static(),
                "Matrix dimensions must match");
};

// === Unary matrix operations
template <typename Op, typename Lhs>
struct ResultNumRows<Op, Lhs, void, std::enable_if_t<details::is_matrix_v<Lhs>>>
    : std::integral_constant<std::size_t,
                             std::decay_t<Lhs>::num_rows_static()> {};

template <typename Op, typename Lhs>
struct ResultNumCols<Op, Lhs, void, std::enable_if_t<details::is_matrix_v<Lhs>>>
    : std::integral_constant<std::size_t,
                             std::decay_t<Lhs>::num_cols_static()> {};

// === Matrix-Scalar operations
template <typename Op, typename Lhs, typename Rhs>
struct ResultNumRows<
    Op, Lhs, Rhs,
    std::enable_if_t<details::is_matrix_v<Lhs> && details::is_scalar_v<Rhs>>>
    : std::integral_constant<std::size_t,
                             std::decay_t<Lhs>::num_rows_static()> {};

template <typename Op, typename Lhs, typename Rhs>
struct ResultNumCols<
    Op, Lhs, Rhs,
    std::enable_if_t<details::is_matrix_v<Lhs> && details::is_scalar_v<Rhs>>>
    : std::integral_constant<std::size_t,
                             std::decay_t<Lhs>::num_cols_static()> {};

// === Scalar-Matrix operations
template <typename Op, typename Lhs, typename Rhs>
struct ResultNumRows<
    Op, Lhs, Rhs,
    std::enable_if_t<details::is_scalar_v<Lhs> && details::is_matrix_v<Rhs>>>
    : std::integral_constant<std::size_t,
                             std::decay_t<Rhs>::num_rows_static()> {};

template <typename Op, typename Lhs, typename Rhs>
struct ResultNumCols<
    Op, Lhs, Rhs,
    std::enable_if_t<details::is_scalar_v<Lhs> && details::is_matrix_v<Rhs>>>
    : std::integral_constant<std::size_t,
                             std::decay_t<Rhs>::num_cols_static()> {};

// === Binary scalar operations
// Scalar operations do not define ResultNumRows or ResultNumCols

// === Unary scalar operations
// Scalar operations do not define ResultNumRows or ResultNumCols

// === Specialized operations
// Matrix multiplication
template <typename Lhs, typename Rhs>
struct ResultNumRows<
    Operation::Multiplication, Lhs, Rhs,
    std::enable_if_t<details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs>>>
    : std::integral_constant<std::size_t,
                             std::decay_t<Lhs>::num_rows_static()> {
  static_assert(std::decay_t<Lhs>::num_cols_static() ==
                    std::decay_t<Rhs>::num_rows_static(),
                "Inner matrix dimensions must match");
};

template <typename Lhs, typename Rhs>
struct ResultNumCols<
    Operation::Multiplication, Lhs, Rhs,
    std::enable_if_t<details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs>>>
    : std::integral_constant<std::size_t,
                             std::decay_t<Rhs>::num_cols_static()> {};

// Vector to diagonal matrix
template <typename Lhs>
struct ResultNumRows<Operation::DiagonalFromVector, Lhs, void,
                     std::enable_if_t<details::is_vector_v<Lhs>>>
    : std::integral_constant<std::size_t,
                             std::max(std::decay_t<Lhs>::num_rows_static(),
                                      std::decay_t<Lhs>::num_cols_static())> {};

template <typename Lhs>
struct ResultNumCols<Operation::DiagonalFromVector, Lhs, void,
                     std::enable_if_t<details::is_vector_v<Lhs>>>
    : std::integral_constant<std::size_t,
                             std::max(std::decay_t<Lhs>::num_rows_static(),
                                      std::decay_t<Lhs>::num_cols_static())> {};

// Diagonal of a matrix to vector
template <typename Lhs>
struct ResultNumRows<Operation::DiagonalToVector, Lhs, void,
                     std::enable_if_t<details::is_matrix_v<Lhs>>>
    : std::integral_constant<std::size_t,
                             std::min(std::decay_t<Lhs>::num_rows_static(),
                                      std::decay_t<Lhs>::num_cols_static())> {};

template <typename Lhs>
struct ResultNumCols<Operation::DiagonalToVector, Lhs, void,
                     std::enable_if_t<details::is_matrix_v<Lhs>>>
    : std::integral_constant<std::size_t, 1> {};

// Diagonal of a matrix to a diagonal matrix
template <typename Lhs>
struct ResultNumRows<Operation::DiagonalMatrix, Lhs, void,
                     std::enable_if_t<details::is_matrix_v<Lhs>>>
    : std::integral_constant<std::size_t,
                             std::min(std::decay_t<Lhs>::num_rows_static(),
                                      std::decay_t<Lhs>::num_cols_static())> {};

template <typename Lhs>
struct ResultNumCols<Operation::DiagonalMatrix, Lhs, void,
                     std::enable_if_t<details::is_matrix_v<Lhs>>>
    : std::integral_constant<std::size_t,
                             std::min(std::decay_t<Lhs>::num_rows_static(),
                                      std::decay_t<Lhs>::num_cols_static())> {};

// Constant matrix
template <std::size_t NumRows, std::size_t NumCols, typename Operand>
struct ResultNumRows<Operation::ConstantMatrix<NumRows, NumCols>, Operand, void,
                     std::enable_if_t<details::is_scalar_v<Operand>>>
    : std::integral_constant<std::size_t, NumRows> {};

template <std::size_t NumRows, std::size_t NumCols, typename Operand>
struct ResultNumCols<Operation::ConstantMatrix<NumRows, NumCols>, Operand, void,
                     std::enable_if_t<details::is_scalar_v<Operand>>>
    : std::integral_constant<std::size_t, NumCols> {};

// Constant diagonal matrix
template <std::size_t NumRows, std::size_t NumCols, typename Operand>
struct ResultNumRows<Operation::ConstantDiagonal<NumRows, NumCols>, Operand,
                     void, std::enable_if_t<details::is_scalar_v<Operand>>>
    : std::integral_constant<std::size_t, NumRows> {};

template <std::size_t NumRows, std::size_t NumCols, typename Operand>
struct ResultNumCols<Operation::ConstantDiagonal<NumRows, NumCols>, Operand,
                     void, std::enable_if_t<details::is_scalar_v<Operand>>>
    : std::integral_constant<std::size_t, NumCols> {};

// Matrix transposition
template <typename Operand>
struct ResultNumRows<Operation::Transpose, Operand, void,
                     std::enable_if_t<details::is_matrix_v<Operand>>>
    : std::integral_constant<std::size_t,
                             std::decay_t<Operand>::num_cols_static()> {};

template <typename Operand>
struct ResultNumCols<Operation::Transpose, Operand, void,
                     std::enable_if_t<details::is_matrix_v<Operand>>>
    : std::integral_constant<std::size_t,
                             std::decay_t<Operand>::num_rows_static()> {};

// Submatrix extraction
template <std::size_t StartRow, std::size_t StartCol, std::size_t NumRows,
          std::size_t NumCols, typename Operand>
struct ResultNumRows<
    Operation::SubmatrixExtraction<StartRow, StartCol, NumRows, NumCols>,
    Operand, void, std::enable_if_t<details::is_matrix_v<Operand>>>
    : std::integral_constant<std::size_t, NumRows> {
  static_assert(StartRow + NumRows <= std::decay_t<Operand>::num_rows_static(),
                "Submatrix extraction out of bounds");
};

template <std::size_t StartRow, std::size_t StartCol, std::size_t NumRows,
          std::size_t NumCols, typename Operand>
struct ResultNumCols<
    Operation::SubmatrixExtraction<StartRow, StartCol, NumRows, NumCols>,
    Operand, void, std::enable_if_t<details::is_matrix_v<Operand>>>
    : std::integral_constant<std::size_t, NumCols> {
  static_assert(StartCol + NumCols <= std::decay_t<Operand>::num_cols_static(),
                "Submatrix extraction out of bounds");
};

}  // namespace optila