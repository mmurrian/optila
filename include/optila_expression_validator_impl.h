#pragma once

#include "optila_operation_impl.h"

namespace optila {
template <typename Op, typename... Operands>
struct ExpressionValidator;

template <typename Lhs, typename Rhs>
struct ExpressionValidator<Operation::ScalarAddition, Lhs, Rhs> {
  using expression_type = details::scalar_tag;

  static_assert(details::is_scalar_v<Lhs> && details::is_scalar_v<Rhs>,
                "Mismatched operands for scalar addition");
};

template <typename Lhs, typename Rhs>
struct ExpressionValidator<Operation::Addition, Lhs, Rhs>
    : details::matrix_tag {
  using expression_type = details::matrix_tag;

  static constexpr auto num_rows_static() {
    return std::decay_t<Lhs>::num_rows_static();
  }

  static constexpr auto num_cols_static() {
    return std::decay_t<Lhs>::num_cols_static();
  }

  static constexpr auto num_rows(const Lhs& lhs, const Rhs& /*rhs*/) {
    return lhs.num_rows();
  }

  static constexpr auto num_cols(const Lhs& lhs, const Rhs& /*rhs*/) {
    return lhs.num_cols();
  }

  static_assert(details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs>,
                "Mismatched operands for addition");

  static constexpr void static_validate() {
    static_assert(std::decay_t<Lhs>::num_rows_static() ==
                          std::decay_t<Rhs>::num_rows_static() &&
                      std::decay_t<Lhs>::num_cols_static() ==
                          std::decay_t<Rhs>::num_cols_static(),
                  "Mismatched operands for addition");
  }

  static constexpr void dynamic_validate(const Lhs& lhs, const Rhs& rhs) {
    assert(lhs.num_rows() == rhs.num_rows() &&
           lhs.num_cols() == rhs.num_cols());
  }
};

// Matrix multiplication
template <typename Lhs, typename Rhs>
struct ExpressionValidator<Operation::Multiplication, Lhs, Rhs> {
  using expression_type = details::matrix_tag;

  static constexpr auto num_rows_static() {
    return std::decay_t<Lhs>::num_rows_static();
  }
  static constexpr auto num_cols_static() {
    return std::decay_t<Rhs>::num_cols_static();
  }
  static constexpr auto num_rows(const Lhs& lhs, const Rhs& /*rhs*/) {
    return lhs.num_rows();
  }
  static constexpr auto num_cols(const Lhs& /*lhs*/, const Rhs& rhs) {
    return rhs.num_cols();
  }

  static_assert(details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs>,
                "Matrix multiplication requires matrix operands");

  static constexpr void static_validate() {
    static_assert(Lhs::num_cols_static() == Rhs::num_rows_static(),
                  "Matrix operand inner dimensions must match");
  }

  static constexpr void dynamic_validate(const Lhs& lhs, const Rhs& rhs) {
    assert(lhs.num_cols() == rhs.num_rows());
  }
};

// Vector dot product
template <typename Lhs, typename Rhs>
struct ExpressionValidator<Operation::DotProduct, Lhs, Rhs> {
  using expression_type = details::scalar_tag;

  static constexpr void static_validate() {
    static_assert(
        details::is_static_vector_v<Lhs> && details::is_static_vector_v<Rhs>,
        "Dot product requires vector operands");
    static_assert(Lhs::num_rows_static() == Rhs::num_rows_static() &&
                      Lhs::num_cols_static() == Rhs::num_cols_static(),
                  "Dot product requires vector operands of the same dimension");
  }

  static constexpr void dynamic_validate(const Lhs& lhs, const Rhs& rhs) {
    assert(lhs.num_rows() == rhs.num_rows() &&
           lhs.num_cols() == rhs.num_cols());
    assert(std::min(lhs.num_rows(), lhs.num_cols()) == 1);
  }
};

// Submatrix extraction
template <std::size_t StartRow, std::size_t StartCol, std::size_t NumRows,
          std::size_t NumCols, typename Lhs>
struct ExpressionValidator<
    Operation::SubmatrixExtraction<StartRow, StartCol, NumRows, NumCols>, Lhs> {
  using expression_type = details::matrix_tag;

  static constexpr auto num_rows_static() { return NumRows; }
  static constexpr auto num_cols_static() { return NumCols; }

  static constexpr auto num_rows(const Lhs& /*lhs*/) { return NumRows; }
  static constexpr auto num_cols(const Lhs& /*lhs*/) { return NumCols; }

  static_assert(details::is_matrix_v<Lhs>,
                "Submatrix extraction requires a "
                "matrix operand");

  static constexpr void static_validate() {
    static_assert(StartRow + NumRows <= Lhs::num_rows_static() &&
                      StartCol + NumCols <= Lhs::num_cols_static(),
                  "Submatrix extraction out of bounds");
  }

  static constexpr void dynamic_validate(const Lhs& lhs) {
    assert(StartRow + NumRows <= lhs.num_rows() &&
           StartCol + NumCols <= lhs.num_cols());
  }
};

}  // namespace optila