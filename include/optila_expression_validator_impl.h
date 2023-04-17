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
struct ExpressionValidator<Operation::Addition, Lhs, Rhs> {
  using expression_type = details::matrix_tag;

  static_assert(details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs> &&
                    std::decay_t<Lhs>::num_rows_static() ==
                        std::decay_t<Rhs>::num_rows_static() &&
                    std::decay_t<Lhs>::num_cols_static() ==
                        std::decay_t<Rhs>::num_cols_static(),
                "Mismatched operands for addition");
};

template <typename Lhs, typename Rhs>
struct ExpressionValidator<Operation::Multiplication, Lhs, Rhs> {
  using expression_type = details::matrix_tag;

  static_assert(details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs>,
                "Matrix multiplication requires matrix operands");
  static_assert(Lhs::num_cols_static() == Dynamic ||
                    Rhs::num_rows_static() == Dynamic ||
                    Lhs::num_cols_static() == Rhs::num_rows_static(),
                "Matrix operand inner dimensions must match");
};

template <typename Lhs, typename Rhs>
struct ExpressionValidator<Operation::DotProduct, Lhs, Rhs> {
  using expression_type = details::scalar_tag;

  static_assert(details::is_vector_v<Lhs> && details::is_vector_v<Rhs> &&
                    (std::decay_t<Lhs>::num_rows_static() ==
                         std::decay_t<Rhs>::num_rows_static() &&
                     std::decay_t<Lhs>::num_cols_static() ==
                         std::decay_t<Rhs>::num_cols_static()),
                "Dot product requires vector operands of the same dimension");
};
}  // namespace optila