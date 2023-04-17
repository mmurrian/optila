#pragma once

#include "optila_matrix.h"
#include "optila_scalar.h"

namespace optila {

template <typename Op, typename... Operands>
class Expression;

namespace details {

template <typename T>
struct is_expression_literal : std::false_type {};

template <typename Op, typename... Operands>
struct is_expression_literal<Expression<Op, Operands...>> : std::true_type {};

template <typename T>
inline constexpr bool is_expression_literal_v = is_expression_literal<T>::value;

template <std::size_t Index, typename Expr>
struct is_operand_expression_literal {
  static constexpr bool value = is_expression_literal_v<
      typename std::decay_t<Expr>::template operand_type<Index>>;
};

template <std::size_t Index, typename Expr>
inline constexpr bool is_operand_expression_literal_v =
    is_operand_expression_literal<Index, Expr>::value;

template <typename Lhs>
struct is_static_expression
    : std::conditional_t<std::decay_t<Lhs>::num_rows_static() != Dynamic &&
                             std::decay_t<Lhs>::num_cols_static() != Dynamic,
                         std::true_type, std::false_type> {};

template <typename Lhs>
inline constexpr bool is_static_expression_v = is_static_expression<Lhs>::value;

template <typename Lhs>
struct is_dynamic_expression
    : std::conditional_t<std::decay_t<Lhs>::num_rows_static() == Dynamic ||
                             std::decay_t<Lhs>::num_cols_static() == Dynamic,
                         std::true_type, std::false_type> {};

template <typename Lhs>
inline constexpr bool is_dynamic_expression_v =
    is_dynamic_expression<Lhs>::value;

}  // namespace details
}  // namespace optila