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

template <typename T>
struct is_matrix_or_scalar_literal
    : std::integral_constant<bool, is_matrix_literal_v<T> ||
                                       is_scalar_literal_v<T>> {};

template <typename T>
inline constexpr bool is_matrix_or_scalar_literal_v =
    is_matrix_or_scalar_literal<T>::value;

template <typename T>
struct is_expression : std::disjunction<is_matrix<T>, is_scalar<T>> {};

template <typename T>
inline constexpr bool is_expression_v = is_expression<T>::value;

template <typename Lhs, typename Rhs>
struct is_unary_operand_pair
    : std::conjunction<is_expression<Lhs>, std::is_void<Rhs>> {};

template <typename Lhs, typename Rhs>
inline constexpr bool is_unary_operand_pair_v =
    is_unary_operand_pair<Lhs, Rhs>::value;

template <typename Lhs, typename Rhs>
struct is_binary_operand_pair
    : std::conjunction<is_expression<Lhs>, is_expression<Rhs>> {};

template <typename Lhs, typename Rhs>
inline constexpr bool is_binary_operand_pair_v =
    is_binary_operand_pair<Lhs, Rhs>::value;

template <typename Lhs, typename Rhs>
struct is_either_scalar : std::disjunction<is_scalar<Lhs>, is_scalar<Rhs>> {};

template <typename Lhs, typename Rhs>
inline constexpr bool is_either_scalar_v = is_either_scalar<Lhs, Rhs>::value;

template <typename Lhs, typename Rhs>
struct is_either_matrix : std::disjunction<is_matrix<Lhs>, is_matrix<Rhs>> {};

template <typename Lhs, typename Rhs>
inline constexpr bool is_either_matrix_v = is_either_matrix<Lhs, Rhs>::value;

template <typename Lhs, typename Rhs>
struct is_binary_matrix_pair
    : std::conjunction<is_matrix<Lhs>, is_matrix<Rhs>> {};

template <typename Lhs, typename Rhs>
inline constexpr bool is_binary_matrix_pair_v =
    is_binary_matrix_pair<Lhs, Rhs>::value;

template <typename Lhs, typename Rhs>
struct is_binary_scalar_pair
    : std::conjunction<is_scalar<Lhs>, is_scalar<Rhs>> {};

template <typename Lhs, typename Rhs>
inline constexpr bool is_binary_scalar_pair_v =
    is_binary_scalar_pair<Lhs, Rhs>::value;

}  // namespace details
}  // namespace optila