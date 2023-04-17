#pragma once

#include <type_traits>

namespace optila::details {

template <typename T, bool SmallTrivial = (sizeof(T) <= sizeof(void*) &&
                                           std::is_trivially_copyable_v<T> &&
                                           alignof(T) <= sizeof(void*))>
struct return_by_value_or_const_ref {
  using type = std::conditional_t<SmallTrivial, T, const T&>;
};

template <typename T>
using return_by_value_or_const_ref_t =
    typename return_by_value_or_const_ref<T>::type;

template <typename T>
struct store_by_value_or_const_ref {
  using type = std::conditional_t<std::is_lvalue_reference_v<T>,
                                  const std::decay_t<T>&, std::decay_t<T>>;
};

template <typename T>
using store_by_value_or_const_ref_t =
    typename store_by_value_or_const_ref<T>::type;

template <typename T>
struct expr_value_type_if_not_void {
  using type = typename std::decay_t<T>::value_type;
};

template <>
struct expr_value_type_if_not_void<void> {
  using type = void;
};

template <typename T>
using expr_value_type_if_not_void_t =
    typename expr_value_type_if_not_void<T>::type;

template <typename Lhs, typename Rhs, bool IsRhsVoid>
struct common_type_if_not_void;

template <typename Lhs, typename Rhs>
struct common_type_if_not_void<Lhs, Rhs, false> {
  using type = std::common_type_t<Lhs, Rhs>;
};

template <typename Lhs, typename Rhs>
struct common_type_if_not_void<Lhs, Rhs, true> {
  using type = Lhs;
};

template <typename Lhs, typename Rhs, bool IsRhsVoid>
using common_type_if_not_void_t =
    typename common_type_if_not_void<Lhs, Rhs, IsRhsVoid>::type;

template <typename... Operands>
struct is_nullary_operand : std::false_type {};

template <>
struct is_nullary_operand<> : std::true_type {};

template <typename... Operands>
inline constexpr bool is_nullary_operand_v =
    is_nullary_operand<Operands...>::value;

template <typename... Operands>
struct is_unary_operand : std::false_type {};

template <typename Operand>
struct is_unary_operand<Operand> : std::true_type {};

template <typename... Operands>
inline constexpr bool is_unary_operand_v = is_unary_operand<Operands...>::value;

template <typename... Operands>
struct is_binary_operand : std::false_type {};

template <typename Lhs, typename Rhs>
struct is_binary_operand<Lhs, Rhs> : std::true_type {};

template <typename... Operands>
inline constexpr bool is_binary_operand_v =
    is_binary_operand<Operands...>::value;

}  // namespace optila::details