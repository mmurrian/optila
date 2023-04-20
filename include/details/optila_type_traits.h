#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

namespace optila::details {

template <typename T>
struct is_small_trivial
    : std::bool_constant<(sizeof(std::decay_t<T>) <= sizeof(void*)) &&
                         std::is_trivially_copyable_v<std::decay_t<T>> &&
                         alignof(std::decay_t<T>) <= sizeof(void*)> {};

template <typename T>
inline constexpr bool is_small_trivial_v = is_small_trivial<T>::value;

template <typename T>
struct efficient_type_qualifiers {
  using type = std::conditional_t<is_small_trivial_v<T>, std::decay_t<T>,
                                  const std::decay_t<T>&>;
};

template <typename T>
using efficient_type_qualifiers_t = typename efficient_type_qualifiers<T>::type;

template <typename T>
struct safe_type_qualifiers {
  using type = std::conditional_t<std::is_lvalue_reference_v<T> &&
                                      !is_small_trivial_v<std::decay_t<T>>,
                                  const std::decay_t<T>&, std::decay_t<T>>;
};

template <typename T>
using safe_type_qualifiers_t = typename safe_type_qualifiers<T>::type;

template <class T, class... Ts>
struct are_same : std::conjunction<std::is_same<T, Ts>...> {};

template <class T, class... Ts>
inline constexpr bool are_same_v = are_same<T, Ts...>::value;

#ifdef OPTILA_ENABLE_IMPLICIT_CONVERSIONS
template <typename... Args>
struct common_value_type : std::common_type<Args...> {};
#else
template <typename T, typename... Ts>
struct common_value_type {
  static_assert(are_same_v<T, Ts...>,
                "Operand type mismatch with implicit conversions disabled.");
  using type = T;
};

template <typename T, typename... Ts>
using common_value_type_t = typename common_value_type<T, Ts...>::type;
#endif

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