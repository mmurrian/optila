#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

namespace optila::details {

template <typename... Args, std::size_t... Is>
constexpr decltype(auto) make_tuple_ref(const std::tuple<Args...>& tuple,
                                        std::index_sequence<Is...>) {
  return std::tie(std::get<Is>(tuple)...);
}

template <typename... Args>
constexpr decltype(auto) make_tuple_ref(const std::tuple<Args...>& tuple) {
  return make_tuple_ref(tuple, std::make_index_sequence<sizeof...(Args)>());
}

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