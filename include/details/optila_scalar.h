#pragma once

#include <type_traits>

namespace optila {

template <typename ValueType>
class Scalar;

namespace details {
struct scalar_tag {};

template <typename T>
struct is_scalar
    : std::integral_constant<bool,
                             std::is_base_of_v<scalar_tag, std::decay_t<T>>> {};

template <typename T>
inline constexpr bool is_scalar_v = is_scalar<T>::value;

template <typename T>
struct is_scalar_literal : std::false_type {};

template <typename ValueType>
struct is_scalar_literal<Scalar<ValueType>> : std::true_type {};

template <typename T>
inline constexpr bool is_scalar_literal_v = is_scalar_literal<T>::value;

}  // namespace details
}  // namespace optila