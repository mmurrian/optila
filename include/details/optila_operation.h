#pragma once

#include <type_traits>

namespace optila::details {

struct operation_tag {};

template <typename T>
struct is_operation : std::is_base_of<details::operation_tag, std::decay_t<T>> {
};

template <typename T>
inline constexpr bool is_operation_v = is_operation<T>::value;

}  // namespace optila::details