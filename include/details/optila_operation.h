#pragma once

#include <type_traits>

namespace optila::details {

struct operation_state_tag {};

template <typename T>
struct has_operation_state
    : std::is_base_of<details::operation_state_tag, std::decay_t<T>> {};

template <typename T>
inline constexpr bool has_operation_state_v = has_operation_state<T>::value;

}  // namespace optila::details