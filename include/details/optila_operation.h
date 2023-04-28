// Copyright (c) 2023 Matthew Murrian
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

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