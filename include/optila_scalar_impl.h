#pragma once

#include <utility>

#include "details/optila_scalar.h"

namespace optila {

template <typename ValueType>
class Scalar : public details::scalar_tag {
 public:
  using value_type = ValueType;

  // Enable this constructor only for arithmetic types, allowing implicit
  // conversion
  template <typename T = ValueType,
            std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
  constexpr explicit Scalar(value_type value) : value_(value) {}

  // Enable this constructor for non-arithmetic types, requiring explicit
  // conversion
  template <typename T = ValueType,
            std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
  constexpr explicit Scalar(value_type&& value)
      : value_(std::forward<value_type>(value)) {}

  constexpr explicit operator decltype(auto)() const { return value_; }
  constexpr decltype(auto) operator()() const { return value_; }

 private:
  ValueType value_;
};

template <typename ValueType>
constexpr Scalar<ValueType> make_scalar(ValueType&& args) {
  return Scalar<ValueType>(std::forward<ValueType>(args));
}

// Deduction guide for Scalar
template <typename ValueType>
Scalar(ValueType) -> Scalar<ValueType>;

}  // namespace optila