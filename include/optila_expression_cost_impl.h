// Copyright (c) 2023 Matthew Murrian
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <cstdint>
#include <type_traits>

namespace optila {

struct OperationCounts {
  std::size_t additions = 0;
  std::size_t multiplications = 0;
  std::size_t divisions = 0;
  std::size_t power_and_root = 0;
  std::size_t expensive_operation = 0;

  constexpr static std::size_t weight_addition = 1;
  constexpr static std::size_t weight_multiplication = 1;
  constexpr static std::size_t weight_division = 4;
  constexpr static std::size_t weight_power_and_root = 6;
  constexpr static std::size_t weight_expensive_operation = 25;

  [[nodiscard]] constexpr std::size_t cost() const {
    return weight_addition * additions +
           weight_multiplication * multiplications +
           weight_division * divisions +
           weight_power_and_root * power_and_root +
           weight_expensive_operation * expensive_operation;
  }
};

constexpr OperationCounts operator*(const OperationCounts& count,
                                    std::size_t factor) {
  return {count.additions * factor, count.multiplications * factor,
          count.divisions * factor, count.power_and_root * factor,
          count.expensive_operation * factor};
}

constexpr decltype(auto) operator+(const OperationCounts count1,
                                   const OperationCounts count2) {
  return OperationCounts{
      count1.additions + count2.additions,
      count1.multiplications + count2.multiplications,
      count1.divisions + count2.divisions,
      count1.power_and_root + count2.power_and_root,
      count1.expensive_operation + count2.expensive_operation};
}

}  // namespace optila