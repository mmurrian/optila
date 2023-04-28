// Copyright (c) 2023 Matthew Murrian
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#pragma once

#include <cmath>
#include <tuple>

#include "details/optila_matrix.h"
#include "details/optila_operation.h"
#include "details/optila_type_traits.h"

namespace optila::Operation {

struct ScalarAddition : public details::operation_tag {
  static constexpr auto to_scalar = [](auto&& lhs, auto&& rhs) {
    return lhs() + rhs();
  };
};

struct Addition : public details::operation_tag {
  static constexpr auto to_matrix_element = [](std::size_t i, std::size_t j,
                                               auto&& lhs, auto&& rhs) {
    return lhs(i, j) + rhs(i, j);
  };
};
struct ScalarSubtraction : public details::operation_tag {
  static constexpr auto to_scalar = [](auto&& lhs, auto&& rhs) {
    return lhs() - rhs();
  };
};
struct Subtraction : public details::operation_tag {
  static constexpr auto to_matrix_element = [](std::size_t i, std::size_t j,
                                               auto&& lhs, auto&& rhs) {
    return lhs(i, j) - rhs(i, j);
  };
};
// Matrix multiplication
struct Multiplication : public details::operation_tag {
  static constexpr auto to_matrix_element = [](std::size_t i, std::size_t j,
                                               auto&& lhs, auto&& rhs) {
    if constexpr (details::is_matrix_v<decltype(lhs)> &&
                  details::is_matrix_v<decltype(rhs)>) {
      using value_type =
          details::common_value_type_t<std::decay_t<decltype(lhs(0, 0))>,
                                       std::decay_t<decltype(rhs(0, 0))>>;
      value_type result = 0;
      for (std::size_t k = 0; k < lhs.num_cols(); ++k) {
        result += lhs(i, k) * rhs(k, j);
      }
      return result;
    } else if constexpr (details::is_matrix_v<decltype(lhs)>) {
      return lhs(i, j) * rhs();
    } else {  // details::is_matrix_v<decltype(rhs)>
      return lhs() * rhs(i, j);
    }
  };
};
struct ScalarMultiplication : public details::operation_tag {
  static constexpr auto to_scalar = [](auto&& lhs, auto&& rhs) {
    return lhs() * rhs();
  };
};
struct MatrixScalarDivision : public details::operation_tag {
  static constexpr auto to_matrix_element = [](std::size_t i, std::size_t j,
                                               auto&& lhs, auto&& rhs) {
    return lhs(i, j) / rhs();
  };
};
struct ScalarDivision : public details::operation_tag {
  static constexpr auto to_scalar = [](auto&& lhs, auto&& rhs) {
    return lhs() / rhs();
  };
};
struct DotProduct : public details::operation_tag {
  static constexpr auto to_scalar = [](auto&& lhs, auto&& rhs) {
    using value_type = typename std::decay_t<decltype(lhs(0, 0))>;
    value_type result = 0;
    for (std::size_t i = 0; i < lhs.num_rows(); ++i) {
      result += lhs(i, 0) * rhs(i, 0);
    }
    return result;
  };
};
struct CrossProduct : public details::operation_tag {};
struct OuterProduct : public details::operation_tag {};
struct Transpose : public details::operation_tag {
  static constexpr auto to_matrix_element =
      [](std::size_t i, std::size_t j, auto&& lhs) { return lhs(j, i); };
};

struct Determinant : public details::operation_tag {};
struct Trace : public details::operation_tag {};
struct Inverse : public details::operation_tag {};
struct Adjoint : public details::operation_tag {};
struct Cofactor : public details::operation_tag {};
struct Rank : public details::operation_tag {};
template <std::size_t StartRow, std::size_t StartCol, std::size_t NumRows,
          std::size_t NumCols>
struct SubmatrixExtraction : public details::operation_tag {
  static constexpr auto to_matrix_element = [](std::size_t i, std::size_t j,
                                               auto&& lhs) {
    return lhs(i + StartRow, j + StartCol);
  };
};
struct Concatenation : public details::operation_tag {};
struct SquareRoot : public details::operation_tag {
  // FIXME: sqrt is not constexpr
  static constexpr auto to_scalar = [](auto&& lhs) {
    using std::sqrt;
    return sqrt(lhs());
  };
};
struct ElementWiseOperation : public details::operation_tag {};

struct StrictEquality : public details::operation_tag {
  static constexpr auto to_scalar = [](auto&& lhs, auto&& rhs) {
    using Lhs = std::decay_t<decltype(lhs)>;
    using Rhs = std::decay_t<decltype(rhs)>;
    if constexpr (details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs>) {
      for (std::size_t i = 0; i < lhs.num_rows(); ++i) {
        for (std::size_t j = 0; j < lhs.num_cols(); ++j) {
          if (lhs(i, j) != rhs(i, j)) {
            return false;
          }
        }
      }
      return true;
    } else {
      return lhs() == rhs();
    }
  };
};

template <typename FromType, typename ToType>
struct StaticConversion : public details::operation_tag {
  static constexpr auto to_scalar = [](auto&& lhs) {
    return static_cast<ToType>(lhs());
  };
  static constexpr auto to_matrix_element = [](std::size_t i, std::size_t j,
                                               auto&& lhs) {
    return static_cast<ToType>(lhs(i, j));
  };
};

// Fill a matrix with a constant value
template <std::size_t NumRows, std::size_t NumCols>
struct ConstantMatrix : public details::operation_tag {
  static constexpr auto to_matrix_element =
      [](std::size_t /*i*/, std::size_t /*j*/, auto&& lhs) { return lhs(); };
};
// Fill a diagonal matrix with a constant value
template <std::size_t NumRows, std::size_t NumCols>
struct ConstantDiagonal : public details::operation_tag {
  static constexpr auto to_matrix_element = [](std::size_t i, std::size_t j,
                                               auto&& lhs) {
    using value_type = decltype(lhs());
    return i == j ? lhs() : value_type{};
  };
};
// Put a vector on the diagonal of a matrix
struct DiagonalFromVector : public details::operation_tag {
  static constexpr auto to_matrix_element = [](std::size_t i, std::size_t j,
                                               auto&& lhs) {
    using value_type = typename std::decay_t<decltype(lhs(0, 0))>::value_type;
    return i == j ? lhs(i, 0) : value_type{};
  };
};
// Extract the diagonal of a matrix into a vector
struct DiagonalToVector : public details::operation_tag {
  static constexpr auto to_matrix_element =
      [](std::size_t i, std::size_t j, auto&& lhs) { return lhs(i, i); };
};
// Extract the diagonal of a matrix into a diagonal matrix
struct DiagonalMatrix : public details::operation_tag {
  static constexpr auto to_matrix_element = [](std::size_t i, std::size_t j,
                                               auto&& lhs) {
    using value_type = typename std::decay_t<decltype(lhs(0, 0))>::value_type;
    return i == j ? lhs(i, i) : value_type{};
  };
};

}  // namespace optila::Operation