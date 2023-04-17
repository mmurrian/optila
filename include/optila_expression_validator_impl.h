#pragma once

#include <type_traits>

#include "details/optila_matrix.h"
#include "details/optila_scalar.h"
#include "details/optila_type_traits.h"
#include "optila_operation_impl.h"

namespace optila {
template <typename Op, typename... Operands>
struct ExpressionValidator;

template <typename Lhs, typename Rhs>
struct ExpressionValidator<Operation::ScalarAddition, Lhs, Rhs> {
  using expression_type = details::scalar_tag;
  using value_type = details::result_type_t<typename Lhs::value_type,
                                            typename Rhs::value_type>;

  static_assert(details::is_scalar_v<Lhs> && details::is_scalar_v<Rhs>,
                "Mismatched operands for scalar addition");
};

template <typename Lhs, typename Rhs>
struct ExpressionValidator<Operation::Addition, Lhs, Rhs>
    : details::matrix_tag {
  using expression_type = details::matrix_tag;
  using value_type = details::result_type_t<typename Lhs::value_type,
                                            typename Rhs::value_type>;

  static constexpr auto num_rows_static() { return Lhs::num_rows_static(); }

  static constexpr auto num_cols_static() { return Lhs::num_cols_static(); }

  static constexpr auto num_rows(const Lhs& lhs, const Rhs& /*rhs*/) {
    return lhs.num_rows();
  }

  static constexpr auto num_cols(const Lhs& lhs, const Rhs& /*rhs*/) {
    return lhs.num_cols();
  }

  static_assert(details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs>,
                "Mismatched operands for addition");

  static constexpr void static_validate() {
    static_assert(Lhs::num_rows_static() == Rhs::num_rows_static() &&
                      Lhs::num_cols_static() == Rhs::num_cols_static(),
                  "Mismatched operands for addition");
  }

  static constexpr void dynamic_validate(const Lhs& lhs, const Rhs& rhs) {
    assert(lhs.num_rows() == rhs.num_rows() &&
           lhs.num_cols() == rhs.num_cols());
  }
};

template <typename Lhs, typename Rhs>
struct ExpressionValidator<Operation::ScalarSubtraction, Lhs, Rhs> {
  using expression_type = details::scalar_tag;
  using value_type = details::result_type_t<typename Lhs::value_type,
                                            typename Rhs::value_type>;

  static_assert(details::is_scalar_v<Lhs> && details::is_scalar_v<Rhs>,
                "Mismatched operands for scalar subtraction");
};

template <typename Lhs, typename Rhs>
struct ExpressionValidator<Operation::Subtraction, Lhs, Rhs>
    : details::matrix_tag {
  using expression_type = details::matrix_tag;
  using value_type = details::result_type_t<typename Lhs::value_type,
                                            typename Rhs::value_type>;

  static constexpr auto num_rows_static() { return Lhs::num_rows_static(); }

  static constexpr auto num_cols_static() { return Lhs::num_cols_static(); }

  static constexpr auto num_rows(const Lhs& lhs, const Rhs& /*rhs*/) {
    return lhs.num_rows();
  }

  static constexpr auto num_cols(const Lhs& lhs, const Rhs& /*rhs*/) {
    return lhs.num_cols();
  }

  static_assert(details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs>,
                "Mismatched operands for subtraction");

  static constexpr void static_validate() {
    static_assert(Lhs::num_rows_static() == Rhs::num_rows_static() &&
                      Lhs::num_cols_static() == Rhs::num_cols_static(),
                  "Mismatched operands for subtraction");
  }

  static constexpr void dynamic_validate(const Lhs& lhs, const Rhs& rhs) {
    assert(lhs.num_rows() == rhs.num_rows() &&
           lhs.num_cols() == rhs.num_cols());
  }
};

// Matrix multiplication
template <typename Lhs, typename Rhs>
struct ExpressionValidator<Operation::Multiplication, Lhs, Rhs> {
  using expression_type = details::matrix_tag;
  using value_type = details::result_type_t<typename Lhs::value_type,
                                            typename Rhs::value_type>;

  static constexpr auto num_rows_static() {
    if constexpr (details::is_matrix_v<Lhs>) {
      return Lhs::num_rows_static();
    } else {
      return Rhs::num_rows_static();
    }
  }
  static constexpr auto num_cols_static() {
    if constexpr (details::is_matrix_v<Rhs>) {
      return Rhs::num_cols_static();
    } else {
      return Lhs::num_cols_static();
    }
  }
  static constexpr auto num_rows(const Lhs& lhs, const Rhs& rhs) {
    if constexpr (details::is_matrix_v<Lhs>) {
      return lhs.num_rows();
    } else {
      return rhs.num_rows();
    }
  }
  static constexpr auto num_cols(const Lhs& lhs, const Rhs& rhs) {
    if constexpr (details::is_matrix_v<Rhs>) {
      return rhs.num_cols();
    } else {
      return lhs.num_cols();
    }
  }

  static constexpr void static_validate() {
    static_assert(details::is_matrix_v<Lhs> || details::is_matrix_v<Rhs>,
                  "Multiplication requires at least one matrix operand");
    if constexpr (details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs>) {
      static_assert(Lhs::num_cols_static() == Rhs::num_rows_static(),
                    "Matrix operand inner dimensions must match");
    }
  }

  static constexpr void dynamic_validate(const Lhs& lhs, const Rhs& rhs) {
    if constexpr (details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs>) {
      assert(lhs.num_cols() == rhs.num_rows());
    }
  }
};

template <typename Lhs, typename Rhs>
struct ExpressionValidator<Operation::ScalarMultiplication, Lhs, Rhs> {
  using expression_type = details::scalar_tag;
  using value_type = details::result_type_t<typename Lhs::value_type,
                                            typename Rhs::value_type>;

  static_assert(details::is_scalar_v<Lhs> && details::is_scalar_v<Rhs>,
                "Mismatched operands for scalar multiplication");

  static constexpr void static_validate() {}

  static constexpr void dynamic_validate(const Lhs& lhs, const Rhs& rhs) {}
};

// Vector dot product
template <typename Lhs, typename Rhs>
struct ExpressionValidator<Operation::DotProduct, Lhs, Rhs> {
  using expression_type = details::scalar_tag;
  using value_type = details::result_type_t<typename Lhs::value_type,
                                            typename Rhs::value_type>;

  static constexpr void static_validate() {
    static_assert(
        details::is_static_vector_v<Lhs> && details::is_static_vector_v<Rhs>,
        "Dot product requires vector operands");
    static_assert(Lhs::num_rows_static() == Rhs::num_rows_static() &&
                      Lhs::num_cols_static() == Rhs::num_cols_static(),
                  "Dot product requires vector operands of the same dimension");
  }

  static constexpr void dynamic_validate(const Lhs& lhs, const Rhs& rhs) {
    assert(lhs.num_rows() == rhs.num_rows() &&
           lhs.num_cols() == rhs.num_cols());
    assert(std::min(lhs.num_rows(), lhs.num_cols()) == 1);
  }
};

// Submatrix extraction
template <std::size_t StartRow, std::size_t StartCol, std::size_t NumRows,
          std::size_t NumCols, typename Lhs>
struct ExpressionValidator<
    Operation::SubmatrixExtraction<StartRow, StartCol, NumRows, NumCols>, Lhs> {
  using expression_type = details::matrix_tag;
  using value_type = details::result_type_t<typename Lhs::value_type>;

  static constexpr auto num_rows_static() { return NumRows; }
  static constexpr auto num_cols_static() { return NumCols; }

  static constexpr auto num_rows(const Lhs& /*lhs*/) { return NumRows; }
  static constexpr auto num_cols(const Lhs& /*lhs*/) { return NumCols; }

  static_assert(details::is_matrix_v<Lhs>,
                "Submatrix extraction requires a "
                "matrix operand");

  static constexpr void static_validate() {
    static_assert(StartRow + NumRows <= Lhs::num_rows_static() &&
                      StartCol + NumCols <= Lhs::num_cols_static(),
                  "Submatrix extraction out of bounds");
  }

  static constexpr void dynamic_validate(const Lhs& lhs) {
    assert(StartRow + NumRows <= lhs.num_rows() &&
           StartCol + NumCols <= lhs.num_cols());
  }
};

// Strict equality
template <typename Lhs, typename Rhs>
struct ExpressionValidator<Operation::StrictEquality, Lhs, Rhs> {
  using expression_type = details::scalar_tag;
  using value_type = bool;

  static_assert((details::is_scalar_v<Lhs> && details::is_scalar_v<Rhs>) ||
                    (details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs>),
                "Mismatched operands for strict equality");

  static constexpr void static_validate() {
    static_assert(Lhs::num_rows_static() == Rhs::num_rows_static() &&
                      Lhs::num_cols_static() == Rhs::num_cols_static(),
                  "Mismatched operands for strict equality");
  }

  static constexpr void dynamic_validate(const Lhs& lhs, const Rhs& rhs) {
    assert(lhs.num_rows() == rhs.num_rows() &&
           lhs.num_cols() == rhs.num_cols());
  }
};

// Static conversion helpers
namespace details {
template <typename Lhs, typename FromType, typename ToType>
struct ScalarStaticConversionExpressionValidator {
  using expression_type = details::scalar_tag;
  using value_type = ToType;

  static_assert(details::is_scalar_v<Lhs>,
                "Static conversion requires a "
                "scalar operand");
};

template <typename Lhs, typename FromType, typename ToType>
struct MatrixStaticConversionExpressionValidator {
  using expression_type = details::matrix_tag;
  using value_type = ToType;

  static constexpr auto num_rows_static() { return Lhs::num_rows_static(); }
  static constexpr auto num_cols_static() { return Lhs::num_cols_static(); }

  static constexpr auto num_rows(const Lhs& lhs) { return lhs.num_rows(); }
  static constexpr auto num_cols(const Lhs& lhs) { return lhs.num_cols(); }

  static_assert(details::is_matrix_v<Lhs>,
                "Static conversion requires a matrix operand");
};
}  // namespace details

template <typename Lhs, typename FromType, typename ToType>
struct ExpressionValidator<Operation::StaticConversion<FromType, ToType>, Lhs>
    : std::conditional_t<details::is_scalar_v<Lhs>,
                         details::ScalarStaticConversionExpressionValidator<
                             Lhs, FromType, ToType>,
                         details::MatrixStaticConversionExpressionValidator<
                             Lhs, FromType, ToType>> {
  static constexpr void static_validate() {
    static_assert(std::is_same_v<FromType, typename Lhs::value_type>,
                  "Operand type does not match static conversion type");
    static_assert(std::is_convertible_v<FromType, ToType>,
                  "Static conversion is not possible between types");
  }

  static constexpr void dynamic_validate(const Lhs& /*lhs*/) {}
};

}  // namespace optila