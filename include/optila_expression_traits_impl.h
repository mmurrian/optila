#pragma once

#include <cstdint>
#include <limits>
#include <ratio>
#include <type_traits>

#include "details/optila_expression.h"
#include "details/optila_matrix.h"
#include "details/optila_scalar.h"
#include "details/optila_type_traits.h"
#include "optila_expression_cost_impl.h"
#include "optila_operation_impl.h"

namespace optila {

template <typename Expr>
struct ExpressionTraits;

template <typename ValueType, std::size_t NumRows, std::size_t NumCols,
          typename MatrixPolicy>
struct ExpressionTraits<Matrix<ValueType, NumRows, NumCols, MatrixPolicy>> {
 private:
  using Expr = Matrix<ValueType, NumRows, NumCols, MatrixPolicy>;

 public:
  using expression_type = details::matrix_tag;
  using value_type = typename Expr::value_type;

  constexpr static auto num_rows_compile_time = Expr::num_rows_compile_time;
  constexpr static auto num_cols_compile_time = Expr::num_cols_compile_time;
  constexpr static auto num_rows_hint = Expr::num_rows_hint;
  constexpr static auto num_cols_hint = Expr::num_cols_hint;

  using result_type = Matrix<value_type, num_rows_compile_time,
                             num_cols_compile_time, MatrixPolicy>;

  using operand_coefficient_ratio = std::tuple<>;
  using operation_counts = OperationCounts<>;

  constexpr static auto num_rows() {
    static_assert(num_rows_compile_time != Dynamic);
    return num_rows_compile_time;
  }

  constexpr static auto num_cols() {
    static_assert(num_cols_compile_time != Dynamic);
    return num_cols_compile_time;
  }

  constexpr static void static_validate() {}

  constexpr static void dynamic_validate() {}
};

template <typename ValueType>
struct ExpressionTraits<Scalar<ValueType>> {
 private:
  using Expr = Scalar<ValueType>;

 public:
  using expression_type = details::scalar_tag;
  using value_type = typename Expr::value_type;
  using result_type = Scalar<value_type>;

  using operand_coefficient_ratio = std::tuple<>;
  using operation_counts = OperationCounts<>;

  constexpr static void static_validate() {}

  constexpr static void dynamic_validate() {}
};

template <typename LhsType>
struct ExpressionTraits<Expression<Operation::Transpose, LhsType>> {
 private:
  using Lhs = std::decay_t<LhsType>;

 public:
  using expression_type = details::matrix_tag;
  using value_type = typename Lhs::value_type;

  constexpr static auto num_rows_compile_time = Lhs::num_cols_compile_time;
  constexpr static auto num_cols_compile_time = Lhs::num_rows_compile_time;
  constexpr static auto num_rows_hint = Lhs::num_cols_hint;
  constexpr static auto num_cols_hint = Lhs::num_rows_hint;

  using result_type = Matrix<value_type, num_rows_compile_time,
                             num_cols_compile_time, DefaultMatrixPolicy>;

  // Each operand coefficient is accessed once in evaluation
  using operand_coefficient_ratio = std::tuple<std::ratio<1>, std::ratio<1>>;
  using operation_counts = OperationCounts<0, 0>;

  constexpr static auto num_rows(const Lhs& lhs) { return lhs.num_cols(); }

  constexpr static auto num_cols(const Lhs& lhs) { return lhs.num_rows(); }

  constexpr static void static_validate() {
    static_assert(details::is_matrix_v<Lhs>,
                  "Transpose can only be applied to matrices");
  }

  constexpr static void dynamic_validate(const Lhs& lhs) {}
};

template <typename LhsType, typename RhsType>
struct ExpressionTraits<
    Expression<Operation::ScalarAddition, LhsType, RhsType>> {
 private:
  using Lhs = std::decay_t<LhsType>;
  using Rhs = std::decay_t<RhsType>;

 public:
  using expression_type = details::scalar_tag;
  using value_type = details::common_value_type_t<typename Lhs::value_type,
                                                  typename Rhs::value_type>;
  using result_type = Scalar<value_type>;
  // Each operand coefficient is accessed once in evaluation
  using operand_coefficient_ratio = std::tuple<std::ratio<1>, std::ratio<1>>;
  using operation_counts = OperationCounts<1, 0>;

  constexpr static void static_validate() {
    static_assert(details::is_scalar_v<Lhs> && details::is_scalar_v<Rhs>,
                  "Mismatched operands for scalar addition");
  }

  constexpr static void dynamic_validate(const Lhs& /*lhs*/,
                                         const Rhs& /*rhs*/) {}
};

template <typename LhsType, typename RhsType>
struct ExpressionTraits<Expression<Operation::Addition, LhsType, RhsType>>
    : details::matrix_tag {
 private:
  using Lhs = std::decay_t<LhsType>;
  using Rhs = std::decay_t<RhsType>;

 public:
  using expression_type = details::matrix_tag;
  using value_type = details::common_value_type_t<typename Lhs::value_type,
                                                  typename Rhs::value_type>;

  constexpr static auto num_rows_compile_time = Lhs::num_rows_compile_time;
  constexpr static auto num_cols_compile_time = Lhs::num_cols_compile_time;
  constexpr static auto num_rows_hint = Lhs::num_rows_hint;
  constexpr static auto num_cols_hint = Lhs::num_cols_hint;

  using result_type = Matrix<value_type, num_rows_compile_time,
                             num_cols_compile_time, DefaultMatrixPolicy>;

  // Each operand coefficient is accessed once in evaluation
  using operand_coefficient_ratio = std::tuple<std::ratio<1>, std::ratio<1>>;
  using operation_counts = OperationCounts<num_rows_hint * num_cols_hint, 0>;

  constexpr static auto num_rows(const Lhs& lhs, const Rhs& /*rhs*/) {
    return lhs.num_rows();
  }

  constexpr static auto num_cols(const Lhs& lhs, const Rhs& /*rhs*/) {
    return lhs.num_cols();
  }

  static_assert(details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs>,
                "Mismatched operands for addition");

  constexpr static void static_validate() {
    static_assert(Lhs::num_rows_compile_time == Rhs::num_rows_compile_time &&
                      Lhs::num_cols_compile_time == Rhs::num_cols_compile_time,
                  "Mismatched operands for addition");
  }

  constexpr static void dynamic_validate(const Lhs& lhs, const Rhs& rhs) {
    assert(lhs.num_rows() == rhs.num_rows() &&
           lhs.num_cols() == rhs.num_cols());
  }
};

template <typename LhsType, typename RhsType>
struct ExpressionTraits<
    Expression<Operation::ScalarSubtraction, LhsType, RhsType>> {
 private:
  using Lhs = std::decay_t<LhsType>;
  using Rhs = std::decay_t<RhsType>;

 public:
  using expression_type = details::scalar_tag;
  using value_type = details::common_value_type_t<typename Lhs::value_type,
                                                  typename Rhs::value_type>;
  using result_type = Scalar<value_type>;
  // Each operand coefficient is accessed once in evaluation
  using operand_coefficient_ratio = std::tuple<std::ratio<1>, std::ratio<1>>;
  using operation_counts = OperationCounts<1, 0>;

  static_assert(details::is_scalar_v<Lhs> && details::is_scalar_v<Rhs>,
                "Mismatched operands for scalar subtraction");
};

template <typename LhsType, typename RhsType>
struct ExpressionTraits<Expression<Operation::Subtraction, LhsType, RhsType>>
    : details::matrix_tag {
 private:
  using Lhs = std::decay_t<LhsType>;
  using Rhs = std::decay_t<RhsType>;

 public:
  using expression_type = details::matrix_tag;
  using value_type = details::common_value_type_t<typename Lhs::value_type,
                                                  typename Rhs::value_type>;

  constexpr static auto num_rows_compile_time = Lhs::num_rows_compile_time;
  constexpr static auto num_cols_compile_time = Lhs::num_cols_compile_time;
  constexpr static auto num_rows_hint = Lhs::num_rows_hint;
  constexpr static auto num_cols_hint = Lhs::num_cols_hint;

  using result_type = Matrix<value_type, num_rows_compile_time,
                             num_cols_compile_time, DefaultMatrixPolicy>;

  constexpr static auto num_rows(const Lhs& lhs, const Rhs& /*rhs*/) {
    return lhs.num_rows();
  }

  constexpr static auto num_cols(const Lhs& lhs, const Rhs& /*rhs*/) {
    return lhs.num_cols();
  }

  // Each operand coefficient is accessed once in evaluation
  using operand_coefficient_ratio = std::tuple<std::ratio<1>, std::ratio<1>>;
  using operation_counts = OperationCounts<num_rows_hint * num_cols_hint, 0>;

  static_assert(details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs>,
                "Mismatched operands for subtraction");

  constexpr static void static_validate() {
    static_assert(Lhs::num_rows_compile_time == Rhs::num_rows_compile_time &&
                      Lhs::num_cols_compile_time == Rhs::num_cols_compile_time,
                  "Mismatched operands for subtraction");
  }

  constexpr static void dynamic_validate(const Lhs& lhs, const Rhs& rhs) {
    assert(lhs.num_rows() == rhs.num_rows() &&
           lhs.num_cols() == rhs.num_cols());
  }
};

// Scalar-Matrix multiplication
template <typename LhsType, typename RhsType>
struct ScalarMatrixMultiplicationExpressionTraits {
 private:
  using Lhs = std::decay_t<LhsType>;
  using Rhs = std::decay_t<RhsType>;

 public:
  using expression_type = details::matrix_tag;
  using value_type = details::common_value_type_t<typename Lhs::value_type,
                                                  typename Rhs::value_type>;

  constexpr static auto num_rows_compile_time = Rhs::num_rows_compile_time;
  constexpr static auto num_cols_compile_time = Rhs::num_cols_compile_time;
  constexpr static auto num_rows_hint = Rhs::num_rows_hint;
  constexpr static auto num_cols_hint = Rhs::num_cols_hint;

  using result_type = Matrix<value_type, num_rows_compile_time,
                             num_cols_compile_time, DefaultMatrixPolicy>;

  constexpr static auto num_rows(const Lhs& /*lhs*/, const Rhs& rhs) {
    return rhs.num_rows();
  }

  constexpr static auto num_cols(const Lhs& /*lhs*/, const Rhs& rhs) {
    return rhs.num_cols();
  }

  // Each operand coefficient is accessed once in evaluation
  using operand_coefficient_ratio = std::tuple<std::ratio<1>, std::ratio<1>>;
  using operation_counts = OperationCounts<num_rows_hint * num_cols_hint, 0>;

  static_assert(details::is_scalar_v<Lhs> && details::is_matrix_v<Rhs>,
                "Mismatched operands for scalar-matrix multiplication");

  constexpr static void static_validate() {}

  constexpr static void dynamic_validate(const Lhs& /*lhs*/, const Rhs& rhs) {
    assert(rhs.num_rows() > 0 && rhs.num_cols() > 0);
  }
};

template <typename LhsType, typename RhsType>
struct MatrixScalarMultiplicationExpressionTraits {
 private:
  using Lhs = std::decay_t<LhsType>;
  using Rhs = std::decay_t<RhsType>;

 public:
  using expression_type = details::matrix_tag;
  using value_type = details::common_value_type_t<typename Lhs::value_type,
                                                  typename Rhs::value_type>;

  constexpr static auto num_rows_compile_time = Lhs::num_rows_compile_time;
  constexpr static auto num_cols_compile_time = Lhs::num_cols_compile_time;
  constexpr static auto num_rows_hint = Lhs::num_rows_hint;
  constexpr static auto num_cols_hint = Lhs::num_cols_hint;

  using result_type = Matrix<value_type, num_rows_compile_time,
                             num_cols_compile_time, DefaultMatrixPolicy>;

  constexpr static auto num_rows(const Lhs& lhs, const Rhs& /* rhs */) {
    return lhs.num_rows();
  }

  constexpr static auto num_cols(const Lhs& lhs, const Rhs& /* rhs */) {
    return lhs.num_cols();
  }

  // Each operand coefficient is accessed once in evaluation
  using operand_coefficient_ratio = std::tuple<std::ratio<1>, std::ratio<1>>;
  using operation_counts = OperationCounts<num_rows_hint * num_cols_hint, 0>;

  static_assert(details::is_matrix_v<Lhs> && details::is_scalar_v<Rhs>,
                "Mismatched operands for matrix-scalar multiplication");

  constexpr static void static_validate() {}

  constexpr static void dynamic_validate(const Lhs& lhs, const Rhs& /* rhs */) {
    assert(lhs.num_rows() > 0 && lhs.num_cols() > 0);
  }
};

// Matrix multiplication
template <typename LhsType, typename RhsType>
struct MatrixMultiplicationExpressionTraits {
 private:
  using Lhs = std::decay_t<LhsType>;
  using Rhs = std::decay_t<RhsType>;

 public:
  using expression_type = details::matrix_tag;
  using value_type = details::common_value_type_t<typename Lhs::value_type,
                                                  typename Rhs::value_type>;

  constexpr static auto num_rows_compile_time = Lhs::num_rows_compile_time;
  constexpr static auto num_cols_compile_time = Rhs::num_cols_compile_time;
  constexpr static auto num_rows_hint = Lhs::num_rows_hint;
  constexpr static auto num_cols_hint = Rhs::num_cols_hint;

  using result_type = Matrix<value_type, num_rows_compile_time,
                             num_cols_compile_time, DefaultMatrixPolicy>;

  constexpr static auto num_rows(const Lhs& lhs, const Rhs& /*rhs*/) {
    return lhs.num_rows();
  }
  constexpr static auto num_cols(const Lhs& /*lhs*/, const Rhs& rhs) {
    return rhs.num_cols();
  }

  // For matrix multiplication A * B, where matrix A has dimension M x K and
  // matrix B has dimension K x N, the number of times each coefficient of A is
  // accessed is equal to the number of columns of B. The number of times each
  // coefficient of B is accessed is equal to the number of rows of A.
  using operand_coefficient_ratio =
      std::tuple<std::ratio<num_cols_hint>, std::ratio<num_rows_hint>>;

  // A dot product is performed for each coefficient of the result matrix (M x
  // N) between the corresponding row of A and the corresponding column of B.
  // Since both the row and column have K coefficients, there are (K-1)
  // additions and K multiplications for a total of 2K-1 operations per
  // coefficient of the result matrix.
  constexpr static auto M = num_rows_hint;
  constexpr static auto N = num_cols_hint;
  constexpr static auto K = Lhs::num_cols_hint;
  using operation_counts = OperationCounts<M * N*(K - 1), M * N * K>;

  constexpr static void static_validate() {
    static_assert(details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs>,
                  "Matrix multiplication requires two matrix operands");
    static_assert(Lhs::num_cols_compile_time == Rhs::num_rows_compile_time,
                  "Matrix operand inner dimensions must match");
  }

  constexpr static void dynamic_validate(const Lhs& lhs, const Rhs& rhs) {
    assert(lhs.num_cols() == rhs.num_rows());
  }
};

template <typename LhsType, typename RhsType>
struct ExpressionTraits<Expression<Operation::Multiplication, LhsType, RhsType>>
    : std::conditional_t<
          details::is_matrix_v<LhsType> && details::is_matrix_v<RhsType>,
          MatrixMultiplicationExpressionTraits<LhsType, RhsType>,
          std::conditional_t<
              details::is_scalar_v<LhsType> && details::is_matrix_v<RhsType>,
              ScalarMatrixMultiplicationExpressionTraits<LhsType, RhsType>,
              MatrixScalarMultiplicationExpressionTraits<LhsType, RhsType>>> {};

template <typename LhsType, typename RhsType>
struct ExpressionTraits<
    Expression<Operation::ScalarMultiplication, LhsType, RhsType>> {
 private:
  using Lhs = std::decay_t<LhsType>;
  using Rhs = std::decay_t<RhsType>;

 public:
  using expression_type = details::scalar_tag;
  using value_type = details::common_value_type_t<typename Lhs::value_type,
                                                  typename Rhs::value_type>;
  using result_type = Scalar<value_type>;

  constexpr static void static_validate() {
    static_assert(details::is_scalar_v<Lhs> && details::is_scalar_v<Rhs>,
                  "Mismatched operands for scalar multiplication");
  }

  constexpr static void dynamic_validate(const Lhs& lhs, const Rhs& rhs) {}
};

template <typename LhsType, typename RhsType>
struct ExpressionTraits<
    Expression<Operation::MatrixScalarDivision, LhsType, RhsType>> {
 private:
  using Lhs = std::decay_t<LhsType>;
  using Rhs = std::decay_t<RhsType>;

 public:
  using expression_type = details::matrix_tag;
  using value_type = details::common_value_type_t<typename Lhs::value_type,
                                                  typename Rhs::value_type>;

  constexpr static auto num_rows_compile_time = Lhs::num_rows_compile_time;
  constexpr static auto num_cols_compile_time = Lhs::num_cols_compile_time;
  constexpr static auto num_rows_hint = Lhs::num_rows_hint;
  constexpr static auto num_cols_hint = Lhs::num_cols_hint;

  using result_type = Matrix<value_type, num_rows_compile_time,
                             num_cols_compile_time, DefaultMatrixPolicy>;

  constexpr static auto num_rows(const Lhs& lhs, const Rhs& /*rhs*/) {
    return lhs.num_rows();
  }

  constexpr static auto num_cols(const Lhs& lhs, const Rhs& /*rhs*/) {
    return lhs.num_cols();
  }

  // Each operand coefficient is accessed once in evaluation
  using operand_coefficient_ratio = std::tuple<std::ratio<1>, std::ratio<1>>;
  using operation_counts = OperationCounts<0, 0, num_rows_hint * num_cols_hint>;

  constexpr static void static_validate() {
    static_assert(details::is_matrix_v<Lhs> && details::is_scalar_v<Rhs>,
                  "Mismatched operands for matrix-scalar division");
  }

  constexpr static void dynamic_validate(const Lhs& lhs, const Rhs& rhs) {}
};

// Vector dot product
template <typename LhsType, typename RhsType>
struct ExpressionTraits<Expression<Operation::DotProduct, LhsType, RhsType>> {
 private:
  using Lhs = std::decay_t<LhsType>;
  using Rhs = std::decay_t<RhsType>;

  // These are for the operands of the dot product, not the result.
  constexpr static auto num_rows_compile_time =
      Lhs::num_rows_compile_time != Dynamic ? Lhs::num_rows_compile_time
                                            : Rhs::num_rows_compile_time;
  constexpr static auto num_cols_compile_time =
      Lhs::num_cols_compile_time != Dynamic ? Lhs::num_cols_compile_time
                                            : Rhs::num_cols_compile_time;
  constexpr static auto num_rows_hint =
      num_rows_compile_time != Dynamic
          ? num_rows_compile_time
          : std::min(Lhs::num_rows_hint, Rhs::num_rows_hint);
  constexpr static auto num_cols_hint =
      num_cols_compile_time != Dynamic
          ? num_cols_compile_time
          : std::min(Lhs::num_cols_hint, Rhs::num_cols_hint);

 public:
  using expression_type = details::scalar_tag;
  using value_type = details::common_value_type_t<typename Lhs::value_type,
                                                  typename Rhs::value_type>;
  using result_type = Scalar<value_type>;

  // Each operand coefficient is accessed once in evaluation
  using operand_coefficient_ratio = std::tuple<std::ratio<1>, std::ratio<1>>;
  using operation_counts = OperationCounts<num_rows_hint * num_cols_hint,
                                           num_rows_hint * num_cols_hint - 1>;

  constexpr static void static_validate() {
    static_assert(
        details::is_static_vector_v<Lhs> && details::is_static_vector_v<Rhs>,
        "Dot product requires vector operands");
    static_assert(Lhs::num_rows_compile_time == Rhs::num_rows_compile_time &&
                      Lhs::num_cols_compile_time == Rhs::num_cols_compile_time,
                  "Dot product requires vector operands of the same dimension");
  }

  constexpr static void dynamic_validate(const Lhs& lhs, const Rhs& rhs) {
    assert(lhs.num_rows() == rhs.num_rows() &&
           lhs.num_cols() == rhs.num_cols());
    assert(std::min(lhs.num_rows(), lhs.num_cols()) == 1);
  }
};

// Square root of a scalar
template <typename LhsType>
struct ExpressionTraits<Expression<Operation::SquareRoot, LhsType>> {
 private:
  using Lhs = std::decay_t<LhsType>;

 public:
  using expression_type = details::scalar_tag;
  using value_type = details::common_value_type_t<typename Lhs::value_type>;
  using result_type = Scalar<value_type>;

  static_assert(details::is_scalar_v<Lhs>,
                "Square root requires a scalar operand");

  // Each operand coefficient is accessed once in evaluation
  using operand_coefficient_ratio = std::tuple<std::ratio<1>>;
  using operation_counts = OperationCounts<0, 0, 0, 1>;

  constexpr static void static_validate() {}

  constexpr static void dynamic_validate(const Lhs& /*lhs*/) {}
};

// Submatrix extraction
template <std::size_t StartRow, std::size_t StartCol, std::size_t NumRows,
          std::size_t NumCols, typename LhsType>
struct ExpressionTraits<Expression<
    Operation::SubmatrixExtraction<StartRow, StartCol, NumRows, NumCols>,
    LhsType>> {
 private:
  using Lhs = std::decay_t<LhsType>;

 public:
  using expression_type = details::matrix_tag;
  using value_type = details::common_value_type_t<typename Lhs::value_type>;

  constexpr static auto num_rows_compile_time = NumRows;
  constexpr static auto num_cols_compile_time = NumCols;
  constexpr static auto num_rows_hint = NumRows;
  constexpr static auto num_cols_hint = NumCols;

  using result_type = Matrix<value_type, num_rows_compile_time,
                             num_cols_compile_time, DefaultMatrixPolicy>;

  constexpr static auto num_rows(const Lhs& /*lhs*/) { return NumRows; }
  constexpr static auto num_cols(const Lhs& /*lhs*/) { return NumCols; }

  // In general, not all operand coefficients are accessed in evaluation of a
  // submatrix.
  using operand_coefficient_ratio = std::tuple<std::ratio<
      std::min(Lhs::num_rows_hint* Lhs::num_cols_hint, NumRows* NumCols),
      std::max(Lhs::num_rows_hint* Lhs::num_cols_hint, NumRows* NumCols)>>;
  // Submatrix extraction requires no operations on its own. The cost of
  // evaluating the submatrix is the cost of evaluating the operand.
  using operation_counts = OperationCounts<0, 0, 0, 0>;

  static_assert(details::is_matrix_v<Lhs>,
                "Submatrix extraction requires a "
                "matrix operand");

  constexpr static void static_validate() {
    static_assert(StartRow + NumRows <= Lhs::num_rows_compile_time &&
                      StartCol + NumCols <= Lhs::num_cols_compile_time,
                  "Submatrix extraction out of bounds");
  }

  constexpr static void dynamic_validate(const Lhs& lhs) {
    assert(StartRow + NumRows <= lhs.num_rows() &&
           StartCol + NumCols <= lhs.num_cols());
  }
};

// Strict equality
template <typename LhsType, typename RhsType>
struct ExpressionTraits<
    Expression<Operation::StrictEquality, LhsType, RhsType>> {
 private:
  using Lhs = std::decay_t<LhsType>;
  using Rhs = std::decay_t<RhsType>;

 public:
  using expression_type = details::scalar_tag;
  using value_type = bool;
  using result_type = Scalar<value_type>;

  // Worst case, all coefficients are accessed in evaluation of a strict
  // equality.
  using operand_coefficient_ratio = std::tuple<std::ratio<1>, std::ratio<1>>;
  using operation_counts = OperationCounts<0, 0, 0, 0>;

  static_assert((details::is_scalar_v<Lhs> && details::is_scalar_v<Rhs>) ||
                    (details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs>),
                "Mismatched operands for strict equality");

  constexpr static void static_validate() {
    static_assert(Lhs::num_rows_compile_time == Rhs::num_rows_compile_time &&
                      Lhs::num_cols_compile_time == Rhs::num_cols_compile_time,
                  "Mismatched operands for strict equality");
  }

  constexpr static void dynamic_validate(const Lhs& lhs, const Rhs& rhs) {
    assert(lhs.num_rows() == rhs.num_rows() &&
           lhs.num_cols() == rhs.num_cols());
  }
};

// Static conversion helpers
namespace details {
template <typename Lhs, typename FromType, typename ToType>
struct ScalarStaticConversionExpressionTraits {
  using expression_type = details::scalar_tag;
  using value_type = ToType;
  using result_type = Scalar<value_type>;

  static_assert(details::is_scalar_v<Lhs>,
                "Static conversion requires a scalar operand");
};

template <typename LhsType, typename FromType, typename ToType>
struct MatrixStaticConversionExpressionTraits {
 private:
  using Lhs = std::decay_t<LhsType>;

 public:
  using expression_type = details::matrix_tag;
  using value_type = ToType;

  constexpr static auto num_rows_compile_time = Lhs::num_rows_compile_time;
  constexpr static auto num_cols_compile_time = Lhs::num_cols_compile_time;
  constexpr static auto num_rows_hint = Lhs::num_rows_hint;
  constexpr static auto num_cols_hint = Lhs::num_cols_hint;

  using result_type = Matrix<value_type, num_rows_compile_time,
                             num_cols_compile_time, DefaultMatrixPolicy>;

  constexpr static auto num_rows(const Lhs& lhs) { return lhs.num_rows(); }
  constexpr static auto num_cols(const Lhs& lhs) { return lhs.num_cols(); }

  static_assert(details::is_matrix_v<Lhs>,
                "Static conversion requires a matrix operand");
};
}  // namespace details

template <typename LhsType, typename FromType, typename ToType>
struct ExpressionTraits<
    Expression<Operation::StaticConversion<FromType, ToType>, LhsType>>
    : std::conditional_t<details::is_scalar_v<LhsType>,
                         details::ScalarStaticConversionExpressionTraits<
                             LhsType, FromType, ToType>,
                         details::MatrixStaticConversionExpressionTraits<
                             LhsType, FromType, ToType>> {
 private:
  using Lhs = std::decay_t<LhsType>;

 public:
  // Each operand coefficient is accessed once in evaluation
  using operand_coefficient_ratio = std::tuple<std::ratio<1>, std::ratio<1>>;
  using operation_counts = OperationCounts<0, 0, 0, 0>;

  constexpr static void static_validate() {
    static_assert(std::is_same_v<FromType, typename Lhs::value_type>,
                  "Operand type does not match static conversion type");
    static_assert(std::is_convertible_v<FromType, ToType>,
                  "Static conversion is not possible between types");
  }

  constexpr static void dynamic_validate(const Lhs& /*lhs*/) {}
};

}  // namespace optila