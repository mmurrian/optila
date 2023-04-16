#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace optila {

namespace details {

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

struct operation_state_tag {};

template <typename T>
struct has_operation_state
    : std::is_base_of<details::operation_state_tag, std::decay_t<T>> {};

template <typename T>
inline constexpr bool has_operation_state_v = has_operation_state<T>::value;

struct matrix_tag {};
struct scalar_tag {};

// Type traits for detecting matrix, vector, and scalar types
template <typename T>
struct is_matrix : std::is_base_of<details::matrix_tag, std::decay_t<T>> {};

template <typename T>
inline constexpr bool is_matrix_v = is_matrix<T>::value;

template <typename T>
struct is_row_vector
    : std::conjunction<is_matrix<T>,
                       std::bool_constant<std::decay_t<T>::num_rows == 1>> {};

template <typename T>
inline constexpr bool is_row_vector_v = is_row_vector<T>::value;

template <typename T>
struct is_column_vector
    : std::conjunction<is_matrix<T>,
                       std::bool_constant<std::decay_t<T>::num_cols == 1>> {};

template <typename T>
inline constexpr bool is_column_vector_v = is_column_vector<T>::value;

template <typename T>
struct is_vector : std::disjunction<is_row_vector<T>, is_column_vector<T>> {};

template <typename T>
inline constexpr bool is_vector_v = is_vector<T>::value;

template <typename T>
struct is_scalar
    : std::integral_constant<bool,
                             std::is_base_of_v<scalar_tag, std::decay_t<T>>> {};

template <typename T>
inline constexpr bool is_scalar_v = is_scalar<T>::value;

template <typename T>
struct is_expression : std::disjunction<is_matrix<T>, is_scalar<T>> {};

template <typename T>
inline constexpr bool is_expression_v = is_expression<T>::value;

template <typename Lhs, typename Rhs>
struct is_unary_operand_pair
    : std::conjunction<is_expression<Lhs>, std::is_void<Rhs>> {};

template <typename Lhs, typename Rhs>
inline constexpr bool is_unary_operand_pair_v =
    is_unary_operand_pair<Lhs, Rhs>::value;

template <typename Lhs, typename Rhs>
struct is_binary_operand_pair
    : std::conjunction<is_expression<Lhs>, is_expression<Rhs>> {};

template <typename Lhs, typename Rhs>
inline constexpr bool is_binary_operand_pair_v =
    is_binary_operand_pair<Lhs, Rhs>::value;

template <typename Lhs, typename Rhs>
struct is_either_scalar : std::disjunction<is_scalar<Lhs>, is_scalar<Rhs>> {};

template <typename Lhs, typename Rhs>
inline constexpr bool is_either_scalar_v = is_either_scalar<Lhs, Rhs>::value;

template <typename Lhs, typename Rhs>
struct is_either_matrix : std::disjunction<is_matrix<Lhs>, is_matrix<Rhs>> {};

template <typename Lhs, typename Rhs>
inline constexpr bool is_either_matrix_v = is_either_matrix<Lhs, Rhs>::value;

template <typename Lhs, typename Rhs>
struct is_binary_matrix_pair
    : std::conjunction<is_matrix<Lhs>, is_matrix<Rhs>> {};

template <typename Lhs, typename Rhs>
inline constexpr bool is_binary_matrix_pair_v =
    is_binary_matrix_pair<Lhs, Rhs>::value;

template <typename Lhs, typename Rhs>
struct is_binary_scalar_pair
    : std::conjunction<is_scalar<Lhs>, is_scalar<Rhs>> {};

template <typename Lhs, typename Rhs>
inline constexpr bool is_binary_scalar_pair_v =
    is_binary_scalar_pair<Lhs, Rhs>::value;

template <typename T, typename = void>
struct has_left_operand : std::false_type {};

template <typename T>
struct has_left_operand<T, std::void_t<typename T::left_operand_type>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_left_operand_v = has_left_operand<T>::value;

template <typename T, typename = void>
struct has_right_operand : std::false_type {};

template <typename T>
struct has_right_operand<T, std::void_t<typename T::right_operand_type>>
    : std::true_type {};

template <typename T>
inline constexpr bool has_right_operand_v = has_right_operand<T>::value;

template <typename T, typename = void>
struct has_operand : std::false_type {};

template <typename T>
struct has_operand<T, std::void_t<typename T::operand_type>> : std::true_type {
};

template <typename T>
inline constexpr bool has_operand_v = has_operand<T>::value;

}  // namespace details

enum class StorageOrder {
  RowMajor,
  ColumnMajor,
};

template <typename ValueType, std::size_t NumRows, std::size_t NumCols,
          StorageOrder Order = StorageOrder::RowMajor>
class Matrix : public details::matrix_tag {
 public:
  using value_type = ValueType;
  constexpr static std::size_t num_rows = NumRows;
  constexpr static std::size_t num_cols = NumCols;

  constexpr Matrix() = default;

  // Constructor for nested-brace syntax
  constexpr Matrix(
      std::initializer_list<std::initializer_list<value_type>> init)
      : data_{} {
    std::size_t i = 0;
    for (const auto& row : init) {
      std::size_t j = 0;
      for (const auto& elem : row) {
        (*this)(i, j++) = elem;
      }
      ++i;
    }
  }

  [[nodiscard]] constexpr value_type& operator()(std::size_t i, std::size_t j) {
    return data_[linear_index(i, j)];
  }

  // Return by value or const reference depending on the size and other
  // characteristics of the value type
  using const_value_type = details::return_by_value_or_const_ref_t<value_type>;
  [[nodiscard]] constexpr const_value_type operator()(std::size_t i,
                                                      std::size_t j) const {
    return data_[linear_index(i, j)];
  }

  [[nodiscard]] constexpr value_type* data() noexcept { return data_.data(); }

  [[nodiscard]] constexpr const value_type* data() const noexcept {
    return data_.data();
  }

 private:
  std::array<value_type, num_rows * num_cols> data_;

  [[nodiscard]] constexpr std::size_t linear_index(std::size_t i,
                                                   std::size_t j) const {
    if constexpr (Order == StorageOrder::RowMajor) {
      return i * num_cols + j;
    } else {  // StorageOrder::ColumnMajor
      return i + j * num_rows;
    }
  }
};

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

template <typename T, typename... Args>
constexpr Scalar<T> make_scalar(Args&&... args) {
  return Scalar<T>(std::forward<Args>(args)...);
}

template <typename Op, typename Lhs = void, typename Rhs = void,
          typename ExprType = void>
class Expression;

namespace details {
template <typename T>
struct is_expression_literal : std::false_type {};

template <typename Op, typename Lhs, typename Rhs, typename ExprType>
struct is_expression_literal<Expression<Op, Lhs, Rhs, ExprType>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_expression_literal_v = is_expression_literal<T>::value;

template <typename T>
struct is_matrix_literal : std::false_type {};

template <typename ValueType, std::size_t NumRows, std::size_t NumCols,
          StorageOrder Order>
struct is_matrix_literal<Matrix<ValueType, NumRows, NumCols, Order>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_matrix_literal_v = is_matrix_literal<T>::value;

template <typename T>
struct is_scalar_literal : std::false_type {};

template <typename ValueType>
struct is_scalar_literal<Scalar<ValueType>> : std::true_type {};

template <typename T>
inline constexpr bool is_scalar_literal_v = is_scalar_literal<T>::value;

template <typename T>
struct is_matrix_or_scalar_literal
    : std::integral_constant<bool, is_matrix_literal_v<T> ||
                                       is_scalar_literal_v<T>> {};

template <typename T>
inline constexpr bool is_matrix_or_scalar_literal_v =
    is_matrix_or_scalar_literal<T>::value;

}  // namespace details

namespace Operation {

// Operation state template structure, specialized for operations with and
// without state
template <typename Op, typename LeftState, typename RightState,
          bool has_left_state = !std::is_void_v<LeftState>,
          bool has_right_state = !std::is_void_v<RightState>,
          bool has_operation_state = details::has_operation_state_v<Op>>
struct State;

template <typename Op, typename LeftState, typename RightState>
struct State<Op, LeftState, RightState, false, false, false> {};

template <typename Op, typename LeftState, typename RightState>
struct State<Op, LeftState, RightState, true, false, false> {
  LeftState leftState;
};

template <typename Op, typename LeftState, typename RightState>
struct State<Op, LeftState, RightState, false, true, false> {
  RightState rightState;
};

template <typename Op, typename LeftState, typename RightState>
struct State<Op, LeftState, RightState, true, true, false> {
  LeftState leftState;
  RightState rightState;
};

template <typename Op, typename LeftState, typename RightState>
struct State<Op, LeftState, RightState, false, false, true> {
  typename Op::State state;
};

template <typename Op, typename LeftState, typename RightState>
struct State<Op, LeftState, RightState, true, false, true> {
  LeftState leftState;
  typename Op::State state;
};

template <typename Op, typename LeftState, typename RightState>
struct State<Op, LeftState, RightState, false, true, true> {
  RightState rightState;
  typename Op::State state;
};

template <typename Op, typename LeftState, typename RightState>
struct State<Op, LeftState, RightState, true, true, true> {
  LeftState leftState;
  RightState rightState;
  typename Op::State state;
};

// Utility function to create the operation state structure
template <typename Expr, bool has_operand = details::has_operand<Expr>::value,
          bool has_left_operand = details::has_left_operand<Expr>::value,
          bool has_right_operand = details::has_right_operand<Expr>::value>
struct BuildState {
  using type = void;
};

template <typename Expr>
struct BuildState<Expr, false, true, true> {
  using type =
      State<typename Expr::operation_type,
            typename BuildState<typename Expr::left_operand_type>::type,
            typename BuildState<typename Expr::right_operand_type>::type>;
};

template <typename Expr>
struct BuildState<Expr, true, false, false> {
  using type =
      State<typename Expr::operation_type,
            typename BuildState<typename Expr::operand_type>::type, void>;
};

template <typename Expr, typename State>
constexpr decltype(auto) evalLeftScalarOperand(Expr&& expr, State&& state) {
  if constexpr (details::is_expression_literal_v<
                    typename std::decay_t<Expr>::left_operand_type>)
    return expr.leftOperand()(state.leftState);
  else
    return expr.leftOperand()();
}

template <typename Expr, typename State>
constexpr decltype(auto) evalRightScalarOperand(Expr&& expr, State&& state) {
  if constexpr (details::is_expression_literal_v<
                    typename std::decay_t<Expr>::right_operand_type>)
    return expr.rightOperand()(state.rightState);
  else
    return expr.rightOperand()();
}

template <typename Expr, typename State>
constexpr decltype(auto) evalScalarOperand(Expr&& expr, State&& state) {
  if constexpr (details::is_expression_literal_v<
                    typename std::decay_t<Expr>::operand_type>)
    return expr.operand()(state.leftState);
  else
    return expr.operand()();
}

template <typename Expr, typename State>
constexpr decltype(auto) evalLeftMatrixOperand(std::size_t i, std::size_t j,
                                               Expr&& expr, State&& state) {
  if constexpr (details::is_expression_literal_v<
                    typename std::decay_t<Expr>::left_operand_type>)
    return expr.leftOperand()(i, j, state.leftState);
  else
    return expr.leftOperand()(i, j);
}

template <typename Expr, typename State>
constexpr decltype(auto) evalRightMatrixOperand(std::size_t i, std::size_t j,
                                                Expr&& expr, State&& state) {
  if constexpr (details::is_expression_literal_v<
                    typename std::decay_t<Expr>::right_operand_type>)
    return expr.rightOperand()(i, j, state.rightState);
  else
    return expr.rightOperand()(i, j);
}

template <typename Expr, typename State>
constexpr decltype(auto) evalMatrixOperand(std::size_t i, std::size_t j,
                                           Expr&& expr, State&& state) {
  if constexpr (details::is_expression_literal_v<
                    typename std::decay_t<Expr>::operand_type>)
    return expr.operand()(i, j, state.leftState);
  else
    return expr.operand()(i, j);
}

struct Addition {
  static constexpr auto apply_scalar = [](auto&& expr, auto&& state) {
    return evalLeftScalarOperand(expr, state) +
           evalRightScalarOperand(expr, state);
  };
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr, auto&& state) {
    return evalLeftMatrixOperand(i, j, expr, state) +
           evalRightMatrixOperand(i, j, expr, state);
  };
};
struct Subtraction {
  static constexpr auto apply_scalar = [](auto&& expr, auto&& state) {
    return evalLeftScalarOperand(expr, state) -
           evalRightScalarOperand(expr, state);
  };
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr, auto&& state) {
    return evalLeftMatrixOperand(i, j, expr, state) -
           evalRightMatrixOperand(i, j, expr, state);
  };
};
// Matrix multiplication
struct Multiplication {
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr, auto&& state) {
    using value_type = typename std::decay_t<decltype(expr)>::value_type;
    value_type result = 0;
    for (std::size_t k = 0;
         k < std::decay_t<decltype(expr.leftOperand())>::num_cols; ++k) {
      result += evalLeftMatrixOperand(i, k, expr, state) *
                evalRightMatrixOperand(k, j, expr, state);
    }
    return result;
  };
};
struct ScalarMultiplication {
  static constexpr auto apply_scalar = [](auto&& expr, auto&& state) {
    return evalLeftScalarOperand(expr, state) *
           evalRightScalarOperand(expr, state);
  };
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr, auto&& state) {
    constexpr bool lhs_is_scalar =
        details::is_scalar_v<decltype(expr.leftOperand())>;
    constexpr bool rhs_is_scalar =
        details::is_scalar_v<decltype(expr.rightOperand())>;
    static_assert(lhs_is_scalar != rhs_is_scalar,
                  "One of the arguments must be a scalar");
    if constexpr (lhs_is_scalar) {
      return evalLeftScalarOperand(expr, state) *
             evalRightMatrixOperand(i, j, expr, state);
    } else {  // rhs_is_scalar
      return evalLeftMatrixOperand(i, j, expr, state) *
             evalRightScalarOperand(expr, state);
    }
  };
};
struct ScalarDivision {
  static constexpr auto apply_scalar = [](auto&& expr, auto&& state) {
    return evalLeftScalarOperand(expr, state) /
           evalRightScalarOperand(expr, state);
  };
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr, auto&& state) {
    constexpr bool rhs_is_scalar =
        details::is_scalar_v<decltype(expr.rightOperand())>;
    static_assert(rhs_is_scalar, "The divisor must be a scalar");

    return evalLeftMatrixOperand(i, j, expr, state) /
           evalRightScalarOperand(expr, state);
  };
};
struct DotProduct {
  static constexpr auto apply_scalar = [](auto&& expr, auto&& state) {
    using value_type = typename std::decay_t<decltype(expr)>::value_type;
    value_type result = 0;
    for (std::size_t i = 0;
         i < std::decay_t<decltype(expr.leftOperand())>::num_rows; ++i) {
      result += evalLeftMatrixOperand(i, 0, expr, state) *
                evalRightMatrixOperand(i, 0, expr, state);
    }
    return result;
  };
};
struct CrossProduct {};
struct OuterProduct {};
struct Transpose {
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr, auto&& state) {
    return evalMatrixOperand(j, i, expr, state);
  };
};

struct Determinant {};
struct Trace {};
struct Inverse {};
struct Adjoint {};
struct Cofactor {};
struct Rank {};
template <std::size_t StartRow, std::size_t StartCol, std::size_t NumRows,
          std::size_t NumCols>
struct SubmatrixExtraction {
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr, auto&& state) {
    return expr.operand()(i + StartRow, j + StartCol, state.leftState);
  };
};
struct Concatenation {};
struct SquareRoot {
  // FIXME: sqrt is not constexpr
  template <typename Expr, typename State>
  static auto apply_scalar(Expr&& expr, State&& state) {
    return sqrt(expr.operand()(state.leftState));
  }
};
struct Norm {};
struct Normalization : public details::operation_state_tag {
  struct State {
    double norm;
  };

  template <typename Expr, typename State>
  static constexpr void precompute(Expr&& expr, State&& state) {
    // TODO: Ideally, this could be implemented after Expression and 'evaluate'
    // so that we can use the Norm operation on expr.operand().
    state.state.norm = 0;
    for (std::size_t i = 0; i < std::decay_t<Expr>::num_rows; ++i) {
      for (std::size_t j = 0; j < std::decay_t<Expr>::num_cols; ++j) {
        state.state.norm += evalMatrixOperand(i, j, expr, state) *
                            evalMatrixOperand(i, j, expr, state);
      }
    }
    state.state.norm = sqrt(state.state.norm);
  }

  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr, auto&& state) {
    return evalMatrixOperand(i, j, expr, state) / state.state.norm;
  };
};
struct ElementWiseOperation {};

// Fill a matrix with a constant value
template <std::size_t NumRows, std::size_t NumCols>
struct ConstantMatrix {
  static constexpr auto apply_matrix = [](std::size_t /*i*/, std::size_t /*j*/,
                                          auto&& expr) {
    return expr.operand()();
  };
};
// Fill a diagonal matrix with a constant value
template <std::size_t NumRows, std::size_t NumCols>
struct ConstantDiagonal {
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr) {
    using value_type = typename std::decay_t<decltype(expr)>::value_type;
    return i == j ? expr.operand()() : value_type{};
  };
};
// Put a vector on the diagonal of a matrix
struct DiagonalFromVector {
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr) {
    using value_type = typename std::decay_t<decltype(expr)>::value_type;
    constexpr bool lhs_is_row_vector =
        details::is_row_vector_v<decltype(expr.operand())>;
    constexpr bool lhs_is_column_vector =
        details::is_column_vector_v<decltype(expr.operand())>;
    static_assert(lhs_is_row_vector != lhs_is_column_vector,
                  "The operand must be a vector");
    if constexpr (lhs_is_row_vector) {
      return i == j ? expr.operand()(0, j) : value_type{};
    } else if constexpr (lhs_is_column_vector) {
      return i == j ? expr.operand()(i, 0) : value_type{};
    }
  };
};
// Extract the diagonal of a matrix into a vector
struct DiagonalToVector {
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr) {
    constexpr bool is_row_vector = details::is_row_vector_v<decltype(expr)>;
    constexpr bool is_column_vector =
        details::is_column_vector_v<decltype(expr)>;
    static_assert(is_row_vector != is_column_vector,
                  "The result must be a vector");
    if constexpr (is_row_vector) {
      return expr.operand()(j, j);
    } else if constexpr (is_column_vector) {
      return expr.operand()(i, i);
    }
  };
};
// Extract the diagonal of a matrix into a diagonal matrix
struct DiagonalMatrix {
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr) {
    using value_type = typename std::decay_t<decltype(expr)>::value_type;
    return i == j ? expr.operand()(i, i) : value_type{};
  };
};

}  // namespace Operation

template <typename Op, typename Lhs = void, typename Rhs = void,
          typename Enable = void>
struct ResultValueType {
  using type = details::common_type_if_not_void_t<
      details::expr_value_type_if_not_void_t<Lhs>,
      details::expr_value_type_if_not_void_t<Rhs>, std::is_void_v<Rhs>>;
};

template <typename Op, typename Lhs = void, typename Rhs = void,
          typename Enable = void>
struct ResultNumRows;

template <typename Op, typename Lhs = void, typename Rhs = void,
          typename Enable = void>
struct ResultNumCols;

// === Binary matrix operations
template <typename Op, typename Lhs, typename Rhs>
struct ResultNumRows<
    Op, Lhs, Rhs,
    std::enable_if_t<details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs>>>
    : std::integral_constant<std::size_t, std::decay_t<Lhs>::num_rows> {
  static_assert(std::decay_t<Lhs>::num_rows == std::decay_t<Rhs>::num_rows,
                "Matrix dimensions must match");
};

template <typename Op, typename Lhs, typename Rhs>
struct ResultNumCols<
    Op, Lhs, Rhs,
    std::enable_if_t<details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs>>>
    : std::integral_constant<std::size_t, std::decay_t<Lhs>::num_cols> {
  static_assert(std::decay_t<Lhs>::num_cols == std::decay_t<Rhs>::num_cols,
                "Matrix dimensions must match");
};

// === Unary matrix operations
template <typename Op, typename Lhs>
struct ResultNumRows<Op, Lhs, void, std::enable_if_t<details::is_matrix_v<Lhs>>>
    : std::integral_constant<std::size_t, std::decay_t<Lhs>::num_rows> {};

template <typename Op, typename Lhs>
struct ResultNumCols<Op, Lhs, void, std::enable_if_t<details::is_matrix_v<Lhs>>>
    : std::integral_constant<std::size_t, std::decay_t<Lhs>::num_cols> {};

// === Matrix-Scalar operations
template <typename Op, typename Lhs, typename Rhs>
struct ResultNumRows<
    Op, Lhs, Rhs,
    std::enable_if_t<details::is_matrix_v<Lhs> && details::is_scalar_v<Rhs>>>
    : std::integral_constant<std::size_t, std::decay_t<Lhs>::num_rows> {};

template <typename Op, typename Lhs, typename Rhs>
struct ResultNumCols<
    Op, Lhs, Rhs,
    std::enable_if_t<details::is_matrix_v<Lhs> && details::is_scalar_v<Rhs>>>
    : std::integral_constant<std::size_t, std::decay_t<Lhs>::num_cols> {};

// === Scalar-Matrix operations
template <typename Op, typename Lhs, typename Rhs>
struct ResultNumRows<
    Op, Lhs, Rhs,
    std::enable_if_t<details::is_scalar_v<Lhs> && details::is_matrix_v<Rhs>>>
    : std::integral_constant<std::size_t, std::decay_t<Rhs>::num_rows> {};

template <typename Op, typename Lhs, typename Rhs>
struct ResultNumCols<
    Op, Lhs, Rhs,
    std::enable_if_t<details::is_scalar_v<Lhs> && details::is_matrix_v<Rhs>>>
    : std::integral_constant<std::size_t, std::decay_t<Rhs>::num_cols> {};

// === Binary scalar operations
// Scalar operations do not define ResultNumRows or ResultNumCols

// === Unary scalar operations
// Scalar operations do not define ResultNumRows or ResultNumCols

// === Specialized operations
// Matrix multiplication
template <typename Lhs, typename Rhs>
struct ResultNumRows<
    Operation::Multiplication, Lhs, Rhs,
    std::enable_if_t<details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs>>>
    : std::integral_constant<std::size_t, std::decay_t<Lhs>::num_rows> {
  static_assert(std::decay_t<Lhs>::num_cols == std::decay_t<Rhs>::num_rows,
                "Inner matrix dimensions must match");
};

template <typename Lhs, typename Rhs>
struct ResultNumCols<
    Operation::Multiplication, Lhs, Rhs,
    std::enable_if_t<details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs>>>
    : std::integral_constant<std::size_t, std::decay_t<Rhs>::num_cols> {};

// Vector to diagonal matrix
template <typename Lhs>
struct ResultNumRows<Operation::DiagonalFromVector, Lhs, void,
                     std::enable_if_t<details::is_vector_v<Lhs>>>
    : std::integral_constant<std::size_t,
                             std::max(std::decay_t<Lhs>::num_rows,
                                      std::decay_t<Lhs>::num_cols)> {};

template <typename Lhs>
struct ResultNumCols<Operation::DiagonalFromVector, Lhs, void,
                     std::enable_if_t<details::is_vector_v<Lhs>>>
    : std::integral_constant<std::size_t,
                             std::max(std::decay_t<Lhs>::num_rows,
                                      std::decay_t<Lhs>::num_cols)> {};

// Diagonal of a matrix to vector
template <typename Lhs>
struct ResultNumRows<Operation::DiagonalToVector, Lhs, void,
                     std::enable_if_t<details::is_matrix_v<Lhs>>>
    : std::integral_constant<std::size_t,
                             std::min(std::decay_t<Lhs>::num_rows,
                                      std::decay_t<Lhs>::num_cols)> {};

template <typename Lhs>
struct ResultNumCols<Operation::DiagonalToVector, Lhs, void,
                     std::enable_if_t<details::is_matrix_v<Lhs>>>
    : std::integral_constant<std::size_t, 1> {};

// Diagonal of a matrix to a diagonal matrix
template <typename Lhs>
struct ResultNumRows<Operation::DiagonalMatrix, Lhs, void,
                     std::enable_if_t<details::is_matrix_v<Lhs>>>
    : std::integral_constant<std::size_t,
                             std::min(std::decay_t<Lhs>::num_rows,
                                      std::decay_t<Lhs>::num_cols)> {};

template <typename Lhs>
struct ResultNumCols<Operation::DiagonalMatrix, Lhs, void,
                     std::enable_if_t<details::is_matrix_v<Lhs>>>
    : std::integral_constant<std::size_t,
                             std::min(std::decay_t<Lhs>::num_rows,
                                      std::decay_t<Lhs>::num_cols)> {};

// Constant matrix
template <std::size_t NumRows, std::size_t NumCols, typename Operand>
struct ResultNumRows<Operation::ConstantMatrix<NumRows, NumCols>, Operand, void,
                     std::enable_if_t<details::is_scalar_v<Operand>>>
    : std::integral_constant<std::size_t, NumRows> {};

template <std::size_t NumRows, std::size_t NumCols, typename Operand>
struct ResultNumCols<Operation::ConstantMatrix<NumRows, NumCols>, Operand, void,
                     std::enable_if_t<details::is_scalar_v<Operand>>>
    : std::integral_constant<std::size_t, NumCols> {};

// Constant diagonal matrix
template <std::size_t NumRows, std::size_t NumCols, typename Operand>
struct ResultNumRows<Operation::ConstantDiagonal<NumRows, NumCols>, Operand,
                     void, std::enable_if_t<details::is_scalar_v<Operand>>>
    : std::integral_constant<std::size_t, NumRows> {};

template <std::size_t NumRows, std::size_t NumCols, typename Operand>
struct ResultNumCols<Operation::ConstantDiagonal<NumRows, NumCols>, Operand,
                     void, std::enable_if_t<details::is_scalar_v<Operand>>>
    : std::integral_constant<std::size_t, NumCols> {};

// Matrix transposition
template <typename Operand>
struct ResultNumRows<Operation::Transpose, Operand, void,
                     std::enable_if_t<details::is_matrix_v<Operand>>>
    : std::integral_constant<std::size_t, std::decay_t<Operand>::num_cols> {};

template <typename Operand>
struct ResultNumCols<Operation::Transpose, Operand, void,
                     std::enable_if_t<details::is_matrix_v<Operand>>>
    : std::integral_constant<std::size_t, std::decay_t<Operand>::num_rows> {};

// Submatrix extraction
template <std::size_t StartRow, std::size_t StartCol, std::size_t NumRows,
          std::size_t NumCols, typename Operand>
struct ResultNumRows<
    Operation::SubmatrixExtraction<StartRow, StartCol, NumRows, NumCols>,
    Operand, void, std::enable_if_t<details::is_matrix_v<Operand>>>
    : std::integral_constant<std::size_t, NumRows> {
  static_assert(StartRow + NumRows <= std::decay_t<Operand>::num_rows,
                "Submatrix extraction out of bounds");
};

template <std::size_t StartRow, std::size_t StartCol, std::size_t NumRows,
          std::size_t NumCols, typename Operand>
struct ResultNumCols<
    Operation::SubmatrixExtraction<StartRow, StartCol, NumRows, NumCols>,
    Operand, void, std::enable_if_t<details::is_matrix_v<Operand>>>
    : std::integral_constant<std::size_t, NumCols> {
  static_assert(StartCol + NumCols <= std::decay_t<Operand>::num_cols,
                "Submatrix extraction out of bounds");
};

namespace details {

template <typename Expr, typename = std::void_t<>>
struct has_left_operand_expression : std::false_type {};

template <typename Expr>
struct has_left_operand_expression<
    Expr, std::void_t<typename std::decay_t<Expr>::left_operand_type>>
    : std::conditional_t<is_expression_literal_v<std::decay_t<
                             typename std::decay_t<Expr>::left_operand_type>>,
                         std::true_type, std::false_type> {};

template <typename Expr>
inline constexpr bool has_left_operand_expression_v =
    has_left_operand_expression<Expr>::value;

template <typename Expr, typename = std::void_t<>>
struct has_right_operand_expression : std::false_type {};

template <typename Expr>
struct has_right_operand_expression<
    Expr, std::void_t<typename std::decay_t<Expr>::right_operand_type>>
    : std::conditional_t<is_expression_literal_v<std::decay_t<
                             typename std::decay_t<Expr>::right_operand_type>>,
                         std::true_type, std::false_type> {};

template <typename Expr>
inline constexpr bool has_right_operand_expression_v =
    has_right_operand_expression<Expr>::value;

template <typename Expr, typename = std::void_t<>>
struct has_operand_expression : std::false_type {};

template <typename Expr>
struct has_operand_expression<
    Expr, std::void_t<typename std::decay_t<Expr>::operand_type>>
    : std::conditional_t<is_expression_literal_v<std::decay_t<
                             typename std::decay_t<Expr>::operand_type>>,
                         std::true_type, std::false_type> {};

template <typename Expr>
inline constexpr bool has_operand_expression_v =
    has_operand_expression<Expr>::value;

}  // namespace details

template <typename Expr, typename State>
constexpr void precompute(const Expr& expr, State& state) {
  static_assert(details::is_expression_literal_v<Expr>,
                "Expression literal expected");
  if constexpr (details::has_left_operand_expression_v<Expr>)
    precompute(expr.leftOperand(), state.leftState);
  if constexpr (details::has_right_operand_expression_v<Expr>)
    precompute(expr.rightOperand(), state.rightState);
  if constexpr (details::has_operand_expression_v<Expr>)
    precompute(expr.operand(), state.leftState);
  if constexpr (details::has_operation_state_v<typename Expr::operation_type>) {
    Expr::operation_type::precompute(expr, state);
  }
}

template <typename Expr, StorageOrder Order = StorageOrder::RowMajor,
          typename = std::enable_if_t<details::is_matrix_v<Expr>>>
constexpr auto evaluate(const Expr& expr) {
  using StateType = typename Operation::BuildState<std::decay_t<Expr>>::type;
  StateType state{};
  precompute(expr, state);

  Matrix<typename Expr::value_type, Expr::num_rows, Expr::num_cols, Order>
      result{};
  for (std::size_t i = 0; i < Expr::num_rows; ++i) {
    for (std::size_t j = 0; j < Expr::num_cols; ++j) {
      result(i, j) = expr(i, j, state);
    }
  }
  return result;
}

template <typename Expr,
          typename = std::enable_if_t<details::is_scalar_v<Expr>>>
constexpr decltype(auto) evaluate(const Expr& expr) {
  using StateType = typename Operation::BuildState<std::decay_t<Expr>>::type;
  StateType state{};
  precompute(expr, state);

  return expr(state);
}

template <typename Op, typename Lhs, typename Rhs = void>
struct ExpressionValidator {
  static constexpr bool value = false;  // Default to invalid expression
};

template <typename Lhs, typename Rhs>
struct ExpressionValidator<Operation::Addition, Lhs, Rhs> {
  static constexpr bool value =
      (details::is_scalar_v<Lhs> && details::is_scalar_v<Rhs>) ||
      (details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs> &&
       std::decay_t<Lhs>::num_rows == std::decay_t<Rhs>::num_rows &&
       std::decay_t<Lhs>::num_cols == std::decay_t<Rhs>::num_cols);
};

template <typename Lhs, typename Rhs>
struct ExpressionValidator<Operation::Multiplication, Lhs, Rhs> {
  static constexpr bool value =
      std::decay_t<Lhs>::num_cols == std::decay_t<Rhs>::num_rows;
};

template <typename Lhs, typename Rhs>
struct ExpressionValidator<Operation::DotProduct, Lhs, Rhs> {
  static constexpr bool value =
      details::is_vector_v<Lhs> && details::is_vector_v<Rhs> &&
      (std::decay_t<Lhs>::num_rows == std::decay_t<Rhs>::num_rows &&
       std::decay_t<Lhs>::num_cols == std::decay_t<Rhs>::num_cols);
};

// Binary matrix-matrix, matrix-scalar, and scalar-matrix expressions
template <typename Op, typename Lhs, typename Rhs>
class Expression<
    Op, Lhs, Rhs,
    typename std::enable_if_t<details::is_binary_operand_pair_v<Lhs, Rhs> &&
                                  details::is_either_matrix_v<Lhs, Rhs>,
                              details::matrix_tag>>
    : public details::matrix_tag {
 public:
  static_assert(ExpressionValidator<Op, Lhs, Rhs>::value,
                "Invalid expression!");

  using value_type = typename ResultValueType<Op, Lhs, Rhs>::type;
  using left_operand_type = std::decay_t<Lhs>;
  using right_operand_type = std::decay_t<Rhs>;
  using operation_type = Op;
  using state_type = typename Operation::BuildState<Expression>::type;

  static constexpr std::size_t num_rows = ResultNumRows<Op, Lhs, Rhs>::value;
  static constexpr std::size_t num_cols = ResultNumCols<Op, Lhs, Rhs>::value;

  constexpr Expression(Lhs&& lhs, Rhs&& rhs)
      : lhs_(std::forward<Lhs>(lhs)), rhs_(std::forward<Rhs>(rhs)) {}

  constexpr decltype(auto) operator()(std::size_t i, std::size_t j,
                                      const state_type& state) const {
    return Op::apply_matrix(i, j, *this, state);
  }

  constexpr const auto& leftOperand() const { return lhs_; }
  constexpr const auto& rightOperand() const { return rhs_; }

 private:
  details::store_by_value_or_const_ref_t<Lhs> lhs_;
  details::store_by_value_or_const_ref_t<Rhs> rhs_;
};

// Binary scalar-scalar expressions
template <typename Op, typename Lhs, typename Rhs>
class Expression<
    Op, Lhs, Rhs,
    typename std::enable_if_t<details::is_binary_scalar_pair_v<Lhs, Rhs>,
                              details::scalar_tag>>
    : public details::scalar_tag {
 public:
  static_assert(ExpressionValidator<Op, Lhs, Rhs>::value,
                "Invalid expression!");

  using value_type = typename ResultValueType<Op, Lhs, Rhs>::type;
  using left_operand_type = std::decay_t<Lhs>;
  using right_operand_type = std::decay_t<Rhs>;
  using operation_type = Op;
  using state_type = typename Operation::BuildState<Expression>::type;

  constexpr Expression(Lhs&& lhs, Rhs&& rhs)
      : lhs_(std::forward<Lhs>(lhs)), rhs_(std::forward<Rhs>(rhs)) {}

  constexpr decltype(auto) operator()(const state_type& state) const {
    return Op::apply_scalar(*this, state);
  }

  constexpr const auto& leftOperand() const { return lhs_; }
  constexpr const auto& rightOperand() const { return rhs_; }

 private:
  details::store_by_value_or_const_ref_t<Lhs> lhs_;
  details::store_by_value_or_const_ref_t<Rhs> rhs_;
};

// Binary scalar-producing matrix-matrix, matrix-scalar, and scalar-matrix
// expressions
template <typename Op, typename Lhs, typename Rhs>
class Expression<
    Op, Lhs, Rhs,
    typename std::enable_if_t<details::is_binary_operand_pair_v<Lhs, Rhs> &&
                                  details::is_either_matrix_v<Lhs, Rhs>,
                              details::scalar_tag>>
    : public details::scalar_tag {
 public:
  static_assert(ExpressionValidator<Op, Lhs, Rhs>::value,
                "Invalid expression!");

  using value_type = typename ResultValueType<Op, Lhs, Rhs>::type;
  using left_operand_type = std::decay_t<Lhs>;
  using right_operand_type = std::decay_t<Rhs>;
  using operation_type = Op;
  using state_type = typename Operation::BuildState<Expression>::type;

  constexpr Expression(Lhs&& lhs, Rhs&& rhs)
      : lhs_(std::forward<Lhs>(lhs)), rhs_(std::forward<Rhs>(rhs)) {}

  constexpr decltype(auto) operator()(const state_type& state) const {
    return Op::apply_scalar(*this, state);
  }

  constexpr const auto& leftOperand() const { return lhs_; }
  constexpr const auto& rightOperand() const { return rhs_; }

 private:
  details::store_by_value_or_const_ref_t<Lhs> lhs_;
  details::store_by_value_or_const_ref_t<Rhs> rhs_;
};

// Unary matrix expressions
template <typename Op, typename Lhs>
class Expression<
    Op, Lhs, void,
    typename std::enable_if_t<details::is_matrix_v<Lhs>, details::matrix_tag>>
    : public details::matrix_tag {
 public:
  static_assert(ExpressionValidator<Op, Lhs>::value, "Invalid expression!");

  using value_type = typename ResultValueType<Op, Lhs>::type;
  using operand_type = std::decay_t<Lhs>;
  using operation_type = Op;
  using state_type = typename Operation::BuildState<Expression>::type;
  static constexpr std::size_t num_rows = ResultNumRows<Op, Lhs>::value;
  static constexpr std::size_t num_cols = ResultNumCols<Op, Lhs>::value;

  constexpr explicit Expression(Lhs&& lhs) : lhs_(std::forward<Lhs>(lhs)) {}

  constexpr decltype(auto) operator()(std::size_t i, std::size_t j,
                                      const state_type& state) const {
    return Op::apply_matrix(i, j, *this, state);
  }

  constexpr const auto& operand() const { return lhs_; }

 private:
  details::store_by_value_or_const_ref_t<Lhs> lhs_;
};

// Unary scalar-producing matrix expressions
template <typename Op, typename Lhs>
class Expression<
    Op, Lhs, void,
    typename std::enable_if_t<details::is_matrix_v<Lhs>, details::scalar_tag>>
    : public details::scalar_tag {
 public:
  static_assert(ExpressionValidator<Op, Lhs>::value, "Invalid expression!");

  using value_type = typename ResultValueType<Op, Lhs>::type;
  using operand_type = std::decay_t<Lhs>;
  using operation_type = Op;
  using state_type = typename Operation::BuildState<Expression>::type;

  constexpr explicit Expression(Lhs&& lhs) : lhs_(std::forward<Lhs>(lhs)) {}

  constexpr decltype(auto) operator()(const state_type& state) const {
    return Op::apply_scalar(*this, state);
  }

  constexpr const auto& operand() const { return lhs_; }

 private:
  details::store_by_value_or_const_ref_t<Lhs> lhs_;
};

// Unary matrix-producing scalar expressions
template <typename Op, typename Lhs>
class Expression<
    Op, Lhs, void,
    typename std::enable_if_t<details::is_scalar_v<Lhs>, details::matrix_tag>>
    : public details::matrix_tag {
 public:
  static_assert(ExpressionValidator<Op, Lhs>::value, "Invalid expression!");

  using value_type = typename ResultValueType<Op, Lhs>::type;
  using operand_type = std::decay_t<Lhs>;
  using operation_type = Op;
  using state_type = typename Operation::BuildState<Expression>::type;
  static constexpr std::size_t num_rows = ResultNumRows<Op, Lhs>::value;
  static constexpr std::size_t num_cols = ResultNumCols<Op, Lhs>::value;

  constexpr explicit Expression(Lhs&& lhs) : lhs_(std::forward<Lhs>(lhs)) {}

  constexpr decltype(auto) operator()(std::size_t i, std::size_t j,
                                      const state_type& state) const {
    return Op::apply_matrix(i, j, *this, state);
  }

  constexpr const auto& operand() const { return lhs_; }

 private:
  details::store_by_value_or_const_ref_t<Lhs> lhs_;
};

// Unary scalar-producing scalar expressions
template <typename Op, typename Lhs>
class Expression<
    Op, Lhs, void,
    typename std::enable_if_t<details::is_scalar_v<Lhs>, details::scalar_tag>>
    : public details::scalar_tag {
 public:
  static_assert(ExpressionValidator<Op, Lhs>::value, "Invalid expression!");

  using value_type = typename ResultValueType<Op, Lhs>::type;
  using operand_type = std::decay_t<Lhs>;
  using operation_type = Op;
  using state_type = typename Operation::BuildState<Expression>::type;

  constexpr explicit Expression(Lhs&& lhs) : lhs_(std::forward<Lhs>(lhs)) {}

  constexpr decltype(auto) operator()(const state_type& state) const {
    return Op::apply_scalar(*this, state);
  }

  constexpr const auto& operand() const { return lhs_; }

 private:
  details::store_by_value_or_const_ref_t<Lhs> lhs_;
};

// Transpose operation
template <typename Lhs>
constexpr auto transpose(Lhs&& mat) {
  return Expression<Operation::Transpose, Lhs, void, details::matrix_tag>(
      std::forward<Lhs>(mat));
}

// Submatrix extraction operation
template <std::size_t StartRow, std::size_t StartCol, std::size_t NewRows,
          std::size_t NewCols, typename Lhs>
constexpr auto submatrix(Lhs&& mat) {
  return Expression<
      Operation::SubmatrixExtraction<StartRow, StartCol, NewRows, NewCols>, Lhs,
      void, details::matrix_tag>(std::forward<Lhs>(mat));
}

// Constant matrix
template <std::size_t NumRows, std::size_t NumCols, typename Lhs>
constexpr auto constant_matrix(Lhs&& scal) {
  return Expression<Operation::ConstantMatrix<NumRows, NumCols>, Lhs, void,
                    details::matrix_tag>(std::forward<Lhs>(scal));
}

// Constant diagonal matrix
template <std::size_t NumRows, std::size_t NumCols, typename Lhs>
constexpr auto constant_diagonal(Lhs&& scal) {
  return Expression<Operation::ConstantDiagonal<NumRows, NumCols>, Lhs, void,
                    details::matrix_tag>(std::forward<Lhs>(scal));
}

// Put a vector on the diagonal of a matrix
template <typename Lhs>
constexpr auto diagonal_from_vector(Lhs&& vec) {
  return Expression<Operation::DiagonalFromVector, Lhs, void,
                    details::matrix_tag>(std::forward<Lhs>(vec));
}

// Extract the diagonal of a matrix into a vector
template <typename Lhs>
constexpr auto diagonal_to_vector(Lhs&& mat) {
  return Expression<Operation::DiagonalToVector, Lhs, void,
                    details::matrix_tag>(std::forward<Lhs>(mat));
}

// Extract the diagonal of a matrix into a diagonal matrix
template <typename Lhs>
constexpr auto diagonal_matrix(Lhs&& mat) {
  return Expression<Operation::DiagonalMatrix, Lhs, void, details::matrix_tag>(
      std::forward<Lhs>(mat));
}

// Dot product operation
template <typename Lhs, typename Rhs>
constexpr auto dot(Lhs&& lhs, Rhs&& rhs) {
  return Expression<Operation::DotProduct, Lhs, Rhs, details::scalar_tag>(
      std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
}

// Vector norm operation
template <typename Lhs>
constexpr auto norm(Lhs&& vec) {
  auto dot_product_expr =
      Expression<Operation::DotProduct, Lhs, Lhs, details::scalar_tag>(
          std::forward<Lhs>(vec), std::forward<Lhs>(vec));
  return Expression<Operation::SquareRoot, decltype(dot_product_expr), void,
                    details::scalar_tag>(std::move(dot_product_expr));
}

// Vector normalization operation
template <typename Lhs>
constexpr auto normalize(Lhs&& vec) {
  return Expression<Operation::Normalization, Lhs, void, details::matrix_tag>(
      std::forward<Lhs>(vec));
}

// Element-wise Addition operation
template <typename Lhs, typename Rhs>
constexpr auto operator+(Lhs&& lhs, Rhs&& rhs) {
  static_assert(details::is_expression_v<Lhs> && details::is_expression_v<Lhs>,
                "Operands must be expressions");
  static_assert(details::is_matrix_v<Lhs> == details::is_matrix_v<Rhs>,
                "The addition operation requires both operands to be matrices "
                "or both to be scalars");
  static_assert(details::is_scalar_v<Lhs> == details::is_scalar_v<Rhs>,
                "The addition operation requires both operands to be matrices "
                "or both to be scalars");

  using ExprType =
      std::conditional_t<details::is_matrix_v<Lhs> || details::is_matrix_v<Rhs>,
                         details::matrix_tag, details::scalar_tag>;

  return Expression<Operation::Addition, Lhs, Rhs, ExprType>(
      std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
}

// Element-wise Subtraction operation
template <typename Lhs, typename Rhs>
constexpr auto operator-(Lhs&& lhs, Rhs&& rhs) {
  using ExprType =
      std::conditional_t<details::is_matrix_v<Lhs> || details::is_matrix_v<Rhs>,
                         details::matrix_tag, details::scalar_tag>;

  return Expression<Operation::Subtraction, Lhs, Rhs, ExprType>(
      std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
}

template <typename Lhs, typename Rhs>
constexpr auto operator*(Lhs&& lhs, Rhs&& rhs) {
  if constexpr (details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs>) {
    // Matrix multiplication operation
    return Expression<Operation::Multiplication, Lhs, Rhs, details::matrix_tag>(
        std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
  } else if constexpr (details::is_scalar_v<Lhs> || details::is_scalar_v<Rhs>) {
    // Scalar multiplication operation
    using ExprType =
        std::conditional_t<details::is_matrix_v<Lhs> ||
                               details::is_matrix_v<Rhs>,
                           details::matrix_tag, details::scalar_tag>;
    return Expression<Operation::ScalarMultiplication, Lhs, Rhs, ExprType>(
        std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
  }
}

// Scalar division operation
template <typename Lhs, typename Rhs,
          typename = std::enable_if_t<details::is_expression_v<Lhs> &&
                                      details::is_scalar_v<Rhs>>>
constexpr auto operator/(Lhs&& lhs, Rhs&& rhs) {
  using ExprType = std::conditional_t<details::is_matrix_v<Lhs>,
                                      details::matrix_tag, details::scalar_tag>;
  return Expression<Operation::ScalarDivision, Lhs, Rhs, ExprType>(
      std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
}

template <typename ValueType, std::size_t NumRows, std::size_t NumCols>
using ConstantMatrix = Expression<Operation::ConstantMatrix<NumRows, NumCols>,
                                  Scalar<ValueType>, void, details::matrix_tag>;

template <typename ValueType, std::size_t NumRows, std::size_t NumCols>
using ConstantDiagonal =
    Expression<Operation::ConstantDiagonal<NumRows, NumCols>, Scalar<ValueType>,
               void, details::matrix_tag>;

template <typename ValueType, std::size_t NumRows, std::size_t NumCols>
inline constexpr auto IdentityMatrix =
    Expression<Operation::ConstantDiagonal<NumRows, NumCols>, Scalar<ValueType>,
               void, details::matrix_tag>{Scalar<ValueType>{1}};

}  // namespace optila