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
                       std::bool_constant<std::decay_t<T>::num_rows() == 1>> {};

template <typename T>
inline constexpr bool is_row_vector_v = is_row_vector<T>::value;

template <typename T>
struct is_column_vector
    : std::conjunction<is_matrix<T>,
                       std::bool_constant<std::decay_t<T>::num_cols() == 1>> {};

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

template <typename... Operands>
struct is_nullary_operand : std::false_type {};

template <>
struct is_nullary_operand<> : std::true_type {};

template <typename... Operands>
inline constexpr bool is_nullary_operand_v =
    is_nullary_operand<Operands...>::value;

template <typename... Operands>
struct is_unary_operand : std::false_type {};

template <typename Operand>
struct is_unary_operand<Operand> : std::true_type {};

template <typename... Operands>
inline constexpr bool is_unary_operand_v = is_unary_operand<Operands...>::value;

template <typename... Operands>
struct is_binary_operand : std::false_type {};

template <typename Lhs, typename Rhs>
struct is_binary_operand<Lhs, Rhs> : std::true_type {};

template <typename... Operands>
inline constexpr bool is_binary_operand_v =
    is_binary_operand<Operands...>::value;

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
  constexpr static std::size_t num_rows() { return NumRows; }
  constexpr static std::size_t num_cols() { return NumCols; }

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
  std::array<value_type, num_rows() * num_cols()> data_;

  [[nodiscard]] constexpr std::size_t linear_index(std::size_t i,
                                                   std::size_t j) const {
    if constexpr (Order == StorageOrder::RowMajor) {
      return i * num_cols() + j;
    } else {  // StorageOrder::ColumnMajor
      return i + j * num_rows();
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

template <typename Op, typename... Operands>
class Expression;

namespace details {
template <typename T>
struct is_expression_literal : std::false_type {};

template <typename Op, typename... Operands>
struct is_expression_literal<Expression<Op, Operands...>> : std::true_type {};

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
template <typename Op, bool HasState, typename... States>
struct StateImpl;

template <typename Op, typename... States>
struct StateImpl<Op, true, States...> : public details::operation_state_tag {
  using operation_type = Op;
  typename operation_type::State state;

  std::tuple<States...> operands;
};

template <typename Op, typename... States>
struct StateImpl<Op, false, States...> {
  using operation_type = Op;

  std::tuple<States...> operands;
};

template <typename Op, typename... States>
struct State
    : public StateImpl<Op, details::has_operation_state_v<Op>, States...> {};

// Utility function to create the operation state structure
template <typename Op, typename... Operands>
struct BuildState {
  struct type {};
};

template <typename Op, typename... Operands>
struct BuildState<Expression<Op, Operands...>> {
  using type = State<Op, typename BuildState<std::decay_t<Operands>>::type...>;
};

template <typename Expr, typename State>
constexpr decltype(auto) evalLeftScalarOperand(Expr&& expr, State&& state) {
  if constexpr (details::is_expression_literal_v<
                    typename std::decay_t<Expr>::template operand_type<0>>)
    return expr.template operand<0>()(std::get<0>(state.operands));
  else
    return expr.template operand<0>()();
}

template <typename Expr, typename State>
constexpr decltype(auto) evalRightScalarOperand(Expr&& expr, State&& state) {
  if constexpr (details::is_expression_literal_v<
                    typename std::decay_t<Expr>::template operand_type<1>>)
    return expr.template operand<1>()(std::get<1>(state.operands));
  else
    return expr.template operand<1>()();
}

template <typename Expr, typename State>
constexpr decltype(auto) evalScalarOperand(Expr&& expr, State&& state) {
  if constexpr (details::is_expression_literal_v<
                    typename std::decay_t<Expr>::template operand_type<0>>)
    return expr.template operand<0>()(std::get<0>(state.operands));
  else
    return expr.template operand<0>()();
}

template <typename Expr, typename State>
constexpr decltype(auto) evalLeftMatrixOperand(std::size_t i, std::size_t j,
                                               Expr&& expr, State&& state) {
  if constexpr (details::is_expression_literal_v<
                    typename std::decay_t<Expr>::template operand_type<0>>)
    return expr.template operand<0>()(i, j, std::get<0>(state.operands));
  else
    return expr.template operand<0>()(i, j);
}

template <typename Expr, typename State>
constexpr decltype(auto) evalRightMatrixOperand(std::size_t i, std::size_t j,
                                                Expr&& expr, State&& state) {
  if constexpr (details::is_expression_literal_v<
                    typename std::decay_t<Expr>::template operand_type<1>>)
    return expr.template operand<1>()(i, j, std::get<1>(state.operands));
  else
    return expr.template operand<1>()(i, j);
}

template <typename Expr, typename State>
constexpr decltype(auto) evalMatrixOperand(std::size_t i, std::size_t j,
                                           Expr&& expr, State&& state) {
  if constexpr (details::is_expression_literal_v<
                    typename std::decay_t<Expr>::template operand_type<0>>)
    return expr.template operand<0>()(i, j, std::get<0>(state.operands));
  else
    return expr.template operand<0>()(i, j);
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
         k < std::decay_t<decltype(expr.template operand<0>())>::num_cols();
         ++k) {
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
        details::is_scalar_v<decltype(expr.template operand<0>())>;
    constexpr bool rhs_is_scalar =
        details::is_scalar_v<decltype(expr.template operand<1>())>;
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
        details::is_scalar_v<decltype(expr.template operand<1>())>;
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
         i < std::decay_t<decltype(expr.template operand<0>())>::num_rows();
         ++i) {
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
    return expr.template operand<0>()(i + StartRow, j + StartCol,
                                      std::get<0>(state.operands));
  };
};
struct Concatenation {};
struct SquareRoot {
  // FIXME: sqrt is not constexpr
  template <typename Expr, typename State>
  static auto apply_scalar(Expr&& expr, State&& state) {
    return sqrt(expr.template operand<0>()(std::get<0>(state.operands)));
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
    // so that we can use the Norm operation on expr.template operand<0>().
    state.state.norm = 0;
    for (std::size_t i = 0; i < std::decay_t<Expr>::num_rows(); ++i) {
      for (std::size_t j = 0; j < std::decay_t<Expr>::num_cols(); ++j) {
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
    return expr.template operand<0>()();
  };
};
// Fill a diagonal matrix with a constant value
template <std::size_t NumRows, std::size_t NumCols>
struct ConstantDiagonal {
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr) {
    using value_type = typename std::decay_t<decltype(expr)>::value_type;
    return i == j ? expr.template operand<0>()() : value_type{};
  };
};
// Put a vector on the diagonal of a matrix
struct DiagonalFromVector {
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr) {
    using value_type = typename std::decay_t<decltype(expr)>::value_type;
    constexpr bool lhs_is_row_vector =
        details::is_row_vector_v<decltype(expr.template operand<0>())>;
    constexpr bool lhs_is_column_vector =
        details::is_column_vector_v<decltype(expr.template operand<0>())>;
    static_assert(lhs_is_row_vector != lhs_is_column_vector,
                  "The operand must be a vector");
    if constexpr (lhs_is_row_vector) {
      return i == j ? expr.template operand<0>()(0, j) : value_type{};
    } else if constexpr (lhs_is_column_vector) {
      return i == j ? expr.template operand<0>()(i, 0) : value_type{};
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
      return expr.template operand<0>()(j, j);
    } else if constexpr (is_column_vector) {
      return expr.template operand<0>()(i, i);
    }
  };
};
// Extract the diagonal of a matrix into a diagonal matrix
struct DiagonalMatrix {
  static constexpr auto apply_matrix = [](std::size_t i, std::size_t j,
                                          auto&& expr) {
    using value_type = typename std::decay_t<decltype(expr)>::value_type;
    return i == j ? expr.template operand<0>()(i, i) : value_type{};
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
    : std::integral_constant<std::size_t, std::decay_t<Lhs>::num_rows()> {
  static_assert(std::decay_t<Lhs>::num_rows() == std::decay_t<Rhs>::num_rows(),
                "Matrix dimensions must match");
};

template <typename Op, typename Lhs, typename Rhs>
struct ResultNumCols<
    Op, Lhs, Rhs,
    std::enable_if_t<details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs>>>
    : std::integral_constant<std::size_t, std::decay_t<Lhs>::num_cols()> {
  static_assert(std::decay_t<Lhs>::num_cols() == std::decay_t<Rhs>::num_cols(),
                "Matrix dimensions must match");
};

// === Unary matrix operations
template <typename Op, typename Lhs>
struct ResultNumRows<Op, Lhs, void, std::enable_if_t<details::is_matrix_v<Lhs>>>
    : std::integral_constant<std::size_t, std::decay_t<Lhs>::num_rows()> {};

template <typename Op, typename Lhs>
struct ResultNumCols<Op, Lhs, void, std::enable_if_t<details::is_matrix_v<Lhs>>>
    : std::integral_constant<std::size_t, std::decay_t<Lhs>::num_cols()> {};

// === Matrix-Scalar operations
template <typename Op, typename Lhs, typename Rhs>
struct ResultNumRows<
    Op, Lhs, Rhs,
    std::enable_if_t<details::is_matrix_v<Lhs> && details::is_scalar_v<Rhs>>>
    : std::integral_constant<std::size_t, std::decay_t<Lhs>::num_rows()> {};

template <typename Op, typename Lhs, typename Rhs>
struct ResultNumCols<
    Op, Lhs, Rhs,
    std::enable_if_t<details::is_matrix_v<Lhs> && details::is_scalar_v<Rhs>>>
    : std::integral_constant<std::size_t, std::decay_t<Lhs>::num_cols()> {};

// === Scalar-Matrix operations
template <typename Op, typename Lhs, typename Rhs>
struct ResultNumRows<
    Op, Lhs, Rhs,
    std::enable_if_t<details::is_scalar_v<Lhs> && details::is_matrix_v<Rhs>>>
    : std::integral_constant<std::size_t, std::decay_t<Rhs>::num_rows()> {};

template <typename Op, typename Lhs, typename Rhs>
struct ResultNumCols<
    Op, Lhs, Rhs,
    std::enable_if_t<details::is_scalar_v<Lhs> && details::is_matrix_v<Rhs>>>
    : std::integral_constant<std::size_t, std::decay_t<Rhs>::num_cols()> {};

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
    : std::integral_constant<std::size_t, std::decay_t<Lhs>::num_rows()> {
  static_assert(std::decay_t<Lhs>::num_cols() == std::decay_t<Rhs>::num_rows(),
                "Inner matrix dimensions must match");
};

template <typename Lhs, typename Rhs>
struct ResultNumCols<
    Operation::Multiplication, Lhs, Rhs,
    std::enable_if_t<details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs>>>
    : std::integral_constant<std::size_t, std::decay_t<Rhs>::num_cols()> {};

// Vector to diagonal matrix
template <typename Lhs>
struct ResultNumRows<Operation::DiagonalFromVector, Lhs, void,
                     std::enable_if_t<details::is_vector_v<Lhs>>>
    : std::integral_constant<std::size_t,
                             std::max(std::decay_t<Lhs>::num_rows(),
                                      std::decay_t<Lhs>::num_cols())> {};

template <typename Lhs>
struct ResultNumCols<Operation::DiagonalFromVector, Lhs, void,
                     std::enable_if_t<details::is_vector_v<Lhs>>>
    : std::integral_constant<std::size_t,
                             std::max(std::decay_t<Lhs>::num_rows(),
                                      std::decay_t<Lhs>::num_cols())> {};

// Diagonal of a matrix to vector
template <typename Lhs>
struct ResultNumRows<Operation::DiagonalToVector, Lhs, void,
                     std::enable_if_t<details::is_matrix_v<Lhs>>>
    : std::integral_constant<std::size_t,
                             std::min(std::decay_t<Lhs>::num_rows(),
                                      std::decay_t<Lhs>::num_cols())> {};

template <typename Lhs>
struct ResultNumCols<Operation::DiagonalToVector, Lhs, void,
                     std::enable_if_t<details::is_matrix_v<Lhs>>>
    : std::integral_constant<std::size_t, 1> {};

// Diagonal of a matrix to a diagonal matrix
template <typename Lhs>
struct ResultNumRows<Operation::DiagonalMatrix, Lhs, void,
                     std::enable_if_t<details::is_matrix_v<Lhs>>>
    : std::integral_constant<std::size_t,
                             std::min(std::decay_t<Lhs>::num_rows(),
                                      std::decay_t<Lhs>::num_cols())> {};

template <typename Lhs>
struct ResultNumCols<Operation::DiagonalMatrix, Lhs, void,
                     std::enable_if_t<details::is_matrix_v<Lhs>>>
    : std::integral_constant<std::size_t,
                             std::min(std::decay_t<Lhs>::num_rows(),
                                      std::decay_t<Lhs>::num_cols())> {};

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
    : std::integral_constant<std::size_t, std::decay_t<Operand>::num_cols()> {};

template <typename Operand>
struct ResultNumCols<Operation::Transpose, Operand, void,
                     std::enable_if_t<details::is_matrix_v<Operand>>>
    : std::integral_constant<std::size_t, std::decay_t<Operand>::num_rows()> {};

// Submatrix extraction
template <std::size_t StartRow, std::size_t StartCol, std::size_t NumRows,
          std::size_t NumCols, typename Operand>
struct ResultNumRows<
    Operation::SubmatrixExtraction<StartRow, StartCol, NumRows, NumCols>,
    Operand, void, std::enable_if_t<details::is_matrix_v<Operand>>>
    : std::integral_constant<std::size_t, NumRows> {
  static_assert(StartRow + NumRows <= std::decay_t<Operand>::num_rows(),
                "Submatrix extraction out of bounds");
};

template <std::size_t StartRow, std::size_t StartCol, std::size_t NumRows,
          std::size_t NumCols, typename Operand>
struct ResultNumCols<
    Operation::SubmatrixExtraction<StartRow, StartCol, NumRows, NumCols>,
    Operand, void, std::enable_if_t<details::is_matrix_v<Operand>>>
    : std::integral_constant<std::size_t, NumCols> {
  static_assert(StartCol + NumCols <= std::decay_t<Operand>::num_cols(),
                "Submatrix extraction out of bounds");
};

namespace details {

template <typename Expr, typename = std::void_t<>>
struct has_left_operand_expression : std::false_type {};

template <typename Expr>
struct has_left_operand_expression<
    Expr, std::void_t<typename std::decay_t<Expr>::template operand_type<0>>>
    : std::conditional_t<
          is_expression_literal_v<std::decay_t<
              typename std::decay_t<Expr>::template operand_type<0>>>,
          std::true_type, std::false_type> {};

template <typename Expr>
inline constexpr bool has_left_operand_expression_v =
    has_left_operand_expression<Expr>::value;

template <typename Expr, typename = std::void_t<>>
struct has_right_operand_expression : std::false_type {};

template <typename Expr>
struct has_right_operand_expression<
    Expr, std::void_t<typename std::decay_t<Expr>::template operand_type<1>>>
    : std::conditional_t<
          is_expression_literal_v<std::decay_t<
              typename std::decay_t<Expr>::template operand_type<1>>>,
          std::true_type, std::false_type> {};

template <typename Expr>
inline constexpr bool has_right_operand_expression_v =
    has_right_operand_expression<Expr>::value;

template <typename Expr, typename = std::void_t<>>
struct has_operand_expression : std::false_type {};

template <typename Expr>
struct has_operand_expression<
    Expr, std::void_t<typename std::decay_t<Expr>::template operand_type<0>>>
    : std::conditional_t<
          is_expression_literal_v<std::decay_t<
              typename std::decay_t<Expr>::template operand_type<0>>>,
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
    precompute(expr.template operand<0>(), std::get<0>(state.operands));
  if constexpr (details::has_right_operand_expression_v<Expr>)
    precompute(expr.template operand<1>(), std::get<1>(state.operands));
  if constexpr (details::has_operand_expression_v<Expr>)
    precompute(expr.template operand<0>(), std::get<0>(state.operands));
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

  Matrix<typename Expr::value_type, Expr::num_rows(), Expr::num_cols(), Order>
      result{};
  for (std::size_t i = 0; i < Expr::num_rows(); ++i) {
    for (std::size_t j = 0; j < Expr::num_cols(); ++j) {
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

template <typename Op, typename... Operands>
struct ExpressionValidator;

template <typename Lhs, typename Rhs>
struct ExpressionValidator<Operation::Addition, Lhs, Rhs> {
  using expression_type =
      std::conditional_t<details::is_scalar_v<Lhs> && details::is_scalar_v<Rhs>,
                         details::scalar_tag, details::matrix_tag>;

  static_assert(
      (details::is_scalar_v<Lhs> && details::is_scalar_v<Rhs>) ||
          (details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs> &&
           std::decay_t<Lhs>::num_rows() == std::decay_t<Rhs>::num_rows() &&
           std::decay_t<Lhs>::num_cols() == std::decay_t<Rhs>::num_cols()),
      "Mismatched operands for addition");
};

template <typename Lhs, typename Rhs>
struct ExpressionValidator<Operation::Multiplication, Lhs, Rhs> {
  using expression_type = details::matrix_tag;

  static_assert(details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs>,
                "Matrix multiplication requires matrix operands");
  static_assert(Lhs::num_cols() == Rhs::num_rows(),
                "Matrix operand inner dimensions must match");
};

template <typename Lhs, typename Rhs>
struct ExpressionValidator<Operation::DotProduct, Lhs, Rhs> {
  using expression_type = details::scalar_tag;

  static_assert(
      details::is_vector_v<Lhs> && details::is_vector_v<Rhs> &&
          (std::decay_t<Lhs>::num_rows() == std::decay_t<Rhs>::num_rows() &&
           std::decay_t<Lhs>::num_cols() == std::decay_t<Rhs>::num_cols()),
      "Dot product requires vector operands of the same dimension");
};

template <typename Op, typename ExprType, typename... Operands>
class ExpressionImpl;

// Partial specialization for matrix_tag
template <typename Op, typename... Operands>
class ExpressionImpl<Op, details::matrix_tag, Operands...>
    : public details::matrix_tag {
 public:
  static constexpr std::size_t num_rows() {
    return ResultNumRows<Op, Operands...>::value;
  }
  static constexpr std::size_t num_cols() {
    return ResultNumCols<Op, Operands...>::value;
  }

  using Derived = Expression<Op, Operands...>;
  using state_type = typename Operation::BuildState<Derived>::type;
  constexpr decltype(auto) operator()(std::size_t i, std::size_t j,
                                      const state_type& state) const {
    return Op::apply_matrix(i, j, static_cast<const Derived&>(*this), state);
  }
};

// Partial specialization for scalar_tag
template <typename Op, typename... Operands>
class ExpressionImpl<Op, details::scalar_tag, Operands...>
    : public details::scalar_tag {
 public:
  using Derived = Expression<Op, Operands...>;
  using state_type = typename Operation::BuildState<Derived>::type;
  constexpr decltype(auto) operator()(const state_type& state) const {
    return Op::apply_scalar(static_cast<const Derived&>(*this), state);
  }
};

template <typename Op, typename... Operands>
class Expression
    : public ExpressionImpl<Op,
                            typename ExpressionValidator<
                                Op, std::decay_t<Operands>...>::expression_type,
                            Operands...> {
  // The Expression class forwards the deduced ExprType to ExpressionImpl.
 public:
  using value_type = typename ResultValueType<Op, Operands...>::type;
  using operation_type = Op;
  template <std::size_t index>
  using operand_type =
      std::tuple_element_t<index, std::tuple<std::decay_t<Operands>...>>;

  constexpr explicit Expression(Operands&&... operands)
      : operands_(std::forward<Operands>(operands)...) {}

  template <std::size_t index>
  constexpr decltype(auto) operand() const {
    return std::get<index>(operands_);
  }
  [[nodiscard]] static constexpr std::size_t num_operands() {
    return std::tuple_size_v<operand_storage_type>;
  }

 private:
  using operand_storage_type =
      std::tuple<details::store_by_value_or_const_ref_t<Operands>...>;
  operand_storage_type operands_;
};

// Transpose operation
template <typename Lhs>
constexpr auto transpose(Lhs&& mat) {
  return Expression<Operation::Transpose, Lhs>(std::forward<Lhs>(mat));
}

// Submatrix extraction operation
template <std::size_t StartRow, std::size_t StartCol, std::size_t NewRows,
          std::size_t NewCols, typename Lhs>
constexpr auto submatrix(Lhs&& mat) {
  return Expression<
      Operation::SubmatrixExtraction<StartRow, StartCol, NewRows, NewCols>,
      Lhs>(std::forward<Lhs>(mat));
}

// Constant matrix
template <std::size_t NumRows, std::size_t NumCols, typename Lhs>
constexpr auto constant_matrix(Lhs&& scal) {
  return Expression<Operation::ConstantMatrix<NumRows, NumCols>, Lhs>(
      std::forward<Lhs>(scal));
}

// Constant diagonal matrix
template <std::size_t NumRows, std::size_t NumCols, typename Lhs>
constexpr auto constant_diagonal(Lhs&& scal) {
  return Expression<Operation::ConstantDiagonal<NumRows, NumCols>, Lhs>(
      std::forward<Lhs>(scal));
}

// Put a vector on the diagonal of a matrix
template <typename Lhs>
constexpr auto diagonal_from_vector(Lhs&& vec) {
  return Expression<Operation::DiagonalFromVector, Lhs>(std::forward<Lhs>(vec));
}

// Extract the diagonal of a matrix into a vector
template <typename Lhs>
constexpr auto diagonal_to_vector(Lhs&& mat) {
  return Expression<Operation::DiagonalToVector, Lhs>(std::forward<Lhs>(mat));
}

// Extract the diagonal of a matrix into a diagonal matrix
template <typename Lhs>
constexpr auto diagonal_matrix(Lhs&& mat) {
  return Expression<Operation::DiagonalMatrix, Lhs>(std::forward<Lhs>(mat));
}

// Dot product operation
template <typename Lhs, typename Rhs>
constexpr auto dot(Lhs&& lhs, Rhs&& rhs) {
  return Expression<Operation::DotProduct, Lhs, Rhs>(std::forward<Lhs>(lhs),
                                                     std::forward<Rhs>(rhs));
}

// Vector norm operation
template <typename Lhs>
constexpr auto norm(Lhs&& vec) {
  auto dot_product_expr = Expression<Operation::DotProduct, Lhs, Lhs>(
      std::forward<Lhs>(vec), std::forward<Lhs>(vec));
  return Expression<Operation::SquareRoot, decltype(dot_product_expr)>(
      std::move(dot_product_expr));
}

// Vector normalization operation
template <typename Lhs>
constexpr auto normalize(Lhs&& vec) {
  return Expression<Operation::Normalization, Lhs>(std::forward<Lhs>(vec));
}

// Element-wise Addition operation
template <typename Lhs, typename Rhs>
constexpr auto operator+(Lhs&& lhs, Rhs&& rhs) {
  return Expression<Operation::Addition, Lhs, Rhs>(std::forward<Lhs>(lhs),
                                                   std::forward<Rhs>(rhs));
}

// Element-wise Subtraction operation
template <typename Lhs, typename Rhs>
constexpr auto operator-(Lhs&& lhs, Rhs&& rhs) {
  return Expression<Operation::Subtraction, Lhs, Rhs>(std::forward<Lhs>(lhs),
                                                      std::forward<Rhs>(rhs));
}

template <typename Lhs, typename Rhs>
constexpr auto operator*(Lhs&& lhs, Rhs&& rhs) {
  if constexpr (details::is_matrix_v<Lhs> && details::is_matrix_v<Rhs>) {
    // Matrix multiplication operation
    return Expression<Operation::Multiplication, Lhs, Rhs>(
        std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
  } else if constexpr (details::is_scalar_v<Lhs> || details::is_scalar_v<Rhs>) {
    // Scalar multiplication operation
    return Expression<Operation::ScalarMultiplication, Lhs, Rhs>(
        std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
  }
}

// Scalar division operation
template <typename Lhs, typename Rhs>
constexpr auto operator/(Lhs&& lhs, Rhs&& rhs) {
  return Expression<Operation::ScalarDivision, Lhs, Rhs>(
      std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
}

template <typename ValueType, std::size_t NumRows, std::size_t NumCols>
using ConstantMatrix =
    Expression<Operation::ConstantMatrix<NumRows, NumCols>, Scalar<ValueType>>;

template <typename ValueType, std::size_t NumRows, std::size_t NumCols>
using ConstantDiagonal =
    Expression<Operation::ConstantDiagonal<NumRows, NumCols>,
               Scalar<ValueType>>;

template <typename ValueType, std::size_t NumRows, std::size_t NumCols>
inline constexpr auto IdentityMatrix =
    Expression<Operation::ConstantDiagonal<NumRows, NumCols>,
               Scalar<ValueType>>{Scalar<ValueType>{1}};

}  // namespace optila