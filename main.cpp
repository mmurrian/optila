#include <cassert>
#include <cstdlib>
#include <ctime>
#include <iostream>

#include "optila.h"

#define OPTILA_TEST_CONSTEXPR

template <typename T, std::size_t NumRows, std::size_t NumCols, typename Policy>
void generate_random_matrix(optila::Matrix<T, NumRows, NumCols, Policy> &A) {
  if constexpr (NumRows == optila::Dynamic || NumCols == optila::Dynamic) {
    A.resize(NumRows, NumCols);
  }
  for (std::size_t i = 0; i < A.num_rows(); ++i) {
    for (std::size_t j = 0; j < A.num_cols(); ++j) {
      A(i, j) = static_cast<double>(rand()) / RAND_MAX;
    }
  }
}

template <typename ArgA, typename ArgB, typename ArgC>
double __attribute__((noinline)) do_my_things(ArgA A, ArgB B, ArgC c) {
  return evaluate((A * B) + c)(0, 0);
}

int random_tests() {
  optila::Matrix<double, 8, 8> A{};
  optila::Matrix<double, 8, 8> B{};
  optila::Matrix<double, 8, 8> c{};
  generate_random_matrix(A);
  generate_random_matrix(B);
  generate_random_matrix(c);

  std::cout << do_my_things(A, B, c) << std::endl;

  return 0;
}

int assignment_tests() {
  optila::Matrix<double, optila::Dynamic, 3> A =
      optila::Matrix<double, 3, 3>{{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}};
  assert(A.num_rows() == 3);
  assert(A.num_cols() == 3);

  optila::Matrix<double, 3, 3> B = A;

  optila::Matrix<double, 3, 3> C = A * B;

  return 0;
}

int active_sub_matrix_tests() {
  static constexpr auto static_A =
      optila::make_matrix<double, 6, 6>({{1, 2, 3, 4, 5, 6},
                                         {7, 8, 9, 10, 11, 12},
                                         {13, 14, 15, 16, 17, 18},
                                         {19, 20, 21, 22, 23, 24},
                                         {25, 26, 27, 28, 29, 30},
                                         {31, 32, 33, 34, 35, 36}});

  static constexpr auto static_B =
      optila::make_matrix<double, 6, 6>({{11, 12, 13, 14, 15, 16},
                                         {17, 18, 19, 20, 21, 22},
                                         {23, 24, 25, 26, 27, 28},
                                         {29, 30, 31, 32, 33, 34},
                                         {35, 36, 37, 38, 39, 40},
                                         {41, 42, 43, 44, 45, 46}});

  static constexpr optila::Matrix static_C = optila::submatrix<3, 3, 2, 2>(
      (static_A + static_B) * (static_B - static_A));
  static_assert(static_C ==
                (optila::Matrix<double, 2, 2>{{{3180, 3180}, {3900, 3900}}}));

  auto A = optila::make_matrix<double, 6, 6>({{1, 2, 3, 4, 5, 6},
                                              {7, 8, 9, 10, 11, 12},
                                              {13, 14, 15, 16, 17, 18},
                                              {19, 20, 21, 22, 23, 24},
                                              {25, 26, 27, 28, 29, 30},
                                              {31, 32, 33, 34, 35, 36}});

  auto B = optila::make_matrix<double, 6, 6>({{11, 12, 13, 14, 15, 16},
                                              {17, 18, 19, 20, 21, 22},
                                              {23, 24, 25, 26, 27, 28},
                                              {29, 30, 31, 32, 33, 34},
                                              {35, 36, 37, 38, 39, 40},
                                              {41, 42, 43, 44, 45, 46}});

  optila::Matrix C = optila::submatrix<3, 3, 2, 2>((A + B) * (B - A));
  assert(C == (optila::Matrix<double, 2, 2>{{{3180, 3180}, {3900, 3900}}}));

  return 0;
}

int other_tests() {
  // Dynamically allocated matrices
  optila::Matrix<double, optila::Dynamic, optila::Dynamic> A{
      {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}};
  optila::Matrix<double, optila::Dynamic, optila::Dynamic> B{
      {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}};

  assert(A.num_rows() == 3);
  assert(A.num_cols() == 3);
  assert(B.num_rows() == 3);
  assert(B.num_cols() == 3);

  auto C = optila::evaluate(A + B);
  auto D = optila::evaluate(A - B);
  auto E = optila::evaluate(A * B);
  auto scalar = optila::Scalar<double>{30.0};

  auto F = optila::evaluate(scalar * A);
  auto G = optila::evaluate(A * scalar);

  assert(C == (optila::Matrix<double, optila::Dynamic, optila::Dynamic>{
                  {{10, 10, 10}, {10, 10, 10}, {10, 10, 10}}}));
  assert(D == (optila::Matrix<double, optila::Dynamic, optila::Dynamic>{
                  {{-8, -6, -4}, {-2, 0, 2}, {4, 6, 8}}}));
  assert(E == (optila::Matrix<double, optila::Dynamic, optila::Dynamic>{
                  {{30, 24, 18}, {84, 69, 54}, {138, 114, 90}}}));
  assert(F == (optila::Matrix<double, optila::Dynamic, optila::Dynamic>{
                  {{30, 60, 90}, {120, 150, 180}, {210, 240, 270}}}));
  assert(G == (optila::Matrix<double, optila::Dynamic, optila::Dynamic>{
                  {{30, 60, 90}, {120, 150, 180}, {210, 240, 270}}}));

#ifdef OPTILA_TEST_CONSTEXPR
  // Statically-known matrices
  static constexpr auto static_A =
      optila::make_matrix<double, 3, 3>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  static constexpr auto static_B =
      optila::make_matrix<double, 3, 3>({{9, 8, 7}, {6, 5, 4}, {3, 2, 1}});

  static_assert(static_A.num_rows() == 3);
  static_assert(static_A.num_cols() == 3);
  static_assert(static_B.num_rows() == 3);
  static_assert(static_B.num_cols() == 3);

  static constexpr optila::Matrix static_C =
      optila::evaluate(static_A + static_B);
  static constexpr optila::Matrix static_D =
      optila::evaluate(static_A - static_B);
  static constexpr optila::Matrix static_E =
      optila::evaluate(static_A * static_B);
  static constexpr auto static_scalar = optila::make_scalar<double>(30.0);

  static constexpr optila::Matrix static_F =
      optila::evaluate(static_scalar * static_A);
  static constexpr optila::Matrix static_G =
      optila::evaluate(static_A * static_scalar);

  static_assert(static_C ==
                decltype(static_C){{{10, 10, 10}, {10, 10, 10}, {10, 10, 10}}});
  static_assert(static_D ==
                decltype(static_D){{{-8, -6, -4}, {-2, 0, 2}, {4, 6, 8}}});
  static_assert(static_E == decltype(static_E){
                                {{30, 24, 18}, {84, 69, 54}, {138, 114, 90}}});

  static_assert(
      static_F ==
      decltype(static_F){{{30, 60, 90}, {120, 150, 180}, {210, 240, 270}}});
  static_assert(
      static_G ==
      decltype(static_G){{{30, 60, 90}, {120, 150, 180}, {210, 240, 270}}});
#endif
  return 0;
}

int constexpr_nested_tests() {
  static constexpr optila::Matrix<double, 3, 3> A{
      {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}};
  static_assert(A.num_rows() == 3);
  static_assert(A.num_cols() == 3);

  static constexpr optila::Matrix<double, 3, 3> B{
      {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}};

  // Perform matrix operations
  static constexpr auto C = A * B;
  static_assert(C == (optila::Matrix<double, 3, 3>{
                         {{30, 24, 18}, {84, 69, 54}, {138, 114, 90}}}));
  static constexpr auto D = B * A;
  static_assert(D == (optila::Matrix<double, 3, 3>{
                         {{90, 114, 138}, {54, 69, 84}, {18, 24, 30}}}));

  // Nested matrix multiplication
  static constexpr auto E = (C * D) * (A * B);
  static_assert(E ==
                (optila::Matrix<double, 3, 3>{{{1516320, 1247076, 977832},
                                               {4304016, 3539781, 2775546},
                                               {7091712, 5832486, 4573260}}}));

  // Additional operations
  static constexpr auto F = E + D - C;

  static constexpr optila::Matrix<double, 3, 1> x{{{1}, {2}, {3}}};
  static constexpr optila::Matrix<double, 1, 3> y{{{1, 2, 3}}};

  static constexpr auto z = optila::dot(x, optila::transpose(y));

  static constexpr auto z2 = optila::dot(x, x);

  // Test implicit evaluation of a matrix expression into a dynamic matrix
  static constexpr optila::Matrix G = F;
  static_assert(G ==
                (optila::Matrix<double, 3, 3>{{{1516380, 1247166, 977952},
                                               {4303986, 3539781, 2775576},
                                               {7091592, 5832396, 4573200}}}));

  return 0;
}

int nested_tests() {
  optila::Matrix<double, optila::Dynamic, optila::Dynamic> A{
      {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}};
  assert(A.num_rows() == 3);
  assert(A.num_cols() == 3);

  optila::Matrix<double, optila::Dynamic, optila::Dynamic> B{
      {{9, 8, 7}, {6, 5, 4}, {3, 2, 1}}};

  // Perform matrix operations
  auto C = A * B;
  auto D = B * A;

  // Nested matrix multiplication
  auto E = (C * D) * (A * B);

  // Additional operations
  auto F = E + D - C;

  optila::Matrix<double, optila::Dynamic, 1> x{{{1}, {2}, {3}}};
  optila::Matrix<double, 1, optila::Dynamic> y{{{1, 2, 3}}};

  auto z = optila::dot(x, optila::transpose(y));
  std::cout << z << std::endl;

  auto z2 = optila::dot(x, x);
  std::cout << optila::Scalar(3.0) * z2 << std::endl;

  // Test implicit evaluation of a matrix expression into a dynamic matrix
  optila::Matrix G = optila::submatrix<0, 0, 3, 3>(F);
  assert(G == (optila::Matrix<double, 3, 3>{{{1516380, 1247166, 977952},
                                             {4303986, 3539781, 2775576},
                                             {7091592, 5832396, 4573200}}}));

  return 0;
}

int dynamic_tests() {
  optila::Matrix<double, optila::Dynamic, optila::Dynamic> A{
      {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}};
  assert(A.num_rows() == 3);
  assert(A.num_cols() == 3);

  // Test copying matrix to matrix.
  optila::Matrix<double, 3, 3> Acopy = A;
  assert(Acopy == A);

  optila::Matrix<double, optila::Dynamic, optila::Dynamic> b{{{1}, {2}, {3}}};
  assert(b.num_rows() == 3);
  assert(b.num_cols() == 1);

  auto C = A * b;

  assert(C.num_rows() == 3);
  assert(C.num_cols() == 1);

  // Test implicit evaluation of a matrix expression into a static matrix
  optila::Matrix<double, 3, 1> D = C;
  assert(D(0, 0) == 14);
  assert(D(1, 0) == 32);
  assert(D(2, 0) == 50);

  // Test implicit evaluation of a matrix expression with template parameters
  // deduced.
  optila::Matrix D2 = C;

  // Test explicit evaluation of a matrix expression.
  const auto D3 = optila::evaluate(C);
  assert(D == D2);
  assert(D == D3);

  // Test implicit evaluation of a scalar expression with template parameters
  // deduced.
  optila::Scalar E = optila::dot(b, b);
  assert(E == 14);

  auto F = optila::evaluate(C + b);
  assert(F(0, 0) == 15);

  auto G = optila::submatrix<1, 1, 2, 2>(A);
  assert(G.num_rows() == 2);
  assert(G.num_cols() == 2);

  auto H = optila::evaluate(G);
  assert(H(0, 0) == 5);
  assert(H(0, 1) == 6);
  assert(H(1, 0) == 8);
  assert(H(1, 1) == 9);

  optila::Scalar bNorm = optila::norm(b);
  assert(bNorm == 3.7416573867739413);
  optila::Matrix I = optila::normalize(b);
  assert(I(0, 0) == 0.2672612419124244);
  assert(I(1, 0) == 0.5345224838248488);
  assert(I(2, 0) == 0.8017837257372732);

  return 0;
}

int constexpr_tests() {
#ifdef OPTILA_TEST_CONSTEXPR
  static constexpr optila::Matrix<double, 3, 3> A{
      {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}};

  // Test copying matrix to matrix.
  static constexpr optila::Matrix<double, 3, 3> Acopy = A;
  static_assert(Acopy == A);

  // Tests deduction guide for static matrices. Note that this constructs a
  // matrix of ints.
  static constexpr optila::Matrix b({{1}, {2}, {3}});
  // Tests static_convert from int to double.
  static constexpr auto C = A * optila::static_convert<double>(b);

  static_assert(C.num_rows() == 3);
  static_assert(C.num_cols() == 1);

  static constexpr auto D = optila::evaluate(C);
  static_assert(D(0, 0) == 14);
  static_assert(D(1, 0) == 32);
  static_assert(D(2, 0) == 50);

  static constexpr optila::Matrix D2 = C;
  static constexpr auto D3 = optila::evaluate(C);
  static_assert(D == D2);
  static_assert(D == D3);

  static constexpr optila::Scalar E = optila::evaluate(optila::dot(b, b));
  static_assert(E == 14);

  static constexpr auto F =
      optila::evaluate(C + optila::static_convert<double>(b));
  static_assert(F(0, 0) == 15);

  static constexpr auto G = optila::submatrix<1, 1, 2, 2>(A);
  static_assert(G.num_rows() == 2);
  static_assert(G.num_cols() == 2);

  static constexpr auto H = optila::evaluate(G);
  static_assert(H(0, 0) == 5);
  static_assert(H(0, 1) == 6);
  static_assert(H(1, 0) == 8);
  static_assert(H(1, 1) == 9);
#endif
  return 0;
}

int main(int argc, char *argv[]) {
  int retval = 0;
  retval -= active_sub_matrix_tests();
  retval -= constexpr_nested_tests();
  retval -= nested_tests();
  retval -= dynamic_tests();
  retval -= constexpr_tests();
  retval -= other_tests();
  retval -= assignment_tests();
  retval -= random_tests();
  return retval;
}