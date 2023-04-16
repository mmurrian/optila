#include "optila.h"

int main(int argc, char *argv[]) {
  static constexpr optila::Matrix<double, 3, 3> A{
      {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}};
  static constexpr optila::Matrix<double, 3, 1> b{{1}, {2}, {3}};
  static constexpr auto C = A * b;

  static_assert(decltype(C)::num_rows() == 3);
  static_assert(decltype(C)::num_cols() == 1);

  static constexpr auto D = optila::evaluate(C);
  static_assert(D(0, 0) == 14);
  static_assert(D(1, 0) == 32);
  static_assert(D(2, 0) == 50);

  static constexpr auto E = optila::evaluate(optila::dot(b, b));
  static_assert(E == 14);

  static constexpr auto F = optila::evaluate(C + b);
  static_assert(F(0, 0) == 15);

  return 0;
}