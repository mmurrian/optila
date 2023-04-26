#include <benchmark/benchmark.h>

#include <Eigen/Dense>
#include <armadillo>
#include <random>
#include <vector>

#include "optila.h"

static constexpr int Nfixed = 4;

// Function to generate a random matrix of size NxN using Eigen
Eigen::MatrixXd eigen_generate_random_matrix(int Nrows, int Ncols) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  Eigen::MatrixXd matrix(Nrows, Ncols);
  for (int i = 0; i < Nrows; ++i) {
    for (int j = 0; j < Ncols; ++j) {
      matrix(i, j) = dis(gen);
    }
  }

  return matrix;
}

optila::Matrix<double, optila::Dynamic, optila::Dynamic>
optila_generate_random_matrix(int Nrows, int Ncols) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  optila::Matrix<double, optila::Dynamic, optila::Dynamic> matrix;
  matrix.resize(Nrows, Ncols);
  for (int i = 0; i < Nrows; ++i) {
    for (int j = 0; j < Ncols; ++j) {
      matrix(i, j) = dis(gen);
    }
  }

  return matrix;
}

arma::mat arma_generate_random_matrix(int Nrows, int Ncols) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  arma::mat matrix(Nrows, Ncols);
  for (int i = 0; i < Nrows; ++i) {
    for (int j = 0; j < Ncols; ++j) {
      matrix(i, j) = dis(gen);
    }
  }

  return matrix;
}

// Benchmark function for matrix multiplication
static void BM_optila_MixedBasic(benchmark::State& state) {
  constexpr int N = Nfixed;  // Matrix size
  optila::Matrix<double, N, N> A = optila_generate_random_matrix(N, N);
  optila::Matrix<double, N, N> B = optila_generate_random_matrix(N, N);
  optila::Matrix<double, N, N> C = optila_generate_random_matrix(N, N);
  optila::Matrix<double, N, N> D = optila_generate_random_matrix(N, N);
  optila::Matrix<double, N, N> E = optila_generate_random_matrix(N, N);
  optila::Matrix<double, N, N> F = optila_generate_random_matrix(N, N);
  optila::Matrix<double, N, N> G = optila_generate_random_matrix(N, N);
  optila::Matrix<double, N, N> H = optila_generate_random_matrix(N, N);

  double result = 0;
  for (auto _ : state) {
    optila::Matrix Z =
        (A * B - C) * optila::transpose(D + E) + F * optila::transpose(G * H);
    result += Z(0, 0);
  }
  benchmark::DoNotOptimize(result);
  state.SetComplexityN(N);
}

// Register the benchmark function with different matrix sizes
BENCHMARK(BM_optila_MixedBasic);

// Benchmark function for matrix multiplication
static void BM_eigen_MixedBasic(benchmark::State& state) {
  constexpr int N = Nfixed;  // Matrix size
  Eigen::Matrix<double, N, N> A = eigen_generate_random_matrix(N, N);
  Eigen::Matrix<double, N, N> B = eigen_generate_random_matrix(N, N);
  Eigen::Matrix<double, N, N> C = eigen_generate_random_matrix(N, N);
  Eigen::Matrix<double, N, N> D = eigen_generate_random_matrix(N, N);
  Eigen::Matrix<double, N, N> E = eigen_generate_random_matrix(N, N);
  Eigen::Matrix<double, N, N> F = eigen_generate_random_matrix(N, N);
  Eigen::Matrix<double, N, N> G = eigen_generate_random_matrix(N, N);
  Eigen::Matrix<double, N, N> H = eigen_generate_random_matrix(N, N);

  double result = 0;
  for (auto _ : state) {
    Eigen::Matrix<double, N, N> Z =
        (A * B - C) * (D + E).transpose() + F * (G * H).transpose();
    result += Z(0, 0);
  }
  benchmark::DoNotOptimize(result);
  state.SetComplexityN(N);
}

// Register the benchmark function with different matrix sizes
BENCHMARK(BM_eigen_MixedBasic);

// Benchmark function for matrix multiplication
static void BM_armadillo_MixedBasic(benchmark::State& state) {
  constexpr int N = Nfixed;  // Matrix size
  arma::mat::fixed<N, N> A = arma_generate_random_matrix(N, N);
  arma::mat::fixed<N, N> B = arma_generate_random_matrix(N, N);
  arma::mat::fixed<N, N> C = arma_generate_random_matrix(N, N);
  arma::mat::fixed<N, N> D = arma_generate_random_matrix(N, N);
  arma::mat::fixed<N, N> E = arma_generate_random_matrix(N, N);
  arma::mat::fixed<N, N> F = arma_generate_random_matrix(N, N);
  arma::mat::fixed<N, N> G = arma_generate_random_matrix(N, N);
  arma::mat::fixed<N, N> H = arma_generate_random_matrix(N, N);

  double result = 0;
  for (auto _ : state) {
    arma::mat::fixed<N, N> Z = (A * B - C) * (D + E).t() + F * (G * H).t();
    result += Z(0, 0);
  }
  benchmark::DoNotOptimize(result);
  state.SetComplexityN(N);
}

// Register the benchmark function with different matrix sizes
BENCHMARK(BM_armadillo_MixedBasic);

// Benchmark function for matrix multiplication
static void BM_optila_FixedMatrixMultiplication(benchmark::State& state) {
  constexpr int N = Nfixed;  // Matrix size
  optila::Matrix<double, N, N> A = optila_generate_random_matrix(N, N);
  optila::Matrix<double, N, N> B = optila_generate_random_matrix(N, N);

  double result = 0;
  for (auto _ : state) {
    optila::Matrix C = A * B;
    result += C(0, 0);
  }
  benchmark::DoNotOptimize(result);
  state.SetComplexityN(N);
}

// Register the benchmark function with different matrix sizes
BENCHMARK(BM_optila_FixedMatrixMultiplication);

// Benchmark function for matrix multiplication
static void BM_eigen_FixedMatrixMultiplication(benchmark::State& state) {
  constexpr int N = Nfixed;  // Matrix size
  Eigen::Matrix<double, N, N> A = eigen_generate_random_matrix(N, N);
  Eigen::Matrix<double, N, N> B = eigen_generate_random_matrix(N, N);

  double result = 0;
  for (auto _ : state) {
    Eigen::Matrix<double, N, N> C = A * B;
    result += C(0, 0);
  }
  benchmark::DoNotOptimize(result);
  state.SetComplexityN(N);
}

// Register the benchmark function with different matrix sizes
BENCHMARK(BM_eigen_FixedMatrixMultiplication);

// Benchmark function for matrix multiplication
static void BM_armadillo_FixedMatrixMultiplication(benchmark::State& state) {
  constexpr int N = Nfixed;  // Matrix size
  arma::mat::fixed<N, N> A = arma_generate_random_matrix(N, N);
  arma::mat::fixed<N, N> B = arma_generate_random_matrix(N, N);

  double result = 0;
  for (auto _ : state) {
    arma::mat::fixed<N, N> C = A * B;
    result += C(0, 0);
  }
  benchmark::DoNotOptimize(result);
  state.SetComplexityN(N);
}

// Register the benchmark function with different matrix sizes
BENCHMARK(BM_armadillo_FixedMatrixMultiplication);

// Benchmark function for matrix multiplication
static void BM_optila_NestedFixedMatrixMultiplication(benchmark::State& state) {
  constexpr int N = Nfixed;  // Matrix size
  optila::Matrix<double, N, N> A = optila_generate_random_matrix(N, N);
  optila::Matrix<double, N, N> B = optila_generate_random_matrix(N, N);

  double result = 0;
  for (auto _ : state) {
    optila::Matrix C = (A * B) * (B * A);
    result += C(0, 0);
  }
  benchmark::DoNotOptimize(result);
  state.SetComplexityN(N);
}

// Register the benchmark function with different matrix sizes
BENCHMARK(BM_optila_NestedFixedMatrixMultiplication);

// Benchmark function for matrix multiplication
static void BM_eigen_NestedFixedMatrixMultiplication(benchmark::State& state) {
  constexpr int N = Nfixed;  // Matrix size
  Eigen::Matrix<double, N, N> A = eigen_generate_random_matrix(N, N);
  Eigen::Matrix<double, N, N> B = eigen_generate_random_matrix(N, N);

  double result = 0;
  for (auto _ : state) {
    Eigen::Matrix<double, N, N> C = (A * B) * (B * A);
    result += C(0, 0);
  }
  benchmark::DoNotOptimize(result);
  state.SetComplexityN(N);
}

// Register the benchmark function with different matrix sizes
BENCHMARK(BM_eigen_NestedFixedMatrixMultiplication);

// Benchmark function for matrix multiplication
static void BM_armadillo_NestedFixedMatrixMultiplication(
    benchmark::State& state) {
  constexpr int N = Nfixed;  // Matrix size
  arma::mat::fixed<N, N> A = arma_generate_random_matrix(N, N);
  arma::mat::fixed<N, N> B = arma_generate_random_matrix(N, N);

  double result = 0;
  for (auto _ : state) {
    arma::mat::fixed<N, N> C = (A * B) * (B * A);
    result += C(0, 0);
  }
  benchmark::DoNotOptimize(result);
  state.SetComplexityN(N);
}

// Register the benchmark function with different matrix sizes
BENCHMARK(BM_armadillo_NestedFixedMatrixMultiplication);

// Benchmark function for matrix multiplication
static void BM_optila_SubMatrixNestedFixedMatrixMultiplication(
    benchmark::State& state) {
  constexpr int N = Nfixed;  // Matrix size
  optila::Matrix<double, N, N> A = optila_generate_random_matrix(N, N);
  optila::Matrix<double, N, N> B = optila_generate_random_matrix(N, N);

  double result = 0;
  for (auto _ : state) {
    optila::Matrix C = optila::submatrix<0, 0, 2, 2>((A * B) * (B * A));
    result += C(0, 0);
  }
  benchmark::DoNotOptimize(result);
  state.SetComplexityN(N);
}

// Register the benchmark function with different matrix sizes
BENCHMARK(BM_optila_SubMatrixNestedFixedMatrixMultiplication);

// Benchmark function for matrix multiplication
static void BM_eigen_SubMatrixNestedFixedMatrixMultiplication(
    benchmark::State& state) {
  constexpr int N = Nfixed;  // Matrix size
  Eigen::Matrix<double, N, N> A = eigen_generate_random_matrix(N, N);
  Eigen::Matrix<double, N, N> B = eigen_generate_random_matrix(N, N);

  double result = 0;
  for (auto _ : state) {
    Eigen::Matrix<double, 2, 2> C = ((A * B) * (B * A)).block<2, 2>(0, 0);
    result += C(0, 0);
  }
  benchmark::DoNotOptimize(result);
  state.SetComplexityN(N);
}

// Register the benchmark function with different matrix sizes
BENCHMARK(BM_eigen_SubMatrixNestedFixedMatrixMultiplication);

// Benchmark function for matrix multiplication
static void BM_armadillo_SubMatrixNestedFixedMatrixMultiplication(
    benchmark::State& state) {
  constexpr int N = Nfixed;  // Matrix size
  arma::mat::fixed<N, N> A = arma_generate_random_matrix(N, N);
  arma::mat::fixed<N, N> B = arma_generate_random_matrix(N, N);

  double result = 0;
  for (auto _ : state) {
    arma::mat::fixed<2, 2> C = ((A * B) * (B * A)).eval().submat(0, 0, 1, 1);
    result += C(0, 0);
  }
  benchmark::DoNotOptimize(result);
  state.SetComplexityN(N);
}

// Register the benchmark function with different matrix sizes
BENCHMARK(BM_armadillo_SubMatrixNestedFixedMatrixMultiplication);

// Benchmark function for matrix multiplication
static void BM_optila_MatrixMultiplicationTranspose(benchmark::State& state) {
  constexpr int N = Nfixed;  // Matrix size
  optila::Matrix<double, N, N> A = optila_generate_random_matrix(N, N);

  double result = 0;
  for (auto _ : state) {
    optila::Matrix C = A * optila::transpose(A);
    result += C(0, 0);
  }
  benchmark::DoNotOptimize(result);
  state.SetComplexityN(N);
}

// Register the benchmark function with different matrix sizes
BENCHMARK(BM_optila_MatrixMultiplicationTranspose);

// Benchmark function for matrix multiplication
static void BM_eigen_MatrixMultiplicationTranspose(benchmark::State& state) {
  constexpr int N = Nfixed;  // Matrix size
  Eigen::Matrix<double, N, N> A = eigen_generate_random_matrix(N, N);

  double result = 0;
  for (auto _ : state) {
    Eigen::Matrix<double, N, N> C = A * A.transpose();
    result += C(0, 0);
  }
  benchmark::DoNotOptimize(result);
  state.SetComplexityN(N);
}

// Register the benchmark function with different matrix sizes
BENCHMARK(BM_eigen_MatrixMultiplicationTranspose);

// Benchmark function for matrix multiplication
static void BM_armadillo_MatrixMultiplicationTranspose(
    benchmark::State& state) {
  constexpr int N = Nfixed;  // Matrix size
  arma::mat::fixed<N, N> A = arma_generate_random_matrix(N, N);

  double result = 0;
  for (auto _ : state) {
    arma::mat::fixed<N, N> C = A * A.t();
    result += C(0, 0);
  }
  benchmark::DoNotOptimize(result);
  state.SetComplexityN(N);
}

// Register the benchmark function with different matrix sizes
BENCHMARK(BM_armadillo_MatrixMultiplicationTranspose);

// Benchmark function for matrix multiplication
static void BM_optila_MatrixMultiplication(benchmark::State& state) {
  int N = state.range(0);  // Matrix size
  optila::Matrix A = optila_generate_random_matrix(N, N);
  optila::Matrix B = optila_generate_random_matrix(N, N);

  double result = 0;
  for (auto _ : state) {
    optila::Matrix C = A * B;
    result += C(0, 0);
  }
  benchmark::DoNotOptimize(result);
  state.SetComplexityN(N);
}

// Register the benchmark function with different matrix sizes
BENCHMARK(BM_optila_MatrixMultiplication)
    ->RangeMultiplier(2)
    ->Range(8, 1024)
    ->Complexity();

// Benchmark function for matrix multiplication
static void BM_eigen_MatrixMultiplication(benchmark::State& state) {
  int N = state.range(0);  // Matrix size
  Eigen::MatrixXd A = eigen_generate_random_matrix(N, N);
  Eigen::MatrixXd B = eigen_generate_random_matrix(N, N);

  double result = 0;
  for (auto _ : state) {
    Eigen::MatrixXd C = A * B;
    result += C(0, 0);
  }
  benchmark::DoNotOptimize(result);
  state.SetComplexityN(N);
}

// Register the benchmark function with different matrix sizes
BENCHMARK(BM_eigen_MatrixMultiplication)
    ->RangeMultiplier(2)
    ->Range(8, 1024)
    ->Complexity();

// Benchmark function for matrix multiplication
static void BM_armadillo_MatrixMultiplication(benchmark::State& state) {
  int N = state.range(0);  // Matrix size
  arma::mat A = arma_generate_random_matrix(N, N);
  arma::mat B = arma_generate_random_matrix(N, N);

  double result = 0;
  for (auto _ : state) {
    arma::mat C = A * B;
    result += C(0, 0);
  }
  benchmark::DoNotOptimize(result);
  state.SetComplexityN(N);
}

// Register the benchmark function with different matrix sizes
BENCHMARK(BM_armadillo_MatrixMultiplication)
    ->RangeMultiplier(2)
    ->Range(8, 1024)
    ->Complexity();

BENCHMARK_MAIN();
