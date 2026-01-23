#ifndef NANOPBM_BENCHMARK_STATISTICS_H
#define NANOPBM_BENCHMARK_STATISTICS_H

#include <cmath>
#include <stdexcept>

namespace NanoPBM {
class BenchmarkStatistics {
 public:
  void push(double timing) {
    ++n;
    double delta = timing - average;
    average += delta / n;
    double delta2 = timing - average;
    second_moment += delta * delta2;
  }

  unsigned int count() const { return n; }

  double mean() const { return average; }

  double variance() const {
    if (n < 2) throw std::logic_error("need at least two samples");
    return second_moment / (n - 1);
  }


  double standard_deviation() const { return std::sqrt(variance()); }


 private:
  unsigned int n       = 0;
  double average       = 0;
  double second_moment = 0;
};
}  // namespace NanoPBM

#endif  // NANOPBM_BENCHMARK_STATISTICS_H