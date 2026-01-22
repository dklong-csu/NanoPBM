#include <nanopbm/particle_growth.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>

#include <array>
#include <boost/ut.hpp>
#include <sundials/sundials_context.hpp>
#include <vector>

namespace testing {
using namespace boost::ut;
using namespace NanoPBM;
template <typename T>
void do_particle_growth_test(const T& kernel, const sunrealtype k1, const sunrealtype k2) {
  sundials::Context sunctx;
  constexpr sunindextype n_odes = 6;

  // ---- exact solutions ----
  std::array<sunrealtype, n_odes> expected_rhs = {-k1 - k2, 1.5 * (k1 + k2), 2.5 * (k1 + k2),
                                                  -k1,      k1 - k2,         k2};

  std::array<sunrealtype, n_odes> expected_Jv = {
      -k1 - k2 - k1 - k2, 1.5 * (k1 + k2 + k1 + k2), 2.5 * (k1 + k2 + k1 + k2),
      -k1 - k1,           k1 - k2 + k1 - k2,         k2 + k2};

  std::array<std::array<sunrealtype, n_odes>, n_odes> expected_Jac = {
      {{-k1 - k2, 0, 0, -k1, -k2, 0},
       {1.5 * (k1 + k2), 0, 0, 1.5 * k1, 1.5 * k2, 0},
       {2.5 * (k1 + k2), 0, 0, 2.5 * k1, 2.5 * k2, 0},
       {-k1, 0, 0, -k1, 0, 0},
       {k1 - k2, 0, 0, k1, -k2, 0},
       {k2, 0, 0, 0, k2, 0}}};


  N_Vector y    = N_VNew_Serial(n_odes, sunctx);
  N_Vector ydot = N_VNew_Serial(n_odes, sunctx);
  N_Vector v    = N_VNew_Serial(n_odes, sunctx);
  N_Vector Jv   = N_VNew_Serial(n_odes, sunctx);
  N_Vector fy   = N_VNew_Serial(n_odes, sunctx);
  N_Vector tmp1 = N_VNew_Serial(n_odes, sunctx);
  N_Vector tmp2 = N_VNew_Serial(n_odes, sunctx);
  N_Vector tmp3 = N_VNew_Serial(n_odes, sunctx);

  N_VConst(0, y);
  N_VConst(0, ydot);
  N_VConst(0, v);
  N_VConst(0, Jv);
  N_VConst(0, fy);
  N_VConst(0, tmp1);
  N_VConst(0, tmp2);
  N_VConst(0, tmp3);

  std::vector<N_Vector> all_n_vecs    = {y, ydot, v, Jv, fy, tmp1, tmp2, tmp3};
  std::vector<N_Vector> n_vec_is_zero = {fy, tmp1, tmp2, tmp3};
  std::vector<N_Vector> n_vec_is_one  = {y, v};

  for (const auto& vec : all_n_vecs) {
    const auto data = N_VGetArrayPointer(vec);
    for (int idx = 0; idx < n_odes; ++idx) {
      expect(data[idx] == 0._d) << "Index: " << idx;
    }
  }

  // make growth

  ParticleGrowth growth(0, {{1, 1.5}, {2, 2.5}}, 2, 3, 3, kernel);

  N_VConst(1, y);
  N_VConst(1, v);

  auto y_data    = N_VGetArrayPointer(y);
  auto v_data    = N_VGetArrayPointer(v);
  auto ydot_data = N_VGetArrayPointer(ydot);
  auto Jv_data   = N_VGetArrayPointer(Jv);

  // ---- Check f(y) ----
  growth.add_to_rhs(0, y, ydot);

  for (int idx = 0; idx < n_odes; ++idx) {
    expect(ydot_data[idx] == _d(expected_rhs.at(idx)))
        << "\n"
        << "Index:" << idx << "\n"
        << "Expected: " << expected_rhs.at(idx) << "\n"
        << "Actual: " << ydot_data[idx] << "\n";
  }

  for (const auto& vec : n_vec_is_zero) {
    const auto data = N_VGetArrayPointer(vec);
    for (int idx = 0; idx < n_odes; ++idx) {
      expect(data[idx] == 0._d) << "\n"
                                << "Index: " << idx;
    }
  }

  for (const auto& vec : n_vec_is_one) {
    const auto data = N_VGetArrayPointer(vec);
    for (int idx = 0; idx < n_odes; ++idx) {
      expect(data[idx] == 1._d) << "\n"
                                << "Index: " << idx;
    }
  }


  // ---- Check Jv product ----
  growth.add_to_jac_times_v(v, Jv, 0, y, ydot, tmp1);

  for (int idx = 0; idx < n_odes; ++idx) {
    expect(Jv_data[idx] == _d(expected_Jv.at(idx))) << "\n"
                                                    << "Index:" << idx << "\n"
                                                    << "Expected: " << expected_Jv.at(idx) << "\n"
                                                    << "Actual: " << Jv_data[idx] << "\n";
  }

  for (const auto& vec : n_vec_is_zero) {
    const auto data = N_VGetArrayPointer(vec);
    for (int idx = 0; idx < n_odes; ++idx) {
      expect(data[idx] == 0._d) << "\n"
                                << "Index: " << idx;
    }
  }

  for (const auto& vec : n_vec_is_one) {
    const auto data = N_VGetArrayPointer(vec);
    for (int idx = 0; idx < n_odes; ++idx) {
      expect(data[idx] == 1._d) << "\n"
                                << "Index: " << idx;
    }
  }

  // ---- Check forming J ----
  gko::matrix_data<sunrealtype, sunindextype> jac_data;
  growth.add_to_jac(jac_data, 0, y, fy, tmp1, tmp2, tmp3);
  std::array<std::array<sunrealtype, n_odes>, n_odes> matrix;

  for (int r = 0; r < n_odes; ++r) {
    for (int c = 0; c < n_odes; ++c) {
      matrix.at(r).at(c) = 0.;
    }
  }

  for (const auto& d : jac_data.nonzeros) {
    const auto c = d.column;
    const auto r = d.row;
    matrix.at(r).at(c) += d.value;
  }

  for (int r = 0; r < n_odes; ++r) {
    for (int c = 0; c < n_odes; ++c) {
      expect(matrix.at(r).at(c) == _d(expected_Jac.at(r).at(c)))
          << "\n"
          << "r: " << r << "  c: " << c << "\n"
          << "Expected: " << expected_Jac.at(r).at(c) << "\n"
          << "Actual: " << matrix.at(r).at(c) << "\n";
    }
  }

  for (const auto& vec : n_vec_is_zero) {
    const auto data = N_VGetArrayPointer(vec);
    for (int idx = 0; idx < n_odes; ++idx) {
      expect(data[idx] == 0._d) << "\n"
                                << "Index: " << idx;
    }
  }

  for (const auto& vec : n_vec_is_one) {
    const auto data = N_VGetArrayPointer(vec);
    for (int idx = 0; idx < n_odes; ++idx) {
      expect(data[idx] == 1._d) << "\n"
                                << "Index: " << idx;
    }
  }

  // ---- Clean up ----
  N_VDestroy(tmp3);
  N_VDestroy(tmp2);
  N_VDestroy(tmp1);
  N_VDestroy(fy);
  N_VDestroy(Jv);
  N_VDestroy(v);
  N_VDestroy(ydot);
  N_VDestroy(y);
}
}  // namespace testing

int main() {
  using namespace boost::ut;
  using namespace NanoPBM;
  using namespace testing;


  "Particle Growth"_test = [&] {
    "Constant Kernel"_test = [&](const auto rate) {
      const ConstantGrowthKernel kernel(rate);
      do_particle_growth_test(kernel, rate, rate);
    } | std::vector<double>{1, 2, 42};

    "Constant x Size Kernel"_test = [&] {
      "Constant"_test = [&](const auto C) {
        "Size Kernel Power"_test = [&](const auto P) {
          const auto sizefcn = [=](const sunindextype i) { return 1.0 * std::pow(i, P); };
          const ConstantTimesSizeGrowthKernel kernel(sizefcn, C);
          do_particle_growth_test(kernel, C * std::pow(2, P), C * std::pow(3, P));
        } | std::vector<int>{1, 2, 3};
      } | std::vector<double>{1, 2, 42};
    };

    "Lambda Fcn Kernel"_test = [&] {
      const auto kernel = [](const sunindextype i) { return 1.0; };
      do_particle_growth_test(kernel, 1, 1);
    };
  };
}