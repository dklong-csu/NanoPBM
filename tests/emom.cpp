
#include "emom.h"

#include <nvector/nvector_serial.h>
#include <sundials/sundials_types.h>

#include <boost/ut.hpp>
#include <cmath>
#include <iomanip>
#include <sundials/sundials_context.hpp>

namespace testing {
using namespace boost::ut;
void do_test_emom(const sunrealtype kg, const sunrealtype vm, const sunrealtype xn,
                  const sunrealtype N, const sunrealtype mu0, const sunrealtype mu1,
                  const sunrealtype mu2, const sunrealtype A, const sunrealtype Pn,
                  const sunrealtype C, const sunrealtype stoich) {
  sundials::Context sunctx;

  // Test growth mechanism is:
  //      A + P_N -> P_continuous + C
  constexpr sunindextype n_odes  = 6;
  constexpr sunindextype idx_mu0 = 0;  // moment 0
  constexpr sunindextype idx_mu1 = 1;  // moment 1
  constexpr sunindextype idx_mu2 = 2;  // moment 2
  constexpr sunindextype idx_A   = 3;  // precursor
  constexpr sunindextype idx_Pn  = 4;  // largest discrete particle
  constexpr sunindextype idx_C   = 5;  // created in growth reaction

  // ---- calculations needed ----
  const sunrealtype k_mu = kg * std::cbrt(2 * vm / (9 * std::numbers::pi));
  const sunrealtype k_bc = kg * std::pow(1. * N, 2. / 3.);
  const sunrealtype k_D  = kg * std::pow(std::numbers::pi / (6 * vm), 2. / 3.);

  // ---- exact solution ----
  std::array<sunrealtype, n_odes> expected_rhs = {k_bc * A * Pn,
                                                  xn * k_bc * A * Pn + k_mu * A * mu0,
                                                  xn * xn * k_bc * A * Pn + 2 * k_mu * A * mu1,
                                                  -k_D * A * mu2,
                                                  0,
                                                  stoich * k_D * A * mu2};

  std::array<sunrealtype, n_odes> expected_Jv = {
      k_bc * Pn + k_bc * A,
      k_mu * A + k_mu * mu0 + xn * k_bc * Pn + xn * k_bc * A,
      2 * k_mu * A + 2 * k_mu * mu1 + xn * xn * k_bc * Pn + xn * xn * k_bc * A,
      -k_D * A - k_D * mu2,
      0,
      stoich * k_D * A + stoich * k_D * mu2};

  std::array<std::array<sunrealtype, n_odes>, n_odes> expected_Jac = {
      {{0, 0, 0, k_bc * Pn, k_bc * A},
       {k_mu * A, 0, 0, k_mu * mu0 + xn * k_bc * Pn, xn * k_bc * A},
       {0, 2 * k_mu * A, 0, 2 * k_mu * mu1 + xn * xn * k_bc * Pn, xn * xn * k_bc * A},
       {0, 0, -k_D * A, -k_D * mu2, 0, 0},
       {0, 0, 0, 0, 0, 0},
       {0, 0, stoich * k_D * A, stoich * k_D * mu2, 0, 0}}};


  // ---- Form vectors ----

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

  N_VConst(1, v);

  auto y_data    = N_VGetArrayPointer(y);
  auto v_data    = N_VGetArrayPointer(v);
  auto ydot_data = N_VGetArrayPointer(ydot);
  auto Jv_data   = N_VGetArrayPointer(Jv);

  y_data[idx_mu0] = mu0;
  y_data[idx_mu1] = mu1;
  y_data[idx_mu2] = mu2;
  y_data[idx_A]   = A;
  y_data[idx_Pn]  = Pn;
  y_data[idx_C]   = C;

  // ---- Make emom object ----
  NanoPBM::EMoM emom(idx_mu0, idx_mu1, idx_mu2, idx_A, idx_Pn, kg, vm, xn, N, {{idx_C, stoich}});

  // ---- Test rhs ----
  emom.add_to_rhs(0, y, ydot);

  for (int idx = 0; idx < n_odes; ++idx) {
    expect(std::abs(ydot_data[idx] - expected_rhs.at(idx)) <= 0.0000000001_d)
        << "\n"
        << std::setprecision(12) << "Index:" << idx << "\n"
        << "Expected: " << expected_rhs.at(idx) << "\n"
        << "Actual  : " << ydot_data[idx] << "\n";
  }

  // ---- Test Jv ----
  emom.add_to_jac_times_v(v, Jv, 0, y, ydot, tmp1);

  for (int idx = 0; idx < n_odes; ++idx) {
    expect(std::abs(Jv_data[idx] - expected_Jv.at(idx)) <= 0.0000000001_d)
        << "\n"
        << std::setprecision(12) << "Index:" << idx << "\n"
        << "Expected: " << expected_Jv.at(idx) << "\n"
        << "Actual  : " << Jv_data[idx] << "\n";
  }

  // ---- Test J ----
  gko::matrix_data<sunrealtype, sunindextype> jac_data;
  emom.add_to_jac(jac_data, 0, y, fy, tmp1, tmp2, tmp3);
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
      expect(std::abs(matrix.at(r).at(c) - expected_Jac.at(r).at(c)) <= 0.0000000001_d)
          << "\n"
          << std::setprecision(12) << "r: " << r << "  c: " << c << "\n"
          << "Expected: " << expected_Jac.at(r).at(c) << "\n"
          << "Actual  : " << matrix.at(r).at(c) << "\n";
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
  using namespace testing;
  "EMoM"_test = [] {
    "Growth rate"_test = [](const auto kg) {
      "Volume"_test = [&](const auto vm) {
        "Size"_test = [&](const auto xn) {
          "N atoms"_test = [&](const auto N) {
            "Moment 0"_test = [&](const auto mu0) {
              "Moment 1"_test = [&](const auto mu1) {
                "Moment 2"_test = [&](const auto mu2) {
                  "Precursor Concentration"_test = [&](const auto A) {
                    "Particle Concentration"_test = [&](const auto Pn) {
                      "Created Species Concentration"_test = [&](const auto C) {
                        "Created Species Stoichiometry"_test = [&](const auto stoich) {
                          do_test_emom(kg, vm, xn, N, mu0, mu1, mu2, A, Pn, C, stoich);
                        } | std::vector<sunrealtype>{0, 1, 2};
                      } | std::vector<sunrealtype>{0, 1, 2};
                    } | std::vector<sunrealtype>{0, 1, 2};
                  } | std::vector<sunrealtype>{0, 1, 2};
                } | std::vector<sunrealtype>{0, 1, 2};
              } | std::vector<sunrealtype>{0, 1, 2};
            } | std::vector<sunrealtype>{0, 1, 2};
          } | std::vector<sunindextype>{1, 2};
        } | std::vector<sunrealtype>{1, 2};
      } | std::vector<sunrealtype>{1, 2};
    } | std::vector<sunrealtype>{0, 1, 2};
  };
}