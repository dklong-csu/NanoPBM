
#include "chemical_reaction.h"

#include <nvector/nvector_serial.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>

#include <array>
#include <boost/ut.hpp>
#include <sundials/sundials_context.hpp>

namespace testing {
using namespace boost::ut;
using namespace NanoPBM;

struct RxnRate {
  RxnRate(const sunrealtype kf, const sunrealtype kb) : kf(kf), kb(kb) {}
  const sunrealtype kf, kb;
  sunrealtype forward(const N_Vector y, const sunrealtype t, const ReactionParameterSet& reactants,
                      const ReactionParameterSet& products) const {
    sunrealtype rate  = kf;
    const auto y_data = N_VGetArrayPointer(y);
    for (const auto& prm : reactants) {
      rate *= std::pow(y_data[prm.index], prm.order);
    }
    return rate;
  }

  sunrealtype backward(const N_Vector y, const sunrealtype t, const ReactionParameterSet& reactants,
                       const ReactionParameterSet& products) const {
    sunrealtype rate  = kb;
    const auto y_data = N_VGetArrayPointer(y);
    for (const auto& prm : products) {
      rate *= std::pow(y_data[prm.index], prm.order);
    }
    return rate;
  }

  sunrealtype df_dr(const ReactionParameters& deriv_prm, const N_Vector y, const sunrealtype t,
                    const ReactionParameterSet& reactants,
                    const ReactionParameterSet& products) const {
    sunrealtype rate  = kf;
    const auto y_data = N_VGetArrayPointer(y);
    bool found        = false;
    for (const auto& prm : reactants) {
      if (prm.index == deriv_prm.index) {
        rate *= prm.order * std::pow(y_data[prm.index], prm.order - 1);
        found = true;
      } else {
        rate *= std::pow(y_data[prm.index], prm.order);
      }
    }
    return rate * found;
  }

  sunrealtype db_dp(const ReactionParameters& deriv_prm, const N_Vector y, const sunrealtype t,
                    const ReactionParameterSet& reactants,
                    const ReactionParameterSet& products) const {
    sunrealtype rate  = kb;
    const auto y_data = N_VGetArrayPointer(y);
    bool found        = false;
    for (const auto& prm : products) {
      if (prm.index == deriv_prm.index) {
        rate *= prm.order * std::pow(y_data[prm.index], prm.order - 1);
        found = true;
      } else {
        rate *= std::pow(y_data[prm.index], prm.order);
      }
    }
    return rate * found;
  }
};

void do_rxn_test(const sunrealtype ca, const sunrealtype cb, const sunrealtype cc,
                 const sunrealtype alpha, const sunrealtype beta, const sunrealtype gamma,
                 const sunrealtype pa, const sunrealtype pb, const sunrealtype pc,
                 const sunrealtype kf, const sunrealtype kb) {
  sundials::Context sunctx;
  constexpr sunindextype n_odes = 3;

  // ---- exact solutions ----
  std::array<sunrealtype, n_odes> expected_rhs = {
      alpha * (kb * std::pow(cc, pc) - kf * std::pow(ca, pa) * std::pow(cb, pb)),
      beta * (kb * std::pow(cc, pc) - kf * std::pow(ca, pa) * std::pow(cb, pb)),
      -gamma * (kb * std::pow(cc, pc) - kf * std::pow(ca, pa) * std::pow(cb, pb)),
  };

  const auto da = kf * pa * std::pow(ca, pa - 1) * std::pow(cb, pb);
  const auto db = kf * std::pow(ca, pa) * pb * std::pow(cb, pb - 1);
  const auto dc = kb * pc * std::pow(cc, pc - 1);

  std::array<sunrealtype, n_odes> expected_Jv = {alpha * (-da - db + dc), beta * (-da - db + dc),
                                                 gamma * (da + db - dc)};

  std::array<std::array<sunrealtype, n_odes>, n_odes> expected_Jac = {
      {{-alpha * da, -alpha * db, alpha * dc},
       {-beta * da, -beta * db, beta * dc},
       {gamma * da, gamma * db, -gamma * dc}}};


  // ---- vector setup ----

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
  std::vector<N_Vector> n_vec_is_one  = {v};

  for (const auto& vec : all_n_vecs) {
    const auto data = N_VGetArrayPointer(vec);
    for (int idx = 0; idx < n_odes; ++idx) {
      expect(data[idx] == 0._d) << "Index: " << idx;
    }
  }

  // ---- make reaction ----
  const RxnRate rxn_rate(kf, kb);
  const ReactionParameters A(0, alpha, pa);
  const ReactionParameters B(1, beta, pb);
  const ReactionParameters C(2, gamma, pc);
  ReactionParameterSet R;
  R->push_back(A);
  R->push_back(B);
  ReactionParameterSet P;
  P->push_back(C);
  const ChemicalReaction rxn(rxn_rate, R, P);

  // ---- Vector values ----
  auto y_data    = N_VGetArrayPointer(y);
  auto v_data    = N_VGetArrayPointer(v);
  auto ydot_data = N_VGetArrayPointer(ydot);
  auto Jv_data   = N_VGetArrayPointer(Jv);

  N_VConst(1, v);
  y_data[0] = ca;
  y_data[1] = cb;
  y_data[2] = cc;

  // ---- test rhs ----
  rxn.add_to_rhs(0, y, ydot);

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

  expect(y_data[0] == _d(ca));
  expect(y_data[1] == _d(cb));
  expect(y_data[2] == _d(cc));

  // ---- test Jv product ----
  rxn.add_to_jac_times_v(v, Jv, 0, y, ydot, tmp1);

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

  expect(y_data[0] == _d(ca));
  expect(y_data[1] == _d(cb));
  expect(y_data[2] == _d(cc));

  // ---- test form J ----
  gko::matrix_data<sunrealtype, sunindextype> jac_data;
  rxn.add_to_jac(jac_data, 0, y, fy, tmp1, tmp2, tmp3);
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

  expect(y_data[0] == _d(ca));
  expect(y_data[1] == _d(cb));
  expect(y_data[2] == _d(cc));

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
  using namespace testing;
  "Chemical Reaction"_test = [] {
    "Concentration A"_test = [](const auto ca) {
      "Concentration B"_test = [&](const auto cb) {
        "Concentration C"_test = [&](const auto cc) {
          "Alpha"_test = [&](const auto alpha) {
            "Beta"_test = [&](const auto beta) {
              "Gamma"_test = [&](const auto gamma) {
                "A order"_test = [&](const auto pa) {
                  "B order"_test = [&](const auto pb) {
                    "C order"_test = [&](const auto pc) {
                      "Forward rate"_test = [&](const auto kf) {
                        "Backward rate"_test = [&](const auto kb) {
                          do_rxn_test(ca, cb, cc, alpha, beta, gamma, pa, pb, pc, kf, kb);
                        } | std::vector<sunrealtype>{0, 1, 2};
                      } | std::vector<sunrealtype>{0, 1, 2};
                    } | std::vector<sunrealtype>{1, 2};
                  } | std::vector<sunrealtype>{1, 2};
                } | std::vector<sunrealtype>{1, 2};
              } | std::vector<sunrealtype>{1, 2};
            } | std::vector<sunrealtype>{1, 2};
          } | std::vector<sunrealtype>{1, 2};
        } | std::vector<sunrealtype>{1, 4};
      } | std::vector<sunrealtype>{1, 3};
    } | std::vector<sunrealtype>{1, 2};
  };
}