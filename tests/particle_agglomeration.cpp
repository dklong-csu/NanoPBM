#include "particle_agglomeration.h"

#include <nvector/nvector_serial.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>

#include <boost/ut.hpp>
#include <sundials/sundials_context.hpp>
#include <vector>

#include "chemical_reaction.h"
#include "ode_contribution.h"

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

template <typename T>
void do_agglom_test(const T& kernel, const sunrealtype a11, const sunrealtype a12,
                    const sunrealtype a13, const sunrealtype a22, const sunrealtype a23,
                    const sunrealtype a33, const sunrealtype p1, const sunrealtype p2,
                    const sunrealtype p3, const sunrealtype p4, const sunrealtype p5,
                    const sunrealtype p6) {
  sundials::Context sunctx;
  constexpr sunindextype n_odes = 6;

  // ---- exact solutions ----
  // Chemical reactions are validated, so just form agglomeration via chemical reactions and check
  ManyOdeContributions agglom_as_rxns;
  {
    const ReactionParameters P1(0, 1, 1);
    const ReactionParameters P2(1, 1, 1);
    const ReactionParameters P3(2, 1, 1);
    const ReactionParameters P4(3, 1, 1);
    const ReactionParameters P5(4, 1, 1);
    const ReactionParameters P6(5, 1, 1);

    const ReactionParameters P1s(0, 2, 2);
    const ReactionParameters P2s(1, 2, 2);
    const ReactionParameters P3s(2, 2, 2);

    const RxnRate a11_rate(a11, 0);
    ReactionParameterSet r11;
    ReactionParameterSet p11;
    r11->push_back(P1s);
    p11->push_back(P2);
    const ChemicalReaction rxn11(a11_rate, r11, p11);
    agglom_as_rxns.add_contribution(rxn11);

    const RxnRate a12_rate(a12, 0);
    ReactionParameterSet r12;
    ReactionParameterSet p12;
    r12->push_back(P1);
    r12->push_back(P2);
    p12->push_back(P3);
    const ChemicalReaction rxn12(a12_rate, r12, p12);
    agglom_as_rxns.add_contribution(rxn12);


    const RxnRate a13_rate(a13, 0);
    ReactionParameterSet r13;
    ReactionParameterSet p13;
    r13->push_back(P1);
    r13->push_back(P3);
    p13->push_back(P4);
    const ChemicalReaction rxn13(a13_rate, r13, p13);
    agglom_as_rxns.add_contribution(rxn13);


    const RxnRate a22_rate(a22, 0);
    ReactionParameterSet r22;
    ReactionParameterSet p22;
    r22->push_back(P2s);
    p22->push_back(P4);
    const ChemicalReaction rxn22(a22_rate, r22, p22);
    agglom_as_rxns.add_contribution(rxn22);


    const RxnRate a23_rate(a23, 0);
    ReactionParameterSet r23;
    ReactionParameterSet p23;
    r23->push_back(P2);
    r23->push_back(P3);
    p23->push_back(P5);
    const ChemicalReaction rxn23(a23_rate, r23, p23);
    agglom_as_rxns.add_contribution(rxn23);


    const RxnRate a33_rate(a33, 0);
    ReactionParameterSet r33;
    ReactionParameterSet p33;
    r33->push_back(P3s);
    p33->push_back(P6);
    const ChemicalReaction rxn33(a33_rate, r33, p33);
    agglom_as_rxns.add_contribution(rxn33);
  }

  // ---- vector setup ----
  N_Vector y          = N_VNew_Serial(n_odes, sunctx);
  N_Vector ydot       = N_VNew_Serial(n_odes, sunctx);
  N_Vector ydot_exact = N_VNew_Serial(n_odes, sunctx);
  N_Vector v          = N_VNew_Serial(n_odes, sunctx);
  N_Vector Jv         = N_VNew_Serial(n_odes, sunctx);
  N_Vector Jv_exact   = N_VNew_Serial(n_odes, sunctx);
  N_Vector fy         = N_VNew_Serial(n_odes, sunctx);
  N_Vector tmp1       = N_VNew_Serial(n_odes, sunctx);
  N_Vector tmp2       = N_VNew_Serial(n_odes, sunctx);
  N_Vector tmp3       = N_VNew_Serial(n_odes, sunctx);

  N_VConst(0, y);
  N_VConst(0, ydot);
  N_VConst(0, ydot_exact);
  N_VConst(0, v);
  N_VConst(0, Jv);
  N_VConst(0, Jv_exact);
  N_VConst(0, fy);
  N_VConst(0, tmp1);
  N_VConst(0, tmp2);
  N_VConst(0, tmp3);


  // ---- Make agglomeration ----
  const ParticleAgglomeration agglom(1, 3, 0, kernel);

  // ---- vector values ----
  auto y_data          = N_VGetArrayPointer(y);
  auto v_data          = N_VGetArrayPointer(v);
  auto ydot_data       = N_VGetArrayPointer(ydot);
  auto Jv_data         = N_VGetArrayPointer(Jv);
  auto ydot_exact_data = N_VGetArrayPointer(ydot_exact);
  auto Jv_exact_data   = N_VGetArrayPointer(Jv_exact);

  N_VConst(1, v);
  y_data[0] = p1;
  y_data[1] = p2;
  y_data[2] = p3;
  y_data[3] = p4;
  y_data[4] = p5;
  y_data[5] = p6;

  // ---- test rhs ----
  agglom.add_to_rhs(0, y, ydot);
  agglom_as_rxns.add_to_rhs(0, y, ydot_exact);
  for (int i = 0; i < n_odes; ++i) {
    expect(ydot_data[i] == _d(ydot_exact_data[i])) << "\nIndex: " << i << "\n"
                                                   << "Expected: " << ydot_exact_data[i] << "\n"
                                                   << "Actual: " << ydot_data[i] << "\n";
  }

  // ---- test Jv product ----
  agglom.add_to_jac_times_v(v, Jv, 0, y, ydot, tmp1);
  agglom_as_rxns.add_to_jac_times_v(v, Jv_exact, 0, y, ydot, tmp1);
  for (int i = 0; i < n_odes; ++i) {
    expect(Jv_data[i] == _d(Jv_exact_data[i])) << "\nIndex: " << i << "\n"
                                               << "Expected: " << Jv_exact_data[i] << "\n"
                                               << "Actual: " << Jv_data[i] << "\n";
  }

  // ---- test form Jac ----
  gko::matrix_data<sunrealtype, sunindextype> jac_data;
  agglom.add_to_jac(jac_data, 0, y, fy, tmp1, tmp2, tmp3);
  gko::matrix_data<sunrealtype, sunindextype> exact_jac_data;
  agglom_as_rxns.add_to_jac(exact_jac_data, 0, y, fy, tmp1, tmp2, tmp3);

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

  std::array<std::array<sunrealtype, n_odes>, n_odes> exact_matrix;

  for (int r = 0; r < n_odes; ++r) {
    for (int c = 0; c < n_odes; ++c) {
      exact_matrix.at(r).at(c) = 0.;
    }
  }

  for (const auto& d : exact_jac_data.nonzeros) {
    const auto c = d.column;
    const auto r = d.row;
    exact_matrix.at(r).at(c) += d.value;
  }

  for (int r = 0; r < n_odes; ++r) {
    for (int c = 0; c < n_odes; ++c) {
      expect(matrix.at(r).at(c) == _d(exact_matrix.at(r).at(c)))
          << "\n"
          << "r: " << r << "  c: " << c << "\n"
          << "Expected: " << exact_matrix.at(r).at(c) << "\n"
          << "Actual: " << matrix.at(r).at(c) << "\n";
    }
  }


  // ---- Clean up ----
  N_VDestroy(tmp3);
  N_VDestroy(tmp2);
  N_VDestroy(tmp1);
  N_VDestroy(fy);
  N_VDestroy(Jv_exact);
  N_VDestroy(Jv);
  N_VDestroy(v);
  N_VDestroy(ydot_exact);
  N_VDestroy(ydot);
  N_VDestroy(y);
}

}  // namespace testing


int main() {
  using namespace boost::ut;
  using namespace testing;
  using namespace NanoPBM;

  "Agglomeration"_test = [] {
    "P1 Concentration"_test = [](const auto p1) {
      "P2 Concentration"_test = [&](const auto p2) {
        "P3 Concentration"_test = [&](const auto p3) {
          "P4 Concentration"_test = [&](const auto p4) {
            "P5 Concentration"_test = [&](const auto p5) {
              "P6 Concentration"_test = [&](const auto p6) {
                "Constant Kernel"_test = [&](const auto C) {
                  const ConstantAgglomerationKernel kernel(C);
                  const auto a11 = C;
                  const auto a12 = C;
                  const auto a13 = C;
                  const auto a22 = C;
                  const auto a23 = C;
                  const auto a33 = C;
                  do_agglom_test(kernel, a11, a12, a13, a22, a23, a33, p1, p2, p3, p4, p5, p6);
                } | std::vector<sunrealtype>{1, 2};

                "Additive Kernel"_test = [&](const auto C) {
                  const auto a2s = [](const sunindextype a) { return 1.0 * a; };
                  const AdditiveAgglomerationKernel kernel(a2s, C);
                  const auto a11 = C * (1 + 1);
                  const auto a12 = C * (1 + 2);
                  const auto a13 = C * (1 + 3);
                  const auto a22 = C * (2 + 2);
                  const auto a23 = C * (2 + 3);
                  const auto a33 = C * (3 + 3);
                  do_agglom_test(kernel, a11, a12, a13, a22, a23, a33, p1, p2, p3, p4, p5, p6);
                } | std::vector<sunrealtype>{1, 2};

                "Multiplicative Kernel"_test = [&](const auto C) {
                  const auto a2s = [](const sunindextype a) { return 1.0 * a; };
                  const MultiplicativeAgglomerationKernel kernel(a2s, C);
                  const auto a11 = C * (1 * 1);
                  const auto a12 = C * (1 * 2);
                  const auto a13 = C * (1 * 3);
                  const auto a22 = C * (2 * 2);
                  const auto a23 = C * (2 * 3);
                  const auto a33 = C * (3 * 3);
                  do_agglom_test(kernel, a11, a12, a13, a22, a23, a33, p1, p2, p3, p4, p5, p6);
                } | std::vector<sunrealtype>{1, 2};

                "Diffusion Kernel"_test = [&] {
                  "Fractal dimension"_test = [&](const auto fd) {
                    const sunrealtype T         = 300;
                    const sunrealtype viscosity = 1.e-6;
                    const sunrealtype kB        = 1.380649e-23;
                    const auto a2s              = [](const sunindextype a) { return 1.0 * a; };
                    const DiffusionLimitedAgglomerationKernel kernel(a2s, T, viscosity, fd);
                    const auto ifd = 1. / fd;
                    const auto a11 = 2. / 3 * kB * T / viscosity *
                                     (std::pow(1., ifd) + std::pow(1., ifd)) *
                                     (std::pow(1., -ifd) + std::pow(1., -ifd));
                    const auto a12 = 2. / 3 * kB * T / viscosity *
                                     (std::pow(1., ifd) + std::pow(2., ifd)) *
                                     (std::pow(1., -ifd) + std::pow(2., -ifd));
                    const auto a13 = 2. / 3 * kB * T / viscosity *
                                     (std::pow(1., ifd) + std::pow(3., ifd)) *
                                     (std::pow(1., -ifd) + std::pow(3., -ifd));
                    const auto a22 = 2. / 3 * kB * T / viscosity *
                                     (std::pow(2., ifd) + std::pow(2., ifd)) *
                                     (std::pow(2., -ifd) + std::pow(2., -ifd));
                    const auto a23 = 2. / 3 * kB * T / viscosity *
                                     (std::pow(2., ifd) + std::pow(3., ifd)) *
                                     (std::pow(2., -ifd) + std::pow(3., -ifd));
                    const auto a33 = 2. / 3 * kB * T / viscosity *
                                     (std::pow(3., ifd) + std::pow(3., ifd)) *
                                     (std::pow(3., -ifd) + std::pow(3., -ifd));
                    do_agglom_test(kernel, a11, a12, a13, a22, a23, a33, p1, p2, p3, p4, p5, p6);
                  } | std::vector<sunrealtype>{1, 2};
                };

                "Reaction Kernel"_test = [&] {
                  "Fractal dimension"_test = [&](const auto fd) {
                    "Fuchs ratio"_test = [&](const auto fuch) {
                      "Exponent"_test = [&](const auto exponent) {
                        const sunrealtype T         = 300;
                        const sunrealtype viscosity = 1.e-6;
                        const sunrealtype kB        = 1.380649e-23;
                        const auto a2s              = [](const sunindextype a) { return 1.0 * a; };
                        const ReactionLimitedAgglomerationKernel kernel(a2s, T, viscosity, exponent,
                                                                        fuch, fd);
                        const auto ifd = 1. / fd;
                        const auto a11 = 2. / 3 * kB * T / viscosity / fuch *
                                         std::pow(1. * 1., exponent) *
                                         (std::pow(1., ifd) + std::pow(1., ifd)) *
                                         (std::pow(1., -ifd) + std::pow(1., -ifd));
                        const auto a12 = 2. / 3 * kB * T / viscosity / fuch *
                                         std::pow(1. * 2., exponent) *
                                         (std::pow(1., ifd) + std::pow(2., ifd)) *
                                         (std::pow(1., -ifd) + std::pow(2., -ifd));
                        const auto a13 = 2. / 3 * kB * T / viscosity / fuch *
                                         std::pow(1. * 3., exponent) *
                                         (std::pow(1., ifd) + std::pow(3., ifd)) *
                                         (std::pow(1., -ifd) + std::pow(3., -ifd));
                        const auto a22 = 2. / 3 * kB * T / viscosity / fuch *
                                         std::pow(2. * 2., exponent) *
                                         (std::pow(2., ifd) + std::pow(2., ifd)) *
                                         (std::pow(2., -ifd) + std::pow(2., -ifd));
                        const auto a23 = 2. / 3 * kB * T / viscosity / fuch *
                                         std::pow(2. * 3., exponent) *
                                         (std::pow(2., ifd) + std::pow(3., ifd)) *
                                         (std::pow(2., -ifd) + std::pow(3., -ifd));
                        const auto a33 = 2. / 3 * kB * T / viscosity / fuch *
                                         std::pow(3. * 3., exponent) *
                                         (std::pow(3., ifd) + std::pow(3., ifd)) *
                                         (std::pow(3., -ifd) + std::pow(3., -ifd));
                        do_agglom_test(kernel, a11, a12, a13, a22, a23, a33, p1, p2, p3, p4, p5,
                                       p6);
                      } | std::vector<sunrealtype>{1, 2};
                    } | std::vector<sunrealtype>{1, 2};
                  } | std::vector<sunrealtype>{1, 2};
                };
              } | std::vector<sunrealtype>{1, 2};
            } | std::vector<sunrealtype>{1, 2};
          } | std::vector<sunrealtype>{1, 2};
        } | std::vector<sunrealtype>{1, 2};
      } | std::vector<sunrealtype>{1, 2};
    } | std::vector<sunrealtype>{1, 2};
  };
}