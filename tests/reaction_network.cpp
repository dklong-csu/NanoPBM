#include "reaction_network.h"

#include <nvector/nvector_serial.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>
#include <sunmatrix/sunmatrix_dense.h>

#include <boost/ut.hpp>
#include <sundials/sundials_context.hpp>
#include <vector>

#include "chemical_reaction.h"
#include "nanopbm/reaction_progress.h"

int main() {
  using namespace boost::ut;
  using namespace NanoPBM;
  sundials::Context sunctx;
  const sunindextype n_odes = 2;

  N_Vector v      = N_VNew_Serial(n_odes, sunctx);
  N_Vector vdot   = N_VClone(v);
  N_Vector vexact = N_VClone(v);

  SUNMatrix M      = SUNDenseMatrix(n_odes, n_odes, sunctx);
  SUNMatrix Mexact = SUNDenseMatrix(n_odes, n_odes, sunctx);

  "Reaction Network"_test = [&] {
    "Reversible kf"_test = [&](const auto& kf) {
      "Reversible kb"_test = [&](const auto& kb) {
        "Irreversible rate"_test = [&](const auto& irr_kf) {
          "Concentrations 0"_test = [&](const auto& conc0) {
            "Concentrations 1"_test = [&](const auto& conc1) {
              "Irreversible order"_test = [&](const auto& irr_ord) {
                "Reversible order forward"_test = [&](const auto& of) {
                  "Reversible order backward"_test = [&](const auto& ob) {
                    "Irreversible stoich R"_test = [&](const auto& irr_sr) {
                      "Irreversible stoich P"_test = [&](const auto& irr_sp) {
                        "Reversible stoich R"_test = [&](const auto& rev_sr) {
                          "Reversible stoich P"_test = [&](const auto& rev_sp) {
                            NV_Ith_S(v, 0) = conc0;
                            NV_Ith_S(v, 1) = conc1;

                            N_VConst(0, vdot);
                            N_VConst(0, vexact);
                            SUNMatZero(M);
                            SUNMatZero(Mexact);

                            ConstantReversibleReactionProgress<1, 1> rev_rate(kf, kb);
                            ChemicalReactionParameters<1, 1> rev_prm({0}, {rev_sr}, {of}, {1},
                                                                     {rev_sp}, {ob});
                            const ChemicalReaction rev_rxn(rev_prm, rev_rate);

                            ConstantIrreversibleReactionProgress<1, 1> irr_rate(irr_kf);
                            ChemicalReactionParameters<1, 1> irr_prm({0}, {irr_sr}, {irr_ord}, {1},
                                                                     {irr_sp});
                            const IrreversibleChemicalReaction irr_rxn(irr_prm, irr_rate);

                            ReactionNetwork rxn_network;
                            rxn_network.add_reaction(rev_rxn);
                            rxn_network.add_reaction(irr_rxn);

                            rev_rxn.add_to_rhs(v, vexact);
                            irr_rxn.add_to_rhs(v, vexact);
                            rxn_network.add_to_rhs(v, vdot);
                            expect(NV_Ith_S(vdot, 0) == _d(NV_Ith_S(vexact, 0)));
                            expect(NV_Ith_S(vdot, 1) == _d(NV_Ith_S(vexact, 1)));

                            rev_rxn.add_to_jacobian(v, Mexact);
                            irr_rxn.add_to_jacobian(v, Mexact);
                            rxn_network.add_to_jacobian(v, M);
                            expect(SM_ELEMENT_D(M, 0, 0) == _d(SM_ELEMENT_D(Mexact, 0, 0)));
                            expect(SM_ELEMENT_D(M, 1, 0) == _d(SM_ELEMENT_D(Mexact, 1, 0)));
                            expect(SM_ELEMENT_D(M, 0, 1) == _d(SM_ELEMENT_D(Mexact, 0, 1)));
                            expect(SM_ELEMENT_D(M, 1, 1) == _d(SM_ELEMENT_D(Mexact, 1, 1)));
                          } | std::vector<sunrealtype>{1, 5};
                        } | std::vector<sunrealtype>{1, 4};
                      } | std::vector<sunrealtype>{1, 3};
                    } | std::vector<sunrealtype>{1, 2};
                  } | std::vector<sunrealtype>{1, 2};
                } | std::vector<sunrealtype>{1, 2};
              } | std::vector<sunrealtype>{1, 2};
            } | std::vector<sunrealtype>{0, 0.75, 1};
          } | std::vector<sunrealtype>{0, 0.25, 1};
        } | std::vector<sunrealtype>{11, 13};
      } | std::vector<sunrealtype>{5, 7};
    } | std::vector<sunrealtype>{1, 3};
  };
}