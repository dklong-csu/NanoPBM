#include "nanopbm/cvode.h"

#include <nvector/nvector_serial.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>

#include <boost/ut.hpp>
#include <functional>
#include <sundials/sundials_context.hpp>

#include "chemical_reaction.h"
#include "reaction_network.h"
#include "reaction_progress.h"


int main() {
  using namespace boost::ut;
  using namespace NanoPBM;
  sundials::Context sunctx;
  const sunindextype n_odes = 2;

  N_Vector y0  = N_VNew_Serial(n_odes, sunctx);
  "CVODE"_test = [&] {
    "Rate 1"_test = [&](const auto r1) {
      "Rate 2"_test = [&](const auto r2) {
        "Reverse indices"_test = [&](const bool switch_idx) {
          if (switch_idx) {
            NV_Ith_S(y0, 0) = 0;
            NV_Ith_S(y0, 1) = 1;
          } else {
            NV_Ith_S(y0, 0) = 1;
            NV_Ith_S(y0, 1) = 0;
          }

          const sunrealtype rate = r1 + r2;
          ReactionNetwork rxn_network;
          if (switch_idx) {
            const ConstantReversibleReactionProgress<1, 1> rxnrate1(r1, 0);
            ChemicalReactionParameters<1, 1> rxnprm1({1}, {1}, {1}, {0}, {1}, {1});
            ChemicalReaction rxn1(rxnprm1, rxnrate1);
            rxn_network.add_reaction(rxn1);
            const ConstantIrreversibleReactionProgress<1, 1> rxnrate2(r2);
            IrreversibleChemicalReaction rxn2(rxnprm1, rxnrate2);
            rxn_network.add_reaction(rxn2);

          } else {
            const ConstantReversibleReactionProgress<1, 1> rxnrate1(r1, 0);
            ChemicalReactionParameters<1, 1> rxnprm1({0}, {1}, {1}, {1}, {1}, {1});
            ChemicalReaction rxn1(rxnprm1, rxnrate1);
            rxn_network.add_reaction(rxn1);
            const ConstantIrreversibleReactionProgress<1, 1> rxnrate2(r2);
            IrreversibleChemicalReaction rxn2(rxnprm1, rxnrate2);
            rxn_network.add_reaction(rxn2);
          }

          std::function<int(sunrealtype, N_Vector, N_Vector)> rhs = [=](sunrealtype t, N_Vector y,
                                                                        N_Vector ydot) {
            N_VConst(0, ydot);
            rxn_network.add_to_rhs(y, ydot);
            return 0;
          };

          std::function<int(sunrealtype, N_Vector, N_Vector, SUNMatrix, N_Vector, N_Vector,
                            N_Vector)>
              jac = [=](sunrealtype t, N_Vector y, N_Vector ydot, SUNMatrix J, N_Vector tmp1,
                        N_Vector tmp2, N_Vector tmp3) {
                SUNMatZero(J);
                rxn_network.add_to_jacobian(y, J);
                return 0;
              };


          CVODESettings cv_settings;
          cv_settings.reltol = 1.e-10;
          cv_settings.abstol = 1.e-10;
          CVODE ode_solver(sunctx, y0, rhs, jac, cv_settings);

          sunrealtype time     = 0;
          const sunrealtype dt = 0.01;
          for (int step = 1; step < 11; ++step) {
            sunrealtype tout = step * dt;
            ode_solver.solve(tout, y0, time);
            const sunrealtype exact = std::exp(-rate * time);
            if (switch_idx) {
              expect(std::abs(NV_Ith_S(y0, 1) - exact) <= 0.000001_d);
              expect(std::abs(NV_Ith_S(y0, 0) - (1. - exact)) <= 0.000001_d);
            } else {
              expect(std::abs(NV_Ith_S(y0, 0) - exact) <= 0.000001_d);
              expect(std::abs(NV_Ith_S(y0, 1) - (1. - exact)) <= 0.000001_d);
            }
          }
        } | std::vector<bool>{false, true};
      } | std::vector<sunrealtype>{1, 2, 3, 5};
    } | std::vector<sunrealtype>{1, 2, 3, 5};
  };
  N_VDestroy(y0);
}