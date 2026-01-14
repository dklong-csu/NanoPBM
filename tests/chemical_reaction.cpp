#include "nanopbm/chemical_reaction.h"

#include <nvector/nvector_serial.h>
#include <sundials/sundials_matrix.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>
#include <sunmatrix/sunmatrix_dense.h>

#include <boost/ut.hpp>
#include <sundials/sundials_context.hpp>
#include <vector>

#include "reaction_progress.h"

int main() {
  sundials::Context sunctx;
  using namespace boost::ut;
  using namespace NanoPBM;
  "RxnParameters"_test = [] {
    ChemicalReactionParameters<1, 1> prm({0}, {1}, {1}, {1}, {1}, {1});
    expect(prm.reactant_indices[0] == 0_i);
    expect(prm.reactant_stoich[0] == 1._d);
    expect(prm.reactant_order[0] == 1._d);
    expect(prm.product_indices[0] == 1_i);
    expect(prm.product_stoich[0] == 1._d);
    expect(prm.product_order[0] == 1._d);

    prm.reactant_indices[0] = 1;
    prm.reactant_stoich[0]  = 2.;
    prm.reactant_order[0]   = 3.;
    prm.product_indices[0]  = 4;
    prm.product_stoich[0]   = 5.;
    prm.product_order[0]    = 6.;

    expect(prm.reactant_indices[0] == 1_i);
    expect(prm.reactant_stoich[0] == 2._d);
    expect(prm.reactant_order[0] == 3._d);
    expect(prm.product_indices[0] == 4_i);
    expect(prm.product_stoich[0] == 5._d);
    expect(prm.product_order[0] == 6._d);
  };

  "ChemicalReaction"_test = [&] {
    const sunindextype n_odes = 2;
    N_Vector v                = N_VNew_Serial(n_odes, sunctx);
    N_Vector vdot             = N_VClone(v);
    auto v_data               = N_VGetArrayPointer(v);
    auto vdot_data            = N_VGetArrayPointer(vdot);

    SUNMatrix M = SUNDenseMatrix(n_odes, n_odes, sunctx);

    "Rates"_test = [&](const auto& rates) {
      "Concentrations"_test = [&](const auto& conc) {
        N_VConst(0, vdot);
        SUNMatZero(M);
        v_data[0] = conc[0];
        v_data[1] = conc[1];

        ConstantReversibleReactionProgress<1, 1> rxn_rate(rates[0], rates[1]);
        ChemicalReactionParameters<1, 1> prm({0}, {1}, {1}, {1}, {1}, {1});
        const ChemicalReaction rxn(prm, rxn_rate);

        rxn.add_to_rhs(v, vdot);
        expect(vdot_data[0] == _d(-rates[0] * conc[0] + rates[1] * conc[1]));
        expect(vdot_data[1] == _d(rates[0] * conc[0] - rates[1] * conc[1]));

        rxn.add_to_jacobian(v, M);
        expect(SM_ELEMENT_D(M, 0, 0) == _d(-rates[0]));
        expect(SM_ELEMENT_D(M, 0, 1) == _d(rates[1]));
        expect(SM_ELEMENT_D(M, 1, 0) == _d(rates[0]));
        expect(SM_ELEMENT_D(M, 1, 1) == _d(-rates[1]));
      } | std::vector<std::array<sunrealtype, 2>>{{1, 0}, {0, 1}, {1, 1}, {10, 0.1}};
    } | std::vector<std::array<sunrealtype, 2>>{{1, 0}, {0, 1}, {1, 1}, {0.25, 0.75}};

    "Higher reaction order"_test = [&]() {
      "Rates"_test = [&](const auto& rates) {
        "Concentrations"_test = [&](const auto& conc) {
          N_VConst(0, vdot);
          SUNMatZero(M);
          v_data[0] = conc[0];
          v_data[1] = conc[1];

          ConstantReversibleReactionProgress<1, 1> rxn_rate(rates[0], rates[1]);
          ChemicalReactionParameters<1, 1> prm({0}, {1}, {2}, {1}, {1}, {3});
          const ChemicalReaction rxn(prm, rxn_rate);

          rxn.add_to_rhs(v, vdot);
          expect(vdot_data[0] ==
                 _d(-rates[0] * conc[0] * conc[0] + rates[1] * conc[1] * conc[1] * conc[1]));
          expect(vdot_data[1] ==
                 _d(rates[0] * conc[0] * conc[0] - rates[1] * conc[1] * conc[1] * conc[1]));

          rxn.add_to_jacobian(v, M);
          expect(SM_ELEMENT_D(M, 0, 0) == _d(-2 * rates[0] * conc[0]));
          expect(SM_ELEMENT_D(M, 0, 1) == _d(3 * rates[1] * conc[1] * conc[1]));
          expect(SM_ELEMENT_D(M, 1, 0) == _d(2 * rates[0] * conc[0]));
          expect(SM_ELEMENT_D(M, 1, 1) == _d(-3 * rates[1] * conc[1] * conc[1]));
        } | std::vector<std::array<sunrealtype, 2>>{{1, 0}, {0, 1}, {1, 1}, {10, 0.1}};
      } | std::vector<std::array<sunrealtype, 2>>{{1, 0}, {0, 1}, {1, 1}, {0.25, 0.75}};
    };
    N_VDestroy(v);
    N_VDestroy(vdot);
    SUNMatDestroy(M);
  };

  "IrreversibleChemicalReaction"_test = [&]{
    const sunindextype n_odes = 2;
    N_Vector v                = N_VNew_Serial(n_odes, sunctx);
    N_Vector vdot             = N_VClone(v);
    N_Vector vdot_expected    = N_VClone(vdot);
    auto v_data               = N_VGetArrayPointer(v);
    auto vdot_data            = N_VGetArrayPointer(vdot);
    auto vdot_expected_data    = N_VGetArrayPointer(vdot_expected);

    SUNMatrix M = SUNDenseMatrix(n_odes, n_odes, sunctx);
    SUNMatrix M_expected = SUNDenseMatrix(n_odes, n_odes, sunctx);

    "Rates"_test = [&](const auto rate){
        "Concentrations"_test = [&](const auto & conc){
            N_VConst(0, vdot);
            N_VConst(0, vdot_expected);
        SUNMatZero(M);
        SUNMatZero(M_expected);
        v_data[0] = conc[0];
        v_data[1] = conc[1];

        ConstantReversibleReactionProgress<1, 1> rxn_rate(rate, 0);
        ChemicalReactionParameters<1, 1> prm({0}, {1}, {1}, {1}, {1}, {1});
        const ChemicalReaction rev_rxn(prm, rxn_rate);

        ConstantIrreversibleReactionProgress<1, 1> irr_rxn_rate(rate);
        const IrreversibleChemicalReaction irr_rxn(prm, irr_rxn_rate);

        irr_rxn.add_to_rhs(v, vdot);
        rev_rxn.add_to_rhs(v, vdot_expected);
        expect(vdot_data[0] == _d(vdot_expected_data[0]));
        expect(vdot_data[1] == _d(vdot_expected_data[1]));

        irr_rxn.add_to_jacobian(v, M);
        rev_rxn.add_to_jacobian(v, M_expected);
        expect(SM_ELEMENT_D(M, 0, 0) == _d(SM_ELEMENT_D(M_expected, 0, 0)));
        expect(SM_ELEMENT_D(M, 0, 1) == _d(SM_ELEMENT_D(M_expected, 0, 1)));
        expect(SM_ELEMENT_D(M, 1, 0) == _d(SM_ELEMENT_D(M_expected, 1, 0)));
        expect(SM_ELEMENT_D(M, 1, 1) == _d(SM_ELEMENT_D(M_expected, 1, 1)));



        } | std::vector<std::array<sunrealtype, 2>>{{1, 0}, {0, 1}, {1, 1}, {10, 0.1}};
    } | std::vector<sunrealtype>{1};


    N_VDestroy(v);
    N_VDestroy(vdot);
    N_VDestroy(vdot_expected);
    SUNMatDestroy(M);

  };
}