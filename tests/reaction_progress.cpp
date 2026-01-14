#include "nanopbm/reaction_progress.h"

#include <nvector/nvector_serial.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>

#include <boost/ut.hpp>
#include <sundials/sundials_context.hpp>


int main() {
  using namespace boost::ut;
  using namespace NanoPBM;
  sundials::Context sunctx;
  const sunindextype n_odes = 2;
  N_Vector v                = N_VNew_Serial(n_odes, sunctx);
  auto v_data               = N_VGetArrayPointer(v);

  "Reversible Constant"_test = [&](const auto& rates) {
    "Concentrations"_test = [&](const auto& conc) {
      "Order forward"_test = [&](const auto& of) {
        "Order backward"_test = [&](const auto& ob) {
          v_data[0] = conc[0];
          v_data[1] = conc[1];

          ConstantReversibleReactionProgress<1, 1> rxn_rate(rates[0], rates[1]);

          auto f = rxn_rate.forward(v, {0}, {of});
          expect(f == _d(rates[0] * std::pow(conc[0], of)));

          auto b = rxn_rate.backward(v, {1}, {ob});
          expect(b == _d(rates[1] * std::pow(conc[1], ob)));

          auto df = rxn_rate.forward_derivatives(v, {0}, {of});
          expect(df[0] == _d(of * rates[0] * std::pow(conc[0], of - 1)));

          auto db = rxn_rate.backward_derivatives(v, {1}, {ob});
          expect(db[0] == _d(ob * rates[1] * std::pow(conc[1], ob - 1)));
        } | std::vector<sunrealtype>{1, 2, 3};
      } | std::vector<sunrealtype>{1, 2, 3};
    } | std::vector<std::array<sunrealtype, 2>>{{1, 1}, {1, 0}, {0, 1}, {3, 7}, {11, 5}};
  } | std::vector<std::array<sunrealtype, 2>>{{1, 1}, {1, 0}, {0, 1}, {0, 0}, {13, 17}, {23, 19}};


  "Irreversible Constant"_test = [&](const auto& rate) {
    "Concentrations"_test = [&](const auto& conc) {
      "Order forward"_test = [&](const auto& of) {
        v_data[0] = conc[0];
        v_data[1] = conc[1];

        ConstantReversibleReactionProgress<1, 1> rev_rxn_rate(rate, 0);
        ConstantIrreversibleReactionProgress<1, 1> irr_rxn_rate(rate);

        auto f = irr_rxn_rate.forward(v, {0}, {of});
        expect(f == _d(rev_rxn_rate.forward(v, {0}, {of})));

        auto b = irr_rxn_rate.backward(v, {1}, {});
        expect(b == 0._d);

        auto df = irr_rxn_rate.forward_derivatives(v, {0}, {of});
        expect(df[0] == _d(rev_rxn_rate.forward_derivatives(v, {0}, {of})[0]));

        auto db = irr_rxn_rate.backward_derivatives(v, {1}, {});
        expect(db[0] == 0._d);
      } | std::vector<sunrealtype>{1, 2, 3};
    } | std::vector<std::array<sunrealtype, 2>>{{1, 1}, {1, 0}, {0, 1}, {3, 7}, {11, 5}};
  } | std::vector<sunrealtype>{0, 2, 3, 5, 7};

  N_VDestroy(v);
}