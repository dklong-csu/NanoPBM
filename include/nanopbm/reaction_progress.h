#ifndef NANOPBM_REACTION_PROGRESS_H
#define NANOPBM_REACTION_PROGRESS_H


#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>

#include <array>
#include <cmath>
namespace NanoPBM {
template <sunindextype n_reactants, sunindextype n_products>
class ConstantReversibleReactionProgress {
 public:
  ConstantReversibleReactionProgress(const sunrealtype kf, const sunrealtype kb) : kf(kf), kb(kb) {}

  sunrealtype forward(const N_Vector y, const std::array<sunindextype, n_reactants>& indices,
                      const std::array<sunrealtype, n_reactants>& orders) const {
    sunrealtype rate  = kf;
    const auto y_data = N_VGetArrayPointer(y);
    for (sunindextype i = 0; i < n_reactants; ++i) {
      const auto idx   = indices[i];
      const auto order = orders[i];
      const auto conc  = y_data[idx];
      rate *= std::pow(conc, order);
    }
    return rate;
  }


  std::array<sunrealtype, n_reactants> forward_derivatives(
      const N_Vector y, const std::array<sunindextype, n_reactants>& indices,
      const std::array<sunrealtype, n_reactants>& orders) const {
    const auto forward_rate = forward(y, indices, orders);

    std::array<sunrealtype, n_reactants> derivs;
    for (int i = 0; i < n_reactants; ++i) {
      derivs[i] = 0;
    }

    const auto y_data = N_VGetArrayPointer(y);

    for (sunindextype i = 0; i < n_reactants; ++i) {
      const auto order                                  = orders[i];
      std::array<sunrealtype, n_reactants> deriv_orders = orders;
      deriv_orders[i] -= 1;
      derivs[i] = order * forward(y, indices, deriv_orders);
    }
    return derivs;
  }


  sunrealtype backward(const N_Vector y, const std::array<sunindextype, n_products>& indices,
                       const std::array<sunrealtype, n_products>& orders) const {
    sunrealtype rate  = kb;
    const auto y_data = N_VGetArrayPointer(y);
    for (sunindextype i = 0; i < n_products; ++i) {
      const auto idx   = indices[i];
      const auto order = orders[i];
      const auto conc  = y_data[idx];
      rate *= std::pow(conc, order);
    }
    return rate;
  }

  std::array<sunrealtype, n_products> backward_derivatives(
      const N_Vector y, const std::array<sunindextype, n_products>& indices,
      const std::array<sunrealtype, n_products>& orders) const {
    const auto backward_rate = backward(y, indices, orders);

    std::array<sunrealtype, n_products> derivs;
    for (int i = 0; i < n_products; ++i) {
      derivs[i] = 0;
    }

    const auto y_data = N_VGetArrayPointer(y);

    for (sunindextype i = 0; i < n_products; ++i) {
      const auto order                                 = orders[i];
      std::array<sunrealtype, n_products> deriv_orders = orders;
      deriv_orders[i] -= 1;
      derivs[i] = order * backward(y, indices, deriv_orders);
    }
    return derivs;
  }

 private:
  const sunrealtype kf, kb;
};
}  // namespace NanoPBM

#endif  // NANOPBM_REACTION_PROGRESS_H