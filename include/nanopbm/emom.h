#ifndef NANOPBM_EMOM_H
#define NANOPBM_EMOM_H


#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>

#include <cmath>
#include <utility>
#include <vector>

#include "ode_contribution.h"
namespace NanoPBM {
class EMoM : public OdeContribution {
 public:
  EMoM() = delete;
  EMoM(const sunindextype idx_mu0, const sunindextype idx_mu1, const sunindextype idx_mu2,
       const sunindextype idx_precursor, const sunindextype idx_particle,
       const sunrealtype growth_rate, const sunrealtype unit_volume,
       const sunrealtype continuous_size_particle_N, const sunindextype discrete_size_particle_N,
       const std::vector<std::pair<sunindextype, sunrealtype>>& species_created_via_growth)
      : idx_mu0(idx_mu0),
        idx_mu1(idx_mu1),
        idx_mu2(idx_mu2),
        idx_precursor(idx_precursor),
        idx_particle(idx_particle),
        k_mu(growth_rate * std::cbrt(2 * unit_volume / 9 / std::numbers::pi)),
        k_bc(growth_rate * std::pow(1. * discrete_size_particle_N, 2. / 3.)),
        k_D(growth_rate * std::pow(std::numbers::pi / 6. / unit_volume, 2. / 3.)),
        xn(continuous_size_particle_N),
        species_created_via_growth(species_created_via_growth) {}

  /// \copydoc OdeContribution::add_to_rhs
  void add_to_rhs(const sunrealtype t, const N_Vector y, N_Vector ydot) const override {
    auto rhs_data     = N_VGetArrayPointer(ydot);
    const auto y_data = N_VGetArrayPointer(y);

    const auto mu0 = y_data[idx_mu0];
    const auto mu1 = y_data[idx_mu1];
    const auto mu2 = y_data[idx_mu2];
    const auto Cm  = y_data[idx_precursor];
    const auto Pn  = y_data[idx_particle];

    // Contribute to moments
    rhs_data[idx_mu0] += k_bc * Cm * Pn;
    rhs_data[idx_mu1] += k_mu * Cm * mu0 + xn * k_bc * Cm * Pn;
    rhs_data[idx_mu2] += 2 * k_mu * Cm * mu1 + xn * xn * k_bc * Cm * Pn;

    // Contribute to discrete
    rhs_data[idx_precursor] -= k_D * Cm * mu2;
    for (const auto& [idx, number] : species_created_via_growth) {
      rhs_data[idx] += number * k_D * Cm * mu2;
    }
  }

  /// \copydoc OdeContribution::add_to_jac_times_v
  void add_to_jac_times_v(const N_Vector v, N_Vector Jv, const sunrealtype t, const N_Vector y,
                          const N_Vector ydot, N_Vector tmp) const override {
    const auto v_data = N_VGetArrayPointer(v);
    auto Jv_data      = N_VGetArrayPointer(Jv);
    const auto y_data = N_VGetArrayPointer(y);

    const auto mu0 = y_data[idx_mu0];
    const auto mu1 = y_data[idx_mu1];
    const auto mu2 = y_data[idx_mu2];
    const auto Cm  = y_data[idx_precursor];
    const auto Pn  = y_data[idx_particle];

    const auto v_mu0 = v_data[idx_mu0];
    const auto v_mu1 = v_data[idx_mu1];
    const auto v_mu2 = v_data[idx_mu2];
    const auto v_Cm  = v_data[idx_precursor];
    const auto v_Pn  = v_data[idx_particle];


    // Moments
    Jv_data[idx_mu0] += k_bc * (Pn * v_Cm + Cm * v_Pn);
    Jv_data[idx_mu1] += k_mu * (Cm * v_mu0 + mu0 * v_Cm) + xn * k_bc * (Pn * v_Cm + Cm * v_Pn);
    Jv_data[idx_mu2] +=
        2 * k_mu * (Cm * v_mu1 + mu1 * v_Cm) + xn * xn * k_bc * (Pn * v_Cm + Cm * v_Pn);

    // Discrete
    Jv_data[idx_precursor] += -k_D * (Cm * v_mu2 + mu2 * v_Cm);
    for (const auto& [idx, number] : species_created_via_growth) {
      Jv_data[idx] += number * k_D * (Cm * v_mu2 + mu2 * v_Cm);
    }
  }


  /// \copydoc OdeContribution::add_to_jac
  void add_to_jac(gko::matrix_data<sunrealtype, sunindextype>& sparsity_pattern,
                  const sunrealtype t, const N_Vector y, const N_Vector fy, N_Vector tmp1,
                  N_Vector tmp2, N_Vector tmp3) const override {
    const auto y_data = N_VGetArrayPointer(y);


    const auto mu0 = y_data[idx_mu0];
    const auto mu1 = y_data[idx_mu1];
    const auto mu2 = y_data[idx_mu2];
    const auto Cm  = y_data[idx_precursor];
    const auto Pn  = y_data[idx_particle];

    // Moments
    sparsity_pattern.nonzeros.emplace_back(idx_mu0, idx_precursor, k_bc * Pn);
    sparsity_pattern.nonzeros.emplace_back(idx_mu0, idx_particle, k_bc * Cm);

    sparsity_pattern.nonzeros.emplace_back(idx_mu1, idx_mu0, k_mu * Cm);
    sparsity_pattern.nonzeros.emplace_back(idx_mu1, idx_precursor, k_mu * mu0 + xn * k_bc * Pn);
    sparsity_pattern.nonzeros.emplace_back(idx_mu1, idx_particle, xn * k_bc * Cm);

    sparsity_pattern.nonzeros.emplace_back(idx_mu2, idx_mu1, 2 * k_mu * Cm);
    sparsity_pattern.nonzeros.emplace_back(idx_mu2, idx_precursor,
                                           2 * k_mu * mu1 + xn * xn * k_bc * Pn);
    sparsity_pattern.nonzeros.emplace_back(idx_mu2, idx_particle, xn * xn * k_bc * Cm);

    // Discrete
    sparsity_pattern.nonzeros.emplace_back(idx_precursor, idx_precursor, -k_D * mu2);
    sparsity_pattern.nonzeros.emplace_back(idx_precursor, idx_mu2, -k_D * Cm);
    for (const auto& [idx, number] : species_created_via_growth) {
      sparsity_pattern.nonzeros.emplace_back(idx, idx_precursor, number * k_D * mu2);
      sparsity_pattern.nonzeros.emplace_back(idx, idx_mu2, number * k_D * Cm);
    }
  }


 private:
  const sunindextype idx_mu0;
  const sunindextype idx_mu1;
  const sunindextype idx_mu2;
  const sunindextype idx_precursor;
  const sunindextype idx_particle;
  const sunrealtype k_mu;
  const sunrealtype k_bc;
  const sunrealtype k_D;
  const sunrealtype xn;  // x_N
  // TODO: created species
  const std::vector<std::pair<sunindextype, sunrealtype>> species_created_via_growth;
};
}  // namespace NanoPBM

#endif  // NANOPBM_EMOM_H