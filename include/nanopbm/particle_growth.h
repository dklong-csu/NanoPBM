#ifndef NANOPBM_PARTICLE_GROWTH_H
#define NANOPBM_PARTICLE_GROWTH_H

#include <sundials/sundials_types.h>

#include <concepts>
#include <vector>

#include "nanopbm/particle_agglomeration.h"
#include "ode_contribution.h"

namespace NanoPBM {
/*
 * Defines the necessary implementation for a class to be used as a kernel.
 */
template <typename T>
concept IsGrowthKernel = requires(const T t, const sunindextype i) {
  { t(i) } -> std::convertible_to<sunrealtype>;
};


/**
 * @brief Represents a constant growth kernel. All particle sizes grow at the same rate.
 *
 * \verbatim embed:rst:leading-asterisk
 *  This is equivalent to
 *
 *  .. math::
 *
 *      G(i) = C
 *
 *  for some constant :math:`C`.
 * \endverbatim
 *
 * @tparam Number Floating point number type used. For example, ``float`` or ``double``.
 */
template <typename Number = sunrealtype>
class ConstantGrowthKernel {
 public:
  ConstantGrowthKernel() = delete;
  /**
   * @brief Constructor.
   *
   * @param rate The growth rate.
   */
  ConstantGrowthKernel(const Number rate = 1) : rate(rate) {}

  /**
   * @brief Calculates the growth rate.
   *
   * @param discrete_size The discrete size of the growing particle.
   * @return The growth rate of the specified particle.
   */
  Number operator()(const sunindextype discrete_size) const { return rate; }

 private:
  const Number rate;
};


/**
 * @brief Represents a growth kernel based on the product of a constant and the particle size.
 *
 * \verbatim embed:rst:leading-asterisk
 *  This class is templated to receive an arbitrary function that converts a discrete size to a
 *  continuous size. Call this conversion function :math:`S`. Then this kernel is
 *
 *  .. math::
 *
 *      G(i) = C \times S(i)
 *
 *  for some constant :math:`C`.
 * \endverbatim
 *
 * @tparam SizeFcn
 * @tparam Number
 */
template <HasAtomsToSizeOperator SizeFcn, typename Number = sunrealtype>
class ConstantTimesSizeGrowthKernel {
 public:
  ConstantTimesSizeGrowthKernel() = delete;

  /**
   * @brief Constructor.
   *
   * @param size_fcn Object that converts discrete to continuous particle size.
   * @param rate Multiplicative factor modifying the growth rate.
   */
  ConstantTimesSizeGrowthKernel(const SizeFcn& size_fcn, const Number rate = 1)
      : to_size(size_fcn), rate(rate) {}


  /**
   * @brief Calculates the growth rate.
   *
   * @param discrete_size The discrete size of the growing particle.
   * @return The growth rate of the specified particle.
   */
  Number operator()(const sunindextype discrete_size) const {
    return rate * to_size(discrete_size);
  }

 private:
  const SizeFcn to_size;
  const Number rate;
};


template <IsGrowthKernel Kernel, typename Number = sunrealtype>
class ParticleGrowth : public OdeContribution {
 public:
  ParticleGrowth(const sunindextype precursor_idx,
                 const std::vector<std::pair<sunindextype, Number>>& created_species,
                 const sunindextype first_size, const sunindextype last_size,
                 const sunindextype first_size_idx, const Kernel& growth_kernel,
                 const sunindextype size_increase = 1)
      : precursor_idx(precursor_idx),
        created_species_idx_and_amount(created_species),
        first_size(first_size),
        last_size(last_size),
        first_size_idx(first_size_idx),
        growth_kernel(growth_kernel),
        size_increase(size_increase) {}


  void add_to_rhs(const sunrealtype t, const N_Vector y, N_Vector ydot) const override {
    const auto y_data = N_VGetArrayPointer(y);
    auto rhs_data     = N_VGetArrayPointer(ydot);
    for (sunindextype size = first_size; size <= last_size; ++size) {
      const sunindextype particle_idx = first_size_idx + (size - first_size);
      const auto rate = growth_kernel(size) * y_data[particle_idx] * y_data[precursor_idx];
      rhs_data[precursor_idx] -= rate;
      rhs_data[particle_idx] -= rate;
      rhs_data[particle_idx + size_increase] += rate;
      for (const auto& [idx, amount] : created_species_idx_and_amount) {
        rhs_data[idx] += amount * rate;
      }
    }
  }

  void add_to_jac_times_v(const N_Vector v, N_Vector Jv, const sunrealtype t, const N_Vector y,
                          const N_Vector ydot, N_Vector tmp) const override {
    const auto v_data = N_VGetArrayPointer(v);
    auto Jv_data      = N_VGetArrayPointer(Jv);
    const auto y_data = N_VGetArrayPointer(y);

    for (sunindextype size = first_size; size <= last_size; ++size) {
      const sunindextype particle_idx = first_size_idx + (size - first_size);
      const auto rate                 = growth_kernel(size);

      const auto jac_times_v = rate * y_data[precursor_idx] * v_data[particle_idx] +
                               rate * y_data[particle_idx] * v_data[precursor_idx];

      Jv_data[particle_idx] -= jac_times_v;
      Jv_data[precursor_idx] -= jac_times_v;
      Jv_data[particle_idx + size_increase] += jac_times_v;
      for (const auto& [idx, amount] : created_species_idx_and_amount) {
        Jv_data[idx] += amount * jac_times_v;
      }
    }
  }

  /// \copydoc OdeContribution::add_to_jac
  void add_to_jac(gko::matrix_data<sunrealtype, sunindextype>& sparsity_pattern,
                  const sunrealtype t, const N_Vector y, const N_Vector fy, N_Vector tmp1,
                  N_Vector tmp2, N_Vector tmp3) const override {
    const auto y_data = N_VGetArrayPointer(y);
    for (sunindextype size = first_size; size <= last_size; ++size) {
      const sunindextype particle_idx = first_size_idx + (size - first_size);
      const auto rate                 = growth_kernel(size);

      const auto df_dprecursor = rate * y_data[particle_idx];
      const auto df_dparticle  = rate * y_data[precursor_idx];

      sparsity_pattern.nonzeros.emplace_back(precursor_idx, precursor_idx, -df_dprecursor);
      sparsity_pattern.nonzeros.emplace_back(precursor_idx, particle_idx, -df_dparticle);

      sparsity_pattern.nonzeros.emplace_back(particle_idx, precursor_idx, -df_dprecursor);
      sparsity_pattern.nonzeros.emplace_back(particle_idx, particle_idx, -df_dparticle);

      sparsity_pattern.nonzeros.emplace_back(particle_idx + size_increase, precursor_idx,
                                             df_dprecursor);
      sparsity_pattern.nonzeros.emplace_back(particle_idx + size_increase, particle_idx,
                                             df_dparticle);

      for (const auto& [idx, amount] : created_species_idx_and_amount) {
        sparsity_pattern.nonzeros.emplace_back(idx, precursor_idx, amount * df_dprecursor);
        sparsity_pattern.nonzeros.emplace_back(idx, particle_idx, amount * df_dparticle);
      }
    }
  }

 private:
  const sunindextype precursor_idx;
  const std::vector<std::pair<sunindextype, Number>> created_species_idx_and_amount;
  const sunindextype first_size;
  const sunindextype last_size;
  const sunindextype first_size_idx;
  const Kernel growth_kernel;
  const sunindextype size_increase;
};

}  // namespace NanoPBM

#endif  // NANOPBM_PARTICLE_GROWTH_H