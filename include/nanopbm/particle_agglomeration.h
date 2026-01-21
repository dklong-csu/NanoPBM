#ifndef NANOPBM_AGGLOMERATION_H
#define NANOPBM_AGGLOMERATION_H

#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>

#include <cmath>
#include <concepts>

#include "ode_contribution.h"

namespace NanoPBM {

/*
 * Defines the necessary implementation for a class to be used as a kernel.
 */
template <typename T>
concept IsAgglomerationKernel = requires(const T t, const sunindextype i, const sunindextype j) {
  { t(i, j) } -> std::convertible_to<sunrealtype>;
};


/*
 * Many agglomeration kernels derived from physical principles require a continuous measure
 * of a particle's size, such as the particle's diameter. The discrete particles we track have
 * a discrete size -- the number of atoms -- rather than this continuous size. If the kernel
 * requires a continuous size, then a conversion function is required. This ``concept`` provides
 * the required functionality for a class to act as the conversion function.
 *
 */
template <typename T>
concept HasAtomsToSizeOperator = requires(const T t, const sunindextype i) {
  { t(i) } -> std::convertible_to<sunrealtype>;
};


/**
 * @brief Represents a constant agglomeration kernel. All particle sizes combine at the same rate.
 *
 * \verbatim embed:rst:leading-asterisk
 *  This is equivalent to
 *
 *  .. math::
 *
 *      A(i,j) = C
 *
 *  for some constant :math:`C`.
 * \endverbatim
 * @tparam Number Floating point number type used. For example, ``float`` or ``double``.
 */
template <typename Number = sunrealtype>
class ConstantAgglomerationKernel {
 public:
  /**
   * @brief Constructor.
   *
   * @param rate The agglomeration rate.
   */
  ConstantAgglomerationKernel(const Number rate) : rate(rate) {}

  /**
   * @brief Constructor. Sets the rate to 1.
   *
   */
  ConstantAgglomerationKernel() : rate(1) {}

  /**
   * @brief Calculates the agglomeration rate.
   *
   * @param n_atoms1 The number of atoms in the first particle that agglomerates.
   * @param n_atoms2 The number of atoms in the second particle that agglomerates.
   * @return The rate of agglomeration between the two particles.
   */
  Number operator()(const sunindextype n_atoms1, const sunindextype n_atoms2) const { return rate; }

 private:
  const Number rate;
};


/**
 * @brief Represents the additive kernel.
 *
 * \verbatim embed:rst:leading-asterisk
 *  This class is templated to receive an arbitrary function that converts a discrete size to a
 *  continuous size. Call this function :math:`S`. Then the kernel is
 *
 *  .. math::
 *
 *      A(i,j) = C (S(i) + S(j))
 *
 * \endverbatim
 *
 * @tparam AtomsToSizeFcn An object that can convert a discrete size to a continuous size.
 * @tparam Number Floating point number type used. For example, ``float`` or ``double``.
 */
template <HasAtomsToSizeOperator AtomsToSizeFcn, typename Number = sunrealtype>
class AdditiveAgglomerationKernel {
 public:
  /**
   * @brief Constructor.
   *
   * @param to_size_fcn Function used to convert from discrete to continuous size.
   * @param rate
   */
  AdditiveAgglomerationKernel(const AtomsToSizeFcn& to_size_fcn, const Number rate = 1)
      : to_size(to_size_fcn), rate(rate) {}

  AdditiveAgglomerationKernel() = delete;

  /**
   * @brief Calculates the agglomeration rate.
   *
   * @param n_atoms1 The number of atoms in the first particle that agglomerates.
   * @param n_atoms2 The number of atoms in the second particle that agglomerates.
   * @return The rate of agglomeration between the two particles.
   */
  Number operator()(const sunindextype n_atoms1, const sunindextype n_atoms2) const {
    return rate * (to_size(n_atoms1) + to_size(n_atoms2));
  }

 private:
  const AtomsToSizeFcn to_size;
  const Number rate;
};


/**
 * @brief Represents the multiplicative kernel.
 *
 * \verbatim embed:rst:leading-asterisk
 *  This class is templated to receive an arbitrary function that converts a discrete size to a
 *  continuous size. Call this function :math:`S`. Then the kernel is
 *
 *  .. math::
 *
 *      A(i,j) = C (S(i) \times S(j))
 *
 * \endverbatim
 *
 * @tparam AtomsToSizeFcn An object that can convert a discrete size to a continuous size.
 * @tparam Number Floating point number type used. For example, ``float`` or ``double``.
 */
template <HasAtomsToSizeOperator AtomsToSizeFcn, typename Number = sunrealtype>
class MultiplicativeAgglomerationKernel {
 public:
  /**
   * @brief Constructor.
   *
   * @param to_size_fcn Function used to convert from discrete to continuous size.
   * @param rate
   */
  MultiplicativeAgglomerationKernel(const AtomsToSizeFcn& to_size_fcn, const Number rate = 1)
      : to_size(to_size_fcn), rate(rate) {}

  MultiplicativeAgglomerationKernel() = delete;

  /**
   * @brief Calculates the agglomeration rate.
   *
   * @param n_atoms1 The number of atoms in the first particle that agglomerates.
   * @param n_atoms2 The number of atoms in the second particle that agglomerates.
   * @return The rate of agglomeration between the two particles.
   */
  Number operator()(const sunindextype n_atoms1, const sunindextype n_atoms2) const {
    return rate * (to_size(n_atoms1) * to_size(n_atoms2));
  }

 private:
  const AtomsToSizeFcn to_size;
  const Number rate;
};


/**
 * @brief Represents the Diffusion limited kernel.
 *
 * \verbatim embed:rst:leading-asterisk
 *  This class is templated to receive an arbitrary function that converts a discrete size to a
 *  continuous size. Call this function :math:`S`. Then the kernel is
 *
 *  .. math::
 *
 *      A(i,j) = \frac23 \frac{k_B T}{\eta}
 *               \left( S(i)^{1/d} + S(j)^{1/d} \right)
 *               \left( S(i)^{-1/d} + S(j)^{-1/d} \right)
 *
 *  where :math:`k_B` is the Boltzmann constant, :math:`T` is the temperature,
 *  :math:`\eta` is the viscosity of the background fluid, and :math:`d` is the fractal dimension
 *  of the particles.
 * \endverbatim
 *
 * @tparam AtomsToSizeFcn An object that can convert a discrete size to a continuous size.
 * @tparam Number Floating point number type used. For example, ``float`` or ``double``.
 */
template <HasAtomsToSizeOperator AtomsToSizeFcn, typename Number = sunrealtype>
class DiffusionLimitedAgglomerationKernel {
 public:
  /**
   * @brief Constructor.
   *
   * @param to_size_fcn Function used to convert from discrete to continuous size.
   * @param temperature Temperature of the system (assumed constant).
   * @param viscosity Viscosity of the background fluid (assumed constant).
   * @param fractal_dim Fractal dimension of the particle.
   */
  DiffusionLimitedAgglomerationKernel(const AtomsToSizeFcn& to_size_fcn, const Number temperature,
                                      const Number viscosity, const Number fractal_dim = 1)
      : to_size(to_size_fcn),
        factor(2. / 3. * kB * temperature / viscosity),
        fractal_dim(fractal_dim) {}

  DiffusionLimitedAgglomerationKernel() = delete;

  /**
   * @brief Calculates the agglomeration rate.
   *
   * @param n_atoms1 The number of atoms in the first particle that agglomerates.
   * @param n_atoms2 The number of atoms in the second particle that agglomerates.
   * @return The rate of agglomeration between the two particles.
   */
  Number operator()(const sunindextype n_atoms1, const sunindextype n_atoms2) const {
    const auto sz1 = std::pow(to_size(n_atoms1), 1. / fractal_dim);
    const auto sz2 = std::pow(to_size(n_atoms2), 1. / fractal_dim);

    return factor * (sz1 + sz2) * (1. / sz1 + 1. / sz2);
  }

 private:
  const AtomsToSizeFcn to_size;
  static constexpr Number kB = 1.380649e-23;  // if based in nm -> 1.380649e-5
  const Number factor;
  const Number fractal_dim;
};


/**
 * @brief Represents the Reaction limited kernel.
 *
 * \verbatim embed:rst:leading-asterisk
 *  This class is templated to receive an arbitrary function that converts a discrete size to a
 *  continuous size. Call this function :math:`S`. Then the kernel is
 *
 *  .. math::
 *
 *      A(i,j) = \frac23 \frac{k_B T}{\eta}
 *               \frac{(S(i)S(j))^\gamma}{W}
 *               \left( S(i)^{1/d} + S(j)^{1/d} \right)
 *               \left( S(i)^{-1/d} + S(j)^{-1/d} \right)
 *
 *  where :math:`k_B` is the Boltzmann constant, :math:`T` is the temperature,
 *  :math:`\eta` is the viscosity of the background fluid, :math:`\gamma` is the exponent of the
 *  product kernel, :math:`W` is the Fuch's stability ratio, and :math:`d` is the fractal dimension
 *  of the particles.
 * \endverbatim
 *
 * @tparam AtomsToSizeFcn An object that can convert a discrete size to a continuous size.
 * @tparam Number Floating point number type used. For example, ``float`` or ``double``.
 */
template <HasAtomsToSizeOperator AtomsToSizeFcn, typename Number = sunrealtype>
class ReactionLimitedAgglomerationKernel {
 public:
  /**
   * @brief Constructor.
   *
   * @param to_size_fcn Function used to convert from discrete to continuous size.
   * @param temperature Temperature of the system (assumed constant).
   * @param viscosity Viscosity of the background fluid (assumed constant).
   * @param fractal_dim Fractal dimension of the particle.
   */
  ReactionLimitedAgglomerationKernel(const AtomsToSizeFcn& to_size_fcn, const Number temperature,
                                     const Number viscosity, const Number kernel_exponent = 1,
                                     const Number fuch_stability_ratio = 1,
                                     const Number fractal_dim          = 1)
      : to_size(to_size_fcn),
        gamma(kernel_exponent),
        factor(2. / 3. * kB * temperature / viscosity / fuch_stability_ratio),
        fractal_dim(fractal_dim) {}

  ReactionLimitedAgglomerationKernel() = delete;

  /**
   * @brief Calculates the agglomeration rate.
   *
   * @param n_atoms1 The number of atoms in the first particle that agglomerates.
   * @param n_atoms2 The number of atoms in the second particle that agglomerates.
   * @return The rate of agglomeration between the two particles.
   */
  Number operator()(const sunindextype n_atoms1, const sunindextype n_atoms2) const {
    const auto sz1 = std::pow(to_size(n_atoms1), 1. / fractal_dim);
    const auto sz2 = std::pow(to_size(n_atoms2), 1. / fractal_dim);

    return factor * std::pow(to_size(n_atoms1) * to_size(n_atoms2), gamma) * (sz1 + sz2) *
           (1. / sz1 + 1. / sz2);
  }

 private:
  const AtomsToSizeFcn to_size;
  const Number gamma;
  static constexpr Number kB = 1.380649e-23;  // if based in nm -> 1.380649e-5
  const Number factor;
  const Number fractal_dim;
};


/**
 * @brief A class to calculate all of the contributions to the ODEs due to one type of
 * agglomeration.
 *
 * \verbatim embed:rst:leading-asterisk
 *  For a given kernel, this computes the contributions representing the following mechanism
 *
 *  .. math::
 *
 *      P_i + P_j \to P_{i+j} \qquad i,j\in [a,b]
 *
 *  where :math:`a,b` are the first and last particle (discrete) size involved in this agglomeration
 *  mechanism. It is assumed the particles are stored in ascending order in the solution vector.
 * \endverbatim
 *
 * @tparam Kernel The object type used to calculate the agglomeration rate.
 * @tparam Number Floating point number type used. For example, ``float`` or ``double``.
 */
template <IsAgglomerationKernel Kernel, typename Number = sunrealtype>
class ParticleAgglomeration : public OdeContribution {
 public:
  /**
   * @brief Constructor. Defines the range of particle sizes involved and the agglomeration kernel.
   *
   * @param first_n_atoms The first discrete size involved in this agglomeration.
   * @param last_n_atoms The last discrete size involved in this agglomeration.
   * @param first_n_atoms_idx The vector index of the first discrete size.
   * @param kernel_fcn The object that calculates the agglomeration rate.
   */
  ParticleAgglomeration(const sunindextype first_n_atoms, const sunindextype last_n_atoms,
                        const sunindextype first_n_atoms_idx, const Kernel& kernel_fcn)
      : first_n_atoms(first_n_atoms),
        last_n_atoms(last_n_atoms),
        first_n_atoms_idx(first_n_atoms_idx),
        agglom_kernel(kernel_fcn) {}


  /// \copydoc OdeContribution::add_to_rhs
  void add_to_rhs(const sunrealtype t, const N_Vector y, N_Vector ydot) const override {
    const auto y_data = N_VGetArrayPointer(y);
    auto rhs_data     = N_VGetArrayPointer(ydot);
    for (sunindextype n_atoms1 = first_n_atoms; n_atoms1 <= last_n_atoms; ++n_atoms1) {
      const sunindextype idx_sz1 = first_n_atoms_idx + (n_atoms1 - first_n_atoms);
      for (sunindextype n_atoms2 = n_atoms1; n_atoms2 <= last_n_atoms; ++n_atoms2) {
        const sunindextype idx_sz2  = first_n_atoms_idx + (n_atoms2 - first_n_atoms);
        const sunindextype n_atoms3 = n_atoms1 + n_atoms2;
        const sunindextype idx_sz3  = first_n_atoms_idx + (n_atoms3 - first_n_atoms);
        const auto rate = agglom_kernel(n_atoms1, n_atoms2) * y_data[idx_sz1] * y_data[idx_sz2];
        rhs_data[idx_sz1] -= rate;
        rhs_data[idx_sz2] -= rate;
        rhs_data[idx_sz3] += rate;
      }
    }
  }


  /// \copydoc OdeContribution::add_to_jac_times_v
  void add_to_jac_times_v(const N_Vector v, N_Vector Jv, const sunrealtype t, const N_Vector y,
                          const N_Vector ydot, N_Vector tmp) const override {
    const auto v_data = N_VGetArrayPointer(v);
    auto Jv_data      = N_VGetArrayPointer(Jv);
    const auto y_data = N_VGetArrayPointer(y);

    for (sunindextype n_atoms1 = first_n_atoms; n_atoms1 <= last_n_atoms; ++n_atoms1) {
      const sunindextype idx_sz1 = first_n_atoms_idx + (n_atoms1 - first_n_atoms);
      for (sunindextype n_atoms2 = n_atoms1; n_atoms2 <= last_n_atoms; ++n_atoms2) {
        const sunindextype idx_sz2  = first_n_atoms_idx + (n_atoms2 - first_n_atoms);
        const sunindextype n_atoms3 = n_atoms1 + n_atoms2;
        const sunindextype idx_sz3  = first_n_atoms_idx + (n_atoms3 - first_n_atoms);

        const auto rate = agglom_kernel(n_atoms1, n_atoms2);

        const auto drate_dsz1 = rate * y_data[idx_sz2];
        const auto drate_dsz2 = rate * y_data[idx_sz1];

        const auto jac_times_v = drate_dsz1 * v_data[idx_sz1] + drate_dsz2 * v_data[idx_sz2];

        // Matrix multiplication: J*v -> Jv
        // Non-zero rows of J  :    idx_sz1, idx_sz2, idx_sz3
        // Non-zero derivatives:    df/didx_sz1, df/didx_sz2
        Jv_data[idx_sz1] -= jac_times_v;
        Jv_data[idx_sz2] -= jac_times_v;
        Jv_data[idx_sz3] += jac_times_v;
      }
    }
  }


  /// \copydoc OdeContribution::add_to_jac
  void add_to_jac(gko::matrix_data<sunrealtype, sunindextype>& sparsity_pattern,
                  const sunrealtype t, const N_Vector y, const N_Vector fy, N_Vector tmp1,
                  N_Vector tmp2, N_Vector tmp3) const override {
    const auto y_data = N_VGetArrayPointer(y);
    for (sunindextype n_atoms1 = first_n_atoms; n_atoms1 <= last_n_atoms; ++n_atoms1) {
      const sunindextype idx_sz1 = first_n_atoms_idx + (n_atoms1 - first_n_atoms);
      for (sunindextype n_atoms2 = n_atoms1; n_atoms2 <= last_n_atoms; ++n_atoms2) {
        const sunindextype idx_sz2  = first_n_atoms_idx + (n_atoms2 - first_n_atoms);
        const sunindextype n_atoms3 = n_atoms1 + n_atoms2;
        const sunindextype idx_sz3  = first_n_atoms_idx + (n_atoms3 - first_n_atoms);

        const auto rate = agglom_kernel(n_atoms1, n_atoms2);

        const auto drate_dsz1 = rate * y_data[idx_sz2];
        const auto drate_dsz2 = rate * y_data[idx_sz1];

        sparsity_pattern.nonzeros.emplace_back(idx_sz1, idx_sz1, -drate_dsz1);
        sparsity_pattern.nonzeros.emplace_back(idx_sz1, idx_sz2, -drate_dsz2);

        sparsity_pattern.nonzeros.emplace_back(idx_sz2, idx_sz1, -drate_dsz1);
        sparsity_pattern.nonzeros.emplace_back(idx_sz2, idx_sz2, -drate_dsz2);

        sparsity_pattern.nonzeros.emplace_back(idx_sz3, idx_sz1, drate_dsz1);
        sparsity_pattern.nonzeros.emplace_back(idx_sz3, idx_sz2, drate_dsz2);
      }
    }
  }

 private:
  const sunindextype first_n_atoms;
  const sunindextype last_n_atoms;
  const sunindextype first_n_atoms_idx;
  const Kernel agglom_kernel;
};
}  // namespace NanoPBM

#endif  // NANOPBM_AGGLOMERATION_H