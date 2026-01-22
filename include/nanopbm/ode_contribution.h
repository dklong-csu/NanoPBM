#ifndef NANOPBM_ODE_CONTRIBUTION_H
#define NANOPBM_ODE_CONTRIBUTION_H

#include <sundials/sundials_matrix.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>

#include <concepts>
#include <ginkgo/core/base/matrix_data.hpp>
#include <memory>
#include <vector>

namespace NanoPBM {


/**
 * @brief A base class representing a contribution to a system of ODEs.
 *
 * \verbatim embed:rst:leading-asterisk
 * This class defines how a contribution to a system of ordinary differential equations (ODEs)
 * is used within this library. This class will not add anything to the ODEs, but instead
 * provides the expected member functions that may be called. A concrete implementation
 * of an ODE contribution **must** be derived from this class.
 *
 * In general, this class provides a minimal interface to work with ``CVODE`` in ``SUNDIALS``.
 * ``CVODE`` solves the equation
 *
 * .. math::
 *    \frac{\mathrm d y}{\mathrm d t} = \dot y = f(t,y)
 *
 *
 * with an implicit method. Therefore, the Jacobian matrix
 *
 * .. math::
 *      (J)_{ij} = \frac{\partial f_i}{\partial y_j}
 *
 *
 * is required as part of the solution algorithm. This interface allows for the computation of
 * a Jacobian vector product for matrix-free methods and forming the Jacobian matrix
 * for a matrix-based solver or preconditioner.
 *
 * Example
 *
 * .. code-block:: cpp
 *
 *      class ExampleOde : public OdeContribution {
 *          public:
 *          void add_to_rhs(...) const override {
 *              // your implementation here
 *          }
 *          void add_to_jac_times(...) const override {
 *              // your implementation here
 *          }
 *          void add_to_jac(...) const override {
 *              // your implementation here
 *          }
 *      };
 * \endverbatim
 */
class OdeContribution {
 public:
  /**
   * @brief Adds to the right hand side function.
   *
   * @param t The time.
   * @param y The values of the solution vector.
   * @param ydot The vector holding the evaluation of the right-hand side function. This function
   *             should add to this vector rather than overwrite values.
   */
  virtual void add_to_rhs(const sunrealtype t, const N_Vector y, N_Vector ydot) const {}

  /**
   * @brief Computes the Jacobian times a vector without forming the Jacobian matrix.
   *
   * @param v The vector the Jacobian is multiplied by.
   * @param Jv The vector holding the result of the Jacobian vector product. This should be added to
   *           by this function.
   * @param t The time.
   * @param y The values of the solution vector.
   * @param ydot The right-hand side vector evaluated at the same ``y`` passed to this function.
   * @param tmp A vector of the same length as ``v``,``y``, etc. that may be used to aid in the
   *            calculation if necessary.
   */
  virtual void add_to_jac_times_v(const N_Vector v, N_Vector Jv, const sunrealtype t,
                                  const N_Vector y, const N_Vector ydot, N_Vector tmp) const {}


  // FIXME: Template this on floating point type to allow for mixed precision?
  // FIXME: Just make this a std::vector instead of the Ginkgo data type?
  /**
   * @brief Adds Jacobian entries to a Ginkgo matrix.
   *
   * If an explicit Jacobian matrix is formed, then this function is used to fill the matrix.
   * In order to support either sparse or dense matrices, the Jacobian values are exported
   * via an intermediate data type
   *
   * @param sparsity_pattern
   * @param t
   * @param y
   * @param fy
   * @param user_data
   * @param tmp1
   * @param tmp2
   * @param tmp3
   */
  virtual void add_to_jac(gko::matrix_data<sunrealtype, sunindextype>& sparsity_pattern,
                          const sunrealtype t, const N_Vector y, const N_Vector fy, N_Vector tmp1,
                          N_Vector tmp2, N_Vector tmp3) const {}
};


/**
 * @brief A container that holds many ODE contributions and applies them one after another.
 *
 */
class ManyOdeContributions : public OdeContribution {
 public:
  /**
   * @brief Adds a contribution to this class.
   *
   * @tparam ContributionType A class representing an ODE contribution.
   * @param fcn The contribution added to this class.
   */
  template <typename ContributionType>
    requires std::derived_from<ContributionType, OdeContribution>
  void add_contribution(const ContributionType& fcn) {
    contributions.push_back(std::make_shared<ContributionType>(fcn));
  }

  /// \copydoc OdeContribution::add_to_rhs
  void add_to_rhs(const sunrealtype t, const N_Vector y, N_Vector ydot) const override {
    for (const auto& contribution : contributions) {
      contribution->add_to_rhs(t, y, ydot);
    }
  }

  /// \copydoc OdeContribution::add_to_jac_times_v
  void add_to_jac_times_v(const N_Vector v, N_Vector Jv, const sunrealtype t, const N_Vector y,
                          const N_Vector ydot, N_Vector tmp) const override {
    for (const auto& contribution : contributions) {
      contribution->add_to_jac_times_v(v, Jv, t, y, ydot, tmp);
    }
  }


  /// \copydoc OdeContribution::add_to_jac
  void add_to_jac(gko::matrix_data<sunrealtype, sunindextype>& sparsity_pattern,
                  const sunrealtype t, const N_Vector y, const N_Vector fy, N_Vector tmp1,
                  N_Vector tmp2, N_Vector tmp3) const override {
    for (const auto& contribution : contributions) {
      contribution->add_to_jac(sparsity_pattern, t, y, fy, tmp1, tmp2, tmp3);
    }
  }

 private:
  std::vector<std::shared_ptr<OdeContribution>> contributions;
};

template <typename T>
  requires std::derived_from<T, OdeContribution>
struct OdeContributionParameters {};
}  // namespace NanoPBM

#endif  // NANOPBM_ODE_CONTRIBUTION_H