#ifndef NANOPBM_CHEMICAL_REACTION_H
#define NANOPBM_CHEMICAL_REACTION_H


#include <sundials/sundials_matrix.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>
#include <sunmatrix/sunmatrix_dense.h>

#include <array>

// TODO: change the vector and matrix types to be wrappers of SUNDIALS objects
namespace NanoPBM {

class ChemicalReactionBase {
 public:
  virtual void add_to_rhs(const N_Vector y, N_Vector rhs) const     = 0;
  virtual void add_to_jacobian(const N_Vector y, SUNMatrix J) const = 0;
  virtual void make_jacobian_sparsity(SUNMatrix J) const            = 0;
};

template <sunindextype n_reactants, sunindextype n_products>
struct ChemicalReactionParameters {
  ChemicalReactionParameters() {}
  ChemicalReactionParameters(std::array<sunindextype, n_reactants> r_indices,
                             std::array<sunrealtype, n_reactants> r_stoich,
                             std::array<sunrealtype, n_reactants> r_order,
                             std::array<sunindextype, n_reactants> p_indices,
                             std::array<sunrealtype, n_reactants> p_stoich,
                             std::array<sunrealtype, n_reactants> p_order)
      : reactant_indices(r_indices),
        reactant_stoich(r_stoich),
        reactant_order(r_order),
        product_indices(p_indices),
        product_stoich(p_stoich),
        product_order(p_order) {}


  std::array<sunindextype, n_reactants> reactant_indices;
  std::array<sunrealtype, n_reactants> reactant_stoich, reactant_order;
  std::array<sunindextype, n_reactants> product_indices;
  std::array<sunrealtype, n_products> product_stoich, product_order;
};


template <sunindextype n_reactants, sunindextype n_products, typename RxnProgress>
class ChemicalReaction : ChemicalReactionBase {
 public:
  ChemicalReaction(const ChemicalReactionParameters<n_reactants, n_products>& parameters,
                   const RxnProgress& rxn_progress_fcn)
      : reactant_indices(parameters.reactant_indices),
        reactant_stoich(parameters.reactant_stoich),
        reactant_order(parameters.reactant_order),
        product_indices(parameters.product_indices),
        product_stoich(parameters.product_stoich),
        product_order(parameters.product_order),
        rxn_rate(rxn_progress_fcn){};


  void add_to_rhs(const N_Vector y, N_Vector rhs) const {
    const auto y_data = N_VGetArrayPointer(y);
    auto rhs_data     = N_VGetArrayPointer(rhs);

    const auto forward_rate  = rxn_rate.forward(y, reactant_indices, reactant_order);
    const auto backward_rate = rxn_rate.backward(y, product_indices, product_order);

    for (sunindextype i = 0; i < n_reactants; ++i) {
      const auto idx    = reactant_indices[i];
      const auto stoich = reactant_stoich[i];
      rhs_data[idx] += stoich * (backward_rate - forward_rate);
    }

    for (sunindextype i = 0; i < n_products; ++i) {
      const auto idx    = product_indices[i];
      const auto stoich = product_stoich[i];
      rhs_data[idx] += stoich * (forward_rate - backward_rate);
    }
  }


  void add_to_jacobian(const N_Vector y, SUNMatrix J) const {
    auto J_data = SUNDenseMatrix_Cols(J);

    const auto dkf_di = rxn_rate.forward_derivatives(y, reactant_indices, reactant_order);
    const auto dkb_di = rxn_rate.backward_derivatives(y, product_indices, product_order);

    // Derivatives with respect to reactants
    for (sunindextype j = 0; j < n_reactants; ++j) {
      const sunindextype idx2 = reactant_indices[j];
      const auto deriv        = dkf_di[j];

      // Reactants
      for (sunindextype i = 0; i < n_reactants; ++i) {
        const sunindextype idx1 = reactant_indices[i];
        const auto stoich       = reactant_stoich[i];
        J_data[idx2][idx1] -= stoich * deriv;
      }

      // Products
      for (sunindextype i = 0; i < n_products; ++i) {
        const sunindextype idx1 = product_indices[i];
        const auto stoich       = product_stoich[i];
        J_data[idx2][idx1] += stoich * deriv;
      }
    }

    // Derivatives with respect to products
    for (sunindextype j = 0; j < n_products; ++j) {
      const sunindextype idx2 = product_indices[j];
      const auto deriv        = dkb_di[j];

      // Reactants
      for (sunindextype i = 0; i < n_reactants; ++i) {
        const sunindextype idx1 = reactant_indices[i];
        const auto stoich       = reactant_stoich[i];
        J_data[idx2][idx1] += stoich * deriv;
      }

      // Products
      for (sunindextype i = 0; i < n_products; ++i) {
        const sunindextype idx1 = product_indices[i];
        const auto stoich       = product_stoich[i];
        J_data[idx2][idx1] -= stoich * deriv;
      }
    }
  }


  void make_jacobian_sparsity(SUNMatrix J) const {
    auto J_data = SUNDenseMatrix_Cols(J);

    // Derivatives with respect to reactants
    for (sunindextype j = 0; j < n_reactants; ++j) {
      const sunindextype idx2 = reactant_indices[j];

      // Reactants
      for (sunindextype i = 0; i < n_reactants; ++i) {
        const sunindextype idx1 = reactant_indices[i];
        const auto stoich       = reactant_stoich[i];
        J_data[idx2][idx1]      = 1;
      }

      // Products
      for (sunindextype i = 0; i < n_products; ++i) {
        const sunindextype idx1 = product_indices[i];
        const auto stoich       = product_stoich[i];
        J_data[idx2][idx1]      = 1;
      }
    }

    // Derivatives with respect to products
    for (sunindextype j = 0; j < n_products; ++j) {
      const sunindextype idx2 = product_indices[j];

      // Reactants
      for (sunindextype i = 0; i < n_reactants; ++i) {
        const sunindextype idx1 = reactant_indices[i];
        const auto stoich       = reactant_stoich[i];
        J_data[idx2][idx1]      = 1;
      }

      // Products
      for (sunindextype i = 0; i < n_products; ++i) {
        const sunindextype idx1 = product_indices[i];
        const auto stoich       = product_stoich[i];
        J_data[idx2][idx1]      = 1;
      }
    }
  }


 private:
  const std::array<sunindextype, n_reactants> reactant_indices;
  const std::array<sunrealtype, n_reactants> reactant_stoich, reactant_order;
  const std::array<sunindextype, n_reactants> product_indices;
  const std::array<sunrealtype, n_products> product_stoich, product_order;

  const RxnProgress rxn_rate;
};
}  // namespace NanoPBM

#endif  // NANOPBM_CHEMICAL_REACTION_H