#ifndef NANOPBM_CHEMICAL_REACTION_H
#define NANOPBM_CHEMICAL_REACTION_H


#include <sundials/sundials_matrix.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>

#include <vector>

#include "ode_contribution.h"

// TODO: change the vector and matrix types to be wrappers of SUNDIALS objects
namespace NanoPBM {

// FIXME: do these need parameters to calculate gibbs
struct ReactionParameters {
  ReactionParameters() = default;
  ReactionParameters(const sunindextype idx, const sunrealtype stoich = 1,
                     const sunrealtype reaction_order = 1)
      : index(idx), stoich(stoich), order(reaction_order) {}
  sunindextype index;
  sunrealtype stoich;
  sunrealtype order;
};


struct ReactionParameterSet {
  std::vector<ReactionParameters> parameters;
  std::vector<ReactionParameters>* operator->() { return &parameters; }
  const std::vector<ReactionParameters>* operator->() const { return &parameters; }
  auto begin() { return parameters.begin(); }
  auto end() { return parameters.end(); }
  auto begin() const { return parameters.begin(); }
  auto end() const { return parameters.end(); }
  auto cbegin() const { return parameters.cbegin(); }
  auto cend() const { return parameters.cend(); }
};


template <typename T>
concept IsReactionRate =
    requires(const T t, const sunrealtype time, const N_Vector y, const ReactionParameterSet& prm1,
             const ReactionParameterSet& prm2, const ReactionParameters& prm3) {
      // FIXME: Do I need to pass both reactants and products to the functions?
      { t.forward(y, time, prm1, prm2) } -> std::convertible_to<sunrealtype>;
      { t.backward(y, time, prm1, prm2) } -> std::convertible_to<sunrealtype>;
      { t.df_dr(prm3, y, time, prm1, prm2) } -> std::convertible_to<sunrealtype>;
      { t.db_dp(prm3, y, time, prm1, prm2) } -> std::convertible_to<sunrealtype>;
    };

template <IsReactionRate RxnRate>
class ChemicalReaction : public OdeContribution {
 public:
  ChemicalReaction(const RxnRate& rate, const ReactionParameterSet& reactant_parameters,
                   const ReactionParameterSet& product_parameters)
      : rxn_rate(rate), reactants(reactant_parameters), products(product_parameters) {}

  /// \copydoc OdeContribution::add_to_rhs
  void add_to_rhs(const sunrealtype t, const N_Vector y, N_Vector ydot) const override {
    auto rhs_data = N_VGetArrayPointer(ydot);

    const auto kf = rxn_rate.forward(y, t, reactants, products);
    const auto kb = rxn_rate.backward(y, t, reactants, products);
    for (const auto& prm : reactants) {
      rhs_data[prm.index] += prm.stoich * (kb - kf);
    }
    for (const auto& prm : products) {
      rhs_data[prm.index] += prm.stoich * (kf - kb);
    }
  }

  /// \copydoc OdeContribution::add_to_jac_times_v
  void add_to_jac_times_v(const N_Vector v, N_Vector Jv, const sunrealtype t, const N_Vector y,
                          const N_Vector ydot, N_Vector tmp) const override {
    const auto v_data = N_VGetArrayPointer(v);
    auto Jv_data      = N_VGetArrayPointer(Jv);

    for (const auto& prm : reactants) {
      // Derivative of f (forward) wrt r (reactants)
      const sunindextype v_idx = prm.index;
      const auto deriv         = rxn_rate.df_dr(prm, y, t, reactants, products);
      for (const auto& prm_react : reactants) {
        const sunindextype Jv_idx = prm_react.index;
        Jv_data[Jv_idx] -= prm_react.stoich * deriv * v_data[v_idx];
      }
      for (const auto& prm_prod : products) {
        const sunindextype Jv_idx = prm_prod.index;
        Jv_data[Jv_idx] += prm_prod.stoich * deriv * v_data[v_idx];
      }
    }

    for (const auto& prm : products) {
      // Derivative of b (backward) wrt p (products)
      const sunindextype v_idx = prm.index;
      const auto deriv         = rxn_rate.db_dp(prm, y, t, reactants, products);
      for (const auto& prm_react : reactants) {
        const sunindextype Jv_idx = prm_react.index;
        Jv_data[Jv_idx] += prm_react.stoich * deriv * v_data[v_idx];
      }
      for (const auto& prm_prod : products) {
        const sunindextype Jv_idx = prm_prod.index;
        Jv_data[Jv_idx] -= prm_prod.stoich * deriv * v_data[v_idx];
      }
    }
  }


  /// \copydoc OdeContribution::add_to_jac
  void add_to_jac(gko::matrix_data<sunrealtype, sunindextype>& sparsity_pattern,
                  const sunrealtype t, const N_Vector y, const N_Vector fy, N_Vector tmp1,
                  N_Vector tmp2, N_Vector tmp3) const override {
    const auto y_data = N_VGetArrayPointer(y);

    // Derivative wrt reactants
    for (const auto& prm_deriv : reactants) {
      const sunindextype column = prm_deriv.index;
      const auto deriv          = rxn_rate.df_dr(prm_deriv, y, t, reactants, products);

      for (const auto& prm : reactants) {
        const sunindextype row = prm.index;
        sparsity_pattern.nonzeros.emplace_back(row, column, -prm.stoich * deriv);
      }

      for (const auto& prm : products) {
        const sunindextype row = prm.index;
        sparsity_pattern.nonzeros.emplace_back(row, column, prm.stoich * deriv);
      }
    }

    // Derivative wrt products
    for (const auto& prm_deriv : products) {
      const sunindextype column = prm_deriv.index;
      const auto deriv          = rxn_rate.db_dp(prm_deriv, y, t, reactants, products);

      for (const auto& prm : reactants) {
        const sunindextype row = prm.index;
        sparsity_pattern.nonzeros.emplace_back(row, column, prm.stoich * deriv);
      }

      for (const auto& prm : products) {
        const sunindextype row = prm.index;
        sparsity_pattern.nonzeros.emplace_back(row, column, -prm.stoich * deriv);
      }
    }
  }


 private:
  const RxnRate rxn_rate;
  ReactionParameterSet reactants;
  ReactionParameterSet products;

  // FIXME: might want this later
  void set(const ReactionParameterSet& new_reactants, const ReactionParameterSet& new_products) {
    reactants = new_reactants;
    products  = new_products;
  }
};


}  // namespace NanoPBM

#endif  // NANOPBM_CHEMICAL_REACTION_H