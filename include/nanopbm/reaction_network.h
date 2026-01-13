#ifndef NANOPBM_REACTION_NETWORK_H
#define NANOPBM_REACTION_NETWORK_H

#include <memory>
#include <vector>

#include "nanopbm/chemical_reaction.h"

namespace NanoPBM {
class ReactionNetwork : ChemicalReactionBase {
 public:
  void add_reaction(const ChemicalReactionBase& rxn) {
    reactions.push_back(std::make_shared<ChemicalReactionBase>(&rxn));
  }

  void add_to_rhs(const N_Vector y, N_Vector rhs) const override {
    for (const auto& rxn : reactions) {
      rxn->add_to_rhs(y, rhs);
    }
  }

  void add_to_jacobian(const N_Vector y, SUNMatrix J) const override {
    for (const auto& rxn : reactions) {
      rxn->add_to_jacobian(y, J);
    }
  }

  void make_jacobian_sparsity(SUNMatrix J) const override {
    for (const auto& rxn : reactions) {
      rxn->make_jacobian_sparsity(J);
    }
  }

 private:
  std::vector<std::shared_ptr<ChemicalReactionBase>> reactions;
};
}  // namespace NanoPBM

#endif  // NANOPBM_REACTION_NETWORK_H