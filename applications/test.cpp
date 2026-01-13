
// #include "nanopbm/sundials_vector.h"
#include <nvector/nvector_serial.h>
#include <sundials/sundials_matrix.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>
#include <sunmatrix/sunmatrix_dense.h>

#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "cvode.h"
#include "nanopbm/chemical_reaction.h"


template <sunindextype n_reactants, sunindextype n_products>
class SimpleRxnRate {
 public:
  SimpleRxnRate(const sunrealtype kf, const sunrealtype kb, const sunrealtype tol = 1.e-12)
      : kf(kf), kb(kb), tol(tol) {}


  sunrealtype forward(const N_Vector y, const std::array<sunindextype, n_reactants> &indices,
                      const std::array<sunrealtype, n_reactants> &orders) const {
    sunrealtype rate  = kf;  // FIXME -- this would be more complex in a real example
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
      const N_Vector y, const std::array<sunindextype, n_reactants> &indices,
      const std::array<sunrealtype, n_reactants> &orders) const {
    const auto forward_rate = forward(y, indices, orders);
    std::array<sunrealtype, n_reactants> derivs;
    const auto y_data = N_VGetArrayPointer(y);
    for (sunindextype i = 0; i < n_reactants; ++i) {
      const auto idx                                    = indices[i];
      const auto order                                  = orders[i];
      std::array<sunrealtype, n_reactants> deriv_orders = orders;
      deriv_orders[i] -= 1;
      derivs[idx] = order * forward(y, indices, deriv_orders);
      // const auto conc = y_data[idx];
      // derivs[idx] = (conc < tol) ? 0.0 : order * forward_rate / conc;
    }
    return derivs;
  }


  sunrealtype backward(const N_Vector y, const std::array<sunindextype, n_products> &indices,
                       const std::array<sunrealtype, n_products> &orders) const {
    sunrealtype rate  = kb;  // FIXME -- this would be more complex in a real example
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
      const N_Vector y, const std::array<sunindextype, n_products> &indices,
      const std::array<sunrealtype, n_products> &orders) const {
    const auto backward_rate = backward(y, indices, orders);
    std::array<sunrealtype, n_products> derivs;
    const auto y_data = N_VGetArrayPointer(y);
    for (sunindextype i = 0; i < n_products; ++i) {
      const auto idx                                   = indices[i];
      const auto order                                 = orders[i];
      std::array<sunrealtype, n_products> deriv_orders = orders;
      deriv_orders[i] -= 1;
      derivs[idx] = order * backward(y, indices, deriv_orders);
      // const auto conc = y_data[idx];
      // derivs[idx] = (conc < tol) ? 0.0 : order * backward_rate / conc;
    }
    return derivs;
  }


 private:
  const sunrealtype kf, kb;
  const sunrealtype tol;
};

int main() {
  try {
    sundials::Context sunctx;

    const sunindextype n_odes = 2;

    // ---- Set initial conditions ----
    N_Vector y0 = N_VNew_Serial(n_odes, sunctx);
    {
      auto ic_data = N_VGetArrayPointer(y0);
      ic_data[0]   = 1;
      ic_data[1]   = 0;
    }

    // ---- Set up reactions ----
    const SimpleRxnRate<1, 1> rxn_rate(1, 0.);
    NanoPBM::ChemicalReactionParameters<1, 1> rxn_prm({0}, {1}, {1}, {1}, {1}, {1});
    const NanoPBM::ChemicalReaction rxn(rxn_prm, rxn_rate);


    // ---- Create Right Hand Side function ----
    std::function<int(sunrealtype, N_Vector, N_Vector)> rhs = [=](sunrealtype t, N_Vector y,
                                                                  N_Vector ydot) {
      rxn.add_to_rhs(y, ydot);
      return 0;
    };

    // ---- Create Jacobian function ----
    std::function<int(sunrealtype, N_Vector, N_Vector, SUNMatrix, N_Vector, N_Vector, N_Vector)>
        jac = [=](sunrealtype t, N_Vector y, N_Vector ydot, SUNMatrix J, N_Vector tmp1,
                  N_Vector tmp2, N_Vector tmp3) {
          rxn.add_to_jacobian(y, J);
          return 0;
        };

    // ---- ODE solver ----
    NanoPBM::CVODESettings cv_settings;
    cv_settings.reltol = 1e-8;
    NanoPBM::CVODE ode_solver(sunctx, y0, rhs, jac, cv_settings);

    // FIXME
    std::vector<std::string> data_names{"A", "B"};
    ode_solver.register_data_interpretation(data_names);

    // ---- Solve a few times ----
    const std::string filename = "test_out.txt";
    ode_solver.prepare_outfile(filename);

    sunrealtype time = 0;
    ode_solver.save_solution(time, y0);
    const sunrealtype dt = 0.01;
    for (int step = 1; step < 1000; ++step) {
      sunrealtype tout = step * dt;
      ode_solver.solve(tout, y0, time);
      ode_solver.save_solution(time, y0);
    }

    ode_solver.print_statistics();
    // ode_solver.print_statistics("test_solver_statistics.txt");

    // ---- Free memory ----
    N_VDestroy(y0);
  } catch (std::exception &exc) {
    std::cerr << std::endl << std::endl << exc.what() << std::endl << "Aborting!" << std::endl;

    return 1;
  } catch (...) {
    std::cerr << std::endl
              << std::endl
              << "------------------------------------------------------------------------------"
              << std::endl
              << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "------------------------------------------------------------------------------"
              << std::endl;

    return 1;
  }

  return 0;
}