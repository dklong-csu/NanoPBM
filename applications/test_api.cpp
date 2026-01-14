
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
#include "nanopbm/reaction_progress.h"


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
    const NanoPBM::ConstantReversibleReactionProgress<1, 1> rxn_rate(1, 0.);
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