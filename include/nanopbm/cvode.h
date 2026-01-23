#ifndef NANOPBM_CVODE_H
#define NANOPBM_CVODE_H

#include <cvode/cvode.h>
#include <cvode/cvode_bandpre.h>
#include <cvode/cvode_bbdpre.h>
#include <cvode/cvode_ls.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_core.h>
#include <sundials/sundials_iterative.h>
#include <sundials/sundials_linearsolver.h>
#include <sundials/sundials_matrix.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>
#include <sunlinsol/sunlinsol_lapackdense.h>
#include <sunlinsol/sunlinsol_spbcgs.h>
#include <sunlinsol/sunlinsol_spgmr.h>
#include <sunlinsol/sunlinsol_sptfqmr.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunmatrix/sunmatrix_sparse.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <sundials/sundials_context.hpp>

#include "ode_contribution.h"

namespace NanoPBM {

enum class LinearAlgebraType { LAPACK, SUNDIALSSPARSE, GINKGO };

// FIXME: make this not be inline
inline void check_cvode_error(const int err_code, const std::string& calling_function,
                              const std::string& err_code_source) {
  if (err_code == CV_SUCCESS || err_code == CVLS_SUCCESS) {
    return;
  }

  const std::string border =
      "----------------------------------------------------------------------------\n";

  std::string msg = border;
  msg += "[" + calling_function + "]\n";
  msg += "  Exception occurred when calling " + err_code_source + "\n";
  if (err_code == CV_MEM_NULL) {
    msg +=
        "  Error code: CV_MEM_NULL\n"
        "  Error: The CVODE memory block was not initialized correctly.\n"
        "         Check the previous call to CVodeCreate().\n";
  } else if (err_code == CV_MEM_FAIL) {
    msg +=
        "  Error code: CV_MEM_FAIL\n"
        "  Error: A memory allocation request has failed.\n";
  } else if (err_code == CV_ILL_INPUT) {
    msg +=
        "  Error code: CV_ILL_INPUT\n"
        "  Error: An input argument has an illegal value.\n";
  } else if (err_code == CV_NO_MALLOC) {
    msg +=
        "  Error code: CV_NO_MALLOC\n"
        "  Error: The allocation function returned NULL\n";
  } else if (err_code == CV_TOO_CLOSE) {
    msg +=
        "  Error code: CV_TOO_CLOSE\n"
        "  Error: The initial time and the output time are to close to each other.\n";
  } else if (err_code == CV_TOO_MUCH_WORK) {
    msg +=
        "  Error code: CV_TOO_MUCH_WORK\n"
        "  Error: CVODE reach the maximum time steps allowed.\n";
  } else if (err_code == CV_TOO_MUCH_ACC) {
    msg +=
        "  Error code: CV_TOO_MUCH_ACC\n"
        "  Error: CVODE could not satisfy the provided accuracy.\n";
  } else if (err_code == CV_ERR_FAILURE) {
    msg +=
        "  Error code: CV_ERR_FAILURE\n"
        "  Error: The CVODE error test failed too many times\n"
        "         or the minimum time step value was reached.\n";
  } else if (err_code == CV_CONV_FAILURE) {
    msg +=
        "  Error code: CV_CONV_FAILURE\n"
        "  Error: The CVODE convergence test failed too many times\n"
        "         or the minimum time step value was reached.\n";
  } else if (err_code == CV_LINIT_FAIL) {
    msg +=
        "  Error code: CV_LINIT_FAIL\n"
        "  Error: The linear solver interface's initialization function failed.\n";
  } else if (err_code == CV_LSETUP_FAIL) {
    msg +=
        "  Error code: CV_LSETUP_FAIL\n"
        "  Error: The linear solver interface's setup function failed in an unrecoverable "
        "manner.\n";
  } else if (err_code == CV_LSOLVE_FAIL) {
    msg +=
        "  Error code: CV_LSOLVE_FAIL\n"
        "  Error: The linear solver interface's solve function failed in an unrecoverable "
        "manner.\n";
  } else if (err_code == CV_CONSTR_FAIL) {
    msg +=
        "  Error code: CV_CONSTR_FAIL\n"
        "  Error: The inequality constraints were violated and the solver was unable to recover.\n";
  } else if (err_code == CV_RHSFUNC_FAIL) {
    msg +=
        "  Error code: CV_RHSFUNC_FAIL\n"
        "  Error: The right-hand side function failed in an unrecoverable manner.\n";
  } else if (err_code == CVLS_MEM_NULL) {
    msg +=
        "  Error code: CVLS_MEM_NULL\n"
        "  Error: The cvode_mem pointer is NULL\n";
  } else if (err_code == CVLS_ILL_INPUT) {
    msg +=
        "  Error code: CVLS_ILL_INPUT\n"
        "  Error: The linear solver interface used by CVODE is not compatible\n"
        "         with the SUNLinearSolver or SUNMatrix input objects \n"
        "         or the N_Vector supplied.\n";
  } else if (err_code == CVLS_SUNLS_FAIL) {
    msg +=
        "  Error code: CVLS_SUNLS_FAIL\n"
        "  Error: A call to the SUNLinearSolver object failed.\n";
  } else if (err_code == CVLS_MEM_FAIL) {
    msg +=
        "  Error code: CVLS_MEM_FAIL\n"
        "  Error: A memory allocation request failed.\n";
  } else if (err_code == CVLS_LMEM_NULL) {
    msg +=
        "  Error code: CVLS_LMEM_NULL\n"
        "  Error: The CVLS linear solver interface has not been initialized.\n"
        "         The function CVodeSetLinearSolver() must be called first!\n";
  } else {
    msg +=
        "  Unknown error code!\n"
        "  Error code: " +
        std::to_string(err_code) + "\n";
  }

  msg += border;
  throw std::runtime_error(msg);
  return;
}

struct CVODESettings {
  sunrealtype start_time = 0.0;
  sunrealtype reltol     = 1.e-6;
  sunrealtype abstol     = 1.e-8;

  // ---- Linear solver ----
  std::string linear_solver       = "direct";
  std::string preconditioner_type = "none";
  int max_linear_iterations       = 30;

  // ---- Optional CVODE settings ----
  int max_order                          = 5;
  long int max_n_steps                   = 100000;
  int max_n_hstep_msgs                   = 10;
  sunbooleantype set_stability_detection = SUNTRUE;
  sunrealtype initial_delta_t            = 0.0;
  sunrealtype min_delta_t                = 0.0;
  sunrealtype max_delta_t                = 0.0;
  sunrealtype stop_time                  = -1.0;
  sunbooleantype interp_stop_time        = SUNTRUE;
  int max_error_test_fails               = 7;

  // ---- Optional CVODE settings for linear solver ----
  sunrealtype max_gamma_change           = -1;  // negative value gives default
  sunrealtype max_gamma_jac_update       = -1;
  long int linear_solver_setup_frequency = 0;  // zero gives default
  long int jacobian_eval_frequency       = 0;
  int prec_upper_bandwidth               = 1;
  int prec_lower_bandwidth               = 1;
};


// FIXME: always use BDF?
template <int METHOD = CV_BDF>
class CVODE {
 private:
 public:
  CVODE() = delete;

  template <typename ContributionType>
    requires std::derived_from<ContributionType, OdeContribution>
  CVODE(const sundials::Context& sunctx, N_Vector initial_condition,
        const ContributionType& ode_contribution, const CVODESettings& settings = CVODESettings{})
      : ode_model(std::make_shared<ContributionType>(ode_contribution)), settings(settings) {
    int err_code     = 0;
    const auto check = [&](const std::string& fcn_name) {
      check_cvode_error(err_code, "CVODE constructor", fcn_name);
    };

    make_cvode_functions();

    cvode_mem = CVodeCreate(METHOD, sunctx);
    err_code  = CVodeInit(cvode_mem, cvode_rhs_fcn, settings.start_time, initial_condition);
    check("CVodeInit");

    n_odes = N_VGetLength(initial_condition);


    make_linear_solver(sunctx, initial_condition);
    apply_settings();

    err_code = CVodeSetUserData(cvode_mem, static_cast<void*>(&ode_model));
    check("CVodeSetUserData");

    // TODO: positivity enforcement
    cvode_constraints = N_VClone(initial_condition);
    N_VConst(1.0, cvode_constraints);
    err_code = CVodeSetConstraints(cvode_mem, cvode_constraints);
    check("CVodeSetConstraints");
  }

  ~CVODE() {
    N_VDestroy(cvode_constraints);
    if (cvode_constraints != nullptr) {
      // N_VDestroy(cvode_constraints);
    }
    if (linear_solver != nullptr) {
      SUNLinSolFree(linear_solver);
    }

    if (template_matrix != nullptr) {
      SUNMatDestroy(template_matrix);
    }
    CVodeFree(&cvode_mem);
  }

  void solve(const sunrealtype tout, N_Vector yout, sunrealtype& t_reached) {
    const int err_code = CVode(cvode_mem, tout, yout, &t_reached, CV_NORMAL);
    check_cvode_error(err_code, "CVODE::solve()", "CVode");
  }


  void print_statistics(const SUNOutputFormat fmt = SUN_OUTPUTFORMAT_TABLE) {
    CVodePrintAllStats(cvode_mem, stdout, fmt);
  }

  void print_statistics(const std::string& filename,
                        const SUNOutputFormat fmt = SUN_OUTPUTFORMAT_TABLE) {
    FILE* fptr = fopen(filename.c_str(), "w");
    if (fptr != nullptr) {
      CVodePrintAllStats(cvode_mem, fptr, fmt);
    }
    fclose(fptr);
  }


 private:
  std::shared_ptr<OdeContribution> ode_model;
  void* cvode_mem = nullptr;

  CVODESettings settings;
  int (*cvode_rhs_fcn)(sunrealtype, N_Vector, N_Vector, void*);
  int (*cvode_jac_fcn)(sunrealtype, N_Vector, N_Vector, SUNMatrix, void*, N_Vector, N_Vector,
                       N_Vector);
  int (*cvode_jac_times_v_fcn)(N_Vector, N_Vector, sunrealtype, N_Vector, N_Vector, void*,
                               N_Vector);

  SUNMatrix template_matrix     = nullptr;
  SUNLinearSolver linear_solver = nullptr;
  N_Vector cvode_constraints    = nullptr;

  sunindextype n_odes = 0;

  void make_cvode_functions();
  void make_linear_solver(const sundials::Context& sunctx, N_Vector initial_condition);
  void make_dense_solver(const sundials::Context& sunctx, N_Vector initial_condition);
  void make_iterative_solver(const sundials::Context& sunctx, N_Vector initial_condition);
  void apply_settings();
};


template <int METHOD>
void CVODE<METHOD>::make_cvode_functions() {
  cvode_rhs_fcn = [](sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) {
    std::shared_ptr<OdeContribution> fcns =
        *static_cast<std::shared_ptr<OdeContribution>*>(user_data);
    N_VConst(0, ydot);
    fcns->add_to_rhs(t, y, ydot);
    return 0;
  };

  cvode_jac_fcn = [](sunrealtype t, N_Vector y, N_Vector ydot, SUNMatrix Jac, void* user_data,
                     N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    std::shared_ptr<OdeContribution> fcns =
        *static_cast<std::shared_ptr<OdeContribution>*>(user_data);
    SUNMatZero(Jac);
    // FIXME:
    gko::matrix_data<sunrealtype, sunindextype> matrix_data;
    fcns->add_to_jac(matrix_data, t, y, ydot, tmp1, tmp2, tmp3);
    for (const auto& data : matrix_data.nonzeros) {
      SM_ELEMENT_D(Jac, data.row, data.column) = data.value;
    }
    return 0;
  };

  cvode_jac_times_v_fcn = [](N_Vector v, N_Vector Jv, sunrealtype t, N_Vector y, N_Vector fy,
                             void* user_data, N_Vector tmp) {
    std::shared_ptr<OdeContribution> fcns =
        *static_cast<std::shared_ptr<OdeContribution>*>(user_data);
    N_VConst(0, Jv);
    fcns->add_to_jac_times_v(v, Jv, t, y, fy, tmp);
    return 0;
  };
}

template <int METHOD>
void CVODE<METHOD>::make_dense_solver(const sundials::Context& sunctx, N_Vector initial_condition) {
  int err_code     = 0;
  const auto check = [&](const std::string& fcn_name) {
    check_cvode_error(err_code, "CVODE constructor", fcn_name);
  };

  template_matrix = SUNDenseMatrix(n_odes, n_odes, sunctx);
  SUNMatZero(template_matrix);
  linear_solver = SUNLinSol_LapackDense(initial_condition, template_matrix, sunctx);

  err_code = CVodeSetLinearSolver(cvode_mem, linear_solver, template_matrix);
  check("CVodeSetLinearSolver");
  err_code = CVodeSetJacFn(cvode_mem, cvode_jac_fcn);
  check("CVodeSetJacFn");
}


template <int METHOD>
void CVODE<METHOD>::make_iterative_solver(const sundials::Context& sunctx,
                                          N_Vector initial_condition) {
  int err_code     = 0;
  const auto check = [&](const std::string& fcn_name) {
    check_cvode_error(err_code, "CVODE constructor", fcn_name);
  };

  const int max_iter       = settings.max_linear_iterations;
  int preconditioning_type = SUN_PREC_NONE;
  if (settings.preconditioner_type == "none") {
    preconditioning_type = SUN_PREC_NONE;
  } else if (settings.preconditioner_type == "right") {
    preconditioning_type = SUN_PREC_RIGHT;
  } else if (settings.preconditioner_type == "left") {
    preconditioning_type = SUN_PREC_LEFT;
  } else {
    const std::string border =
        "----------------------------------------------------------------------------\n";
    std::string err_msg = border;
    err_msg += "  [CVODE] -> [make_iterative_solver]\n";
    err_msg += "    Error: Invalid preconditioning type specified.\n";
    err_msg += "    Additional details:\n";
    err_msg += "      Provided setting: " + settings.preconditioner_type + "\n";
    err_msg += "      Valid options   : none, right, left\n";
    err_msg += border;
    throw std::runtime_error(err_msg);
  }
  if (settings.linear_solver == "gmres") {
    linear_solver = SUNLinSol_SPGMR(initial_condition, preconditioning_type, max_iter, sunctx);
    // FIXME
    SUNLinSol_SPGMRSetGSType(linear_solver, SUN_MODIFIED_GS);
    // FIXME
    SUNLinSol_SPGMRSetMaxRestarts(linear_solver, -1);

  } else if (settings.linear_solver == "bicgstab") {
    linear_solver = SUNLinSol_SPBCGS(initial_condition, preconditioning_type, max_iter, sunctx);
  } else if (settings.linear_solver == "tfqmr") {
    linear_solver = SUNLinSol_SPTFQMR(initial_condition, preconditioning_type, max_iter, sunctx);
  } else {
    const std::string border =
        "----------------------------------------------------------------------------\n";
    std::string err_msg = border;
    err_msg += "  [CVODE] -> [make_iterative_solver]\n";
    err_msg += "    Error: Invalid iterative solver specified.\n";
    err_msg += "    Additional details:\n";
    err_msg += "      Provided setting: " + settings.linear_solver + "\n";
    err_msg += "      Valid options   : GMRES, BICGSTAB, TFQMR\n";
    err_msg += border;
    throw std::runtime_error(err_msg);
  }


  err_code = CVodeSetLinearSolver(cvode_mem, linear_solver, template_matrix);
  check("CVodeSetLinearSolver");
  err_code = CVodeSetJacTimes(cvode_mem, nullptr, cvode_jac_times_v_fcn);
  check("CVodeSetJacTimes");

  // FIXME: add logic for GINKGO preconditioner
  if (preconditioning_type != SUN_PREC_NONE) {
    // Use SUNDIALS preconditioner
    const int upper_bandwidth = settings.prec_upper_bandwidth;
    const int lower_bandwidth = settings.prec_lower_bandwidth;
    err_code                  = CVBandPrecInit(cvode_mem, n_odes, upper_bandwidth, lower_bandwidth);
  }
}

template <int METHOD>
void CVODE<METHOD>::make_linear_solver(const sundials::Context& sunctx,
                                       N_Vector initial_condition) {
  // make solver name lowercase for robustness
  std::transform(settings.linear_solver.begin(), settings.linear_solver.end(),
                 settings.linear_solver.begin(), [](unsigned char c) { return std::tolower(c); });

  std::transform(settings.preconditioner_type.begin(), settings.preconditioner_type.end(),
                 settings.preconditioner_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });


  if (settings.linear_solver == "direct") {
    make_dense_solver(sunctx, initial_condition);
  } else {
    make_iterative_solver(sunctx, initial_condition);
  }
  return;
}


template <int METHOD>
void CVODE<METHOD>::apply_settings() {
  int err_code     = 0;
  const auto check = [&](const std::string& fcn_name) {
    check_cvode_error(err_code, "CVODE constructor", fcn_name);
  };


  // TODO: logic to vary the tolerance function called
  err_code = CVodeSStolerances(cvode_mem, settings.reltol, settings.abstol);
  check("CVodeSStolerances");


  // Monitor the solution and do something at predefined intervals
  // TODO

  err_code = CVodeSetMaxOrd(cvode_mem, settings.max_order);
  check("CVodeSetMaxOrd");
  err_code = CVodeSetMaxNumSteps(cvode_mem, settings.max_n_steps);
  check("CVodeSetMaxNumSteps");
  err_code = CVodeSetMaxHnilWarns(cvode_mem, settings.max_n_hstep_msgs);
  check("CVodeSetMaxHnilWarns");
  err_code = CVodeSetStabLimDet(cvode_mem, settings.set_stability_detection);
  check("CVodeSetStabLimDet");
  err_code = CVodeSetInitStep(cvode_mem, settings.initial_delta_t);
  check("CVodeSetInitStep");
  err_code = CVodeSetMinStep(cvode_mem, settings.min_delta_t);
  check("CVodeSetMinStep");
  err_code = CVodeSetMaxStep(cvode_mem, settings.max_delta_t);
  check("CVodeSetMaxStep");
  if (settings.stop_time > 0.0) {
    err_code = CVodeSetStopTime(cvode_mem, settings.stop_time);
    check("CVodeSetStopTime");
  }
  err_code = CVodeSetInterpolateStopTime(cvode_mem, settings.interp_stop_time);
  check("CVodeSetInterpolateStopTime");
  err_code = CVodeSetMaxErrTestFails(cvode_mem, settings.max_error_test_fails);
  check("CVodeSetMaxErrTestFails");


  // ---- Optional CVODE settings for linear solver interface ----
  err_code = CVodeSetDeltaGammaMaxLSetup(cvode_mem, settings.max_gamma_change);
  check("CVodeSetDeltaGammaMaxLSetup");
  err_code = CVodeSetDeltaGammaMaxBadJac(cvode_mem, settings.max_gamma_jac_update);
  check("CVodeSetDeltaGammaMaxBadJac");
  err_code = CVodeSetLSetupFrequency(cvode_mem, settings.linear_solver_setup_frequency);
  check("CVodeSetLSetupFrequency");
  err_code = CVodeSetJacEvalFrequency(cvode_mem, settings.jacobian_eval_frequency);
  check("CVodeSetJacEvalFrequency");


  // TODO: if doing matrix free solver, call appropriate functions


  // ---- Nonlinear solver ----
  // TODO: do nothing right now -> this means Newton method is used -> good default
}

}  // namespace NanoPBM

#endif  // NANOPBM_CVODE_H