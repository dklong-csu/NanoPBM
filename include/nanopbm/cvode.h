#ifndef NANOPBM_CVODE_H
#define NANOPBM_CVODE_H

#include <cstdio>
#include <cvode/cvode_ls.h>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sundials/sundials_context.hpp>
#include <sundials/sundials_core.h>
#include <cvode/cvode.h>
#include <sundials/sundials_linearsolver.h>
#include <sundials/sundials_matrix.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>

#include <nvector/nvector_serial.h>
#include <sunlinsol/sunlinsol_lapackdense.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <vector>

namespace NanoPBM {

    enum class MatrixTypes {DENSE};
    enum class LinearSolverTypes {LAPACK};

    // FIXME: make this not be inline
    inline void check_cvode_error(const int err_code, 
                           const std::string & calling_function,
                           const std::string & err_code_source){
        if (err_code == CV_SUCCESS
            || err_code == CVLS_SUCCESS
        ){return;}

        const std::string border 
            = "----------------------------------------------------------------------------\n";
        
        std::string msg = border;
        msg += "[" + calling_function + "]\n";
        msg += "  Exception occurred when calling " + err_code_source + "\n";
        if (err_code == CV_MEM_NULL){
            msg += "  Error code: CV_MEM_NULL\n"
                   "  Error: The CVODE memory block was not initialized correctly.\n"
                   "         Check the previous call to CVodeCreate().\n";
        } else if (err_code == CV_MEM_FAIL){
            msg += "  Error code: CV_MEM_FAIL\n"
                   "  Error: A memory allocation request has failed.\n";
        } else if (err_code == CV_ILL_INPUT){
            msg += "  Error code: CV_ILL_INPUT\n"
                   "  Error: An input argument has an illegal value.\n";
        } else if (err_code == CV_NO_MALLOC){
            msg += "  Error code: CV_NO_MALLOC\n"
                   "  Error: The allocation function returned NULL\n";
        } else if (err_code == CV_TOO_CLOSE) {
            msg += "  Error code: CV_TOO_CLOSE\n"
                   "  Error: The initial time and the output time are to close to each other.\n";
        } else if (err_code == CV_TOO_MUCH_WORK) {
            msg += "  Error code: CV_TOO_MUCH_WORK\n"
                   "  Error: CVODE reach the maximum time steps allowed.\n";
        } else if (err_code == CV_TOO_MUCH_ACC) {
            msg += "  Error code: CV_TOO_MUCH_ACC\n"
                   "  Error: CVODE could not satisfy the provided accuracy.\n";
        } else if (err_code == CV_ERR_FAILURE) {
            msg += "  Error code: CV_ERR_FAILURE\n"
                   "  Error: The CVODE error test failed too many times\n"
                   "         or the minimum time step value was reached.\n";
        } else if (err_code == CV_CONV_FAILURE) {
            msg += "  Error code: CV_CONV_FAILURE\n"
                   "  Error: The CVODE convergence test failed too many times\n"
                   "         or the minimum time step value was reached.\n";
        } else if (err_code == CV_LINIT_FAIL) {
            msg += "  Error code: CV_LINIT_FAIL\n"
                   "  Error: The linear solver interface's initialization function failed.\n";
        } else if (err_code == CV_LSETUP_FAIL) {
            msg += "  Error code: CV_LSETUP_FAIL\n"
                   "  Error: The linear solver interface's setup function failed in an unrecoverable manner.\n";
        } else if (err_code == CV_LSOLVE_FAIL) {
            msg += "  Error code: CV_LSOLVE_FAIL\n"
                   "  Error: The linear solver interface's solve function failed in an unrecoverable manner.\n";
        } else if (err_code == CV_CONSTR_FAIL) {
            msg += "  Error code: CV_CONSTR_FAIL\n"
                   "  Error: The inequality constraints were violated and the solver was unable to recover.\n";
        } else if (err_code == CV_RHSFUNC_FAIL) {
            msg += "  Error code: CV_RHSFUNC_FAIL\n"
                   "  Error: The right-hand side function failed in an unrecoverable manner.\n";
        }   else if (err_code == CVLS_MEM_NULL){
            msg += "  Error code: CVLS_MEM_NULL\n"
                   "  Error: The cvode_mem pointer is NULL\n";
        } else if (err_code == CVLS_ILL_INPUT){
            msg += "  Error code: CVLS_ILL_INPUT\n"
                   "  Error: The linear solver interface used by CVODE is not compatible\n"
                   "         with the SUNLinearSolver or SUNMatrix input objects \n"
                   "         or the N_Vector supplied.\n";
        } else if(err_code == CVLS_SUNLS_FAIL){
            msg += "  Error code: CVLS_SUNLS_FAIL\n"
                   "  Error: A call to the SUNLinearSolver object failed.\n";
        } else if (err_code == CVLS_MEM_FAIL){
            msg += "  Error code: CVLS_MEM_FAIL\n"
                   "  Error: A memory allocation request failed.\n";
        } else if (err_code == CVLS_LMEM_NULL){
            msg += "  Error code: CVLS_LMEM_NULL\n"
                   "  Error: The CVLS linear solver interface has not been initialized.\n"
                   "         The function CVodeSetLinearSolver() must be called first!\n";
        }
        else {
            msg += "  Unknown error code!\n"
                   "  Error code: " + std::to_string(err_code) + "\n";
        }

        msg += border;
        throw std::runtime_error(msg);
        return;
    }

    struct CVODESettings {
        sunrealtype start_time = 0.0;
        sunrealtype reltol = 1.e-4;
        sunrealtype abstol = 1.e-8;

        // ---- Linear solver ----
        sunbooleantype matrix_free = false;
        MatrixTypes matrix_type = MatrixTypes::DENSE;
        LinearSolverTypes ls_type = LinearSolverTypes::LAPACK;

        // ---- Optional CVODE settings ----
        int max_order = 5;
        long int max_n_steps = 100000;
        int max_n_hstep_msgs = 10;
        sunbooleantype set_stability_detection = SUNTRUE;
        sunrealtype initial_delta_t = 0.0;
        sunrealtype min_delta_t = 0.0;
        sunrealtype max_delta_t = 0.0;
        sunrealtype stop_time = -1.0;
        sunbooleantype interp_stop_time = SUNTRUE;
        int max_error_test_fails = 7;

        // ---- Optional CVODE settings for linear solver ----
        sunrealtype max_gamma_change = -1; // negative value gives default 
        sunrealtype max_gamma_jac_update = -1;
        long int linear_solver_setup_frequency = 0; // zero gives default
        long int jacobian_eval_frequency = 0;

    };



    template <typename RHSFcn, typename JacFcn, int METHOD = CV_BDF>
    class CVODE {
        private:
        struct CVODEUserData {
            RHSFcn rhs;
            JacFcn jac;
        };
        public:
        CVODE() = delete;

        CVODE(const sundials::Context & sunctx, N_Vector initial_condition, 
              const RHSFcn & rhs, const JacFcn & jac,
              const CVODESettings & settings = CVODESettings{})
        : rhs(rhs), jac(jac), settings(settings)
        {
            int err_code = 0;
            const auto check = [&](const std::string & fcn_name){
                check_cvode_error(err_code, "CVODE constructor", fcn_name);
            };

            // ---- Basic setup of CVODE ----
            cvode_mem = CVodeCreate(METHOD, sunctx);

            cvode_rhs_fcn = [](sunrealtype t, N_Vector y, N_Vector ydot, void* user_data){
                const CVODEUserData* fcns 
                    = static_cast<CVODEUserData*>(user_data);
                N_VConst(0, ydot);
                return fcns->rhs(t, y, ydot);
            };
            err_code = CVodeInit(cvode_mem, cvode_rhs_fcn,settings.start_time,initial_condition);
            // TODO:
            check("CVodeInit");

            // TODO: logic to vary the tolerance function called
            err_code = CVodeSStolerances(cvode_mem, settings.reltol, settings.abstol);
            check("CVodeSStolerances");

            // Create a template matrix for the linear solver. 
            // TODO: It can remain as a nullptr if a matrix free method is used.
            n_odes = N_VGetLength(initial_condition);
            // TODO: logic for other matrix types
            template_matrix = SUNDenseMatrix(n_odes, n_odes, sunctx);
            SUNMatZero(template_matrix);
            // TODO: logic for other solver types
            linear_solver = SUNLinSol_LapackDense(initial_condition, template_matrix, sunctx);

            err_code = CVodeSetLinearSolver(cvode_mem, linear_solver, template_matrix);
            check("CVodeSetLinearSolver");
            // TODO: optional arguments for linear solver

            // ---- Optional CVODE settings ----
            user_data.rhs = rhs;
            user_data.jac = jac;
            err_code = CVodeSetUserData(cvode_mem, static_cast<void*>(&user_data));
            check("CVodeSetUserData");
            
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
            if (settings.stop_time > 0.0){
                err_code = CVodeSetStopTime(cvode_mem, settings.stop_time);
                check("CVodeSetStopTime");
            }
            err_code = CVodeSetInterpolateStopTime(cvode_mem, settings.interp_stop_time);
            check("CVodeSetInterpolateStopTime");
            err_code = CVodeSetMaxErrTestFails(cvode_mem, settings.max_error_test_fails);
            check("CVodeSetMaxErrTestFails");
            // TODO: positivity enforcement
            err_code = CVodeSetConstraints(cvode_mem, cvode_constraints);
            check("CVodeSetConstraints");

            // ---- Optional CVODE settings for linear solver interface ----
            err_code = CVodeSetDeltaGammaMaxLSetup(cvode_mem, settings.max_gamma_change);
            check("CVodeSetDeltaGammaMaxLSetup");
            err_code = CVodeSetDeltaGammaMaxBadJac(cvode_mem, settings.max_gamma_jac_update);
            check("CVodeSetDeltaGammaMaxBadJac");
            err_code = CVodeSetLSetupFrequency(cvode_mem, settings.linear_solver_setup_frequency);
            check("CVodeSetLSetupFrequency");
            err_code = CVodeSetJacEvalFrequency(cvode_mem, settings.jacobian_eval_frequency);
            check("CVodeSetJacEvalFrequency");

            cvode_jac_fcn = [](sunrealtype t, N_Vector y, N_Vector ydot, SUNMatrix Jac,
                               void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3){
                const CVODEUserData* fcns 
                    = static_cast<CVODEUserData*>(user_data);
                SUNMatZero(Jac);
                return fcns->jac(t, y, ydot, Jac, tmp1, tmp2, tmp3);
            };
            err_code = CVodeSetJacFn(cvode_mem, cvode_jac_fcn);
            check("CVodeSetJacFn");
            // TODO: if doing matrix free solver, call appropriate functions




            // ---- Nonlinear solver ----
            // TODO: do nothing right now -> this means Newton method is used -> good default
        }
        
        ~CVODE(){
            out_file.close();
            N_VDestroy(cvode_constraints);
            SUNLinSolFree(linear_solver);
            SUNMatDestroy(template_matrix);
            CVodeFree(&cvode_mem);
        }

        void solve(const sunrealtype tout, N_Vector yout, sunrealtype &t_reached){
            const int err_code = CVode(cvode_mem, tout, yout, &t_reached, CV_NORMAL);
            check_cvode_error(err_code, "CVODE::solve()", "CVode");
        }

        void save_solution(const sunrealtype time, N_Vector solution){
            if (outfile_initialized){
                out_file.open(filename,std::ios::app);
                out_file << time << ",";
                const auto size = N_VGetLength(solution);
                const auto data = N_VGetArrayPointer(solution);
                for (sunindextype i=0;i<size;++i){
                    out_file << data[i];
                    if (i<size-1){
                        out_file << ",";
                    }
                }
                out_file << "\n";
                out_file.close();
            }
        }

        void print_statistics(const SUNOutputFormat fmt = SUN_OUTPUTFORMAT_TABLE) {
            CVodePrintAllStats(cvode_mem, stdout, fmt);
        }

        void print_statistics(const std::string & filename, const SUNOutputFormat fmt = SUN_OUTPUTFORMAT_TABLE){
            FILE * fptr = fopen(filename.c_str(),"w");
            if (fptr != nullptr){
                CVodePrintAllStats(cvode_mem, fptr, fmt);
            }
            fclose(fptr);
        }

        void register_data_interpretation(const std::vector<std::string> & names){
            component_names = names;
        }

        void prepare_outfile(const std::string & output_filename){
            filename = output_filename;
            out_file = std::ofstream(filename, std::ios::out);
            out_file << "time,";
            for (sunindextype i=0;i<n_odes;++i){
                out_file << component_names[i];
                if (i<n_odes-1){
                    out_file << ",";
                }
            }
            out_file << "\n";
            out_file.close();
            outfile_initialized = true;
        }


        

        private:
        void* cvode_mem = nullptr;

        const RHSFcn rhs;
        const JacFcn jac;
        const CVODESettings settings;
        int (*cvode_rhs_fcn)(sunrealtype, N_Vector, N_Vector, void*);
        int (*cvode_jac_fcn)(sunrealtype, N_Vector, N_Vector, SUNMatrix, void*, N_Vector, N_Vector, N_Vector);
        CVODEUserData user_data;

        SUNMatrix template_matrix = nullptr;
        SUNLinearSolver linear_solver = nullptr;
        N_Vector cvode_constraints = nullptr;

        sunindextype n_odes = 0;
        std::vector<std::string> component_names;
        std::ofstream out_file;
        std::string filename;
        bool outfile_initialized = false;

    };


}

#endif // NANOPBM_CVODE_H