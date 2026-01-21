#include "nanopbm/cvode.h"

#include <nvector/nvector_serial.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>

#include <boost/ut.hpp>
#include <cmath>
#include <cstdlib>
#include <sundials/sundials_context.hpp>

#include "nanopbm/ode_contribution.h"

namespace testing {
class FakeODE : public NanoPBM::OdeContribution {
 public:
  void add_to_rhs(const sunrealtype t, const N_Vector y, N_Vector ydot) const override {
    const auto y_data = N_VGetArrayPointer(y);
    auto rhs_data     = N_VGetArrayPointer(ydot);
    rhs_data[0] -= y_data[0];
    rhs_data[1] += y_data[0];
  }

  void add_to_jac_times_v(const N_Vector v, N_Vector Jv, const sunrealtype t, const N_Vector y,
                          const N_Vector ydot, N_Vector tmp) const override {
    auto Jv_data      = N_VGetArrayPointer(Jv);
    const auto v_data = N_VGetArrayPointer(v);
    Jv_data[0] -= v_data[0];
    Jv_data[1] += v_data[0];
  }


  void add_to_jac(gko::matrix_data<sunrealtype, sunindextype>& sparsity_pattern,
                  const sunrealtype t, const N_Vector y, const N_Vector fy, N_Vector tmp1,
                  N_Vector tmp2, N_Vector tmp3) const override {
    sparsity_pattern.nonzeros.emplace_back(0, 0, -1.);
    sparsity_pattern.nonzeros.emplace_back(1, 0, 1.);
  }

 private:
};

void make_ic(N_Vector v) {
  auto vd = N_VGetArrayPointer(v);
  vd[0]   = 1;
  vd[1]   = 0;
}

void check_solution(const double time, const N_Vector sol, const std::string& solver,
                    const std::string& prec_type) {
  using namespace boost::ut;
  const auto data          = N_VGetArrayPointer(sol);
  const sunrealtype exact0 = std::exp(-time);
  const sunrealtype exact1 = 1. - exact0;

  const sunrealtype diff0 = std::abs(data[0] - exact0);
  const sunrealtype diff1 = std::abs(data[1] - exact1);
  expect(diff0 <= 0.000001_d) << "\n  [ Solver =" << solver
                              << "] [ Preconditioner type=" << prec_type << "]"
                              << "\n  [0] Exact: " << exact0 << " | Result: " << data[0];
  expect(diff1 <= 0.000001_d) << "\n  [ Solver =" << solver
                              << "] [ Preconditioner type=" << prec_type << "]"
                              << "\n  [1] Exact: " << exact1 << " | Result: " << data[1];
  ;
}

void do_test(const std::string& solver, const std::string& prec_type = "none") {
  using namespace NanoPBM;

  try {
    sundials::Context sunctx;
    const sunindextype n_odes = 2;
    N_Vector y0               = N_VNew_Serial(n_odes, sunctx);
    const testing::FakeODE ode_model;

    CVODESettings settings;
    settings.linear_solver       = solver;
    settings.preconditioner_type = prec_type;
    settings.reltol              = 1.e-8;
    settings.abstol              = 1.e-14;

    testing::make_ic(y0);
    CVODE ode_solver(sunctx, y0, ode_model, settings);
    sunrealtype time     = 0;
    const sunrealtype dt = 0.1;
    for (int i = 0; i < 5; ++i) {
      const auto tout = (i + 1) * dt;
      ode_solver.solve(tout, y0, time);
      testing::check_solution(tout, y0, solver, prec_type);
    }

    N_VDestroy(y0);
  } catch (std::exception& exc) {
    boost::ut::expect(false) << "\n  [ Solver =" << solver
                             << "] [ Preconditioner type=" << prec_type << "]"
                             << "Exception encountered! Test failed!\n"
                             << exc.what();
  } catch (...) {
    boost::ut::expect(false) << "\n  [ Solver =" << solver
                             << "] [ Preconditioner type=" << prec_type << "]"
                             << "Exception encountered! Test failed!\n"
                             << "Unknown exception!";
  }
};
}  // namespace testing

int main() {
  using namespace boost::ut;
  using namespace testing;


  "CVODE"_test = [&] {
    "Direct linear solver"_test = [&] {
      do_test("direct");
      do_test("DIRECT");
      do_test("dIrEct");
    };
    "Iterative linear solver"_test = [&](const auto& solver) {
      "Preconditioner"_test = [&](const auto& prec_type) {
        do_test(solver, prec_type);
      } | std::vector<std::string>{"none", "NONE", "right", "RIGHT", "left", "LEFT"};
    } | std::vector<std::string>{"gmres", "GMRES", "bicgstab", "BICGSTAB", "tfqmr", "TFQMR"};
  };
}