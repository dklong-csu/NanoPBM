#include "nanopbm/ode_contribution.h"

#include <nvector/nvector_serial.h>
#include <sundials/sundials_nvector.h>

#include <boost/ut.hpp>
#include <cmath>
#include <ginkgo/core/base/matrix_data.hpp>
#include <sundials/sundials_context.hpp>
#include <vector>


namespace testing {
class FakeODE1 : public NanoPBM::OdeContribution {
 public:
  FakeODE1(const sunrealtype rate) : rate(rate) {}

  void add_to_rhs(const sunrealtype t, const N_Vector y, N_Vector ydot) const override {
    const auto y_data = N_VGetArrayPointer(y);
    auto rhs_data     = N_VGetArrayPointer(ydot);
    rhs_data[0] -= rate * y_data[0];
    rhs_data[1] += rate * y_data[0];
  }

  void add_to_jac_times_v(const N_Vector v, N_Vector Jv, const sunrealtype t, const N_Vector y,
                          const N_Vector ydot, N_Vector tmp) const override {
    auto Jv_data      = N_VGetArrayPointer(Jv);
    const auto v_data = N_VGetArrayPointer(v);
    Jv_data[0] -= rate * v_data[0];
    Jv_data[1] += rate * v_data[0];
  }


  void add_to_jac(gko::matrix_data<sunrealtype, sunindextype>& sparsity_pattern,
                  const sunrealtype t, const N_Vector y, const N_Vector fy, N_Vector tmp1,
                  N_Vector tmp2, N_Vector tmp3) const override {
    sparsity_pattern.nonzeros.emplace_back(0, 0, -rate);
    sparsity_pattern.nonzeros.emplace_back(1, 0, rate);
  }

 private:
  const sunrealtype rate;
};


class FakeODE2 : public NanoPBM::OdeContribution {
 public:
  FakeODE2(const sunrealtype rate1, const sunrealtype rate2) : rate1(rate1), rate2(rate2) {}

  void add_to_rhs(const sunrealtype t, const N_Vector y, N_Vector ydot) const override {
    const auto y_data = N_VGetArrayPointer(y);
    auto rhs_data     = N_VGetArrayPointer(ydot);
    rhs_data[0] += rate2 * y_data[1] - rate1 * y_data[0];
    rhs_data[1] += rate1 * y_data[0] - rate2 * y_data[1];
  }

  void add_to_jac_times_v(const N_Vector v, N_Vector Jv, const sunrealtype t, const N_Vector y,
                          const N_Vector ydot, N_Vector tmp) const override {
    auto Jv_data      = N_VGetArrayPointer(Jv);
    const auto v_data = N_VGetArrayPointer(v);
    Jv_data[0] += rate2 * v_data[1] - rate1 * v_data[0];
    Jv_data[1] += rate1 * v_data[0] - rate2 * v_data[1];
  }


  void add_to_jac(gko::matrix_data<sunrealtype, sunindextype>& sparsity_pattern,
                  const sunrealtype t, const N_Vector y, const N_Vector fy, N_Vector tmp1,
                  N_Vector tmp2, N_Vector tmp3) const override {
    sparsity_pattern.nonzeros.emplace_back(0, 0, -rate1);
    sparsity_pattern.nonzeros.emplace_back(0, 1, rate2);
    sparsity_pattern.nonzeros.emplace_back(1, 0, rate1);
    sparsity_pattern.nonzeros.emplace_back(1, 1, -rate2);
  }

 private:
  const sunrealtype rate1;
  const sunrealtype rate2;
};

}  // namespace testing


int main() {
  using namespace boost::ut;
  using namespace testing;
  "Many ODEs"_test = [] {
    "ODE 1 rate"_test = [](const auto& ode1_rate) {
      "ODE 2 rate 1"_test = [&](const auto& ode2_rate1) {
        "ODE 2 rate 2"_test = [&](const auto& ode2_rate2) {
          sundials::Context sunctx;
          N_Vector y    = N_VNew_Serial(2, sunctx);
          N_Vector ydot = N_VNew_Serial(2, sunctx);
          N_Vector v    = N_VNew_Serial(2, sunctx);
          N_Vector Jv   = N_VNew_Serial(2, sunctx);
          N_Vector fy   = N_VNew_Serial(2, sunctx);
          N_Vector tmp1 = N_VNew_Serial(2, sunctx);
          N_Vector tmp2 = N_VNew_Serial(2, sunctx);
          N_Vector tmp3 = N_VNew_Serial(2, sunctx);

          N_VConst(0, y);
          N_VConst(0, ydot);
          N_VConst(0, v);
          N_VConst(0, Jv);
          N_VConst(0, fy);
          N_VConst(0, tmp1);
          N_VConst(0, tmp2);
          N_VConst(0, tmp3);

          std::vector<N_Vector> all_n_vecs    = {y, ydot, v, Jv, fy, tmp1, tmp2, tmp3};
          std::vector<N_Vector> n_vec_is_zero = {fy, tmp1, tmp2, tmp3};

          for (const auto& vec : all_n_vecs) {
            const auto data = N_VGetArrayPointer(vec);
            expect(data[0] == 0._d);
            expect(data[1] == 0._d);
          }

          FakeODE1 ode1(ode1_rate);
          FakeODE2 ode2(ode2_rate1, ode2_rate2);
          NanoPBM::ManyOdeContributions ode_system;
          ode_system.add_contribution(ode1);
          ode_system.add_contribution(ode2);

          auto y_data = N_VGetArrayPointer(y);
          y_data[0]   = 1;
          y_data[1]   = 1;

          auto v_data = N_VGetArrayPointer(v);
          v_data[0]   = 1;
          v_data[1]   = 1;

          auto ydot_data = N_VGetArrayPointer(ydot);
          auto Jv_data   = N_VGetArrayPointer(Jv);

          // ---- Check f(y) ----
          ode_system.add_to_rhs(0, y, ydot);

          expect(ydot_data[0] == _d(ode2_rate2 - ode2_rate1 - ode1_rate));
          expect(ydot_data[1] == _d(-ode2_rate2 + ode2_rate1 + ode1_rate));

          expect(y_data[0] == 1._d);
          expect(y_data[1] == 1._d);

          for (const auto& vec : n_vec_is_zero) {
            const auto data = N_VGetArrayPointer(vec);
            expect(data[0] == 0._d);
            expect(data[1] == 0._d);
          }

          // ---- Check Jv product ----
          ode_system.add_to_jac_times_v(v, Jv, 0, y, ydot, tmp1);

          expect(Jv_data[0] == _d(-ode1_rate - ode2_rate1 + ode2_rate2));
          expect(Jv_data[1] == _d(ode1_rate + ode2_rate1 - ode2_rate2));

          expect(ydot_data[0] == _d(ode2_rate2 - ode2_rate1 - ode1_rate));
          expect(ydot_data[1] == _d(-ode2_rate2 + ode2_rate1 + ode1_rate));

          expect(y_data[0] == 1._d);
          expect(y_data[1] == 1._d);

          expect(v_data[0] == 1._d);
          expect(v_data[1] == 1._d);

          for (const auto& vec : n_vec_is_zero) {
            const auto data = N_VGetArrayPointer(vec);
            expect(data[0] == 0._d);
            expect(data[1] == 0._d);
          }

          // ---- Check forming J ----
          gko::matrix_data<sunrealtype, sunindextype> jac_data;
          ode_system.add_to_jac(jac_data, 0, y, fy, tmp1, tmp2, tmp3);

          sunrealtype J00 = 0;
          sunrealtype J01 = 0;
          sunrealtype J10 = 0;
          sunrealtype J11 = 0;

          for (const auto& d : jac_data.nonzeros) {
            const auto c = d.column;
            const auto r = d.row;
            if (r == 0 && c == 0) {
              J00 += d.value;
            } else if (r == 0 && c == 1) {
              J01 += d.value;
            } else if (r == 1 && c == 0) {
              J10 += d.value;
            } else if (r == 1 && c == 1) {
              J11 += d.value;
            } else {
              expect(false) << "The Jacobain should be 2x2, but contains an element outside those "
                               "dimensions.\n"
                            << "Row: " << r << "\n"
                            << "Column: " << c << "\n"
                            << "Value: " << d.value << "\n";
            }
          }

          expect(J00 == _d(-ode1_rate - ode2_rate1));
          expect(J01 == _d(ode2_rate2));
          expect(J10 == _d(ode1_rate + ode2_rate1));
          expect(J11 == _d(-ode2_rate2));

          expect(Jv_data[0] == _d(-ode1_rate - ode2_rate1 + ode2_rate2));
          expect(Jv_data[1] == _d(ode1_rate + ode2_rate1 - ode2_rate2));

          expect(ydot_data[0] == _d(ode2_rate2 - ode2_rate1 - ode1_rate));
          expect(ydot_data[1] == _d(-ode2_rate2 + ode2_rate1 + ode1_rate));

          expect(y_data[0] == 1._d);
          expect(y_data[1] == 1._d);

          expect(v_data[0] == 1._d);
          expect(v_data[1] == 1._d);

          for (const auto& vec : n_vec_is_zero) {
            const auto data = N_VGetArrayPointer(vec);
            expect(data[0] == 0._d);
            expect(data[1] == 0._d);
          }


          N_VDestroy(tmp3);
          N_VDestroy(tmp2);
          N_VDestroy(tmp1);
          N_VDestroy(fy);
          N_VDestroy(Jv);
          N_VDestroy(v);
          N_VDestroy(ydot);
          N_VDestroy(y);
        } | std::vector<sunrealtype>{0, 1, 2, 3};
      } | std::vector<sunrealtype>{0, 1, 2, 3};
    } | std::vector<sunrealtype>{0, 1, 2, 3};
  };
}