#include <nvector/nvector_openmp.h>
#include <nvector/nvector_serial.h>
#include <sundials/sundials_matrix.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>
#include <sunmatrix/sunmatrix_dense.h>

#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sundials/sundials_context.hpp>
#include <vector>

#include "benchmark_statistics.h"
#include "cvode.h"
#include "data_out.h"
#include "emom.h"
#include "input_parameters.h"
#include "nanopbm/chemical_reaction.h"
#include "ode_contribution.h"
#include "particle_agglomeration.h"
#include "particle_growth.h"


namespace Iridium {
using namespace NanoPBM;
sunrealtype iridium_diameter(const sunrealtype n_atoms) { return 0.3 * std::pow(n_atoms, 1. / 3.); }
sunrealtype iridium_radius(const sunrealtype n_atoms) { return iridium_diameter(n_atoms) / 2; }
sunrealtype surf_area_to_volume(const sunrealtype n_atoms) {
  return 2.677 * std::pow(n_atoms, 2. / 3.);
}

class SourceTerm : public OdeContribution {
 public:
  SourceTerm() = delete;
  SourceTerm(const sunindextype index, const sunrealtype value = 0) : index(index), value(value) {}

  void add_to_rhs(const sunrealtype t, const N_Vector y, N_Vector ydot) const override {
    auto data = N_VGetArrayPointer(ydot);
    data[index] += value;
  }


 private:
  const sunindextype index;
  const sunrealtype value;
};


// TODO: make this part of the library
struct RxnRate {
  RxnRate(const sunrealtype kf, const sunrealtype kb) : kf(kf), kb(kb) {}
  const sunrealtype kf, kb;
  sunrealtype forward(const N_Vector y, const sunrealtype t, const ReactionParameterSet& reactants,
                      const ReactionParameterSet& products) const {
    sunrealtype rate  = kf;
    const auto y_data = N_VGetArrayPointer(y);
    for (const auto& prm : reactants) {
      rate *= std::pow(y_data[prm.index], prm.order);
    }
    return rate;
  }

  sunrealtype backward(const N_Vector y, const sunrealtype t, const ReactionParameterSet& reactants,
                       const ReactionParameterSet& products) const {
    sunrealtype rate  = kb;
    const auto y_data = N_VGetArrayPointer(y);
    for (const auto& prm : products) {
      rate *= std::pow(y_data[prm.index], prm.order);
    }
    return rate;
  }

  sunrealtype df_dr(const ReactionParameters& deriv_prm, const N_Vector y, const sunrealtype t,
                    const ReactionParameterSet& reactants,
                    const ReactionParameterSet& products) const {
    sunrealtype rate  = kf;
    const auto y_data = N_VGetArrayPointer(y);
    bool found        = false;
    for (const auto& prm : reactants) {
      if (prm.index == deriv_prm.index) {
        rate *= prm.order * std::pow(y_data[prm.index], prm.order - 1);
        found = true;
      } else {
        rate *= std::pow(y_data[prm.index], prm.order);
      }
    }
    return rate * found;
  }

  sunrealtype db_dp(const ReactionParameters& deriv_prm, const N_Vector y, const sunrealtype t,
                    const ReactionParameterSet& reactants,
                    const ReactionParameterSet& products) const {
    sunrealtype rate  = kb;
    const auto y_data = N_VGetArrayPointer(y);
    bool found        = false;
    for (const auto& prm : products) {
      if (prm.index == deriv_prm.index) {
        rate *= prm.order * std::pow(y_data[prm.index], prm.order - 1);
        found = true;
      } else {
        rate *= std::pow(y_data[prm.index], prm.order);
      }
    }
    return rate * found;
  }
};
}  // namespace Iridium

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      const std::string border =
          "------------------------------------------------------------------------------\n";
      std::string err_msg = border;
      err_msg += "Error: No input file specified!\n";
      err_msg += "Usage: " + std::string(argv[0]) + " /path/to/input_file.yaml\n";
      err_msg += "  Tip: Check the folder 'example_input_files' for an example.\n";
      err_msg += border;
      throw std::runtime_error(err_msg);
    }

    const std::string input_filename(argv[1]);
    // TODO: print some info
    NanoPBM::InputReader input(input_filename);

    int n_benchmark_runs = 1;
    try {
      n_benchmark_runs = input.get<int>("Number of runs");
    } catch (...) {
      // some message
      n_benchmark_runs = 1;
    }


    sundials::Context sunctx;


    // ---- Keep track of settings and indices ----
    const bool include_agglom = (input.get<std::string>("Mechanism") == "4step");
    const bool include_emom   = (input.get<std::string>("Use eMoM") == "yes");


    const sunindextype max_particle_size = input.get<sunindextype>("Maximum number of atoms");
    const sunindextype max_agglomeration_size =
        input.get<sunindextype>("Maximum small particle size");
    const sunindextype nucleated_particle_size = 3;

    // Also track "A", "L", and "As"
    const sunindextype n_nonparticles_tracked =
        include_agglom ? 6 : 3;  // FIXME: should be different if emom used
    const sunindextype P3_idx = n_nonparticles_tracked;
    const sunindextype n_odes =
        (max_particle_size - nucleated_particle_size + 1) + n_nonparticles_tracked;

    constexpr sunindextype A_idx  = 0;
    constexpr sunindextype L_idx  = 1;
    constexpr sunindextype As_idx = 2;
    // TODO: consider putting emom before A_idx or right after
    constexpr sunindextype mu0_idx       = 3;  // only used if emom selected
    constexpr sunindextype mu1_idx       = 4;  // only used if emom selected
    constexpr sunindextype mu2_idx       = 5;  // only used if emom selected
    const sunindextype last_particle_idx = n_odes - 1;


    // ---- Set up the solution vector ----
    N_Vector y0             = nullptr;
    const std::string vtype = input.get<std::string>("N Vector type");
    if (vtype == "serial") {
      y0 = N_VNew_Serial(n_odes, sunctx);
    } else if (vtype == "openmp") {
      const int n_threads = input.get<int>("Number of threads");
      y0                  = N_VNew_OpenMP(n_odes, n_threads, sunctx);
    }


    // ---- Give names to all solution vector components ----
    std::vector<std::string> data_names(n_odes);
    data_names[A_idx]  = "Precursor";
    data_names[L_idx]  = "Ligand";
    data_names[As_idx] = "Solvated complex";
    if (include_emom) {
      data_names[mu0_idx] = "Moment 0";
      data_names[mu1_idx] = "Moment 1";
      data_names[mu2_idx] = "Moment 2";
    }
    // FIXME: emom names
    for (sunindextype idx = n_nonparticles_tracked; idx < n_odes; ++idx) {
      const sunindextype n_atoms = idx - n_nonparticles_tracked + nucleated_particle_size;
      data_names[idx]            = "P" + std::to_string(n_atoms);
    }


    // ---- Add contributions to the ODE system ----

    NanoPBM::ManyOdeContributions ode_model;

    // Optional source term
    const sunrealtype src_term = input.get<sunrealtype>("Precursor source");
    Iridium::SourceTerm src(A_idx, src_term);
    ode_model.add_contribution(src);


    // Reaction rate parameters
    const sunrealtype S  = input.get<sunrealtype>("Solvent concentration");
    const sunrealtype kb = input.get<sunrealtype>("kb");
    const sunrealtype kf = kb * input.get<sunrealtype>("kf to kb ratio");
    const sunrealtype k1 = input.get<sunrealtype>("k1");
    const sunrealtype k2 = input.get<sunrealtype>("k2");
    const sunrealtype k3 = input.get<sunrealtype>("k3");

    // ---- Reaction ----
    {
      //    A <=> As + L
      Iridium::RxnRate rxn_rate(S * S * kf, kb);
      NanoPBM::ReactionParameters prm_rxn1_A(A_idx, 1, 1);
      NanoPBM::ReactionParameters prm_rxn1_As(As_idx, 1, 1);
      NanoPBM::ReactionParameters prm_rxn1_L(L_idx, 1, 1);

      NanoPBM::ReactionParameterSet rxn1_reactants;
      rxn1_reactants->push_back(prm_rxn1_A);

      NanoPBM::ReactionParameterSet rxn1_products;
      rxn1_products->push_back(prm_rxn1_As);
      rxn1_products->push_back(prm_rxn1_L);

      NanoPBM::ChemicalReaction rxn1(rxn_rate, rxn1_reactants, rxn1_products);
      ode_model.add_contribution(rxn1);
    }

    // ---- Nucleation ----
    {
      // 2As + A -> P3 + L
      Iridium::RxnRate nuc_rate(k1, 0);
      NanoPBM::ReactionParameters prm_nuc_As(As_idx, 2, 2);
      NanoPBM::ReactionParameters prm_nuc_A(A_idx);
      NanoPBM::ReactionParameters prm_nuc_p3(P3_idx);
      NanoPBM::ReactionParameters prm_nuc_L(L_idx);

      NanoPBM::ReactionParameterSet nuc_reactants;
      nuc_reactants->push_back(prm_nuc_As);
      nuc_reactants->push_back(prm_nuc_A);

      NanoPBM::ReactionParameterSet nuc_products;
      nuc_products->push_back(prm_nuc_p3);
      nuc_products->push_back(prm_nuc_L);

      NanoPBM::ChemicalReaction nuc(nuc_rate, nuc_reactants, nuc_products);
      ode_model.add_contribution(nuc);
    }


    // Growth due to precursor
    const auto atoms_to_size = [](const sunindextype atoms) {
      return Iridium::surf_area_to_volume(atoms);
    };
    NanoPBM::ConstantTimesSizeGrowthKernel small_growth_kernel(atoms_to_size, k2);
    NanoPBM::ConstantTimesSizeGrowthKernel large_growth_kernel(atoms_to_size, k3);

    NanoPBM::ParticleGrowth small_growth_mechanism(A_idx, {{L_idx, 1}}, nucleated_particle_size,
                                                   max_agglomeration_size, P3_idx,
                                                   small_growth_kernel);
    ode_model.add_contribution(small_growth_mechanism);

    const sunindextype smallest_large_particle_idx =
        max_agglomeration_size + 1 - nucleated_particle_size + P3_idx;
    NanoPBM::ParticleGrowth large_growth_mechanism(
        A_idx, {{L_idx, 1}}, max_agglomeration_size + 1, max_particle_size - 1,
        smallest_large_particle_idx, large_growth_kernel);
    ode_model.add_contribution(large_growth_mechanism);


    // Agglomeration with other particles
    if (include_agglom) {
      const sunrealtype k4     = input.get<sunrealtype>("k4");
      const auto agglom_kernel = [=](const sunindextype size1, const sunindextype size2) {
        return k4 * Iridium::surf_area_to_volume(size1) * Iridium::surf_area_to_volume(size2);
      };

      NanoPBM::ParticleAgglomeration agglomeration_mechanism(
          nucleated_particle_size, max_agglomeration_size, P3_idx, agglom_kernel);
      ode_model.add_contribution(agglomeration_mechanism);
    }

    // Exact method of moments
    if (include_emom) {
      const sunrealtype Vm = 4. / 3. * std::numbers::pi * std::pow(Iridium::iridium_radius(1), 3);
      const sunrealtype xN = std::pow(6. / std::numbers::pi * Vm * max_particle_size, 1. / 3.);
      NanoPBM::EMoM emom(mu0_idx, mu1_idx, mu2_idx, A_idx, last_particle_idx,
                         k3 /*use same growth rate as large*/, Vm, xN, max_particle_size,
                         {{L_idx, 1}});
      ode_model.add_contribution(emom);
    }


    // Make ODE solver
    NanoPBM::CVODESettings settings;
    settings.reltol               = 1.e-8;
    settings.abstol               = 1.e-14;
    settings.linear_solver        = input.get<std::string>("Linear solver type");
    settings.preconditioner_type  = input.get<std::string>("Preconditioner type");
    settings.prec_upper_bandwidth = input.get<int>("Preconditioner upper bandwidth");
    settings.prec_lower_bandwidth = input.get<int>("Preconditioner lower bandwidth");

    // First solve is without timing and does the data output
    const std::string do_data_output_run = input.get<std::string>("Output data");
    if (do_data_output_run == "yes") {
      // Initial conditions
      N_VConst(0, y0);
      {
        auto y0_data    = N_VGetArrayPointer(y0);
        y0_data[A_idx]  = input.get<sunrealtype>("Precursor initial condition");
        y0_data[L_idx]  = input.get<sunrealtype>("Ligand initial condition");
        y0_data[As_idx] = input.get<sunrealtype>("Solvated complex initial condition");
      }

      NanoPBM::CVODE ode_solver(sunctx, y0, ode_model, settings);
      // Construct object to handle saving data
      const std::string filename = input.get<std::string>("Output file name");
      NanoPBM::DataOut data_out(filename);
      data_out.set_columns(data_names);

      sunrealtype time = 0;
      auto start       = std::chrono::steady_clock::now();
      data_out.write(time, y0);

      // Solve ODE at specified times
      const sunrealtype dt            = input.get<sunrealtype>("Time step size");
      const sunindextype n_time_steps = input.get<sunindextype>("Number of time steps") + 1;
      for (int step = 1; step < n_time_steps; ++step) {
        sunrealtype tout = step * dt;
        ode_solver.solve(tout, y0, time);

        data_out.write(time, y0);
      }
      auto end                              = std::chrono::steady_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      std::cout << "-----------------------------------------------------------------" << std::endl;
      ode_solver.print_statistics();
      std::cout << "-----------------------------------------------------------------" << std::endl;
      std::cout << "Data written to: " << input.get<std::string>("Output file name") << std::endl;
      std::cout << "ODE solve time + data output time: " << elapsed.count() << std::endl;
    }

    NanoPBM::BenchmarkStatistics benchmark;
    for (int run = 0; run < n_benchmark_runs; ++run) {
      // Initial conditions
      N_VConst(0, y0);
      {
        auto y0_data    = N_VGetArrayPointer(y0);
        y0_data[A_idx]  = input.get<sunrealtype>("Precursor initial condition");
        y0_data[L_idx]  = input.get<sunrealtype>("Ligand initial condition");
        y0_data[As_idx] = input.get<sunrealtype>("Solvated complex initial condition");
      }

      NanoPBM::CVODE ode_solver(sunctx, y0, ode_model, settings);
      sunrealtype time = 0;
      // Solve ODE at specified times
      const sunrealtype dt            = input.get<sunrealtype>("Time step size");
      const sunindextype n_time_steps = input.get<sunindextype>("Number of time steps") + 1;
      auto start                      = std::chrono::steady_clock::now();
      for (int step = 1; step < n_time_steps; ++step) {
        sunrealtype tout = step * dt;
        ode_solver.solve(tout, y0, time);
      }
      auto end                              = std::chrono::steady_clock::now();
      std::chrono::duration<double> elapsed = end - start;
      benchmark.push(elapsed.count());
    }
    N_VDestroy(y0);


    const int n_runs = benchmark.count();
    if (n_runs > 0) {
      std::cout << "ODE solve time: " << benchmark.mean() << " seconds" << std::endl;
      if (n_runs > 2) {
        std::cout << "  Average of " << n_runs << " solves.\n"
                  << "  Standard deviation: " << benchmark.standard_deviation() << std::endl;
      }
      std::cout << std::endl << benchmark.mean() << std::endl;
    }


  } catch (std::exception& exc) {
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