#include <omp.h>
#include <yaml-cpp/yaml.h>

#include <cstddef>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "yaml-cpp/node/parse.h"

namespace PrmStudy {
struct ParameterSet {
  std::string key;
  std::vector<std::string> values;
};

void generate_yaml_files(
    const std::string& template_path, const std::vector<ParameterSet>& parameters, int param_index,
    std::vector<int>& current_indices, const std::string& output_prefix,
    std::vector<std::vector<std::pair<std::string, std::string>>>& settings_list, int* file_num) {
  if (param_index == parameters.size()) {
    // Load template
    YAML::Node config = YAML::LoadFile(template_path);

    // Apply all parameter values
    const std::string filename =
        output_prefix + "input_deck_" + std::to_string(*file_num) + ".yaml";

    std::vector<std::pair<std::string, std::string>> current_settings;
    current_settings.push_back({"Filename", filename});
    current_settings.push_back({"Scenario", std::to_string(*file_num)});
    for (size_t i = 0; i < parameters.size(); i++) {
      const auto& param        = parameters[i];
      const std::string& value = param.values[current_indices[i]];

      // Set the value in the YAML (handles nested keys with dots)
      YAML::Node current = config;
      std::stringstream ss(param.key);
      std::string part;
      std::vector<std::string> keyParts;

      while (std::getline(ss, part, '.')) {
        keyParts.push_back(part);
      }

      for (size_t j = 0; j < keyParts.size() - 1; j++) {
        current = current[keyParts[j]];
      }
      current[keyParts.back()] = value;

      current_settings.push_back({param.key, value});
    }
    settings_list.push_back(current_settings);


    // Write the output file
    std::ofstream fout(filename);
    fout << config;
    fout.close();

    std::cout << "    Generated: " << filename << std::endl;
    ++(*file_num);
    return;
  }

  // Recursively iterate through all combinations
  for (size_t i = 0; i < parameters[param_index].values.size(); i++) {
    current_indices[param_index] = i;
    generate_yaml_files(template_path, parameters, param_index + 1, current_indices, output_prefix,
                        settings_list, file_num);
  }
}

std::string run_command_and_get_output(const std::string& command) {
  constexpr size_t BUFFER_SIZE = 4096;  // Larger buffer for robustness
  std::array<char, BUFFER_SIZE> buffer;
  std::string currentLine;
  std::string lastLine;

  // Open pipe to read command output
  // Using a lambda avoids decltype(&pclose) which can trigger compiler warnings
  auto pclose_deleter = [](FILE* fp) {
    if (fp) pclose(fp);
  };
  std::unique_ptr<FILE, decltype(pclose_deleter)> pipe(popen(command.c_str(), "r"), pclose_deleter);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }

  // Read output, handling lines of arbitrary length
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    currentLine += buffer.data();

    // Check if we've read a complete line (ends with newline)
    if (!currentLine.empty() && currentLine.back() == '\n') {
      currentLine.pop_back();  // Remove trailing newline
      if (!currentLine.empty()) {
        lastLine = currentLine;
      }
      currentLine.clear();
    }
  }

  // Handle case where last line doesn't end with newline
  if (!currentLine.empty()) {
    lastLine = currentLine;
  }

  return lastLine;
}
}  // namespace PrmStudy


int main(int argc, char** argv) {
  try {
    std::cout << "===================================================================" << std::endl;
    std::cout << "Setting up a parameter matrix study\n";
    std::cout << "===================================================================" << std::endl;
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
    YAML::Node config = YAML::LoadFile(input_filename);
    std::cout << "YAML read: " << input_filename << std::endl;

    // Get file information
    const std::string template_path = config["Template input path"].as<std::string>();
    std::cout << "  - Template YAML file: " << template_path << std::endl;
    const std::string executable_path = config["Executable path"].as<std::string>();
    std::cout << "  - Executable: " << executable_path << std::endl;
    const std::string output_directory = config["Output directory"].as<std::string>();
    std::cout << "  - Output directory: " << output_directory << std::endl;
    // FIXME: make the directory if it does not exist

    std::vector<PrmStudy::ParameterSet> parameters;
    const YAML::Node& params = config["parameters"];
    int total_combos         = 1;
    for (const auto& prm : params) {
      PrmStudy::ParameterSet ps;
      ps.key                   = prm["key"].as<std::string>();
      const YAML::Node& values = prm["values"];
      for (const auto& v : values) {
        ps.values.push_back(v.as<std::string>());
      }

      parameters.push_back(ps);
      std::cout << "  - Key: " << ps.key << " (" << ps.values.size() << " values)" << std::endl;
      total_combos *= ps.values.size();
    }
    std::cout << "Generating " << total_combos << " files..." << std::endl;

    // Generate all combination
    std::vector<int> current_indices(parameters.size(), 0);
    std::vector<std::vector<std::pair<std::string, std::string>>> settings_list;
    int file_num = 0;
    PrmStudy::generate_yaml_files(template_path, parameters, 0, current_indices, output_directory,
                                  settings_list, &file_num);

    std::cout << "done!" << std::endl;


    std::cout << "Performing parameter study:" << std::endl;
    const std::string summary_filename = output_directory + "summary.txt";
    {
      std::ofstream out(summary_filename, std::ios::out);
      for (int i = 1; i < settings_list.at(0).size(); ++i) {
        out << settings_list.at(0).at(i).first << ",";
      }
      out << "Result" << std::endl;
    }


    int finished = 0;
    omp_set_num_threads(config["Parallel jobs"].as<int>());
#pragma omp parallel for
    for (int i = 0; i < settings_list.size(); ++i) {
      const std::string command = executable_path + " " + settings_list.at(i).at(0).second;
      const auto exe_result     = PrmStudy::run_command_and_get_output(command);
#pragma omp critical
      {
        ++finished;
        std::ofstream out(summary_filename, std::ios::app);
        for (int j = 1; j < settings_list.at(i).size(); ++j) {
          out << settings_list.at(i).at(j).second << ",";
        }
        out << exe_result << std::endl;
        std::cout << "  " << finished << "/" << total_combos << " done." << std::endl;
      }
    }

    std::cout << "All iterations complete. Results written to: " << summary_filename << std::endl;

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