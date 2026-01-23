#ifndef NANOPBM_INPUT_PARAMETERS_H
#define NANOPBM_INPUT_PARAMETERS_H

#include <yaml-cpp/yaml.h>

#include "yaml-cpp/node/parse.h"

namespace NanoPBM {
class InputReader {
 public:
  InputReader() = delete;
  InputReader(const std::string& filename) { input_file = YAML::LoadFile(filename); }

  template <typename T>
  T get(const std::string& key) const {
    return input_file[key].as<T>();
  }


 private:
  YAML::Node input_file;
};
}  // namespace NanoPBM

#endif  // NANOPBM_INPUT_PARAMETERS_H