#ifndef NANOPBM_DATAOUT_H
#define NANOPBM_DATAOUT_H

#include <H5Cpp.h>
#include <H5public.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>

#include <cmath>
#include <format>
#include <stdexcept>
#include <string>
#include <vector>


namespace NanoPBM {
class DataOut {
 private:
  H5::H5File file;
  H5::DataSet dataset;
  hsize_t current_row;
  hsize_t num_cols;
  hsize_t chunk_size;

 public:
  // Constructor: creates HDF5 file and initializes structure
  DataOut(const std::string& filename, const hsize_t chunk_size = 100)
      : file(filename, H5F_ACC_TRUNC), current_row(0), num_cols(0), chunk_size(chunk_size) {}

  // Set column names and create the dataset
  void set_columns(const std::vector<std::string>& cols) {
    const bool first_col_is_time = (cols[0] == "time");
    std::vector<std::string> tmp_cols;
    tmp_cols.reserve(cols.size() + 1);
    if (!first_col_is_time) {
      tmp_cols.push_back("time");
    }
    tmp_cols.insert(tmp_cols.end(), cols.begin(), cols.end());
    num_cols = tmp_cols.size();


    // Create dataspace with unlimited rows and fixed columns
    hsize_t dims[2]       = {0, num_cols};
    hsize_t max_dims[2]   = {H5S_UNLIMITED, num_cols};
    hsize_t chunk_dims[2] = {chunk_size, num_cols};  // Chunk size for performance

    H5::DataSpace dataspace(2, dims, max_dims);

    // Set chunking and compression properties
    H5::DSetCreatPropList props;
    props.setChunk(2, chunk_dims);
    props.setDeflate(6);  // Compression level 6

    // Create dataset
    dataset = file.createDataSet("data", H5::PredType::NATIVE_DOUBLE, dataspace, props);

    // Store column names as attributes
    for (size_t i = 0; i < num_cols; ++i) {
      H5::DataSpace attr_space(H5S_SCALAR);
      H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
      //   std::string attr_name = "col_" + std::to_string(i);
      int pad_width         = std::log10(10. * num_cols);
      std::string attr_name = "col_" + std::format("{:0>{}}", i, pad_width);
      H5::Attribute attr    = dataset.createAttribute(attr_name, str_type, attr_space);
      const auto cstr       = tmp_cols[i].c_str();
      attr.write(str_type, &cstr);
    }
  }

  // Write a row: first value is a double, rest come from array
  void write(double time, N_Vector vector) {
    if (num_cols == 0) {
      throw std::runtime_error("Column names must be set before writing data");
    }

    auto data = N_VGetArrayPointer(vector);

    // Check that sizes are correct
    const auto vec_size = N_VGetLength(vector);
    if (num_cols != vec_size + 1) {
      const std::string border =
          "----------------------------------------------------------------------------\n";
      std::string err_msg = border;
      err_msg += "[class = DataOut] [method = write()]\n";
      err_msg +=
          "  Error: The number of columns in the output data set does not match the N_Vector.\n";
      err_msg +=
          "         The output data should have exactly 1 more column than the N_Vector has "
          "entries.\n";
      err_msg += "         The extra column represents time.\n";
      err_msg += "  Additional information:\n";
      err_msg += "  -----------------------\n";
      err_msg += "  Number of columns in dataset: " + std::to_string(num_cols) + "\n";
      err_msg += "  Size of N_Vector:             " + std::to_string(vec_size) + "\n";
      err_msg += border;
      throw std::runtime_error(err_msg);
      return;
    }

    // Prepare row data
    std::vector<double> row_data(num_cols);
    row_data[0] = time;
    for (sunindextype idx = 0; idx < vec_size; ++idx) {
      row_data[idx + 1] = data[idx];
    }

    // Extend dataset by one row
    hsize_t newDims[2] = {current_row + 1, num_cols};
    dataset.extend(newDims);

    // Select the hyperslab for the new row
    H5::DataSpace filespace = dataset.getSpace();
    hsize_t offset[2]       = {current_row, 0};
    hsize_t count[2]        = {1, num_cols};
    filespace.selectHyperslab(H5S_SELECT_SET, count, offset);

    // Define memory dataspace
    H5::DataSpace memspace(2, count);

    // Write the row
    dataset.write(row_data.data(), H5::PredType::NATIVE_DOUBLE, memspace, filespace);

    current_row++;
  }

  // Destructor: closes file
  ~DataOut() {
    dataset.close();
    file.close();
  }
};
}  // namespace NanoPBM

#endif  // NANOPBM_DATAOUT_H