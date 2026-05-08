#include "isvd.hpp"
#include <pybind11/detail/common.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>

namespace py = pybind11;

template <typename scalar>
void declare_isvd(py::module &m, const std::string &typestr) {
  using Class = IncrementalSVD<scalar>;
  std::string pyclass_name = "IncrementalSVD_" + typestr;

  py::class_<Class>(m, pyclass_name.c_str())
      .def(py::init<int, scalar>(), py::arg("r"), py::arg("ff") = 1.0)
      .def("fit", &Class::fit)
      .def("increment", &Class::increment)
      .def_readonly("U", &Class::U)
      .def_readonly("S", &Class::S);
}

PYBIND11_MODULE(incremental_svd_lib, m) {
  declare_isvd<float>(m, "F");
  declare_isvd<double>(m, "D");
}
