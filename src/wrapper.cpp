#include "isvd.hpp"
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(incremental_svd_lib, m) {
    py::class_<IncrementalSVD>(m, "IncrementalSVD")
        .def(py::init<int, float>(), py::arg("r"), py::arg("ff") = 1.0)
        .def("fit", &IncrementalSVD::fit)
        .def("increment", &IncrementalSVD::increment)
        .def_readonly("U", &IncrementalSVD::U)
        .def_readonly("S", &IncrementalSVD::S);
}
