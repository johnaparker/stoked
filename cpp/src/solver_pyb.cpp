#include "solver.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace pybind11::literals;

    //void bind_drag_t(py::module &m) {
        //py::class_<drag_t>(m, "drag_t")
            //.def(py::init<double>())
            //.def("drag_T", &drag_t::drag_T)
            //.def("drag_R", &drag_t::drag_R);
    //}

    void bind_drag_sphere(py::module &m) {
        py::class_<drag_sphere>(m, "drag_sphere")
            .def(py::init<const Ref<Array>&, double>())
            .def("drag_T", &drag_sphere::drag_T)
            .def("drag_R", &drag_sphere::drag_R);
    }

    void bind_brownian_dynamics(py::module &m) {
        py::class_<brownian_dynamics>(m, "brownian_dynamics")
            .def(py::init<double, double, Matrix>())
            .def("step", &brownian_dynamics::step, "Nsteps"_a=1)
            .def_property_readonly("position", &brownian_dynamics::get_position);
    }
