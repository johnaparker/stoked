#include <pybind11/pybind11.h>

namespace py = pybind11;

// special submodule
void bind_drag_t(py::module &);
void bind_drag_sphere(py::module &);
void bind_brownian_dynamics(py::module &);

PYBIND11_MODULE(cpp, m) {
    m.doc() = R"pbdoc(
        C++ submodule of Pedesis
        -----------------------

        .. currentmodule:: cpp

        .. autosummary::
           :toctree: _generate
    )pbdoc";

    // special submodule
    py::module solver_m = m.def_submodule("solver", "Equation of motion solvers");

    //bind_drag_t(m);
    bind_drag_sphere(m);
    bind_brownian_dynamics(m);
}
