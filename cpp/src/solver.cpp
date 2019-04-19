#include "solver.hpp"
#include <random>

    Array drag_sphere::drag_T() const {
        int Nparticles = radius.size();
        Array drag(Nparticles);
        for (int i = 0; i < Nparticles; i++) {
            drag[i] = 6*PI*viscosity*radius[i];
        }

        return drag;
    }

    Array drag_sphere::drag_R() const {
        int Nparticles = radius.size();
        Array drag(Nparticles);
        for (int i = 0; i < Nparticles; i++) {
            drag[i] = 8*PI*viscosity*pow(radius[i],3);
        }

        return drag;
    }

    brownian_dynamics::brownian_dynamics(double temperature, double dt, Matrix position):
        temperature(temperature), dt(dt), position(position) {
            Nparticles = position.rows();
            ndim = position.cols();
            alpha_T = Array::Ones(Nparticles);

            distribution = std::normal_distribution<double>(0.0, 1.0);

    }

    void brownian_dynamics::step(int Nsteps) {
        for (int i = 0; i < Nsteps; i++) {
            Matrix Rn(position.rows(), position.cols());
            for (int n = 0; n < Nparticles; n++) {
                for (int d = 0; d < ndim; d++) {
                    Rn(n,d) = distribution(generator);
                    position(n,d) += sqrt(2*KB*temperature*alpha_T(n)*dt)*Rn(n,d);
                    
                }
            }
        }
    }
