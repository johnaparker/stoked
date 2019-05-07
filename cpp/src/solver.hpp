#ifndef GUARD_solver_h
#define GUARD_solver_h

#include <complex>
#include <eigen3/Eigen/Core>
#include <vector>
#include "vec.hpp"

    class drag_t {
        public:
            drag_t(double viscosity): viscosity(viscosity) {};
            virtual Array drag_T() const = 0;
            virtual Array drag_R() const = 0;
        protected:
            double viscosity;
    };

    class drag_sphere: public drag_t {
        public:
            drag_sphere(const Ref<Array>& radius, double viscosity): drag_t(viscosity), radius(radius) {};
            Array drag_T() const override;
            Array drag_R() const override;
        private:
            Array radius;
    };

    class stokesian_dynamics {
        public:
            stokesian_dynamics(double temperature, double dt, Matrix position);
            void step(int Nsteps=1);
            void run_until(double final_time) {};
            const Matrix& get_position() const { return position; }
        private:
            Matrix get_velocity() const;
            void update_interactions();
        private:
            std::default_random_engine generator;
            std::normal_distribution<double> distribution;

            double temperature;
            double dt;
            Matrix position;
            //drag_t drag;

            Array alpha_T;
            int Nparticles;
            int ndim;
            int time = 0;
    };



#endif
