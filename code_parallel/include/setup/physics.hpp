#ifndef PHYSICS_HPP
#define PHYSICS_HPP

#include "core/config.hpp"
#include "setup/fluid.hpp"

class physics {
public:
	physics();
	void get_physical_fluxes(const fluid_cell &fluid, fluxes_cell &fluxes, const parallelisation::direction &local_direction);
	void get_lambda_min_max(double &lambda_min, double &lambda_max, const fluid_cell &fluid_left, const fluid_cell &fluid_right,
	                        const parallelisation::direction &local_direction);
	double get_sound_speed(double density, double pressure);
	double get_lambda_abs_max(const fluid_cell &fluid);
	double get_pressure(const fluid_cell &fluid);
	void transform_characteristic_to_conservative(fluid_cell &fluid);
	void transform_conservative_to_characteristic(fluid_cell &fluid);
	double get_e_total(const fluid_cell &fluid);

private:
	double adiabatic_index;
};

#endif