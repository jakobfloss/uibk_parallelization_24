#include "setup/physics.hpp"
#include "setup/fluid.hpp"
#include "util/utility_functions.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace sim_util;

physics::physics() { adiabatic_index = 1.4; }

double physics::get_pressure(const fluid_cell &fluid) { return (adiabatic_index - 1.0) * fluid.fluid_data[fluid.get_index_energy()]; }

void physics::transform_characteristic_to_conservative(fluid_cell &fluid) {
	assert(!fluid.is_conservative());
	double density = fluid.fluid_data[fluid.get_index_density()];
	double v_x = fluid.fluid_data[fluid.get_index_v_x()];
	double v_y = fluid.fluid_data[fluid.get_index_v_y()];
	double v_z = fluid.fluid_data[fluid.get_index_v_z()];
	double e_thermal = fluid.fluid_data[fluid.get_index_energy()];

	double e_kin = 0.5 * density * (square(v_x) + square(v_y) + square(v_z));
	double momentum_x = density * v_x;
	double momentum_y = density * v_y;
	double momentum_z = density * v_z;
	double e_total = e_thermal + e_kin;

	fluid.fluid_data[fluid.get_index_v_x()] = momentum_x;
	fluid.fluid_data[fluid.get_index_v_y()] = momentum_y;
	fluid.fluid_data[fluid.get_index_v_z()] = momentum_z;
	fluid.fluid_data[fluid.get_index_energy()] = e_total;
	fluid.set_conservative();
}

void physics::transform_conservative_to_characteristic(fluid_cell &fluid) {
	assert(fluid.is_conservative());
	double density = fluid.fluid_data[fluid.get_index_density()];
	double momentum_x = fluid.fluid_data[fluid.get_index_v_x()];
	double momentum_y = fluid.fluid_data[fluid.get_index_v_y()];
	double momentum_z = fluid.fluid_data[fluid.get_index_v_z()];
	double e_total = fluid.fluid_data[fluid.get_index_energy()];

	double v_x = momentum_x / density;
	double v_y = momentum_y / density;
	double v_z = momentum_z / density;
	double e_kin = 0.5 * density * (square(v_x) + square(v_y) + square(v_z));
	double e_thermal = e_total - e_kin;

	fluid.fluid_data[fluid.get_index_v_x()] = v_x;
	fluid.fluid_data[fluid.get_index_v_y()] = v_y;
	fluid.fluid_data[fluid.get_index_v_z()] = v_z;
	fluid.fluid_data[fluid.get_index_energy()] = e_thermal;
	fluid.set_conservative();
}

double physics::get_e_total(const fluid_cell &fluid) {
	double density = fluid.fluid_data[fluid.get_index_density()];
	double v_x = fluid.fluid_data[fluid.get_index_v_x()];
	double v_y = fluid.fluid_data[fluid.get_index_v_y()];
	double v_z = fluid.fluid_data[fluid.get_index_v_z()];
	double e_therm = fluid.fluid_data[fluid.get_index_energy()];

	double e_total = 0.5 * density * (sim_util::square(v_x) + sim_util::square(v_y) + sim_util::square(v_z)) + e_therm;
	return e_total;
}

void physics::get_physical_fluxes(const fluid_cell &fluid, fluxes_cell &fluxes, const parallelisation::direction &local_direction) {
	// Set physical fluxes for hydrodynamics
	assert(!fluid.is_conservative());
	double density = fluid.fluid_data[fluid.get_index_density()];
	double v_x = fluid.fluid_data[fluid.get_index_v_x()];
	double v_y = fluid.fluid_data[fluid.get_index_v_y()];
	double v_z = fluid.fluid_data[fluid.get_index_v_z()];
	double tracer = fluid.fluid_data[fluid.get_index_tracer()];

	double pressure = get_pressure(fluid);
	double e_total = get_e_total(fluid);

	// fluxes.flux_data[fluid.get_index_density()] = fluid_cell.fluid_data[]
	if (local_direction == parallelisation::direction::x) {
		fluxes.flux_data[fluid.get_index_density()] = density * v_x;
		fluxes.flux_data[fluid.get_index_v_x()] = density * v_x * v_x + pressure;
		fluxes.flux_data[fluid.get_index_v_y()] = density * v_x * v_y;
		fluxes.flux_data[fluid.get_index_v_z()] = density * v_x * v_z;
		fluxes.flux_data[fluid.get_index_energy()] = (e_total + pressure) * v_x;
		fluxes.flux_data[fluid.get_index_tracer()] = tracer * v_x;
	} else if (local_direction == parallelisation::direction::y) {
		fluxes.flux_data[fluid.get_index_density()] = density * v_y;
		fluxes.flux_data[fluid.get_index_v_x()] = density * v_y * v_x;
		fluxes.flux_data[fluid.get_index_v_y()] = density * v_y * v_y + pressure;
		fluxes.flux_data[fluid.get_index_v_z()] = density * v_y * v_z;
		fluxes.flux_data[fluid.get_index_energy()] = (e_total + pressure) * v_y;
		fluxes.flux_data[fluid.get_index_tracer()] = tracer * v_y;
	} else {
		fluxes.flux_data[fluid.get_index_density()] = density * v_z;
		fluxes.flux_data[fluid.get_index_v_x()] = density * v_z * v_x;
		fluxes.flux_data[fluid.get_index_v_y()] = density * v_z * v_y;
		fluxes.flux_data[fluid.get_index_v_z()] = density * v_z * v_z + pressure;
		fluxes.flux_data[fluid.get_index_energy()] = (e_total + pressure) * v_z;
		fluxes.flux_data[fluid.get_index_tracer()] = tracer * v_z;
	}
}

double physics::get_sound_speed(double density, double pressure) { return sqrt(adiabatic_index * pressure / density); }

double physics::get_lambda_abs_max(const fluid_cell &fluid) {
	assert(!fluid.is_conservative());
	double density = fluid.fluid_data[fluid.get_index_density()];
	double v_x = fluid.fluid_data[fluid.get_index_v_x()];
	double v_y = fluid.fluid_data[fluid.get_index_v_y()];
	double v_z = fluid.fluid_data[fluid.get_index_v_z()];
	double pressure = get_pressure(fluid);

	double sound_speed = get_sound_speed(density, pressure);
	double velo_abs = sqrt(sim_util::square(v_x) + sim_util::square(v_y) + sim_util::square(v_z));
	double lambda_abs_max = velo_abs + sound_speed;
	// std::cout << " lambda " << density << "\n";
	return lambda_abs_max;
}

void physics::get_lambda_min_max(double &lambda_min, double &lambda_max, const fluid_cell &fluid_left_cell, const fluid_cell &fluid_right_cell,
                                 const parallelisation::direction &local_direction) {

	assert(!fluid_left_cell.is_conservative());
	assert(!fluid_right_cell.is_conservative());

	int index_density = fluid_left_cell.get_index_density();
	int index_velocity_parallel = fluid_left_cell.get_index_v_x();
	if (local_direction == parallelisation::direction::y) {
		index_velocity_parallel = fluid_left_cell.get_index_v_y();
	} else if (local_direction == parallelisation::direction::z) {
		index_velocity_parallel = fluid_left_cell.get_index_v_z();
	}

	double density_left = fluid_left_cell.fluid_data[index_density];
	double density_right = fluid_right_cell.fluid_data[index_density];

	double v_parallel_left = fluid_left_cell.fluid_data[index_velocity_parallel];
	double v_parallel_right = fluid_right_cell.fluid_data[index_velocity_parallel];

	double pressure_left = get_pressure(fluid_left_cell);
	double pressure_right = get_pressure(fluid_right_cell);

	double sound_speed_left = get_sound_speed(density_left, pressure_left);
	double sound_speed_right = get_sound_speed(density_right, pressure_right);

	// lambda_max = std::max(std::max( v_parallel_left + sound_speed_left, v_parallel_right + sound_speed_right), 0.0);
	// lambda_min = std::min(std::min( v_parallel_left - sound_speed_left, v_parallel_right - sound_speed_right), 0.0);
	lambda_max = std::max(v_parallel_left + sound_speed_left, v_parallel_right + sound_speed_right);
	lambda_min = std::min(v_parallel_left - sound_speed_left, v_parallel_right - sound_speed_right);
}
