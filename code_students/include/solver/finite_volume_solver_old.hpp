#ifndef FINITE_VOLUME_SOLVER_HPP
#define FINITE_VOLUME_SOLVER_HPP

#include "core/config.hpp"
#include "setup/fluid.hpp"
#include "setup/grid.hpp"
#include "setup/physics.hpp"
#include "solver/Riemann_solvers.hpp"
#include "solver/reconstruction.hpp"

#include <functional>
#include <memory>

class finite_volume_solver {
public:
	finite_volume_solver(fluid &current_fluid);
	double singlestep(grid_3D &spatial_grid, fluid &current_fluid, fluid &current_changes);
	int run(grid_3D &spatial_grid, fluid &current_fluid, double time_final, double delta_t_output);
	void set_init_function(std::function<void(fluid_cell &, double, double, double)>);
	matrix<double, 3> get_data_computational_volume(grid_3D &spatial_grid, fluid &current_fluid, int index_data);
	void set_verbosity(int verbosity);

private:
	double get_CFL(grid_3D &spatial_grid, fluid &current_fluid);
	void set_initial_conditions(grid_3D &spatial_grid, fluid &current_fluid);
	void transform_fluid_to_conservative(fluid &current_fluid);
	void transform_fluid_to_characteristic(fluid &current_fluid);
	void apply_boundary_conditions(grid_3D &spatial_grid, fluid &current_fluid);
	void store_timestep(grid_3D &spatial_grid, fluid &current_fluid);
	double compute_delta_t_next(grid_3D &spatial_grid, fluid &current_fluid);
	physics fluid_physics;
	std::unique_ptr<HLL_solver> Riemann;
	reconsctruction_second_order reconst;
	// std::unique_ptr<Riemann_solver> Riemann;
	fluxes_cell num_flux_left_x, num_flux_right_x, num_flux_left_y, num_flux_right_y;
	fluxes_cell num_flux_left_z, num_flux_right_z;
	fluxes_cell fluxes_local, fluxes_previous;
	fluid_cell quantities_local, quantities_previous;
	fluid_cell quantities_temp_left, quantities_temp_right;
	std::function<void(fluid_cell &, double, double, double)> init_function;
	parallelisation::BoundaryType boundary_type;
	/**
	 * Maximum allowd CFL number for scheme
	 */
	double CFL_max;
	double time;
	double time_output_next, delta_t_output;
	int num_time_steps, verbosity;
	bool init_set, write_next_step;
};

#endif