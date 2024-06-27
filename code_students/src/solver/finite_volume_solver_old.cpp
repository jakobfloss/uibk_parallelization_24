#include "solver/finite_volume_solver.hpp"
#include "IO/data_storage.hpp"
#include "core/config.hpp"
#include "mpi.h"
#include "setup/fluid.hpp"
#include "setup/physics.hpp"
#include "solver/time_integrator.hpp"

#include <cassert>
#include <iostream>
#include <sstream>

finite_volume_solver::finite_volume_solver(fluid &current_fluid)
: reconst(current_fluid.get_fluid_type()), num_flux_left_x(current_fluid.get_fluid_type()), num_flux_right_x(current_fluid.get_fluid_type()),
  num_flux_left_y(current_fluid.get_fluid_type()), num_flux_right_y(current_fluid.get_fluid_type()), num_flux_left_z(current_fluid.get_fluid_type()),
  num_flux_right_z(current_fluid.get_fluid_type()), fluxes_local(current_fluid.get_fluid_type()), fluxes_previous(current_fluid.get_fluid_type()),
  quantities_local(current_fluid.get_fluid_type()), quantities_previous(current_fluid.get_fluid_type()), quantities_temp_left(current_fluid.get_fluid_type()),
  quantities_temp_right(current_fluid.get_fluid_type()) {

	CFL_max = 0.4;

	init_set = false;
	write_next_step = false;
	verbosity = 0;

	boundary_type = parallelisation::BoundaryType::constant_extrapolation;

	// Get number of fields:
	size_t num_fields = quantities_local.get_num_fields();

	// Choose a specific Riemann solver (here: HLL)
	Riemann = std::make_unique<HLL_solver>(num_fields);
}

void finite_volume_solver::set_init_function(std::function<void(fluid_cell &, double, double, double)> init_function_user) {
	init_function = init_function_user;
	init_set = true;
}

void finite_volume_solver::set_verbosity(int verbosity) { this->verbosity = verbosity; }

double finite_volume_solver::compute_delta_t_next(grid_3D &spatial_grid, fluid &current_fluid) {
	double CFL_number = get_CFL(spatial_grid, current_fluid);
	double delta_t = CFL_max / CFL_number;

	write_next_step = false;

	double delta_t_output_next = time_output_next - time;
	if (delta_t_output_next < delta_t) {
		delta_t = delta_t_output_next;
		time_output_next += delta_t_output;
		write_next_step = true;
	}

	#ifdef PARALLEL_VERSION 
		double delta_t_global;
		MPI_Allreduce(&delta_t, &delta_t_global, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
		return delta_t;
	#endif
	
	// sequential version
	return delta_t;
}

int finite_volume_solver::run(grid_3D &spatial_grid, fluid &current_fluid, double time_final, double delta_t_output) {
	assert(init_set);

	this->delta_t_output = delta_t_output;

	// Set initial conditions
	set_initial_conditions(spatial_grid, current_fluid);

	// prepare
	fluid fluid_changes(current_fluid.get_fluid_type());
	fluid_changes.setup(spatial_grid);
	RungeKutta2 time_stepper(spatial_grid, current_fluid);

	time = 0.0;
	num_time_steps = 0;

    // Store initial time step
	store_timestep(spatial_grid, current_fluid);

	time_output_next = delta_t_output;

	// Store initial time step
	if (num_time_steps % 10 == 0) {
		store_timestep(spatial_grid, current_fluid);
	}

	while (time < time_final) {

		double delta_t = compute_delta_t_next(spatial_grid, current_fluid);

		std::cout << " Integrating time step " << num_time_steps << " at t =" << time << "\n";
		std::cout << " \t\t step size " << delta_t << "\n";

		// Do individual Runge-Kutta steps
		int n_RK_steps = time_stepper.get_number_substeps();

		for (int i_RK_step = 0; i_RK_step < n_RK_steps; ++i_RK_step) {

			singlestep(spatial_grid, current_fluid, fluid_changes);

			// Transform to conservative variables
			transform_fluid_to_conservative(current_fluid);
			time_stepper.do_sub_step(spatial_grid, fluid_changes, current_fluid, delta_t, i_RK_step);
			// Transform back to characteristic variables
			transform_fluid_to_characteristic(current_fluid);

			apply_boundary_conditions(spatial_grid, current_fluid);
		}

		num_time_steps += 1;
		time += delta_t;

		// Write some test outputs
		// if(num_time_steps%10 == 0) {
		if (write_next_step) {
			store_timestep(spatial_grid, current_fluid);
		}
	}

	return num_time_steps;
}

void finite_volume_solver::set_initial_conditions(grid_3D &spatial_grid, fluid &current_fluid) {
	assert(init_set);

	for (int ix = 0; ix < spatial_grid.get_num_cells(0); ++ix) {
		double x_position = spatial_grid.x_grid.get_center(ix);
		for (int iy = 0; iy < spatial_grid.get_num_cells(1); ++iy) {
			double y_position = spatial_grid.y_grid.get_center(iy);
			for (int iz = 0; iz < spatial_grid.get_num_cells(2); ++iz) {
				double z_position = spatial_grid.z_grid.get_center(iz);

				init_function(quantities_local, x_position, y_position, z_position);
				current_fluid.set_cell_values(quantities_local, ix, iy, iz);
			}
		}
	}

	apply_boundary_conditions(spatial_grid, current_fluid);
}

void finite_volume_solver::apply_boundary_conditions(grid_3D &spatial_grid, fluid &current_fluid) {

	if (boundary_type == parallelisation::BoundaryType::constant_extrapolation) {

		for (size_t i_field = 0; i_field < current_fluid.fluid_data.size(); ++i_field) {

			// Lower x boundary
			for (int ix = -1; ix >= -2; --ix) {
				for (int iy = 0; iy < spatial_grid.get_num_cells(1); ++iy) {
					for (int iz = 0; iz < spatial_grid.get_num_cells(2); ++iz) {
						current_fluid.fluid_data[i_field](ix, iy, iz) = current_fluid.fluid_data[i_field](ix + 1, iy, iz);
					}
				}
			}
			// Upper x boundary
			for (int ix = spatial_grid.get_num_cells(0); ix < spatial_grid.get_num_cells(0) + 2; ++ix) {
				for (int iy = 0; iy < spatial_grid.get_num_cells(1); ++iy) {
					for (int iz = 0; iz < spatial_grid.get_num_cells(2); ++iz) {
						current_fluid.fluid_data[i_field](ix, iy, iz) = current_fluid.fluid_data[i_field](ix - 1, iy, iz);
					}
				}
			}

			// Lower y boundary
			for (int ix = 0; ix < spatial_grid.get_num_cells(0); ++ix) {
				for (int iy = -1; iy >= -2; --iy) {
					for (int iz = 0; iz < spatial_grid.get_num_cells(2); ++iz) {
						current_fluid.fluid_data[i_field](ix, iy, iz) = current_fluid.fluid_data[i_field](ix, iy + 1, iz);
					}
				}
			}
			// Upper y boundary
			for (int ix = 0; ix < spatial_grid.get_num_cells(0); ++ix) {
				for (int iy = spatial_grid.get_num_cells(1); iy < spatial_grid.get_num_cells(1) + 2; ++iy) {
					for (int iz = 0; iz < spatial_grid.get_num_cells(2); ++iz) {
						current_fluid.fluid_data[i_field](ix, iy, iz) = current_fluid.fluid_data[i_field](ix, iy - 1, iz);
					}
				}
			}

			// Lower z boundary
			for (int ix = 0; ix < spatial_grid.get_num_cells(0); ++ix) {
				for (int iy = 0; iy < spatial_grid.get_num_cells(1); ++iy) {
					for (int iz = -1; iz >= -2; --iz) {
						current_fluid.fluid_data[i_field](ix, iy, iz) = current_fluid.fluid_data[i_field](ix, iy, iz + 1);
					}
				}
			}
			// Upper z boundary
			for (int ix = 0; ix < spatial_grid.get_num_cells(0); ++ix) {
				for (int iy = 0; iy < spatial_grid.get_num_cells(1); ++iy) {
					for (int iz = spatial_grid.get_num_cells(2); iz < spatial_grid.get_num_cells(2) + 2; ++iz) {
						current_fluid.fluid_data[i_field](ix, iy, iz) = current_fluid.fluid_data[i_field](ix, iy, iz - 1);
					}
				}
			}
		}
	}
}

void finite_volume_solver::transform_fluid_to_conservative(fluid &current_fluid) {
	size_t num_cells = current_fluid.fluid_data[0].get_size();

	for (size_t i_cell = 0; i_cell < num_cells; ++i_cell) {
		current_fluid.get_fluid_cell_raw(quantities_local, i_cell);
		fluid_physics.transform_characteristic_to_conservative(quantities_local);
		current_fluid.set_fluid_cell_raw(quantities_local, i_cell);
	}
	current_fluid.set_conservative();
}

void finite_volume_solver::transform_fluid_to_characteristic(fluid &current_fluid) {
	size_t num_cells = current_fluid.fluid_data[0].get_size();

	for (size_t i_cell = 0; i_cell < num_cells; ++i_cell) {
		current_fluid.get_fluid_cell_raw(quantities_local, i_cell);
		fluid_physics.transform_conservative_to_characteristic(quantities_local);
		current_fluid.set_fluid_cell_raw(quantities_local, i_cell);
	}
	current_fluid.set_characteristic();
}

double finite_volume_solver::get_CFL(grid_3D &spatial_grid, fluid &current_fluid) {
	double CFL_number = 0.0;
	double delta_min = std::min(spatial_grid.x_grid.get_dx(), std::min(spatial_grid.y_grid.get_dx(), spatial_grid.z_grid.get_dx()));
	std::cout << " delta: " << delta_min << "\n";
	// Loop over full grid and compute local CFL value
	for (int ix = 0; ix < spatial_grid.get_num_cells(0); ++ix) {
		for (int iy = 0; iy < spatial_grid.get_num_cells(1); ++iy) {
			for (int iz = 0; iz < spatial_grid.get_num_cells(2); ++iz) {
				current_fluid.get_fluid_cell(quantities_local, ix, iy, iz);
				double v_max = fluid_physics.get_lambda_abs_max(quantities_local);
				CFL_number = std::max(CFL_number, v_max / delta_min);
			}
		}
	}
	return CFL_number;
}

void finite_volume_solver::store_timestep(grid_3D &spatial_grid, fluid &current_fluid) {
	std::ostringstream oss_file_name;
	oss_file_name << "output_step" << num_time_steps << ".h5";
	std::string file_name = oss_file_name.str();

	data_storage storage(file_name);
	storage.AddGlobalAttr<int>("Output step", num_time_steps);
	storage.AddGlobalAttr<double>("Output time", time);

	// Store the computational grid
	std::vector<double> grid_positions_center, grid_positions_left;
	std::vector<size_t> extent_grid;

	// x direction
	for (int index_x = 0; index_x < spatial_grid.get_num_cells(0); ++index_x) {
		grid_positions_center.push_back(spatial_grid.x_grid.get_center(index_x));
		grid_positions_left.push_back(spatial_grid.x_grid.get_left(index_x));
	}
	grid_positions_left.push_back(spatial_grid.x_grid.get_left(spatial_grid.get_num_cells(0)));
	extent_grid.push_back(spatial_grid.get_num_cells(0));
	storage.write_dataset(grid_positions_center, extent_grid, "x_grid_centers");
	extent_grid[0] += 1;
	storage.write_dataset(grid_positions_left, extent_grid, "x_grid_left");

	// y direction
	grid_positions_center.clear();
	grid_positions_left.clear();
	for (int index_y = 0; index_y < spatial_grid.get_num_cells(1); ++index_y) {
		grid_positions_center.push_back(spatial_grid.y_grid.get_center(index_y));
		grid_positions_left.push_back(spatial_grid.y_grid.get_left(index_y));
	}
	grid_positions_left.push_back(spatial_grid.y_grid.get_left(spatial_grid.get_num_cells(2)));
	extent_grid[0] = spatial_grid.get_num_cells(1);
	storage.write_dataset(grid_positions_center, extent_grid, "y_grid_centers");
	extent_grid[0] += 1;
	storage.write_dataset(grid_positions_left, extent_grid, "y_grid_left");

	// z direction
	grid_positions_center.clear();
	grid_positions_left.clear();
	for (int index_z = 0; index_z < spatial_grid.get_num_cells(2); ++index_z) {
		grid_positions_center.push_back(spatial_grid.z_grid.get_center(index_z));
		grid_positions_left.push_back(spatial_grid.z_grid.get_left(index_z));
	}
	grid_positions_left.push_back(spatial_grid.z_grid.get_left(spatial_grid.get_num_cells(2)));
	extent_grid[0] = spatial_grid.get_num_cells(2);
	storage.write_dataset(grid_positions_center, extent_grid, "z_grid_centers");
	extent_grid[0] += 1;
	storage.write_dataset(grid_positions_left, extent_grid, "z_grid_left");

	std::vector<size_t> extents;
	// extents.push_back(current_fluid.fluid_data[0].get_highest(0)-current_fluid.fluid_data[0].get_lowest(0)+1);
	// extents.push_back(current_fluid.fluid_data[0].get_highest(1)-current_fluid.fluid_data[0].get_lowest(1)+1);
	// extents.push_back(current_fluid.fluid_data[0].get_highest(2)-current_fluid.fluid_data[0].get_lowest(2)+1);

	extents.push_back(spatial_grid.get_num_cells(0));
	extents.push_back(spatial_grid.get_num_cells(1));
	extents.push_back(spatial_grid.get_num_cells(2));

	// Store the actual data
	matrix<double, 3> storage_data = get_data_computational_volume(spatial_grid, current_fluid, current_fluid.get_index_density());
	storage.write_dataset(storage_data.get_raw_data(), extents, "Density");
	storage_data = get_data_computational_volume(spatial_grid, current_fluid, current_fluid.get_index_v_x());
	storage.write_dataset(storage_data.get_raw_data(), extents, "v_x");
	storage_data = get_data_computational_volume(spatial_grid, current_fluid, current_fluid.get_index_v_y());
	storage.write_dataset(storage_data.get_raw_data(), extents, "v_y");
	storage_data = get_data_computational_volume(spatial_grid, current_fluid, current_fluid.get_index_v_z());
	storage.write_dataset(storage_data.get_raw_data(), extents, "v_z");
	storage_data = get_data_computational_volume(spatial_grid, current_fluid, current_fluid.get_index_energy());
	storage.write_dataset(storage_data.get_raw_data(), extents, "Energy");
	storage_data = get_data_computational_volume(spatial_grid, current_fluid, current_fluid.get_index_tracer());
	storage.write_dataset(storage_data.get_raw_data(), extents, "Tracer");
}

matrix<double, 3> finite_volume_solver::get_data_computational_volume(grid_3D &spatial_grid, fluid &current_fluid, int index_data) {
	matrix<double, 3> return_data;
	int index_low[3], index_hi[3];
	index_low[0] = 0;
	index_low[1] = 0;
	index_low[2] = 0;
	index_hi[0] = spatial_grid.get_num_cells(0) - 1;
	index_hi[1] = spatial_grid.get_num_cells(1) - 1;
	index_hi[2] = spatial_grid.get_num_cells(2) - 1;
	return_data.resize(index_low, index_hi);

	for (int ix = 0; ix < spatial_grid.get_num_cells(0); ++ix) {
		for (int iy = 0; iy < spatial_grid.get_num_cells(1); ++iy) {
			for (int iz = 0; iz < spatial_grid.get_num_cells(2); ++iz) {
				return_data(ix, iy, iz) = current_fluid.fluid_data[index_data](ix, iy, iz);
			}
		}
	}
	return return_data;
}