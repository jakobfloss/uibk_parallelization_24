#include "solver/finite_volume_solver.hpp"
#include "util/matrix.hpp"
#include <cmath>
#ifdef PARALLEL_VERSION
#include "IO/data_storage_parallel.hpp"
#else
#include "IO/data_storage.hpp"
#endif
#include "core/config.hpp"
#include "setup/fluid.hpp"
#include "setup/physics.hpp"
#include "solver/time_integrator.hpp"

#include <cassert>
#include <iostream>
#include <sstream>

#ifdef PARALLEL_VERSION
	finite_volume_solver::finite_volume_solver(fluid &current_fluid, mpi_handler &parallel_stuff, grid_3D &global_grid)
#else
	finite_volume_solver::finite_volume_solver(fluid &current_fluid)
#endif
: reconst(current_fluid.get_fluid_type()), num_flux_left_x(current_fluid.get_fluid_type()), num_flux_right_x(current_fluid.get_fluid_type()),
  num_flux_left_y(current_fluid.get_fluid_type()), num_flux_right_y(current_fluid.get_fluid_type()), num_flux_left_z(current_fluid.get_fluid_type()),
  num_flux_right_z(current_fluid.get_fluid_type()), fluxes_local(current_fluid.get_fluid_type()), fluxes_previous(current_fluid.get_fluid_type()),
  quantities_local(current_fluid.get_fluid_type()), quantities_previous(current_fluid.get_fluid_type()), quantities_temp_left(current_fluid.get_fluid_type()),
  quantities_temp_right(current_fluid.get_fluid_type()) 
#ifdef PARALLEL_VERSION
	, parallel_handler(parallel_stuff), _global_grid(global_grid)
#endif
  {

	CFL_max = 0.4;

	rank=0;
#ifdef PARALLEL_VERSION
	rank = parallel_handler.get_rank();
#endif


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
#ifdef PARALLEL_VERSION
    store_timestep_parallel(spatial_grid, current_fluid);
#else 
    store_timestep(spatial_grid, current_fluid);
#endif


	time_output_next = delta_t_output;



	while (time < time_final) {

		double delta_t = compute_delta_t_next(spatial_grid, current_fluid);

		if(rank==0) {
			std::cout << " Integrating time step " << num_time_steps << " at t =" << time << "\n";
			std::cout << " \t\t step size " << delta_t << "\n";
		}

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
#ifdef PARALLEL_VERSION
			store_timestep_parallel(spatial_grid, current_fluid);
#else
			store_timestep(spatial_grid, current_fluid);
#endif
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


#ifdef PARALLEL_VERSION

void finite_volume_solver::apply_boundary_conditions(grid_3D &spatial_grid, fluid &current_fluid) {

	int Nx = spatial_grid.get_num_cells(0);
	int Ny = spatial_grid.get_num_cells(1);
	int Nz = spatial_grid.get_num_cells(2);


	if (boundary_type == parallelisation::BoundaryType::constant_extrapolation) {

		for (size_t i_field = 0; i_field < current_fluid.fluid_data.size(); ++i_field) {

			// Lower x boundary
			
			// first get data from neighbouring rank on the left and send data to rank on the right.

			// where necessary, do parallel boundaries

			// Prepare buffer -> size 2 x Ny x Nz
			int size_buff = 2 * Ny * Nz;
			std::vector<double> buff_send_x(size_buff);
			std::vector<double> buff_recv_x(size_buff);

			int buff_lo[3] = {0, 0, 0};
			int buff_hi[3] = {1, Ny-1, Nz-1};
			// matrix<double, 3> buff_Send_x, buff_Recv_x;
			// buff_Send_x.resize(buff_lo, buff_hi);
			// buff_Recv_x.resize(buff_lo, buff_hi);


			// get distination rank
			int dest_rank = parallel_handler.get_right();
			int src_rank = parallel_handler.get_left();

			int tag_send = 0;
			int tag_recv = 0;

			// prepare buffer to be send
			size_t i_buff=0;
			for(int ix=0; ix<2; ix++) {
				for (int iy = 0; iy < spatial_grid.get_num_cells(1); ++iy) {
					for (int iz = 0; iz < spatial_grid.get_num_cells(2); ++iz) {
						buff_send_x[i_buff] = current_fluid.fluid_data[i_field](Nx - 2 + ix, iy, iz);
						// buff_Send_x(ix, iy, iz) = current_fluid.fluid_data[i_field](Nx - 2 + ix, iy, iz);
						i_buff++;
					}
				}
			}

			MPI_Status status;
			// Send and receive data
			MPI_Sendrecv(&buff_send_x[0], size_buff, MPI_DOUBLE, dest_rank, tag_send,
	            &buff_recv_x[0], size_buff, MPI_DOUBLE, src_rank, tag_recv, parallel_handler.comm3D, &status);
			// MPI_Sendrecv(&buff_send_x[0], size_buff, MPI_DOUBLE, dest_rank, tag_send,
	        //     &buff_recv_x[0], size_buff, MPI_DOUBLE, src_rank, tag_recv, parallel_handler.comm3D, &status);

			// Finally, assign data - either directly or from receive buffer
			if(parallel_handler.get_left()==MPI_PROC_NULL) {
				for (int ix = -1; ix >= -2; --ix) {
					for (int iy = 0; iy < spatial_grid.get_num_cells(1); ++iy) {
						for (int iz = 0; iz < spatial_grid.get_num_cells(2); ++iz) {
							current_fluid.fluid_data[i_field](ix, iy, iz) = current_fluid.fluid_data[i_field](ix + 1, iy, iz);
						}
					}
				}
			} else {
				i_buff=0;
				for (int ix = 0; ix < 2; ++ix) {
					for (int iy = 0; iy < spatial_grid.get_num_cells(1); ++iy) {
						for (int iz = 0; iz < spatial_grid.get_num_cells(2); ++iz) {
							current_fluid.fluid_data[i_field](ix - 2, iy, iz) = buff_recv_x[i_buff];
							i_buff++;
						}
					}
				}
			}





			// Upper x boundary

			// next, get data from neighbouring rank on the right and send data to rank on the left.

			// get distination rank
			dest_rank = parallel_handler.get_left();
			src_rank = parallel_handler.get_right();

			tag_send = 1;
			tag_recv = 1;

			// prepare buffer to be send
			i_buff=0;
			for(int ix=0; ix<2; ix++) {
				for (int iy = 0; iy < spatial_grid.get_num_cells(1); ++iy) {
					for (int iz = 0; iz < spatial_grid.get_num_cells(2); ++iz) {
						buff_send_x[i_buff] = current_fluid.fluid_data[i_field](ix, iy, iz);
						i_buff++;
					}
				}
			}

			// Send and receive data
			MPI_Sendrecv(&buff_send_x[0], size_buff, MPI_DOUBLE, dest_rank, tag_send,
	            &buff_recv_x[0], size_buff, MPI_DOUBLE, src_rank, tag_recv, parallel_handler.comm3D, &status);

			// Finally, assign data - either directly or from receive buffer
			if(parallel_handler.get_right()==MPI_PROC_NULL) {
				for (int ix = spatial_grid.get_num_cells(0); ix < spatial_grid.get_num_cells(0) + 2; ++ix) {
					for (int iy = 0; iy < spatial_grid.get_num_cells(1); ++iy) {
						for (int iz = 0; iz < spatial_grid.get_num_cells(2); ++iz) {
							current_fluid.fluid_data[i_field](ix, iy, iz) = current_fluid.fluid_data[i_field](ix - 1, iy, iz);
						}
					}
				}	
			} else {
				i_buff=0;
				for (int ix = 0; ix < 2; ix++) {
					for (int iy = 0; iy < spatial_grid.get_num_cells(1); ++iy) {
						for (int iz = 0; iz < spatial_grid.get_num_cells(2); ++iz) {
							current_fluid.fluid_data[i_field](Nx + ix, iy, iz) = buff_recv_x[i_buff];
							i_buff++;
						}
					}
				}
			}

			




			// Lower y boundary

			// first, get data from neighbouring rank at the front and send data to rank at the back.

			// where necessary, do parallel boundaries

			// Prepare buffer -> size 2 x Nx x Nz
			size_buff = 2 * Nx * Nz;
			std::vector<double> buff_send_y(size_buff);
			std::vector<double> buff_recv_y(size_buff);


			// get distination rank
			dest_rank = parallel_handler.get_back();
			src_rank = parallel_handler.get_front();

			tag_send = 2;
			tag_recv = 2;

			// prepare buffer to be send
			i_buff=0;
			for (int ix = 0; ix < spatial_grid.get_num_cells(0); ++ix) {
				for(int iy=0; iy<2; iy++) {
					for (int iz = 0; iz < spatial_grid.get_num_cells(2); ++iz) {
						buff_send_y[i_buff] = current_fluid.fluid_data[i_field](ix, Ny - 2 + iy, iz);
						i_buff++;
					}
				}
			}

			// Send and receive data
			MPI_Sendrecv(&buff_send_y[0], size_buff, MPI_DOUBLE, dest_rank, tag_send,
	            &buff_recv_y[0], size_buff, MPI_DOUBLE, src_rank, tag_recv, parallel_handler.comm3D, &status);


			// Finally, assign data - either directly or from receive buffer
			if(parallel_handler.get_front()==MPI_PROC_NULL) {
				for (int ix = 0; ix < spatial_grid.get_num_cells(0); ++ix) {
					for (int iy = -1; iy >= -2; --iy) {
						for (int iz = 0; iz < spatial_grid.get_num_cells(2); ++iz) {
							current_fluid.fluid_data[i_field](ix, iy, iz) = current_fluid.fluid_data[i_field](ix, iy + 1, iz);
						}
					}
				}
			} else {
				i_buff=0;
				for (int ix = 0; ix < spatial_grid.get_num_cells(0); ++ix) {
					for (int iy = 0; iy < 2; ++iy) {
						for (int iz = 0; iz < spatial_grid.get_num_cells(2); ++iz) {
							current_fluid.fluid_data[i_field](ix, iy - 2, iz) = buff_recv_y[i_buff];
							i_buff++;
						}
					}
				}
			}




			// Upper y boundary

			// next, get data from neighbouring rank at the back and send data to rank in front.

			// where necessary, do parallel boundaries

			// get distination rank
			dest_rank = parallel_handler.get_front();
			src_rank = parallel_handler.get_back();

			tag_send = 3;
			tag_recv = 3;

			// prepare buffer to be send
			i_buff=0;
			for (int ix = 0; ix < spatial_grid.get_num_cells(0); ++ix) {
				for(int iy=0; iy<2; iy++) {
					for (int iz = 0; iz < spatial_grid.get_num_cells(2); ++iz) {
						buff_send_y[i_buff] = current_fluid.fluid_data[i_field](ix, iy - 2, iz);
						i_buff++;
					}
				}
			}

			// Send and receive data
			MPI_Sendrecv(&buff_send_y[0], size_buff, MPI_DOUBLE, dest_rank, tag_send,
	            &buff_recv_y[0], size_buff, MPI_DOUBLE, src_rank, tag_recv, parallel_handler.comm3D, &status);


			// Finally, assign data - either directly or from receive buffer
			if(parallel_handler.get_back()==MPI_PROC_NULL) {
				for (int ix = 0; ix < spatial_grid.get_num_cells(0); ++ix) {
					for (int iy = spatial_grid.get_num_cells(1); iy < spatial_grid.get_num_cells(1) + 2; ++iy) {
						for (int iz = 0; iz < spatial_grid.get_num_cells(2); ++iz) {
							current_fluid.fluid_data[i_field](ix, iy, iz) = current_fluid.fluid_data[i_field](ix, iy - 1, iz);
						}
					}
				}
			} else {
				i_buff=0;
				for (int ix = 0; ix < spatial_grid.get_num_cells(0); ++ix) {
					for (int iy = 0; iy < 2; ++iy) {
						for (int iz = 0; iz < spatial_grid.get_num_cells(2); ++iz) {
							current_fluid.fluid_data[i_field](ix, Ny + iy, iz) = buff_recv_y[i_buff];
							i_buff++;
						}
					}
				}
			}




			// Lower z boundary

			// first get data from neighbouring rank at the bottom and send data to rank at the top.

			// where necessary, do parallel boundaries

			// TBD by students 

		}
	}

}



#else

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

#endif

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
	double CLF_number = 0.0;
	double delta_min = std::min(spatial_grid.x_grid.get_dx(), std::min(spatial_grid.y_grid.get_dx(), spatial_grid.z_grid.get_dx()));
	// if(rank==0) {
	// 	std::cout << " delta (CFL): " << delta_min << "\n";
	// }
	// Loop over full grid and compute local CFL value
	for (int ix = 0; ix < spatial_grid.get_num_cells(0); ++ix) {
		for (int iy = 0; iy < spatial_grid.get_num_cells(1); ++iy) {
			for (int iz = 0; iz < spatial_grid.get_num_cells(2); ++iz) {
				current_fluid.get_fluid_cell(quantities_local, ix, iy, iz);
				double v_max = fluid_physics.get_lambda_abs_max(quantities_local);
				CLF_number = std::max(CLF_number, v_max / delta_min);
			}
		}
	}
#ifdef PARALLEL_VERSION
	// TBD by students 

#endif

	if(rank==0) {
		std::cout << " CFL number " << CLF_number << "\n";
	}

	return CLF_number;
}

#ifdef PARALLEL_VERSION

void finite_volume_solver::store_timestep_parallel(grid_3D &spatial_grid, fluid &current_fluid) {
	std::ostringstream oss_file_name;
	oss_file_name << "output_step" << num_time_steps << ".h5";
	std::string file_name = oss_file_name.str();

	data_storage_parallel storage(file_name);

	storage.AddGlobalAttr<int>("Output step", num_time_steps);
	storage.AddGlobalAttr<double>("Output time", time);


	// Store the computational grid
	std::vector<double> grid_positions_center, grid_positions_left;
	std::vector<size_t> extent_grid;

	// x direction
	for (int index_x = 0; index_x < _global_grid.get_num_cells(0); ++index_x) {
		grid_positions_center.push_back(_global_grid.x_grid.get_center(index_x));
		grid_positions_left.push_back(_global_grid.x_grid.get_left(index_x));
	}
	grid_positions_left.push_back(_global_grid.x_grid.get_left(_global_grid.get_num_cells(0)));
	extent_grid.push_back(_global_grid.get_num_cells(0));
	storage.write_dataset(grid_positions_center, extent_grid, "x_grid_centers");
	extent_grid[0] += 1;
	storage.write_dataset(grid_positions_left, extent_grid, "x_grid_left");


	// y direction
	grid_positions_center.clear();
	grid_positions_left.clear();
	for (int index_y = 0; index_y < _global_grid.get_num_cells(1); ++index_y) {
		grid_positions_center.push_back(_global_grid.y_grid.get_center(index_y));
		grid_positions_left.push_back(_global_grid.y_grid.get_left(index_y));
	}
	grid_positions_left.push_back(_global_grid.y_grid.get_left(_global_grid.get_num_cells(2)));
	extent_grid[0] = _global_grid.get_num_cells(1);
	storage.write_dataset(grid_positions_center, extent_grid, "y_grid_centers");
	extent_grid[0] += 1;
	storage.write_dataset(grid_positions_left, extent_grid, "y_grid_left");


	// z direction
	grid_positions_center.clear();
	grid_positions_left.clear();
	for (int index_z = 0; index_z < _global_grid.get_num_cells(2); ++index_z) {
		grid_positions_center.push_back(_global_grid.z_grid.get_center(index_z));
		grid_positions_left.push_back(_global_grid.z_grid.get_left(index_z));
	}
	grid_positions_left.push_back(_global_grid.z_grid.get_left(_global_grid.get_num_cells(2)));
	extent_grid[0] = _global_grid.get_num_cells(2);
	storage.write_dataset(grid_positions_center, extent_grid, "z_grid_centers");
	extent_grid[0] += 1;
	storage.write_dataset(grid_positions_left, extent_grid, "z_grid_left");



	std::vector<size_t> extents_local;
	// extents.push_back(current_fluid.fluid_data[0].get_highest(0)-current_fluid.fluid_data[0].get_lowest(0)+1);
	// extents.push_back(current_fluid.fluid_data[0].get_highest(1)-current_fluid.fluid_data[0].get_lowest(1)+1);
	// extents.push_back(current_fluid.fluid_data[0].get_highest(2)-current_fluid.fluid_data[0].get_lowest(2)+1);

	extents_local.push_back(spatial_grid.get_num_cells(0));
	extents_local.push_back(spatial_grid.get_num_cells(1));
	extents_local.push_back(spatial_grid.get_num_cells(2));

	std::vector<size_t> extents_global;
	extents_global.push_back(spatial_grid.get_num_cells(0) * parallel_handler.get_num_tasks(0));
	extents_global.push_back(spatial_grid.get_num_cells(1) * parallel_handler.get_num_tasks(1));
	extents_global.push_back(spatial_grid.get_num_cells(2) * parallel_handler.get_num_tasks(2));

	std::vector<size_t> rank_shifts;
	rank_shifts.push_back(parallel_handler.get_shift_cells(0));
	rank_shifts.push_back(parallel_handler.get_shift_cells(1));
	rank_shifts.push_back(parallel_handler.get_shift_cells(2));




	// Store the actual data
	matrix<double, 3> storage_data = get_data_computational_volume(spatial_grid, current_fluid, current_fluid.get_index_density());
	storage.write_dataset_parallel(storage_data.get_raw_data(), extents_global, extents_local, rank_shifts, "Density");
	storage_data = get_data_computational_volume(spatial_grid, current_fluid, current_fluid.get_index_v_x());
	storage.write_dataset_parallel(storage_data.get_raw_data(), extents_global, extents_local, rank_shifts, "v_x");
	storage_data = get_data_computational_volume(spatial_grid, current_fluid, current_fluid.get_index_v_y());
	storage.write_dataset_parallel(storage_data.get_raw_data(), extents_global, extents_local, rank_shifts, "v_y");
	storage_data = get_data_computational_volume(spatial_grid, current_fluid, current_fluid.get_index_v_z());
	storage.write_dataset_parallel(storage_data.get_raw_data(), extents_global, extents_local, rank_shifts, "v_z");
	storage_data = get_data_computational_volume(spatial_grid, current_fluid, current_fluid.get_index_energy());
	storage.write_dataset_parallel(storage_data.get_raw_data(), extents_global, extents_local, rank_shifts, "Energy");
	storage_data = get_data_computational_volume(spatial_grid, current_fluid, current_fluid.get_index_tracer());
	storage.write_dataset_parallel(storage_data.get_raw_data(), extents_global, extents_local, rank_shifts, "Tracer");
}

#else

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

#endif


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
				if(std::isnan(return_data(ix,iy,iz))) {
					std::cerr << " Error at " << rank << " " << index_data << " " << ix << " " << iy << " " << iz << "\n";
					exit(232);
				}
			}
		}
	}
	return return_data;
}
