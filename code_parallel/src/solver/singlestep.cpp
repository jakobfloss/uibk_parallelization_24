#include "core/config.hpp"
#include "setup/physics.hpp"
#include "solver/finite_volume_solver.hpp"

#include <iostream>

double finite_volume_solver::singlestep(grid_3D &spatial_grid, fluid &current_fluid, fluid &current_changes) {

	size_t num_fields = current_fluid.get_number_fields();

	for (size_t i_field = 0; i_field < num_fields; ++i_field) {
		current_changes.fluid_data[i_field].clear();
	}

	for (int ix = 0; ix <= spatial_grid.get_num_cells(0); ++ix) {
		for (int iy = 0; iy <= spatial_grid.get_num_cells(1); ++iy) {
			for (int iz = 0; iz <= spatial_grid.get_num_cells(2); ++iz) {

				// x-dimension

				// Get local values using a reconstruction
				// previous cell
				reconst.compute_point_values(current_fluid, quantities_temp_left, quantities_temp_right, parallelisation::direction::x, ix - 1, iy, iz);
				quantities_previous = quantities_temp_right;
				// local cell
				reconst.compute_point_values(current_fluid, quantities_temp_left, quantities_temp_right, parallelisation::direction::x, ix, iy, iz);
				quantities_local = quantities_temp_left;

				// compute physical fluxes
				fluid_physics.get_physical_fluxes(quantities_previous, fluxes_previous, parallelisation::direction::x);
				fluid_physics.get_physical_fluxes(quantities_local, fluxes_local, parallelisation::direction::x);

				// Compute characteristic velocities from point values
				double lambda_min_x(0.0), lambda_max_x(0.0);
				fluid_physics.get_lambda_min_max(lambda_min_x, lambda_max_x, quantities_previous, quantities_local, parallelisation::direction::x);

				// Need conservative variables for numerical fluxes
				fluid_physics.transform_characteristic_to_conservative(quantities_previous);
				fluid_physics.transform_characteristic_to_conservative(quantities_local);

				// Compute numerical fluxes from point values and physical fluxes
				// Compute left-handed flux in x-direction
				Riemann->get_num_flux(quantities_previous, quantities_local, fluxes_previous.flux_data, fluxes_local.flux_data, num_flux_left_x.flux_data, lambda_min_x,
				                      lambda_max_x);

				// Compute changes of cell - left handed flux applies to previous and local cell
				for (size_t i_field = 0; i_field < num_fields; ++i_field) {
					if (ix > 0) {
						current_changes.fluid_data[i_field](ix - 1, iy, iz) -= num_flux_left_x.flux_data[i_field] * spatial_grid.x_grid.get_inv_dx();
					}
					if (ix < spatial_grid.get_num_cells(0)) {
						current_changes.fluid_data[i_field](ix, iy, iz) += num_flux_left_x.flux_data[i_field] * spatial_grid.x_grid.get_inv_dx();
					}
				}

				// y-dimension

				// Get local values using a reconstruction
				// previous cell
				reconst.compute_point_values(current_fluid, quantities_temp_left, quantities_temp_right, parallelisation::direction::y, ix, iy - 1, iz);
				quantities_previous = quantities_temp_right;
				// local cell
				reconst.compute_point_values(current_fluid, quantities_temp_left, quantities_temp_right, parallelisation::direction::y, ix, iy, iz);
				quantities_local = quantities_temp_left;

				// compute physical fluxes
				fluid_physics.get_physical_fluxes(quantities_previous, fluxes_previous, parallelisation::direction::y);
				fluid_physics.get_physical_fluxes(quantities_local, fluxes_local, parallelisation::direction::y);

				// Compute characteristic velocities from point values
				double lambda_min_y(0.0), lambda_max_y(0.0);
				fluid_physics.get_lambda_min_max(lambda_min_y, lambda_max_y, quantities_previous, quantities_local, parallelisation::direction::y);

				// Need conservative variables for numerical fluxes
				fluid_physics.transform_characteristic_to_conservative(quantities_previous);
				fluid_physics.transform_characteristic_to_conservative(quantities_local);

				// Compute numerical fluxes from point values and physical fluxes
				// Compute left-handed flux in x-direction
				Riemann->get_num_flux(quantities_previous, quantities_local, fluxes_previous.flux_data, fluxes_local.flux_data, num_flux_left_y.flux_data, lambda_min_y,
				                      lambda_max_y);

				// Compute changes of cell
				for (size_t i_field = 0; i_field < num_fields; ++i_field) {
					if (iy > 0) {
						current_changes.fluid_data[i_field](ix, iy - 1, iz) -= num_flux_left_y.flux_data[i_field] * spatial_grid.y_grid.get_inv_dx();
					}
					if (iy < spatial_grid.get_num_cells(1)) {
						current_changes.fluid_data[i_field](ix, iy, iz) += num_flux_left_y.flux_data[i_field] * spatial_grid.y_grid.get_inv_dx();
					}
				}

				// z-dimension

				// Get local values using a reconstruction
				// previous cell
				reconst.compute_point_values(current_fluid, quantities_temp_left, quantities_temp_right, parallelisation::direction::z, ix, iy, iz - 1);
				quantities_previous = quantities_temp_right;
				// local cell
				reconst.compute_point_values(current_fluid, quantities_temp_left, quantities_temp_right, parallelisation::direction::z, ix, iy, iz);
				quantities_local = quantities_temp_left;

				// compute physical fluxes
				fluid_physics.get_physical_fluxes(quantities_previous, fluxes_previous, parallelisation::direction::z);
				fluid_physics.get_physical_fluxes(quantities_local, fluxes_local, parallelisation::direction::z);

				// Compute characteristic velocities from point values
				double lambda_min_z(0.0), lambda_max_z(0.0);
				fluid_physics.get_lambda_min_max(lambda_min_z, lambda_max_z, quantities_previous, quantities_local, parallelisation::direction::z);

				// Need conservative variables for numerical fluxes
				fluid_physics.transform_characteristic_to_conservative(quantities_previous);
				fluid_physics.transform_characteristic_to_conservative(quantities_local);

				// Compute numerical fluxes from point values and physical fluxes
				// Compute left-handed flux in z-direction
				Riemann->get_num_flux(quantities_previous, quantities_local, fluxes_previous.flux_data, fluxes_local.flux_data, num_flux_left_z.flux_data, lambda_min_z,
				                      lambda_max_z);

				// Compute changes of cell
				for (size_t i_field = 0; i_field < num_fields; ++i_field) {
					if (iz > 0) {
						current_changes.fluid_data[i_field](ix, iy, iz - 1) -= num_flux_left_z.flux_data[i_field] * spatial_grid.z_grid.get_inv_dx();
					}
					if (iz < spatial_grid.get_num_cells(2)) {
						current_changes.fluid_data[i_field](ix, iy, iz) += num_flux_left_z.flux_data[i_field] * spatial_grid.z_grid.get_inv_dx();
					}
				}
			}
		}
	}

	if (verbosity > 10) {
		std::cout << " Values: ";
		std::cout << current_fluid.fluid_data[0](14, 16, 16) << " ";
		std::cout << current_fluid.fluid_data[4](13, 16, 16) << " ";
		std::cout << current_fluid.fluid_data[4](14, 16, 16) << " ";
		std::cout << current_fluid.fluid_data[4](15, 16, 16) << " ";
		std::cout << "\n";
		std::cout << " Changes ";
		std::cout << current_changes.fluid_data[0](13, 16, 16) << " ";
		std::cout << current_changes.fluid_data[0](14, 16, 16) << " ";
		std::cout << current_changes.fluid_data[0](15, 16, 16) << " ";
		std::cout << "\n";
		std::cout << current_changes.fluid_data[1](12, 16, 16) << " ";
		std::cout << current_changes.fluid_data[1](13, 16, 16) << " ";
		std::cout << current_changes.fluid_data[1](14, 16, 16) << " ";
		std::cout << current_changes.fluid_data[1](15, 16, 16) << " ";
		std::cout << "\n";
		std::cout << current_changes.fluid_data[2](13, 16, 16) << " ";
		std::cout << current_changes.fluid_data[2](14, 16, 16) << " ";
		std::cout << current_changes.fluid_data[2](15, 16, 16) << " ";
		std::cout << "\n";
		std::cout << current_changes.fluid_data[3](13, 16, 16) << " ";
		std::cout << current_changes.fluid_data[3](14, 16, 16) << " ";
		std::cout << current_changes.fluid_data[3](15, 16, 16) << " ";
		std::cout << "\n";
		std::cout << current_changes.fluid_data[4](12, 16, 16) << " ";
		std::cout << current_changes.fluid_data[4](13, 16, 16) << " ";
		std::cout << current_changes.fluid_data[4](14, 16, 16) << " ";
		std::cout << current_changes.fluid_data[4](15, 16, 16) << " ";
		std::cout << "\n";
	}

	return 0.0;
}