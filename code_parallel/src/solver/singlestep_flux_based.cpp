#include "solver/finite_volume_solver.hpp"

double finite_volume_solver::singlestep(grid_3D &spatial_grid, fluid &current_fluid, fluid &current_changes) {

	size_t num_fields = current_fluid.get_number_fields();

	for (int iz = 0; iz < spatial_grid.get_num_cells(2); ++iz) {
		for (int iy = 0; iy < spatial_grid.get_num_cells(1); ++iy) {
			for (int ix = 0; ix < spatial_grid.get_num_cells(0); ++ix) {

				// compute physical fluxes

				// Compute characteristic velocities from point values

				// Compute numerical fluxes from point values and physical fluxes

				// double num_flux_x_plus = 0.0;
				// double num_flux_x_minus = 0.0;

				// double num_flux_y_plus = 0.0;
				// double num_flux_y_minus = 0.0;

				// double num_flux_z_plus = 0.0;
				// double num_flux_z_minus = 0.0;

				// Compute changes for cell ix, iy, iz
				std::vector<double> num_flux_left_x(num_fields);
				std::vector<double> num_flux_left_y(num_fields);
				std::vector<double> num_flux_left_z(num_fields);

				// Apply fluxes to affected cells:
				for (size_t i_field = 0; i_field < num_fields; ++i_field) {
					current_changes.fluid_data[i_field](ix, iy, iz) += num_flux_left_x[i_field] / spatial_grid.x_grid.get_dx();
					current_changes.fluid_data[i_field](ix - 1, iy, iz) -= num_flux_left_x[i_field] / spatial_grid.x_grid.get_dx();

					current_changes.fluid_data[i_field](ix, iy, iz) += num_flux_left_y[i_field] / spatial_grid.y_grid.get_dx();
					current_changes.fluid_data[i_field](ix, iy - 1, iz) -= num_flux_left_y[i_field] / spatial_grid.y_grid.get_dx();

					current_changes.fluid_data[i_field](ix, iy, iz) += num_flux_left_z[i_field] / spatial_grid.z_grid.get_dx();
					current_changes.fluid_data[i_field](ix, iy, iz - 1) -= num_flux_left_z[i_field] / spatial_grid.z_grid.get_dx();
				}
			}
		}
	}

	return 0.0;
}