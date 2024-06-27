#include "solver/Riemann_solvers.hpp"

HLL_solver::HLL_solver(std::size_t num_fields_) : Riemann_solver(num_fields_) { epsilon = 1.e-50; }

void HLL_solver::get_num_flux(fluid_cell &fluid_left_cell, fluid_cell &fluid_right_cell, const std::vector<double> &phys_flux_left_cell,
                              const std::vector<double> &phys_flux_right_cell, std::vector<double> &num_flux, double v_char_slowest, double v_char_fastest) {

	// Apply HLL fluxes
	if (v_char_slowest > 0.0) {

		for (std::size_t i_field = 0; i_field < num_fields; i_field++) {
			num_flux[i_field] = phys_flux_left_cell[i_field];
		}

	} else if (v_char_fastest < 0.0) {

		for (std::size_t i_field = 0; i_field < num_fields; i_field++) {
			num_flux[i_field] = phys_flux_right_cell[i_field];
		}

	} else {

		for (std::size_t i_field = 0; i_field < num_fields; i_field++) {
			double factor = 1.0 / (v_char_fastest - v_char_slowest + epsilon);

			num_flux[i_field] = (v_char_fastest * phys_flux_left_cell[i_field] - v_char_slowest * phys_flux_right_cell[i_field] +
			                     v_char_fastest * v_char_slowest * (fluid_right_cell.fluid_data[i_field] - fluid_left_cell.fluid_data[i_field])) *
			                    factor;
		}
	}
}
