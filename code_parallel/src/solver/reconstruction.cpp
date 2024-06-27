#include "solver/reconstruction.hpp"
#include "core/config.hpp"
#include "solver/limiter.hpp"

reconstruction::reconstruction() : limiter(1.0) {}

void reconstruction::get_derivatives(const fluid &fluid3D, fluid_cell &derivatives, const parallelisation::direction &local_direction, int index_x, int index_y,
                                     int index_z) {

	size_t num_fields = fluid3D.get_number_fields();

	for (size_t index_field = 0; index_field < num_fields; ++index_field) {
		double delta_left(0.0), delta_right(0.0);
		if (local_direction == parallelisation::direction::x) {
			delta_left = fluid3D.fluid_data[index_field](index_x, index_y, index_z) - fluid3D.fluid_data[index_field](index_x - 1, index_y, index_z);
			delta_right = fluid3D.fluid_data[index_field](index_x + 1, index_y, index_z) - fluid3D.fluid_data[index_field](index_x, index_y, index_z);
		} else if (local_direction == parallelisation::direction::y) {
			delta_left = fluid3D.fluid_data[index_field](index_x, index_y, index_z) - fluid3D.fluid_data[index_field](index_x, index_y - 1, index_z);
			delta_right = fluid3D.fluid_data[index_field](index_x, index_y + 1, index_z) - fluid3D.fluid_data[index_field](index_x, index_y, index_z);
		} else {
			delta_left = fluid3D.fluid_data[index_field](index_x, index_y, index_z) - fluid3D.fluid_data[index_field](index_x, index_y, index_z - 1);
			delta_right = fluid3D.fluid_data[index_field](index_x, index_y, index_z + 1) - fluid3D.fluid_data[index_field](index_x, index_y, index_z);
		}
		double delta_central = (delta_left + delta_right)*0.5;

		// Now, apply the limiter
		double derivative = limiter.compute(delta_left, delta_central, delta_right);
		derivatives.fluid_data[index_field] = derivative;
	}
}

reconsctruction_second_order::reconsctruction_second_order(const fluid &fluid3D) : derivatives(fluid3D.get_fluid_type()) {}

void reconsctruction_second_order::compute_point_values(const fluid &fluid3D, fluid_cell &values_left, fluid_cell &values_right,
                                                        const parallelisation::direction &local_direction, int index_x, int index_y, int index_z) {

	// Start by computing the derivatives:
	get_derivatives(fluid3D, derivatives, local_direction, index_x, index_y, index_z);

	// Now, compute point values
	for (size_t index_field = 0; index_field < fluid3D.get_number_fields(); ++index_field) {
		double value_local = fluid3D.fluid_data[index_field](index_x, index_y, index_z);

		values_left.fluid_data[index_field] = value_local - 0.5 * derivatives.fluid_data[index_field];
		values_right.fluid_data[index_field] = value_local + 0.5 * derivatives.fluid_data[index_field];
	}
}
