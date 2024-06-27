#include "setup/grid1D.hpp"

#include <cassert>
#include <iostream>

grid_1D::grid_1D() {
	is_set = false;
	my_type = GridType::none;
}

grid_1D::grid_1D(const GridType &type, double min, double max, size_t num_cells, int num_ghost_cells) { make_grid(type, min, max, num_cells, num_ghost_cells); }

void grid_1D::make_grid(const GridType &type, double min, double max, size_t num_cells, int num_ghost_cells) {

	my_type = type;
	if (type == GridType::linear) {
		build_lin_axis(min, max, num_cells, num_ghost_cells);
		delta_x = grid_left(1) - grid_left(0);
		inv_delta_x = 1. / delta_x;
	}

	is_set = true;
}

double grid_1D::get_center(int i_cell) const {
	assert(is_set);
	return grid_centers(i_cell);
}

double grid_1D::get_left(int i_cell) const {
	assert(is_set);
	return grid_left(i_cell);
}

int grid_1D::get_index_lowest() const {
	assert(is_set);
	return grid_centers.get_lowest(0);
}

int grid_1D::get_index_highest() const {
	assert(is_set);
	return grid_centers.get_highest(0);
}

double grid_1D::get_dx() const {
	assert(is_set);
	assert(my_type == GridType::linear);
	// return grid_left(1) - grid_left(0);
	return delta_x;
}

double grid_1D::get_inv_dx() const {
	assert(is_set);
	assert(my_type == GridType::linear);
	return inv_delta_x;
}

double grid_1D::get_dx(int i_cell) const {
	assert(is_set);
	return grid_left(i_cell + 1) - grid_left(i_cell);
}

double grid_1D::get_inv_dx(int i_cell) const {
	assert(is_set);
	return 1. / (grid_left(i_cell + 1) - grid_left(i_cell));
}

void grid_1D::build_lin_axis(double min, double max, size_t num_cells, int num_ghost_cells) {

	assert(min < max);            // Min must be smaller than max
	assert(num_cells > 1);        // size must be larger than a single cell
	assert(num_ghost_cells >= 0); // number of ghoscells must not be smaller than zero

	this->num_ghostcells = num_ghost_cells;

	int grid_beg = -num_ghostcells;
	int grid_end = num_cells + num_ghostcells;

	std::cout << " Making linear grid from ";
	std::cout << min << " " << max << " with " << num_cells << " cells\n";
	grid_left.resize(&grid_beg, &grid_end);

	const auto dx = (max - min) / static_cast<double>(num_cells);
	for (int i = grid_beg; i <= grid_end; ++i) {
		const auto value = min + dx * i;
		grid_left(i) = value;
	}

	// Compute positions of cell centers
	get_centers();
}

void grid_1D::get_centers() {
	int cen_grid_beg = grid_left.get_lowest(0);
	int cen_grid_end = grid_left.get_highest(0) - 1;
	grid_centers.resize(&cen_grid_beg, &cen_grid_end);

	for (int i = cen_grid_beg; i <= cen_grid_end; ++i) {
		const auto pos_left = grid_left(i);
		const auto pos_right = grid_left(i + 1);
		const auto pos_center = 0.5 * (pos_left + pos_right);
		grid_centers(i) = pos_center;
	}
}