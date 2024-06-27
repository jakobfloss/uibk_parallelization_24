#include "setup/grid.hpp"

#include <cassert>
#include <iostream>

grid_3D::grid_3D(std::vector<double> &lower_boundary, std::vector<double> &upper_boundary, std::vector<int> &num_cells, int num_ghostcells) {

	dim = 3;
	delta.resize(3);

	std::cout << " Constructor\n";

	this->num_cells = num_cells;
	this->num_ghostcells = num_ghostcells;

	std::cout << " axes\n";

	make_axes(lower_boundary, upper_boundary);
}

int grid_3D::get_num_cells(int i_dir) const { return num_cells[i_dir]; }

void grid_3D::make_axes(std::vector<double> &lower_boundary, std::vector<double> &upper_boundary) {
	std::cout << "wtf\n";

	const grid_1D::GridType type_x_grid = grid_1D::GridType::linear;
	x_grid.make_grid(type_x_grid, lower_boundary[0], upper_boundary[0], num_cells[0], num_ghostcells);

	const grid_1D::GridType type_y_grid = grid_1D::GridType::linear;
	y_grid.make_grid(type_y_grid, lower_boundary[1], upper_boundary[1], num_cells[1], num_ghostcells);

	const grid_1D::GridType type_z_grid = grid_1D::GridType::linear;
	z_grid.make_grid(type_z_grid, lower_boundary[2], upper_boundary[2], num_cells[2], num_ghostcells);

	// // Test x-axis
	// std::cout << " X-Grid: \n";
	// for(int i_x=-num_ghostcells; i_x<num_cells[0]+num_ghostcells; ++i_x) {
	//     std::cout << " Writing point " << i_x << "\n";
	//     std::cout << i_x << " " << x_grid.get_center(i_x) << "\n";
	// }
}