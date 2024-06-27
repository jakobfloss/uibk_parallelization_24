#ifndef GRID_HPP
#define GRID_HPP

#include "setup/grid1D.hpp"
#include "util/matrix.hpp"

class grid_3D {
public:
	grid_3D(std::vector<double> &lower_boundary, std::vector<double> &upper_boundary, std::vector<int> &num_cells, int num_ghostcells);
	int get_num_cells(int i_dir) const;
	grid_1D x_grid, y_grid, z_grid;

private:
	/**
	 * Compute positions of gridpoints in all directions
	 */
	void make_axes(std::vector<double> &lower_boundary, std::vector<double> &upper_boundary);
	matrix<double, 1> xCen, yCen, zCen;
	std::vector<double> delta;
	std::vector<int> num_cells;
	int dim, num_ghostcells;
};
#endif