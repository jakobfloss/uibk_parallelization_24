#ifndef GRID1D_HPP
#define GRID1D_HPP

#include "util/matrix.hpp"

class grid_1D {
public:
	/**
	 * List of available grid types
	 */
	enum class GridType { linear, none };

	grid_1D();

	/**
	 * Constructor which directly produces a grid
	 * @param type type of grid selected from GridType enum
	 * @param min left boundary of physical domain
	 * @param max right boundary of physical domain
	 * @param num_cells number of cells to be distributed between min and max
	 * @param num_ghost_cells number of boundary cells outside physical domain
	 */
	grid_1D(const GridType &type, double min, double max, size_t num_cells, int num_ghost_cells = 0);

	/**
	 * make a grid arrays
	 * @param type type of grid selected from GridType enum
	 * @param min left boundary of physical domain
	 * @param max right boundary of physical domain
	 * @param num_cells number of cells to be distributed between min and max
	 * @param num_ghost_cells number of boundary cells outside physical domain
	 */
	void make_grid(const GridType &type, double min, double max, size_t num_cells, int num_ghost_cells = 0);

	/**
	 * get position of cell center / left edge
	 * @param i_cell index of cell
	 */
	double get_center(int i_cell) const;
	double get_left(int i_cell) const;

	/**
	 * get first / last index of grid cells
	 */
	int get_index_lowest() const;
	int get_index_highest() const;

	/**
	 * Return size of grid cell for homogeneous grid
	 */
	double get_dx() const;

	/**
	 * Return inverse of size of grid cell for homogeneous grid
	 */
	double get_inv_dx() const;

	/**
	 * Return size of grid cell for arbitrary grid
	 */
	double get_dx(int i_cell) const;

	/**
	 * Return inverse of size of grid cell for arbitrary grid
	 */
	double get_inv_dx(int i_cell) const;

protected:
	int num_ghostcells;

private:
	void build_lin_axis(double min, double max, size_t num_cells, int num_ghost_cells = 0);
	/**
	 * get grid centers from left-handed grid boundareis
	 */
	void get_centers();

	matrix<double, 1> grid_centers, grid_left;

	double delta_x, inv_delta_x;

	GridType my_type;
	bool is_set;
};

#endif