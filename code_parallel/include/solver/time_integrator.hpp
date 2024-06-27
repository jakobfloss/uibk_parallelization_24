#ifndef TIME_INTEGRATOR_HPP
#define TIME_INTEGRATOR_HPP

#include "setup/fluid.hpp"
#include "setup/grid.hpp"

class RungeKutta2 {
public:
	RungeKutta2(const grid_3D &grid, const fluid &fluid3D);
	void save_data(const fluid &fluid3D);
	void do_sub_step(const grid_3D &grid, const fluid &changes, fluid &fluid3D, double delta_t, int sub_step);
	int get_number_substeps() const;

private:
	grid_3D simulation_grid;
	fluid fluid_storage;
};

#endif