#include "IO/data_storage.hpp"
#include "core/config.hpp"
#include "setup/fluid.hpp"
#include "setup/grid.hpp"
#include "solver/finite_volume_solver.hpp"
#include "solver/time_integrator.hpp"
#include "util/matrix.hpp"

#include <iostream>

int main() {
	std::cout << " Trying to create a matrix \n";
	size_t dims_2D[2] = {10, 10};
	size_t dims_1D[1] = {10};
	matrix<double, 1> my_matr_1D(dims_1D);
	matrix<double, 2> my_matr_2D(dims_2D);

	// my_matr_2D(2,3) = 22.2;
	// std::cout << " Values " << my_matr_2D(1,1) << " ";
	// std::cout << my_matr_2D(2,3) << "\n";

	std::vector<double> bound_low(3), bound_up(3);
	bound_low[0] = 0.0;
	bound_low[1] = 0.0;
	bound_low[2] = 0.0;

	bound_up[0] = 1.0;
	bound_up[1] = 1.0;
	bound_up[2] = 1.0;

	std::vector<int> num_cells(3);
	num_cells[0] = 40;
	num_cells[1] = 30;
	num_cells[2] = 20;

	grid_3D my_grid(bound_low, bound_up, num_cells, 2);

	// Now, I will create a HD fluid
	fluid hd_fluid(parallelisation::FluidType::adiabatic);
	hd_fluid.setup(my_grid);

	fluid hd_changes(hd_fluid.get_fluid_type());
	hd_changes.setup(my_grid);

	finite_volume_solver solver(hd_fluid);

	RungeKutta2 time_steper(my_grid, hd_fluid);

	double delta_t = 0.1;

	// Do two RK steps:
	solver.singlestep(my_grid, hd_fluid, hd_changes);
	time_steper.do_sub_step(my_grid, hd_changes, hd_fluid, delta_t, 0);
	solver.singlestep(my_grid, hd_fluid, hd_changes);
	time_steper.do_sub_step(my_grid, hd_changes, hd_fluid, delta_t, 1);

	// Testing IO
	data_storage my_storage("test.h5");
	std::vector<double> raw_data = hd_fluid.fluid_data[0].get_raw_data();
	std::vector<size_t> extent = hd_fluid.fluid_data[0].get_dims();
	my_storage.write_dataset(raw_data, extent, "Density");

	return 0;
}