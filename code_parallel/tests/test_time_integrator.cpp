#include "setup/grid.hpp"
#include "solver/time_integrator.hpp"

#include <gtest/gtest.h>
#include <iostream>
#include <memory>

// Choice of accuracy requirement for passing a test
const double eps = 1.e-8;

class TestTimeIntegrator : public ::testing::Test {
protected:
	virtual void SetUp() {

		num_cells_1D = 4;

		std::vector<double> bound_low(3), bound_up(3);
		bound_low[0] = 0.0;
		bound_low[1] = 0.0;
		bound_low[2] = 0.0;

		bound_up[0] = 1.0;
		bound_up[1] = 1.0;
		bound_up[2] = 1.0;

		std::vector<int> num_cells_3D(3);
		num_cells_3D[0] = num_cells_1D;
		num_cells_3D[1] = num_cells_1D;
		num_cells_3D[2] = num_cells_1D;

		grid = std::make_unique<grid_3D>(bound_low, bound_up, num_cells_3D, 0);

		std::cout << " Making fluid\n";
		hd_fluid = std::make_unique<fluid>(parallelisation::FluidType::adiabatic);
		std::cout << " setup of fluid\n";
		hd_fluid->setup(*grid);

		std::cout << " Making time stepper\n";
		time_stepper = std::make_unique<RungeKutta2>(*grid, *hd_fluid);
		std::cout << " Finished making time stepper\n";
	}
	virtual void TearDown() {}

	size_t num_cells_1D;
	std::unique_ptr<grid_3D> grid;
	std::unique_ptr<fluid> hd_fluid;
	std::unique_ptr<RungeKutta2> time_stepper;
};

TEST_F(TestTimeIntegrator, exponential) {
	std::cout << " Testing time integration of an exponential\n";
	fluid fluid_changes(hd_fluid->get_fluid_type());
	fluid_changes.setup(*grid);

	for(size_t ix=0; ix<num_cells_1D; ++ix) {
		for(size_t iy=0; iy<num_cells_1D; ++iy) {
			for(size_t iz=0; iz<num_cells_1D; ++iz) {
				hd_fluid->fluid_data[0](ix, iy, iz) = 1.0;
			}
		}
	}

	double lambda = -2.5;
	double delta_t = 0.2;

	for(size_t ix=0; ix<num_cells_1D; ++ix) {
		for(size_t iy=0; iy<num_cells_1D; ++iy) {
			for(size_t iz=0; iz<num_cells_1D; ++iz) {
	
				// RK step 0:
				fluid_changes.fluid_data[0](ix, iy, iz) = lambda * hd_fluid->fluid_data[0](ix, iy, iz);

			}
		}
	}

	time_stepper->do_sub_step(*grid, fluid_changes, *hd_fluid, delta_t, 0);

	for(size_t ix=0; ix<num_cells_1D; ++ix) {
		for(size_t iy=0; iy<num_cells_1D; ++iy) {
			for(size_t iz=0; iz<num_cells_1D; ++iz) {

				EXPECT_NEAR(hd_fluid->fluid_data[0](ix, iy, iz), 1.0 + delta_t * lambda, eps);

			}
		}
	}

	std::cout << " Result(RK step 1): " << hd_fluid->fluid_data[0](0, 0, 0) << "\n";

	for(size_t ix=0; ix<num_cells_1D; ++ix) {
		for(size_t iy=0; iy<num_cells_1D; ++iy) {
			for(size_t iz=0; iz<num_cells_1D; ++iz) {
	
				// RK step 1:
				fluid_changes.fluid_data[0](ix, iy, iz) = lambda * hd_fluid->fluid_data[0](ix, iy, iz);

			}
		}
	}

	time_stepper->do_sub_step(*grid, fluid_changes, *hd_fluid, delta_t, 1);

	for(size_t ix=0; ix<num_cells_1D; ++ix) {
		for(size_t iy=0; iy<num_cells_1D; ++iy) {
			for(size_t iz=0; iz<num_cells_1D; ++iz) {
	
				EXPECT_NEAR(hd_fluid->fluid_data[0](ix, iy, iz), 1.0 + delta_t * lambda + 0.5 * delta_t * delta_t * lambda * lambda, eps);

			}
		}
	}

	std::cout << " Result(RK step 2): " << hd_fluid->fluid_data[0](0, 0, 0) << "\n";
}

int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
