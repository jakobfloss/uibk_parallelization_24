#include "core/config.hpp"
#include "setup/fluid.hpp"
#include "setup/physics.hpp"
#include "solver/Riemann_solvers.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>

// Choice of accuracy requirement for passing a test
const double eps = 1.e-8;

class TestRiemannSolvers : public ::testing::Test {
protected:
	virtual void SetUp() {
		fluid_cell hd_fluid(parallelisation::FluidType::adiabatic);
		Riemann_HLL = std::make_unique<HLL_solver>(hd_fluid.get_num_fields());
		ptr_physics = std::make_unique<physics>();
		fluid_type = parallelisation::FluidType::adiabatic;
	}
	virtual void TearDown() {}
	std::unique_ptr<HLL_solver> Riemann_HLL;
	std::unique_ptr<physics> ptr_physics;
	parallelisation::FluidType fluid_type;
};

TEST_F(TestRiemannSolvers, hll) {
	std::cout << " Testing the Riemann solver\n";
	fluid_cell quantities_previous(fluid_type);
	quantities_previous.fluid_data[quantities_previous.get_index_density()] = 1.0;
	quantities_previous.fluid_data[quantities_previous.get_index_v_x()] = 1.5;
	quantities_previous.fluid_data[quantities_previous.get_index_v_y()] = 2.5;
	quantities_previous.fluid_data[quantities_previous.get_index_v_z()] = 4.5;
	quantities_previous.fluid_data[quantities_previous.get_index_energy()] = 0.1;
	quantities_previous.fluid_data[quantities_previous.get_index_tracer()] = -2.2;

	fluid_cell quantities_local(fluid_type);
	quantities_local.fluid_data[quantities_local.get_index_density()] = 2.0;
	quantities_local.fluid_data[quantities_local.get_index_v_x()] = 2.5;
	quantities_local.fluid_data[quantities_local.get_index_v_y()] = 4.5;
	quantities_local.fluid_data[quantities_local.get_index_v_z()] = 8.5;
	quantities_local.fluid_data[quantities_local.get_index_energy()] = 0.2;
	quantities_local.fluid_data[quantities_local.get_index_tracer()] = -3.2;

	fluxes_cell fluxes_previous(fluid_type), fluxes_local(fluid_type);
	fluxes_previous.flux_data[quantities_local.get_index_density()] = 1.5;
	fluxes_previous.flux_data[quantities_local.get_index_v_x()] = 2.5;
	fluxes_previous.flux_data[quantities_local.get_index_v_y()] = 3.5;
	fluxes_previous.flux_data[quantities_local.get_index_v_z()] = 4.5;
	fluxes_previous.flux_data[quantities_local.get_index_energy()] = 5.5;
	fluxes_previous.flux_data[quantities_local.get_index_tracer()] = 6.5;

	fluxes_local.flux_data[quantities_local.get_index_density()] = 6.5;
	fluxes_local.flux_data[quantities_local.get_index_v_x()] = 7.5;
	fluxes_local.flux_data[quantities_local.get_index_v_y()] = 8.5;
	fluxes_local.flux_data[quantities_local.get_index_v_z()] = 9.5;
	fluxes_local.flux_data[quantities_local.get_index_energy()] = 10.5;
	fluxes_local.flux_data[quantities_local.get_index_tracer()] = 11.5;

	// Test Riemann fan tipped to the right
	double lambda_min_x = 2.0;
	double lambda_max_x = 3.0;

	fluxes_cell num_flux(quantities_local.get_fluid_type());
	Riemann_HLL->get_num_flux(quantities_previous, quantities_local, fluxes_previous.flux_data, fluxes_local.flux_data, num_flux.flux_data, lambda_min_x,
	                          lambda_max_x);

	EXPECT_NEAR(num_flux.flux_data[0], 1.5, eps);
	EXPECT_NEAR(num_flux.flux_data[3], 4.5, eps);

	std::cout << " Left-handed flux " << num_flux.flux_data[0] << "\n";

	// Test Riemann fan tipped to the left
	lambda_min_x = -3.0;
	lambda_max_x = -2.0;

	Riemann_HLL->get_num_flux(quantities_previous, quantities_local, fluxes_previous.flux_data, fluxes_local.flux_data, num_flux.flux_data, lambda_min_x,
	                          lambda_max_x);

	EXPECT_NEAR(num_flux.flux_data[1], 7.5, eps);
	EXPECT_NEAR(num_flux.flux_data[4], 10.5, eps);

	std::cout << " Right-handed flux " << num_flux.flux_data[1] << "\n";

	// Test local Riemann fan
	lambda_min_x = -2.0;
	lambda_max_x = 3.0;

	Riemann_HLL->get_num_flux(quantities_previous, quantities_local, fluxes_previous.flux_data, fluxes_local.flux_data, num_flux.flux_data, lambda_min_x,
	                          lambda_max_x);

	EXPECT_NEAR(num_flux.flux_data[2], 3.1, eps);
	EXPECT_NEAR(num_flux.flux_data[5], 9.7, eps);

	std::cout << " Riemann-fan-internal flux " << num_flux.flux_data[2] << "\n";
}

int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
