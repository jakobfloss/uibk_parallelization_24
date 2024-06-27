#include "core/config.hpp"
#include "setup/fluid.hpp"
#include "setup/physics.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>

// Choice of accuracy requirement for passing a test
const double eps = 1.e-8;

class TestLimiter : public ::testing::Test {
protected:
	virtual void SetUp() { ptr_physics = std::make_unique<physics>(); }
	virtual void TearDown() {}

	std::unique_ptr<physics> ptr_physics;
};

TEST_F(TestLimiter, variable_transforms) {
	std::cout << " Testing computation of thermal pressure\n";

	fluid_cell quantities_local(parallelisation::FluidType::adiabatic);
	quantities_local.fluid_data[quantities_local.get_index_density()] = 2.0;
	quantities_local.fluid_data[quantities_local.get_index_v_x()] = 2.5;
	quantities_local.fluid_data[quantities_local.get_index_v_y()] = 4.5;
	quantities_local.fluid_data[quantities_local.get_index_v_z()] = 8.5;
	quantities_local.fluid_data[quantities_local.get_index_energy()] = 0.2;
	quantities_local.fluid_data[quantities_local.get_index_tracer()] = -3.2;

	std::cout << " Testing computation of pressure\n";
	double pressure = ptr_physics->get_pressure(quantities_local);

	EXPECT_NEAR(pressure, 0.08, eps);

	std::cout << " Testing computation of total energy\n";
	double e_total = ptr_physics->get_e_total(quantities_local);

	EXPECT_NEAR(e_total, 98.95, eps);

	std::cout << " Testing transform to conservative variables\n";
	ptr_physics->transform_characteristic_to_conservative(quantities_local);
	EXPECT_NEAR(quantities_local.fluid_data[quantities_local.get_index_density()], 2.0, eps);
	EXPECT_NEAR(quantities_local.fluid_data[quantities_local.get_index_v_x()], 5.0, eps);
	EXPECT_NEAR(quantities_local.fluid_data[quantities_local.get_index_v_y()], 9.0, eps);
	EXPECT_NEAR(quantities_local.fluid_data[quantities_local.get_index_v_z()], 17.0, eps);
	EXPECT_NEAR(quantities_local.fluid_data[quantities_local.get_index_energy()], 98.95, eps);
	EXPECT_NEAR(quantities_local.fluid_data[quantities_local.get_index_tracer()], -3.2, eps);

	std::cout << " Testing transform to characteristic variables\n";
	ptr_physics->transform_conservative_to_characteristic(quantities_local);
	EXPECT_NEAR(quantities_local.fluid_data[quantities_local.get_index_density()], 2.0, eps);
	EXPECT_NEAR(quantities_local.fluid_data[quantities_local.get_index_v_x()], 2.5, eps);
	EXPECT_NEAR(quantities_local.fluid_data[quantities_local.get_index_v_y()], 4.5, eps);
	EXPECT_NEAR(quantities_local.fluid_data[quantities_local.get_index_v_z()], 8.5, eps);
	EXPECT_NEAR(quantities_local.fluid_data[quantities_local.get_index_energy()], 0.2, eps);
	EXPECT_NEAR(quantities_local.fluid_data[quantities_local.get_index_tracer()], -3.2, eps);
}

TEST_F(TestLimiter, physical_fluxes) {
	std::cout << " Testing physical fluxes\n";
	fluid_cell quantities_local(parallelisation::FluidType::adiabatic);
	quantities_local.fluid_data[quantities_local.get_index_density()] = 1.0;
	quantities_local.fluid_data[quantities_local.get_index_v_x()] = 2.5;
	quantities_local.fluid_data[quantities_local.get_index_v_y()] = 4.5;
	quantities_local.fluid_data[quantities_local.get_index_v_z()] = 8.5;
	quantities_local.fluid_data[quantities_local.get_index_energy()] = 0.2;
	quantities_local.fluid_data[quantities_local.get_index_tracer()] = -3.2;

	fluxes_cell fluxes_local(quantities_local.get_fluid_type());

	ptr_physics->get_physical_fluxes(quantities_local, fluxes_local, parallelisation::direction::x);

	EXPECT_NEAR(fluxes_local.flux_data[quantities_local.get_index_density()], 2.5, eps);
	EXPECT_NEAR(fluxes_local.flux_data[quantities_local.get_index_v_x()], 6.33, eps);
	EXPECT_NEAR(fluxes_local.flux_data[quantities_local.get_index_v_y()], 11.25, eps);
	EXPECT_NEAR(fluxes_local.flux_data[quantities_local.get_index_v_z()], 21.25, eps);
	EXPECT_NEAR(fluxes_local.flux_data[quantities_local.get_index_energy()], 124.1375, eps);
	EXPECT_NEAR(fluxes_local.flux_data[quantities_local.get_index_tracer()], -8.0, eps);

	ptr_physics->get_physical_fluxes(quantities_local, fluxes_local, parallelisation::direction::y);

	EXPECT_NEAR(fluxes_local.flux_data[quantities_local.get_index_density()], 4.5, eps);
	EXPECT_NEAR(fluxes_local.flux_data[quantities_local.get_index_v_x()], 11.25, eps);
	EXPECT_NEAR(fluxes_local.flux_data[quantities_local.get_index_v_y()], 20.33, eps);
	EXPECT_NEAR(fluxes_local.flux_data[quantities_local.get_index_v_z()], 38.25, eps);
	EXPECT_NEAR(fluxes_local.flux_data[quantities_local.get_index_energy()], 223.4475, eps);
	EXPECT_NEAR(fluxes_local.flux_data[quantities_local.get_index_tracer()], -14.4, eps);

	ptr_physics->get_physical_fluxes(quantities_local, fluxes_local, parallelisation::direction::z);

	EXPECT_NEAR(fluxes_local.flux_data[quantities_local.get_index_density()], 8.5, eps);
	EXPECT_NEAR(fluxes_local.flux_data[quantities_local.get_index_v_x()], 21.25, eps);
	EXPECT_NEAR(fluxes_local.flux_data[quantities_local.get_index_v_y()], 38.25, eps);
	EXPECT_NEAR(fluxes_local.flux_data[quantities_local.get_index_v_z()], 72.33, eps);
	EXPECT_NEAR(fluxes_local.flux_data[quantities_local.get_index_energy()], 422.0675, eps);
	EXPECT_NEAR(fluxes_local.flux_data[quantities_local.get_index_tracer()], -27.2, eps);
}

TEST_F(TestLimiter, characteristic_speeds) {
	std::cout << " Testing characteristic speeds\n";
	fluid_cell quantities_local(parallelisation::FluidType::adiabatic);
	quantities_local.fluid_data[quantities_local.get_index_density()] = 2.0;
	quantities_local.fluid_data[quantities_local.get_index_v_x()] = 2.5;
	quantities_local.fluid_data[quantities_local.get_index_v_y()] = 4.5;
	quantities_local.fluid_data[quantities_local.get_index_v_z()] = 8.5;
	quantities_local.fluid_data[quantities_local.get_index_energy()] = 0.2;
	quantities_local.fluid_data[quantities_local.get_index_tracer()] = -3.2;

	double pressure = ptr_physics->get_pressure(quantities_local);
	double c_sound = ptr_physics->get_sound_speed(quantities_local.fluid_data[quantities_local.get_index_density()], pressure);

	EXPECT_NEAR(c_sound * c_sound, 0.056, eps);

	double lambda_abs_max = ptr_physics->get_lambda_abs_max(quantities_local);

	double v_abs = sqrt(2.5 * 2.5 + 4.5 * 4.5 + 8.5 * 8.5);
	EXPECT_NEAR(lambda_abs_max, v_abs + sqrt(0.056), eps);

	fluid_cell quantities_previous(parallelisation::FluidType::adiabatic);
	quantities_previous.fluid_data[quantities_previous.get_index_density()] = 1.0;
	quantities_previous.fluid_data[quantities_previous.get_index_v_x()] = 1.5;
	quantities_previous.fluid_data[quantities_previous.get_index_v_y()] = 2.5;
	quantities_previous.fluid_data[quantities_previous.get_index_v_z()] = 4.5;
	quantities_previous.fluid_data[quantities_previous.get_index_energy()] = 0.1;
	quantities_previous.fluid_data[quantities_previous.get_index_tracer()] = -2.2;

	double lambda_min, lambda_max;
	ptr_physics->get_lambda_min_max(lambda_min, lambda_max, quantities_previous, quantities_local, parallelisation::direction::x);
	EXPECT_NEAR(lambda_max, 2.5 + sqrt(0.056), eps);
	//     EXPECT_NEAR( lambda_min, 0.0, eps);
	EXPECT_NEAR(lambda_min, 1.5 - sqrt(0.056), eps);

	ptr_physics->get_lambda_min_max(lambda_min, lambda_max, quantities_previous, quantities_local, parallelisation::direction::y);
	EXPECT_NEAR(lambda_max, 4.5 + sqrt(0.056), eps);
	//     EXPECT_NEAR( lambda_min, 0.0, eps);
	EXPECT_NEAR(lambda_min, 2.5 - sqrt(0.056), eps);

	ptr_physics->get_lambda_min_max(lambda_min, lambda_max, quantities_previous, quantities_local, parallelisation::direction::z);
	EXPECT_NEAR(lambda_max, 8.5 + sqrt(0.056), eps);
	//     EXPECT_NEAR( lambda_min, 0.0, eps);
	EXPECT_NEAR(lambda_min, 4.5 - sqrt(0.056), eps);

	quantities_previous.fluid_data[quantities_local.get_index_density()] = 1.0;
	quantities_previous.fluid_data[quantities_local.get_index_v_x()] = 1.5;
	quantities_previous.fluid_data[quantities_local.get_index_v_y()] = 2.5;
	quantities_previous.fluid_data[quantities_local.get_index_v_z()] = 4.5;
	quantities_previous.fluid_data[quantities_local.get_index_energy()] = 22.0;
	quantities_previous.fluid_data[quantities_local.get_index_tracer()] = -2.2;

	// p=0.4*22=8.8 c_s**2 = 8.8/1.0*1.4=12.32
	ptr_physics->get_lambda_min_max(lambda_min, lambda_max, quantities_previous, quantities_local, parallelisation::direction::x);
	EXPECT_NEAR(lambda_max, 1.5 + sqrt(12.32), eps);
	EXPECT_NEAR(lambda_min, 1.5 - sqrt(12.32), eps);

	ptr_physics->get_lambda_min_max(lambda_min, lambda_max, quantities_previous, quantities_local, parallelisation::direction::y);
	EXPECT_NEAR(lambda_max, 2.5 + sqrt(12.32), eps);
	EXPECT_NEAR(lambda_min, 2.5 - sqrt(12.32), eps);

	ptr_physics->get_lambda_min_max(lambda_min, lambda_max, quantities_previous, quantities_local, parallelisation::direction::z);
	EXPECT_NEAR(lambda_max, 8.5 + sqrt(0.056), eps); // left-handed value still higher
	EXPECT_NEAR(lambda_min, 4.5 - sqrt(12.32), eps);
}

int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
