#ifndef FLUID_HPP
#define FLUID_HPP

#include "core/config.hpp"
#include "setup/grid.hpp"
#include "util/matrix.hpp"

#include <map>
#include <memory>
#include <vector>

class base_fluid {

	enum class FieldName { density, v_x, v_y, v_z, energy, tracer, none };

public:
	base_fluid();
	base_fluid(const parallelisation::FluidType &type);
	void setup();
	int get_field_index(const FieldName &field);
	int get_index_density() const;
	int get_index_v_x() const;
	int get_index_v_y() const;
	int get_index_v_z() const;
	int get_index_energy() const;
	int get_index_tracer() const;
	int get_num_fields() const;
	bool is_adiabatic() const;
	void set_characteristic();
	void set_conservative();
	bool is_conservative() const;
	parallelisation::FluidType get_fluid_type() const;

private:
	parallelisation::FluidType type;
	std::map<FieldName, int> map_fields;
	int index_density, index_v_x, index_v_y, index_v_z;
	int index_energy, index_tracer;
	bool uses_characteristic_quantities;
};

class fluid_cell : public base_fluid {
public:
	fluid_cell(const parallelisation::FluidType &type);

	void setup();

	/**
	 * Array holding fluid data
	 */
	std::vector<double> fluid_data;

private:
};

class fluxes_cell : base_fluid {
public:
	fluxes_cell(const parallelisation::FluidType &type);

	void setup();

	/**
	 * Array holding fluxes
	 */
	std::vector<double> flux_data;
};

class fluid : public base_fluid {
public:
	fluid(const parallelisation::FluidType &type);

	void setup(grid_3D &spatial_grid);
	/**
	 * Array of matrices holding the actual data
	 */
	std::vector<matrix<double, 3>> fluid_data;

	/**
	 * Get number of fields stored in fluid_data
	 */
	size_t get_number_fields() const;
	void get_fluid_cell(fluid_cell &local_fluid, int index_x, int index_y, int index_z) const;
	void set_cell_values(const fluid_cell &local_fluid, int index_x, int index_y, int index_z);
	void get_fluid_cell_raw(fluid_cell &local_fluid, int index_1D) const;
	void set_fluid_cell_raw(fluid_cell &local_fluid, int index_1D);

private:
};

#endif