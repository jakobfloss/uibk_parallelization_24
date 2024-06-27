#include "setup/fluid.hpp"
#include "util/matrix.hpp"

#include <cassert>
#include <iostream>

base_fluid::base_fluid() {
	this->type = parallelisation::FluidType::none;
	set_characteristic();
}

base_fluid::base_fluid(const parallelisation::FluidType &type) {
	this->type = type;
	this->setup();
	set_characteristic();
}

void base_fluid::setup() {
	assert(this->type != parallelisation::FluidType::none);

	// Set up isothermal or adiabatic fluid:

	map_fields.clear(); // Make sure that map is empty

	map_fields.insert(std::pair<FieldName, int>(FieldName::density, 0));
	map_fields.insert(std::pair<FieldName, int>(FieldName::v_x, 1));
	map_fields.insert(std::pair<FieldName, int>(FieldName::v_y, 2));
	map_fields.insert(std::pair<FieldName, int>(FieldName::v_z, 3));

	if (type == parallelisation::FluidType::adiabatic) {
		map_fields.insert(std::pair<FieldName, int>(FieldName::energy, 4));
	}

	map_fields.insert(std::pair<FieldName, int>(FieldName::tracer, 5));

	// Set individual indices:
	index_density = get_field_index(FieldName::density);
	index_v_x = get_field_index(FieldName::v_x);
	index_v_y = get_field_index(FieldName::v_y);
	index_v_z = get_field_index(FieldName::v_z);

	if (type == parallelisation::FluidType::adiabatic) {
		index_energy = get_field_index(FieldName::energy);
	}

	index_tracer = get_field_index(FieldName::tracer);
}

int base_fluid::get_field_index(const FieldName &field) {
	assert(field != FieldName::none);

	if (map_fields.find(field) != map_fields.end()) {
		int map_index = map_fields.at(field);
		return map_index;
	} else {
		return -1;
	}
}

parallelisation::FluidType base_fluid::get_fluid_type() const { return type; }

void base_fluid::set_characteristic() { uses_characteristic_quantities = true; }

void base_fluid::set_conservative() { uses_characteristic_quantities = false; }

bool base_fluid::is_conservative() const { return !uses_characteristic_quantities; }

int base_fluid::get_index_density() const { return index_density; }

int base_fluid::get_index_v_x() const { return index_v_x; }

int base_fluid::get_index_v_y() const { return index_v_y; }

int base_fluid::get_index_v_z() const { return index_v_z; }

int base_fluid::get_index_energy() const {
	assert(type == parallelisation::FluidType::adiabatic);
	return index_energy;
}

int base_fluid::get_index_tracer() const { return index_tracer; }

int base_fluid::get_num_fields() const {
	// std::cerr << " This still needs to be tested\n" << map_fields.size() << "\n";
	return map_fields.size();
}

bool base_fluid::is_adiabatic() const {
	if (type == parallelisation::FluidType::adiabatic) {
		return true;
	} else {
		return false;
	}
}

fluid::fluid(const parallelisation::FluidType &type) : base_fluid(type) {}

void fluid::setup(grid_3D &spatial_grid) {

	// Get number of fields:
	size_t num_fields = get_num_fields();

	// Make fluid arrays
	// Append spatial grids
	for (size_t i_field = 0; i_field < num_fields; ++i_field) {
		fluid_data.push_back(matrix<double, 3>());
	}

	// fluid_data.push_back(matrix<double, 3>()); // density
	// fluid_data.push_back(matrix<double, 3>()); // velocity (x)
	// fluid_data.push_back(matrix<double, 3>()); // velocity (y)
	// fluid_data.push_back(matrix<double, 3>()); // velocity (z)
	// fluid_data.push_back(matrix<double, 3>()); // energy density
	// fluid_data.push_back(matrix<double, 3>()); // Tracer

	// Get extent of grid
	int index_low[3], index_hi[3];
	index_low[0] = spatial_grid.x_grid.get_index_lowest();
	index_hi[0] = spatial_grid.x_grid.get_index_highest();

	index_low[1] = spatial_grid.y_grid.get_index_lowest();
	index_hi[1] = spatial_grid.y_grid.get_index_highest();

	index_low[2] = spatial_grid.z_grid.get_index_lowest();
	index_hi[2] = spatial_grid.z_grid.get_index_highest();

	// Resize grids
	for (size_t i_field = 0; i_field < fluid_data.size(); ++i_field) {
		std::cout << " resizing field " << i_field << ":\n";
		fluid_data[i_field].resize(index_low, index_hi);
		std::cout << " -> " << fluid_data[i_field].get_size() << "\n";
	}
}

void fluid::get_fluid_cell(fluid_cell &local_fluid, int index_x, int index_y, int index_z) const {
	for (size_t i_field = 0; i_field < fluid_data.size(); ++i_field) {
		local_fluid.fluid_data[i_field] = fluid_data[i_field](index_x, index_y, index_z);
	}
	if (is_conservative()) {
		local_fluid.set_conservative();
	} else {
		local_fluid.set_characteristic();
	}
}

void fluid::set_cell_values(const fluid_cell &local_fluid, int index_x, int index_y, int index_z) {
	for (size_t i_field = 0; i_field < fluid_data.size(); ++i_field) {
		fluid_data[i_field](index_x, index_y, index_z) = local_fluid.fluid_data[i_field];
	}
}

void fluid::get_fluid_cell_raw(fluid_cell &local_fluid, int index_1D) const {
	for (size_t i_field = 0; i_field < fluid_data.size(); ++i_field) {
		local_fluid.fluid_data[i_field] = fluid_data[i_field].get_raw_data(index_1D);
	}
	if (is_conservative()) {
		local_fluid.set_conservative();
	} else {
		local_fluid.set_characteristic();
	}
}

void fluid::set_fluid_cell_raw(fluid_cell &local_fluid, int index_1D) {
	for (size_t i_field = 0; i_field < fluid_data.size(); ++i_field) {
		fluid_data[i_field].set_raw_data(index_1D, local_fluid.fluid_data[i_field]);
	}
}

size_t fluid::get_number_fields() const { return fluid_data.size(); }

fluid_cell::fluid_cell(const parallelisation::FluidType &type) : base_fluid(type) { setup(); }

void fluid_cell::setup() { fluid_data.resize(get_num_fields()); }

fluxes_cell::fluxes_cell(const parallelisation::FluidType &type) : base_fluid(type) { setup(); }

void fluxes_cell::setup() { flux_data.resize(get_num_fields()); }
