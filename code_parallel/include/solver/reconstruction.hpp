#ifndef RECONSTRUCTION_HPP
#define RECONSTRUCTION_HPP

#include "core/config.hpp"
#include "setup/fluid.hpp"
#include "solver/limiter.hpp"

class reconstruction {
public:
	reconstruction();
	virtual void compute_point_values(const fluid &fluid3D, fluid_cell &values_left, fluid_cell &values_right, const parallelisation::direction &local_direction,
	                                  int index_x, int index_y, int index_z) = 0;
	void get_derivatives(const fluid &fluid, fluid_cell &derivatives, const parallelisation::direction &local_direction, int index_x, int index_y, int index_z);

protected:
	limiter_minmod limiter;
};

class reconsctruction_second_order : public reconstruction {
public:
	reconsctruction_second_order(const fluid &fluid3D);
	void compute_point_values(const fluid &fluid3D, fluid_cell &values_left, fluid_cell &values_right, const parallelisation::direction &local_direction,
	                          int index_x, int index_y, int index_z);

private:
	fluid_cell derivatives;
};

#endif
