#ifndef RIEMANN_SOLVERS_HPP
#define RIEMANN_SOLVERS_HPP

#include "setup/fluid.hpp"

#include <vector>

class Riemann_solver {
public:
	Riemann_solver(std::size_t num_fields_);
	virtual ~Riemann_solver();
	virtual void get_num_flux(fluid_cell &fluid_left_cell, fluid_cell &fluid_right_cell, const std::vector<double> &phys_flux_left_cell,
	                          const std::vector<double> &phys_flux_right_cell, std::vector<double> &num_flux, double v_char_slowest, double v_char_fastest) = 0;

protected:
	std::size_t num_fields;
};

class HLL_solver : public Riemann_solver {
public:
	HLL_solver(std::size_t num_fields_);
	void get_num_flux(fluid_cell &fluid_left_cell, fluid_cell &fluid_right_cell, const std::vector<double> &phys_flux_left_cell,
	                  const std::vector<double> &phys_flux_right_cell, std::vector<double> &num_flux, double v_char_slowest, double v_char_fastest);

private:
	double epsilon;
};

#endif