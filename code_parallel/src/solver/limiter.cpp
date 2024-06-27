#include "solver/limiter.hpp"

#include <algorithm>
#include <iostream>

limiter_base::limiter_base() {}

limiter_minmod::limiter_minmod(double theta) { this->theta = theta; }

double limiter_minmod::compute(double delta_left, double delta_central, double delta_right) {

	delta_left *= theta;
	delta_right *= theta;

	if (delta_left * delta_right > 0.0) {
		if (delta_right > 0.0) {
			return std::min(delta_right, std::min(delta_central, delta_left));
		} else if (delta_right < 0.0) {
			return std::max(delta_right, std::max(delta_central, delta_left));
		} else {
			std::cerr << " Error in minmod limiter\n";
			exit(3);
		}
	} else {
		return 0.0;
	}
}