#include "solver/limiter.hpp"

#include <algorithm>
#include <iostream>
#include <math.h>

limiter_base::limiter_base() {}

limiter_minmod::limiter_minmod(double theta) { this->theta = theta; }

double limiter_minmod::compute(double first, double second, double third) {

	// TBD by students
	signed char sign = 1;

	// flip sign of all values if first is negative, store sign
	if (first<0) {
		first = -first;
		second = -second;
		third = -third;
		sign = -1;
	}

	// not all values have the same sign -> return 0
	if ((second < 0) or (third < 0)) {
		return 0;
	}

	// find min and restore sign
	return sign * fmin(fmin(first, second), third);
}