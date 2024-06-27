#ifndef LIMITER_HPP
#define LIMITER_HPP

class limiter_base {
public:
	limiter_base();
	virtual double compute(double delta_left, double delta_central, double delta_right) = 0;
};

class limiter_minmod : public limiter_base {
public:
	limiter_minmod(double theta);
	double compute(double delta_left, double delta_central, double delta_right);

private:
	double theta;
};

#endif