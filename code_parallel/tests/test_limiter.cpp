#include "solver/limiter.hpp"

#include <gtest/gtest.h>
#include <iostream>
#include <memory>

// Choice of accuracy requirement for passing a test
const double eps = 1.e-8;

class TestLimiter : public ::testing::Test {
protected:
	virtual void SetUp() { ptr_limiter = std::make_unique<limiter_minmod>(1.0); }
	virtual void TearDown() {}

	std::unique_ptr<limiter_minmod> ptr_limiter;
};

TEST_F(TestLimiter, same_sign) {
	// delta_right > 0.0
	double derivative = ptr_limiter->compute(1.5, 3.0, 2.0);
	EXPECT_NEAR(derivative, 1.5, eps);

	derivative = ptr_limiter->compute(2.0, 3.0, 1.5);
	EXPECT_NEAR(derivative, 1.5, eps);

	derivative = ptr_limiter->compute(3.0, 2.0, 1.5);
	EXPECT_NEAR(derivative, 1.5, eps);

	// delta_right < 0.0
	derivative = ptr_limiter->compute(-1.5, -3.0, -2.0);
	EXPECT_NEAR(derivative, -1.5, eps);

	ptr_limiter->compute(-2.0, -3.0, -1.5);
	EXPECT_NEAR(derivative, -1.5, eps);

	derivative = ptr_limiter->compute(-3.0, -2.0, -1.5);
	EXPECT_NEAR(derivative, -1.5, eps);
}

TEST_F(TestLimiter, different_sign) {
	double derivative = ptr_limiter->compute(1.5, 3.0, -2.0);
	EXPECT_NEAR(derivative, 0.0, eps);

	derivative = ptr_limiter->compute(-2.0, 3.0, 1.5);
	EXPECT_NEAR(derivative, 0.0, eps);

	derivative = ptr_limiter->compute(-3.0, -2.0, 1.5);
	EXPECT_NEAR(derivative, 0.0, eps);
}

int main(int argc, char **argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
