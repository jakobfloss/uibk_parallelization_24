#include <bits/chrono.h>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <sys/time.h>
#include <tuple>
#include <vector>
#include <mpi.h>

// Include that allows to print result as an image
// Also, ignore some warnings that pop up when compiling this as C++ mode
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#pragma GCC diagnostic pop

constexpr int default_size_x = 1344;
constexpr int default_size_y = 768;

// RGB image will hold 3 color channels
constexpr int num_channels = 3;
// max iterations cutoff
constexpr int max_iterations = 10000;

#define IND(Y, X, SIZE_Y, SIZE_X, CHANNEL) (Y * SIZE_X * num_channels + X * num_channels + CHANNEL)

size_t index(int y, int x, int /*size_y*/, int size_x, int channel) {
	return y * size_x * num_channels + x * num_channels + channel;
}

using Image = std::vector<uint8_t>;

auto HSVToRGB(double H, const double S, double V) {
	if (H >= 1.0) {
		V = 0.0;
		H = 0.0;
	}

	const double step = 1.0 / 6.0;
	const double vh = H / step;

	const int i = (int)floor(vh);

	const double f = vh - i;
	const double p = V * (1.0 - S);
	const double q = V * (1.0 - (S * f));
	const double t = V * (1.0 - (S * (1.0 - f)));
	double R = 0.0;
	double G = 0.0;
	double B = 0.0;

	// clang-format off
	switch (i) {
	case 0: { R = V; G = t; B = p; break; }
	case 1: { R = q; G = V; B = p; break; }
	case 2: { R = p; G = V; B = t; break; }
	case 3: { R = p; G = q; B = V; break; }
	case 4: { R = t; G = p; B = V; break; }
	case 5: { R = V; G = p; B = q; break; }
	}
	// clang-format on

	return std::make_tuple(R, G, B);
}

void calcMandelbrot(Image &image, int size_x, int size_y) {
	int mpi_size;
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	auto time_start = std::chrono::high_resolution_clock::now();

	const float left = -2.5, right = 1;
	const float bottom = -1, top = 1;

	// TODOs for MPI parallelization
	// 1) domain decomposition
	//   - decide how to split the image into multiple parts
	//   - ensure every rank is computing its own part only
	// 2) result aggregation
	//   - aggregate the individual parts of the ranks into a single, complete image on the root rank (rank 0)

	// 1) separate vertically (along y axis) into subdomains

	// calculate the size along x
	int dom_size_y = size_y / mpi_size;

	// distribute remainder along  first ranks
	int dom_y_min = rank * dom_size_y;
	int dom_y_max = dom_y_min + dom_size_y;

	std::cout << "dom_y_min: " << dom_y_min << ", dom_y_max: " << dom_y_max << std::endl;

	for (int pixel_y = dom_y_min; pixel_y < dom_y_max; pixel_y++) {
		// scale y pixel into mandelbrot coordinate system
		const float cy = (pixel_y / (float)size_y) * (top - bottom) + bottom;
		for (int pixel_x = 0; pixel_x < size_x; pixel_x++) {
			// scale x pixel into mandelbrot coordinate system
			const float cx = (pixel_x / (float)size_x) * (right - left) + left;
			float x = 0;
			float y = 0;
			int num_iterations = 0;

			// Check if the distance from the origin becomes
			// greater than 2 within the max number of iterations.
			while ((x * x + y * y <= 2 * 2) && (num_iterations < max_iterations)) {
				float x_tmp = x * x - y * y + cx;
				y = 2 * x * y + cy;
				x = x_tmp;
				num_iterations += 1;
			}

			// Normalize iteration and write it to pixel position
			double value = fabs((num_iterations / (float)max_iterations)) * 200;

			auto [red, green, blue] = HSVToRGB(value, 1.0, 1.0);

			int channel = 0;
			image[index(pixel_y, pixel_x, size_y, size_x, channel++)] = (uint8_t)(red * UINT8_MAX);
			image[index(pixel_y, pixel_x, size_y, size_x, channel++)] = (uint8_t)(green * UINT8_MAX);
			image[index(pixel_y, pixel_x, size_y, size_x, channel++)] = (uint8_t)(blue * UINT8_MAX);
		}
	}


	std::cout << "Domain calculated succesfully, merging calculated images" << std::endl;
	fflush(stdout);
	MPI_Barrier(MPI_COMM_WORLD);

	int send_size = num_channels * size_x * dom_size_y;
	int recv_size = send_size;
	Image recv_buff(recv_size * mpi_size);

	MPI_Gather(&image[rank * send_size], send_size, MPI_UINT8_T, 
	&recv_buff[0], recv_size, MPI_UINT8_T,
	0, MPI_COMM_WORLD);

	if (rank == 0){
		image = recv_buff;
		auto time_end = std::chrono::high_resolution_clock::now();
		auto time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
		
		std::cout << "Mandelbrot set calculation for " << size_x << "x" << size_y << " took: " << time_elapsed << " ms." << std::endl;
	}
}

int main(int argc, char **argv) {

	MPI_Init(&argc, &argv);

	int mpi_size;
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	std::cout << "I am rank " << rank << " of " << mpi_size << " ranks" << std::endl;
	
	int size_x = default_size_x;
	int size_y = default_size_y;

	if (argc == 3) {
		size_x = atoi(argv[1]);
		size_y = atoi(argv[2]);
		if (rank == 0) {
			std::cout << "Using size " << size_x << "x" << size_y << std::endl;
		}
	} else if (rank == 0) {
		std::cout << "No arguments given, using default size " << size_x << "x" << size_y << std::endl;
	}

	// if not possible to divide even crash
	if ((rank == 0) && (size_x % mpi_size != 0)) {
		std::cout << "ERROR: Wrong number of processes" << std::endl;
		std::cout << "Please make sure size_x is divisible by process count" << std::endl;
		std::cout << "(size_x: " << size_x << ")" << std::endl;
		return EXIT_FAILURE;
	}

	Image image(num_channels * size_x * size_y);

	calcMandelbrot(image, size_x, size_y);

	if (rank == 0) {
		constexpr int stride_bytes = 0;
		stbi_write_png("mandelbrot_mpi_gather.png", size_x, size_y, num_channels, image.data(), stride_bytes);
	}

	MPI_Finalize();

	return EXIT_SUCCESS;
}
