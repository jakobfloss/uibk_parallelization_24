#include "setup/mpi_handler.hpp"
#include <iostream>


mpi_handler::mpi_handler(const std::vector<int> &_num_tasks) {
	// Determine the rank of the current task
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Get number of ranks from MPI
	int ntasks;
	MPI_Comm_size(MPI_COMM_WORLD, &ntasks);


	num_tasks.resize(_num_tasks.size());
	for(size_t i_dim=0; i_dim<_num_tasks.size(); ++i_dim) {
		num_tasks[i_dim] = _num_tasks[i_dim];
	}

	// Check the correct total number of dimensional sub-tasks
	if(ntasks != num_tasks[0]*num_tasks[1]*num_tasks[2]){
		std::cerr << " Wrong number of processes " << std::endl;
		std::cout << ntasks << " " << num_tasks[0]*num_tasks[1]*num_tasks[2] << std::endl;
		MPI_Finalize();
		exit(51);
	}

	if(rank==0) {
		std::cout << " Running in parallel with " << ntasks << " tasks \n";
	}

	// Create a Cartesian MPI communicator
	// Grid is not periodic
	int periods[3] = {false, false, false};
	int reorder = false;
	// If all is okay: Create new communicator "comm3d"
	MPI_Cart_create(MPI_COMM_WORLD, 3, &num_tasks[0], periods, reorder, &comm3D);

	//   Determine rank again for cartesian communicator -> overwrite rank
	MPI_Comm_rank(comm3D, &rank);
	std::cout << " Created new tasks " << rank << "\n";

	// Obtain other ranks:
	// Syntax: comm3D, shift direction, displacement, source, destination
	MPI_Cart_shift(comm3D, 0, 1, &left , &right);
	MPI_Cart_shift(comm3D, 1, 1, &front, &back);
	MPI_Cart_shift(comm3D, 2, 1, &bottom, &top);

	// Get coordinates of local rank
	coords.resize(3);
	int coords_local[3];
	MPI_Cart_coords(comm3D, rank, 3, coords_local);
	for(size_t i_dim=0; i_dim<_num_tasks.size(); ++i_dim) {
		coords[i_dim] = coords_local[i_dim];
	}
	std::cout << " MPI coordinates for rank " << rank << " -> " << coords[0] << " " << coords[1] << " " << coords[2] << "\n";


}


grid_3D mpi_handler::make_local_grid(const grid_3D &global_grid) {

	// Obtain properties of global grid
	std::vector<int> num_cells_global(3);
	// TBD by students
	num_cells_global[0] = global_grid.get_num_cells(0);
	num_cells_global[1] = global_grid.get_num_cells(1);
	num_cells_global[2] = global_grid.get_num_cells(2);

	std::vector<double> bound_low_global(3);
	// TBD by students
	bound_low_global[0] = global_grid.x_grid.get_left(0);
	bound_low_global[1] = global_grid.y_grid.get_left(0);
	bound_low_global[2] = global_grid.z_grid.get_left(0);

	std::vector<double> bound_up_global(3);
	// TBD by students
	bound_up_global[0] = global_grid.x_grid.get_left(num_cells_global[0]);
	bound_up_global[1] = global_grid.y_grid.get_left(num_cells_global[1]);
	bound_up_global[2] = global_grid.z_grid.get_left(num_cells_global[2]);


	std::vector<double> size_cell(3);
	size_cell[0] = global_grid.x_grid.get_dx();
	size_cell[1] = global_grid.y_grid.get_dx();
	size_cell[2] = global_grid.z_grid.get_dx();


	// Start making a new grid
	std::vector<int> num_cells_local(3);
	std::vector<double> bound_low_local(3), bound_up_local(3);


	// Now, compute the local number of cells in each dimension
	for(int i_dim=0; i_dim<3; ++i_dim) {
		// First check if division is possible
		if(num_cells_global[i_dim] % num_tasks[i_dim] == 0) {
			// TBD by students
			num_cells_local[i_dim] = num_cells_global[i_dim] / num_tasks[i_dim];
		} else {
			std::cerr << " Only considering homogeneous distribution \n";
			exit(3);
		}
	}

	std::vector<double> spatial_shift(3);
	shift_cells.resize(3);
	// Get shift of cells
	for(int i_dim=0; i_dim<3; ++i_dim) {
		shift_cells[i_dim] = num_cells_local[i_dim] * coords[i_dim];
		spatial_shift[i_dim] = size_cell[i_dim] * shift_cells[i_dim];
		// Compute lower boundary for current task in current direction
		bound_low_local[i_dim] = bound_low_global[i_dim] + spatial_shift[i_dim];
		// Compute upper boundary for current task in current dimension
		// TBD by students
		bound_up_local[i_dim] = bound_low_local[i_dim] + size_cell[i_dim] * num_cells_local[i_dim];
	}

	// std::cout << " Local sizes " << num_cells_local[0] << " " << num_cells_local[1] << " " << num_cells_local[2] << "\n";
	// std::cout << " spatial shift " << rank << " " << spatial_shift[0] << " " << spatial_shift[1] << " " << spatial_shift[2] << "\n";
	// std::cout << " cell sizes " << size_cell[0] << " " << size_cell[1] << " " << size_cell[2] << "\n";

	std::cout << " Properties of grid " << rank << " " << bound_low_local[0] << " " << bound_low_local[1] << " " << bound_low_local[2] << " ";
	std::cout << bound_up_local[0] << " " << bound_up_local[1] << " " << bound_up_local[2] << " " << "\n";


	// With this, we can create a local grid
	// TBD by students.
	grid_3D local_grid(bound_low_local, bound_up_local, num_cells_local, 2);

	return local_grid;
}



int mpi_handler::get_rank() const {
	return rank;
}


int mpi_handler::get_left() const {
	return left;
}

int mpi_handler::get_right() const {
	return right;
}

int mpi_handler::get_front() const {
	return front;
}

int mpi_handler::get_back() const {
	return back;
}

int mpi_handler::get_top() const {
	return top;
}

int mpi_handler::get_bottom() const {
	return bottom;
}


int mpi_handler::get_num_tasks(int dir) const {
	return num_tasks[dir];
}

int mpi_handler::get_coords(int dir) const {
	return coords[dir];
}

int mpi_handler::get_shift_cells(int dir) const {
	return shift_cells[dir];
}

mpi_handler::mpi_handler(mpi_handler &other_handler) {
	num_tasks = other_handler.num_tasks;
	coords = other_handler.coords;
	shift_cells = other_handler.shift_cells;

	left = other_handler.left;
	right = other_handler.right;
	front = other_handler.front;
	back = other_handler.back;
	bottom = other_handler.bottom;
	top = other_handler.top;
}