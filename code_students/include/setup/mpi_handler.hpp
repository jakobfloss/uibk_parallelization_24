#pragma once

#include "setup/grid.hpp"

#include <mpi.h>
#include <vector>

class mpi_handler {
public:
    mpi_handler(const std::vector<int> &num_tasks);
    mpi_handler(mpi_handler &other_handler);
    MPI_Comm comm3D;
    int get_left() const;
    int get_right() const;
    int get_front() const;
    int get_back() const;
    int get_bottom() const;
    int get_top() const;
    int get_rank() const;
    int get_num_tasks(int dir) const;
    int get_coords(int dir) const;
    int get_shift_cells(int dir) const;
    grid_3D make_local_grid(const grid_3D &global_grid);
private:
    std::vector<int> num_tasks, coords, shift_cells;
    int left, right, front, back, bottom, top;
    int rank;
};