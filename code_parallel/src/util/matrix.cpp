#include "util/matrix.hpp"

#include <cassert>
#include <iostream>

template <class T, int dim> matrix<T, dim>::matrix() {
	data.resize(0);
	extent.resize(dim);
	index_lo.resize(dim);
	index_hi.resize(dim);
	size = 0;
	for (size_t i_dir = 0; i_dir < dim; ++i_dir) {
		extent[i_dir] = 0;
		index_lo[i_dir] = 0;
		index_hi[i_dir] = 0;
	}
}

template <class T, int dim> matrix<T, dim>::matrix(size_t *size_dim) {

	int ind_lo[dim] = {0};
	int ind_hi[dim];

	for (size_t i_dir = 0; i_dir < dim; ++i_dir) {
		ind_hi[i_dir] = size_dim[i_dir];
	}
	reset_size(ind_lo, ind_hi);
}

template <class T, int dim> matrix<T, dim>::matrix(int index_lo[dim], int index_hi[dim]) { reset_size(index_lo, index_hi); }

template <class T, int dim> void matrix<T, dim>::resize(size_t *size_dim) {
	int ind_lo[dim] = {0};
	int ind_hi[dim];

	for (size_t i_dir = 0; i_dir < dim; ++i_dir) {
		ind_hi[i_dir] = size_dim[i_dir];
	}
	reset_size(ind_lo, ind_hi);
}

template <class T, int dim> void matrix<T, dim>::resize(int _index_lo[dim], int index_hi[dim]) { reset_size(_index_lo, index_hi); }

template <class T, int dim> void matrix<T, dim>::reset_size(int index_lo[dim], int index_hi[dim]) {
	size = 1;
	extent.resize(dim);
	this->index_lo.resize(dim);
	this->index_hi.resize(dim);
	for (size_t i_dir = 0; i_dir < dim; ++i_dir) {
		this->index_lo[i_dir] = index_lo[i_dir];
		this->index_hi[i_dir] = index_hi[i_dir];
		extent[i_dir] = index_hi[i_dir] - index_lo[i_dir] + 1;
		size *= extent[i_dir];
	}

	data.resize(size);
}

template <class T, int dim> T &matrix<T, dim>::operator()(int i) {
	assert(dim == 1);
	assert(i >= index_lo[0] && i <= index_hi[0]);
	return data[i - index_lo[0]];
}

template <class T, int dim> T matrix<T, dim>::operator()(int i) const {
	assert(dim == 1);
	assert(i >= index_lo[0] && i <= index_hi[0]);
	return data[i - index_lo[0]];
}

template <class T, int dim> T &matrix<T, dim>::operator()(int i, int j) {
	assert(dim == 2);
	assert(i >= index_lo[0] && i <= index_hi[0]);
	assert(j >= index_lo[1] && j <= index_hi[1]);
	return data[(i - index_lo[0]) * extent[1] + (j - index_lo[1])];
}

template <class T, int dim> T matrix<T, dim>::operator()(int i, int j) const {
	assert(dim == 2);
	assert(i >= index_lo[0] && i <= index_hi[0]);
	assert(j >= index_lo[1] && j <= index_hi[1]);
	return data[(i - index_lo[0]) * extent[1] + (j - index_lo[1])];
}

template <class T, int dim> T &matrix<T, dim>::operator()(int i, int j, int k) {
	assert(dim == 3);
	assert(i >= index_lo[0] && i <= index_hi[0]);
	assert(j >= index_lo[1] && j <= index_hi[1]);
	assert(k >= index_lo[2] && k <= index_hi[2]);
	return data[((i - index_lo[0]) * extent[1] + (j - index_lo[1])) * extent[2] + (k - index_lo[2])];
}

template <class T, int dim> T matrix<T, dim>::operator()(int i, int j, int k) const {
	assert(dim == 3);
	assert(i >= index_lo[0] && i <= index_hi[0]);
	assert(j >= index_lo[1] && j <= index_hi[1]);
	assert(k >= index_lo[2] && k <= index_hi[2]);
	return data[((i - index_lo[0]) * extent[1] + (j - index_lo[1])) * extent[2] + (k - index_lo[2])];
}

template <class T, int dim> void matrix<T, dim>::set_raw_data(size_t index_1D, T value) {
	assert(index_1D < size);
	data[index_1D] = value;
}

template <class T, int dim> T matrix<T, dim>::get_raw_data(size_t index_1D) const {
	assert(index_1D < size);
	return data[index_1D];
}

template <class T, int dim> int matrix<T, dim>::get_lowest(size_t i_dir) const {
	assert(i_dir < dim); // index of direction must be within bounds of dim
	return index_lo[i_dir];
}

template <class T, int dim> int matrix<T, dim>::get_highest(size_t i_dir) const {
	assert(i_dir < dim); // index of direction must be within bounds of dim
	return index_hi[i_dir];
}

template <class T, int dim> size_t matrix<T, dim>::get_size() const { return size; }

template <class T, int dim> std::vector<size_t> matrix<T, dim>::get_dims() const {
	std::vector<size_t> dim_extents;
	for (size_t i_dir = 0; i_dir < dim; ++i_dir) {
		dim_extents.push_back(extent[i_dir]);
	}
	return dim_extents;
}

template <class T, int dim> void matrix<T, dim>::clear() {
	for (size_t i_entry = 0; i_entry < size; ++i_entry) {
		data[i_entry] = 0;
	}
}

// template<class T, int dim>
// const T* matrix<T,dim>::raw_data() const {
//     return &data[0];
// }

template <class T, int dim> std::vector<T> matrix<T, dim>::get_raw_data() const {
	std::vector<T> raw_data;
	raw_data = data;
	return raw_data;
}

// // Make instances of matrix:
template class matrix<double, 1>;
template class matrix<double, 2>;
template class matrix<double, 3>;