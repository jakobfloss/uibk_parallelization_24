#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <array>
#include <memory>
#include <stddef.h>
#include <vector>

template <class T, int rank> class matrix;

template <class T, int dim> class matrix {
public:
	matrix();
	matrix(size_t *size_dim);
	matrix(int index_lo[dim], int index_hi[dim]);

	void resize(size_t *size_dim);
	void resize(int index_lo[dim], int index_hi[dim]);

	/** index operator, writing - 1D version */
	T &operator()(int i);
	/** index operator, reading - 1D version */
	T operator()(int i) const;
	/** index operator, writing - 2D version */
	T &operator()(int i, int j);
	/** index operator, reading - 2D version */
	T operator()(int i, int j) const;
	/** index operator, writing - 3D version */
	T &operator()(int i, int j, int k);
	/** index operator, reading - 3D version */
	T operator()(int i, int j, int k) const;

	/** Setter that accesses the global 1D array */
	void set_raw_data(size_t index_1D, T value);

	/** Getter that accesses the global 1D array */
	T get_raw_data(size_t index_1D) const;

	/**
	 * Read access to raw data
	 */
	// const T* raw_data() const;
	std::vector<T> get_raw_data() const;

	/**
	 * get lowest / highest index in direction i_dir
	 * @param i_dir direction
	 */
	int get_lowest(size_t i_dir) const;
	int get_highest(size_t i_dir) const;

	/**
	 * get total number of entries
	 */
	size_t get_size() const;

	/**
	 * get extent of all dimensions
	 */
	std::vector<size_t> get_dims() const;

	/**
	 * set all entries to zero
	 */
	void clear();

private:
	void reset_size(int index_lo[dim], int index_hi[dim]);
	std::vector<T> data;
	//    std::unique_ptr<T> data;
	std::vector<int> index_lo, index_hi, extent;
	size_t size;
};

#endif