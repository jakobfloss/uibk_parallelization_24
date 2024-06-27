#include "IO/data_storage.hpp"

#include <iostream>

data_storage::data_storage(std::string file_name) {
	this->file_name = file_name;
	hdf5file = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	hdf5group = H5Gcreate2(hdf5file, "/Data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Gclose(hdf5group);
	H5Fclose(hdf5file);
}

void data_storage::open_file() {
	// std::cerr << " open file (serial version)\n\n\n";
	hdf5file = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
	hdf5group = H5Gopen2(hdf5file, "/Data", H5P_DEFAULT);
}

void data_storage::close_file() {
	H5Gclose(hdf5group);
	H5Fclose(hdf5file);
}

template <typename T, typename> void data_storage::AddGlobalAttr(const std::string &AttrName, T AttrData) {
	// Open the file
	this->open_file();

	// little endian of size 1
	hid_t datatype = get_hdf5_data_type<T>();
	if (!std::is_same<T, std::string>::value) {
		// This is not allowed for strings
		H5Tset_order(datatype, H5T_ORDER_LE);
	}

	// Create dataspace
	hid_t AttrSpace = H5Screate(H5S_SCALAR);

	// Create Attribute
	hid_t info = H5Acreate2(hdf5group, AttrName.c_str(), datatype, AttrSpace, H5P_DEFAULT, H5P_DEFAULT);

	if (std::is_same<T, std::string>::value) {
		std::string *data = (std::string *)(&AttrData); // Workaround under c++11, since we dont have if constexpr
		const char *temp = data->c_str();
		H5Awrite(info, datatype, &temp);
	} else {
		H5Awrite(info, datatype, &AttrData);
	}

	// Close Attribute
	H5Aclose(info);
	// Close Dataspace
	H5Sclose(AttrSpace);

	close_file();
}

template void data_storage::AddGlobalAttr(const std::string &AttrName, int AttrData);
template void data_storage::AddGlobalAttr(const std::string &AttrName, float AttrData);
template void data_storage::AddGlobalAttr(const std::string &AttrName, double AttrData);

void data_storage::write_dataset(const std::vector<double> &raw_data, const std::vector<size_t> &extents, std::string dataset_name) {
	// Open the file
	open_file();

	int dataset_dimension = extents.size();
	std::vector<hsize_t> DimsData;
	// hsize_t DimsData[dataset_dimension];
	for (int i_dir = 0; i_dir < dataset_dimension; ++i_dir) {
		DimsData.push_back(extents[i_dir]);
	}

	// Make dataspace:
	hid_t dataspace = H5Screate_simple(dataset_dimension, &DimsData[0], NULL);

	// Set datatype to float
	hid_t datatype = H5Tcopy(H5T_NATIVE_DOUBLE);

	// Create dataset
	hid_t dataset = H5Dcreate2(hdf5group, dataset_name.c_str(), datatype, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

	H5Dwrite(dataset, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &raw_data[0]);

	H5Dclose(dataset);
	H5Sclose(dataspace);

	close_file();
}
