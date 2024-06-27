#include "IO/data_storage_parallel.hpp"

#include <iostream>

data_storage_parallel::data_storage_parallel(std::string file_name) {
    // Prepare MPI-related stuff
    plist_file_id = H5Pcreate(H5P_FILE_ACCESS);
    MPI_Info info_mpi  = MPI_INFO_NULL;
	H5Pset_fapl_mpio(plist_file_id, MPI_COMM_WORLD, info_mpi);

    // property list for dataset access
	plist_dset_id = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plist_dset_id, H5FD_MPIO_COLLECTIVE);

	this->file_name = file_name;

    hdf5file =  H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_file_id);
	hdf5group = H5Gcreate2(hdf5file, "/Data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Gclose(hdf5group);
	H5Fclose(hdf5file);
}


void data_storage_parallel::open_file() {
    // std::cerr << " open file in parallel \n\n\n";
	hdf5file = H5Fopen(file_name.c_str(),  H5F_ACC_RDWR, plist_file_id);
	hdf5group = H5Gopen2(hdf5file, "/Data", H5P_DEFAULT);
}


void data_storage_parallel::write_dataset_parallel(const std::vector<double> &raw_data, const std::vector<size_t> &extents_global, 
		const std::vector<size_t> &extents_local, const std::vector<size_t> &rank_shifts,
		std::string dataset_name) {

	// Open the file
	open_file();

    // Extent of dataset given by GLOBAL grid
	int dataset_dimension = extents_global.size();
	std::vector<hsize_t> DimsData;
	// hsize_t DimsData[dataset_dimension];
	for (int i_dir = 0; i_dir < dataset_dimension; ++i_dir) {
		DimsData.push_back(extents_global[i_dir]);
	}

	// Make dataspace:
	hid_t dataspace = H5Screate_simple(dataset_dimension, &DimsData[0], NULL);

	// Set datatype to float
	hid_t datatype = H5Tcopy(H5T_NATIVE_DOUBLE);

	// Create dataset
	hid_t dataset = H5Dcreate2(hdf5group, dataset_name.c_str(), datatype, dataspace,
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Now make distinction from global data via hyperslab definition:
    std::vector<hsize_t> sizeLocal;
    for (int i_dir = 0; i_dir < dataset_dimension; ++i_dir) {
        sizeLocal.push_back(extents_local[i_dir]);
    }

    std::vector<hsize_t> offset;
    for (int i_dir = 0; i_dir < dataset_dimension; ++i_dir) {
        offset.push_back(rank_shifts[i_dir]);
    }

    hid_t dataspaceLocal = H5Screate_simple(dataset_dimension, &sizeLocal[0], NULL);

    // Select local data as hyperslab in dataset
	H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, &offset[0], NULL, &sizeLocal[0], NULL);

    // Set up parallel IO
	hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    H5Dwrite(dataset, datatype, dataspaceLocal, dataspace, plist_id, &raw_data[0]);

    H5Pclose(plist_id);
	H5Dclose(dataset);
    H5Sclose(dataspaceLocal);
	H5Sclose(dataspace);

	close_file();
}
