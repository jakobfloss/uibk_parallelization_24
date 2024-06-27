#pragma once

#include <mpi.h>


#include "IO/data_storage.hpp"





class data_storage_parallel : public data_storage {
public:
	data_storage_parallel(std::string file_name);
	void write_dataset_parallel(const std::vector<double> &raw_data, const std::vector<size_t> &extents_global, 
		const std::vector<size_t> &extents_local, const std::vector<size_t> &rank_shifts,
		std::string dataset_name);
private:
	void open_file() override ;
	hid_t plist_file_id, plist_dset_id;
	int rank;
};
