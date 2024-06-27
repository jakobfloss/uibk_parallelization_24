#pragma once

#include "IO/hdf5_definitions.hpp"



class data_storage {
public:
	data_storage(std::string file_name);
	data_storage() {};
	void write_dataset(const std::vector<double> &raw_data, const std::vector<size_t> &extents, std::string dataset_name);
	template_noPointer void AddGlobalAttr(const std::string &AttrName, T AttrData);

protected:
	virtual void open_file();
	void close_file();
	hid_t hdf5file, hdf5group;
	std::string file_name;
};

