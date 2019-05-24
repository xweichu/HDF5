#include "/usr/local/hdf5/include/hdf5.h"
#include "hdf5creat.h"

int * creat_file_1_svc(char ** name, struct svc_req * req){
	hid_t file_id = H5Fcreate(*name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	H5Fclose(file_id);
	static int result = 0;
	return &result;
}

int * creat_dataset_1_svc(list * lst, struct svc_req * req){
	list *ptr;
    ptr = lst;
	
	hid_t file_id = H5Fopen(ptr->name,H5F_ACC_RDWR,H5P_DEFAULT);
	hid_t dataspace = H5Sdecode(ptr->data.data_val);
	printf("dataspace:%d \n", dataspace);
	hid_t dataset_id = H5Dcreate(file_id, ptr->dsname, H5T_STD_I32BE, dataspace, 
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	
    H5Dclose(dataset_id);
	H5Sclose(dataspace);
	H5Fclose(file_id);
	static int result = 0;
	return &result;
}