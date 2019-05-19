#include "/usr/local/hdf5/include/hdf5.h"
#include "hdf5creat.h"

int * creat_file_1_svc(char ** name, struct svc_req * req){
	H5Fcreate(*name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	static int result = 0 ;
	return &result;
}

