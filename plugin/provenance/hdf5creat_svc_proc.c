#include "/usr/local/hdf5/include/hdf5.h"
#include "hdf5creat.h"
#include <pthread.h>
#include <stdlib.h>

void creatFile(void * n){
	char* name = (char*)n;
	hid_t file_id = H5Fcreate(name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	H5Fclose(file_id);
}

int * creat_file_1_svc(char ** name, struct svc_req * req){
	pthread_t thread_id;
	pthread_create(&thread_id, NULL, creatFile, *name); 

	char* pline = "test1.h5";

	pthread_t thread_id_1;
	pthread_create(&thread_id_1, NULL, creatFile, pline); 



    // pthread_join(thread_id, NULL); 
	// pthread_join(thread_id_1, NULL); 

	// hid_t file_id = H5Fcreate(*name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	// H5Fclose(file_id);
	static int result = 0;
	return &result;
}

int * creat_dataset_1_svc(list * lst, struct svc_req * req){
	list *ptr;
    ptr = lst;
	hid_t file_id = H5Fopen(ptr->name,H5F_ACC_RDWR,H5P_DEFAULT);
	hid_t dataspace = H5Sdecode(ptr->data.data_val);
	printf("dataspace:%d \n", (int)dataspace);
	hid_t dataset_id = H5Dcreate(file_id, ptr->dsname, H5T_STD_I32BE, dataspace, 
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	
    H5Dclose(dataset_id);
	H5Sclose(dataspace);
	H5Fclose(file_id);
	static int result = 0;
	return &result;
}

int * open_dataset_1_svc(list * lst, struct svc_req * req){
	list *ptr;
    ptr = lst;
	
	hid_t file_id = H5Fopen(ptr->name,H5F_ACC_RDWR,H5P_DEFAULT);
	hid_t dataset_id = H5Dopen2(file_id, ptr->dsname, H5P_DEFAULT);

	hid_t dspace = H5Dget_space(dataset_id);
	const int ndims = H5Sget_simple_extent_ndims(dspace);
	hsize_t dims[ndims];
	H5Sget_simple_extent_dims(dspace, dims, NULL);

	static int result = 1;

	for(int i=0; i<ndims; i++){
		result = result * dims[i];
	}
	
    H5Dclose(dataset_id);
	H5Fclose(file_id);
	
	return &result;
}

int * open_file_1_svc(char ** name, struct svc_req * req){
	hid_t file_id = H5Fopen(*name, H5F_ACC_RDWR, H5P_DEFAULT);
	H5Fclose(file_id);
	static int result = 0;
	return &result;
}

dataset * read_dataset_1_svc(list * lst, struct svc_req * req){

	list *ptr;
    ptr = lst;
	static dataset res;
	hid_t file_id = H5Fopen(ptr->name,H5F_ACC_RDWR,H5P_DEFAULT);
	hid_t dataset_id = H5Dopen2(file_id, ptr->dsname, H5P_DEFAULT);

	hid_t dspace = H5Dget_space(dataset_id);
	const int ndims = H5Sget_simple_extent_ndims(dspace);
	hsize_t dims[ndims];
	H5Sget_simple_extent_dims(dspace, dims, NULL);



	int size = 1;
	for(int i=0; i<ndims; i++){
		size = size * dims[i];
	}

	res.data.data_val = (int*)malloc(size*sizeof(int));
	int * buf = (int*)malloc(size*sizeof(int));
	res.data.data_len = size;

	H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, 
                    buf);

	for(int i=0; i<size; i++){
		res.data.data_val[i] = buf[i];
	}

	H5Dclose(dataset_id);
	H5Fclose(file_id);
	return (&res);
}

int * write_dataset_1_svc(list * lst, struct svc_req * req){
	list *ptr;
    ptr = lst;
	
	hid_t file_id = H5Fopen(ptr->name,H5F_ACC_RDWR,H5P_DEFAULT);
	hid_t dataset_id = H5Dopen2(file_id, ptr->dsname, H5P_DEFAULT);
	H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     lst->data.data_val);
	
    H5Dclose(dataset_id);
	H5Fclose(file_id);
	static int result = 0;
	return &result;
}

int * writefile(char* filename, char* dsname,list *lst){

	hid_t file_id = H5Fopen(filename,H5F_ACC_RDWR,H5P_DEFAULT);
	hid_t dataset_id = H5Dopen2(file_id, dsname, H5P_DEFAULT);
	H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     lst->data.data_val);
	
    H5Dclose(dataset_id);
	H5Fclose(file_id);
	static int result = 0;
	return &result;
}


