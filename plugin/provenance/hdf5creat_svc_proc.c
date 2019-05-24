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
	
    H5Dclose(dataset_id);
	H5Fclose(file_id);
	static int result = 0;
	return &result;
}

int * open_file_1_svc(char ** name, struct svc_req * req){
	hid_t file_id = H5Fopen(*name, H5F_ACC_RDWR, H5P_DEFAULT);
	H5Fclose(file_id);
	static int result = 0;
	return &result;
}

list * read_dataset_1_svc(list * lst, struct svc_req * req){
	list *ptr;
    ptr = lst;

	list *res;
	res = (list*)malloc(sizeof(list));

	char* new_name = strdup(ptr->name);
	char* new_dsname = strdup(ptr->dsname);
	res->name = new_name;
	res->dsname = new_dsname;

	hid_t file_id = H5Fopen(ptr->name,H5F_ACC_RDWR,H5P_DEFAULT);
	hid_t dataset_id = H5Dopen2(file_id, ptr->dsname, H5P_DEFAULT);

	int* dset_data = (int*)malloc(5*sizeof(int));
	H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, 
                    dset_data);

	res->data.data_len = 5*sizeof(int);
	res->data.data_val = dset_data;

	printf("data:");
    for(int i =0; i<5; i++){
        printf("%d,",dset_data[i]);
    }
   	printf("\n");
	
    // H5Dclose(dataset_id);
	// H5Fclose(file_id);

	return NULL;
}