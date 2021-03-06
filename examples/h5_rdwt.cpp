#include "hdf5.h"
#define FILE "groups.h5"

int main() {

   hid_t       file_id, dataset_id;  /* identifiers */
   herr_t      status;
   int         dset_data[9];


   /* Open an existing file. */
   file_id = H5Fopen(FILE, H5F_ACC_RDWR, H5P_DEFAULT);

   /* Open an existing dataset. */
   dataset_id = H5Dopen2(file_id, "/dset1", H5P_DEFAULT);

   status = H5Dread(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, 
                    dset_data);

   printf("data:");
   for(int i =0; i<9; i++){
        printf("%d,",dset_data[i]);
   }
   printf("\n");
 
   /* Close the dataset. */
   status = H5Dclose(dataset_id);

   /* Close the file. */
   status = H5Fclose(file_id);
}