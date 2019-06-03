
#include "hdf5.h"
#define FILE "groups.h5"

int main() {

   hid_t       file_id, dataset_id, dataspace_id;  /* identifiers */
   hsize_t     dims[2];
   herr_t      status;
   int         i, j, dset1_data[100][100];

   /* Initialize the first dataset. */
   for (i = 0; i < 50; i++)
      for (j = 0; j < 50; j++)
         dset1_data[i][j] = j + i + 1;

   /* Create a file. */
   file_id = H5Fcreate(FILE, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

   /* Create the data space for the first dataset. */
   dims[0] = 100;
   dims[1] = 100;
   dataspace_id = H5Screate_simple(2, dims, NULL);

   /* Create a dataset in group "MyGroup". */
   dataset_id = H5Dcreate2(file_id, "/dset1", H5T_STD_I32BE, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

   /* Write the first dataset. */
   status = H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     dset1_data);

   /* Close the data space for the first dataset. */
   status = H5Sclose(dataspace_id);

   /* Close the first dataset. */
   status = H5Dclose(dataset_id);

   /* Close the file. */
   status = H5Fclose(file_id);
}

