
#include "hdf5.h"
#include <time.h>
#include <sys/time.h>
#define FILE "groups.h5"
#define DATASIZE 1200

unsigned long get_time_usec(void) {
    struct timeval tp;

    gettimeofday(&tp, NULL);
    return (unsigned long)((1000000 * tp.tv_sec) + tp.tv_usec);
}

int main() {

//    unsigned long start = get_time_usec();
   hid_t       file_id, dataset_id, dataspace_id;  /* identifiers */
   hsize_t     dims[2];
   herr_t      status;
   int         i, j, dset1_data[DATASIZE][DATASIZE];

   /* Initialize the first dataset. */
   for (i = 0; i < DATASIZE; i++)
      for (j = 0; j < DATASIZE; j++)
         dset1_data[i][j] = j + i + 1;

   unsigned long start = get_time_usec();
   /* Create a file. */
   file_id = H5Fcreate(FILE, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

   unsigned long stop = get_time_usec();
   printf("time to create the file: %d\n", stop - start);
   
   start = get_time_usec();

   /* Create the data space for the first dataset. */
   dims[0] = DATASIZE;
   dims[1] = DATASIZE;
   dataspace_id = H5Screate_simple(2, dims, NULL);




   /* Create a dataset in group "MyGroup". */
   dataset_id = H5Dcreate2(file_id, "/dset1", H5T_STD_I32BE, dataspace_id,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   stop = get_time_usec();
   
   printf("time to create the dataset: %d \n", stop - start);

   start = get_time_usec();

   /* Write the first dataset. */
   status = H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     dset1_data);


    stop = get_time_usec();
       printf("time to write the dataset: %d\n", stop - start);
   /* Close the data space for the first dataset. */
   status = H5Sclose(dataspace_id);

   /* Close the first dataset. */
   status = H5Dclose(dataset_id);

   /* Close the file. */
   status = H5Fclose(file_id);
}

