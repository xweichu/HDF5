/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Board of Trustees of the University of Illinois.         *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of HDF5.  The full HDF5 copyright notice, including     *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the root of the source code       *
 * distribution tree, or in https://support.hdfgroup.org/ftp/HDF5/releases.  *
 * If you do not have access to either file, you may request a copy from     *
 * help@hdfgroup.org.                                                        *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/*
 *  This example illustrates how to create a dataset that is a 4 x 6 
 *  array.  It is used in the HDF5 Tutorial.
 */

#include "/usr/local/hdf5/include/hdf5.h"
#include "iostream"
#define FILE "dset2.h5"



using namespace std;

int main() {

   hid_t       file_id, dataset_id, dataspace_id;  /* identifiers */
   hsize_t     dims[2];
   herr_t      status;

   cout<< "test 0" <<endl;

   /* Create a new file using default properties. */
   //file_id = 
   file_id = H5Fcreate(FILE, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

   // /* Create the data space for the dataset. */
   dims[0] = 4; 
   dims[1] = 6; 

   // dataspace_id = H5Screate(H5S_SIMPLE);
   dataspace_id = H5Screate_simple(2, dims, NULL);
   printf("space id: %d \n", dataspace_id);
   // size_t size = 0;
   // H5Sencode2(dataspace_id, NULL, &size, H5P_DEFAULT);
   // printf("size is %d \n", size);

   /* Create the dataset. */
   dataset_id = H5Dcreate(file_id, "/dset2", H5T_STD_I32BE, dataspace_id, 
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   
   // dataset_id = H5Dcreate(file_id, "/dset3", H5T_STD_I32BE, dataspace_id, 
   //                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   
   // dataset_id = H5Dcreate(file_id, "/dset4", H5T_STD_I32BE, dataspace_id, 
   //                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

   // dataset_id = H5Dcreate(file_id, "/dset5", H5T_STD_I32BE, dataspace_id, 
   //                        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
   printf("test\n");
   // /* End access to the dataset and release resources used by it. */
   // status = H5Dclose(dataset_id);

   // /* Terminate access to the data space. */ 
   // status = H5Sclose(dataspace_id);

   // /* Close the file. */
   // status = H5Fclose(file_id);
}

