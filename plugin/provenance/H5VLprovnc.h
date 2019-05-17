/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright by The HDF Group.                                               *
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
 * Purpose:	The public header file for the pass-through VOL connector.
 */

#ifndef _H5VLprovnc_H
#define _H5VLprovnc_H

/* Identifier for the pass-through VOL connector */
#define H5VL_PROVNC	(H5VL_provenance_register())

/* Characteristics of the pass-through VOL connector */
#define H5VL_PROVNC_NAME        "provenance"
#define H5VL_PROVNC_VALUE       509           /* VOL connector ID */
#define H5VL_PROVNC_VERSION     0


typedef enum ProvLevel {
    Default, //no file write, only screen print
    Print_only,
    File_only,
    File_and_print,
    Level3,
    Level4,
    Disabled
}Prov_level;

/* Pass-through VOL connector info */
typedef struct H5VL_provenance_info_t {
    hid_t under_vol_id;         /* VOL ID for under VOL */
    void *under_vol_info;       /* VOL info for under VOL */
    char* prov_file_path;   // TODO: Move into prov_helper_t
    Prov_level prov_level;  // TODO: Move into prov_helper_t
    char* prov_line_format; // TODO: Move into prov_helper_t
//    int ds_created;         // TODO: Move into dataset_prov_info
//    int ds_accessed;        // TODO: Move into dataset_prov_info
} H5VL_provenance_info_t;


typedef enum ProvenanceOutputDST{
    TEXT,
    BINARY,
    CSV
}prov_out_dst;

typedef struct ProvenanceFormat{
    prov_out_dst dst_format;

} prov_format;

typedef struct H5VL_prov_file_info_t file_prov_info_t;

typedef struct ProvenanceHelper {
    /* Provenance properties */
    char* prov_file_path;
    FILE* prov_file_handle;
    Prov_level prov_level;
    char* prov_line_format;
    char user_name[32];
    int pid;
    pthread_t tid;
    char proc_name[64];
    int ptr_cnt;
    int opened_files_cnt;
    file_prov_info_t* opened_files;//linkedlist,
} prov_helper_t;

prov_helper_t* prov_helper_init( char* file_path, Prov_level prov_level, char* prov_line_format);
int prov_write(prov_helper_t* helper_in, const char* msg, unsigned long duration);

#ifdef __cplusplus
extern "C" {
#endif

H5_DLL hid_t H5VL_provenance_register(void);

#ifdef __cplusplus
}
#endif

#endif /* _H5VLprovnc_H */

