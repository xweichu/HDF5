/*
 * Purpose:	The public header file for the demo VOL connector.
 */
#include <assert.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "/usr/local/hdf5/include/hdf5.h"
#ifndef _H5VLdemo_H
#define _H5VLdemo_H

/* Identifier for the demo VOL connector */
#define H5VL_DEMO (H5VL_demo_register())

/* Characteristics of the demo VOL connector */
#define H5VL_DEMO_NAME        "demo"
#define H5VL_DEMO_VALUE       509           /* VOL connector ID */
#define H5VL_DEMO_VERSION     0

/* Common object and attribute information */
typedef struct H5VL_demo_item_t {
    H5I_type_t type;
    struct H5VL_demo_file_t *file;
    int rc;
} H5VL_demo_item_t;

/* Common object information */
typedef struct H5VL_demo_obj_t {
    H5VL_demo_item_t item; /* Must be first */
    uint64_t bin_oid;
    char *oid;
} H5VL_demo_obj_t;

/* The file struct */
typedef struct H5VL_demo_file_t {
    H5VL_demo_item_t item; /* Must be first */
    char *file_name;
    size_t file_name_len;
    unsigned flags;
    char *glob_md_oid;
    struct H5VL_demo_group_t *root_grp;
    uint64_t max_oid;
    hbool_t max_oid_dirty;
    hid_t fcpl_id;
    hid_t fapl_id;
    int my_rank;
    int num_procs;
    hbool_t collective;
} H5VL_demo_file_t;

/* The group struct */
typedef struct H5VL_demo_group_t {
    H5VL_demo_obj_t obj; /* Must be first */
    hid_t gcpl_id;
    hid_t gapl_id;
} H5VL_demo_group_t;

/* The dataset struct */
typedef struct H5VL_demo_dset_t {
    H5VL_demo_obj_t obj; /* Must be first */
    hid_t type_id;
    hid_t space_id;
    hid_t dcpl_id;
    hid_t dapl_id;
} H5VL_demo_dset_t;

/* The datatype struct */
/* Note we could speed things up a bit by caching the serialized datatype.  We
 * may also not need to keep the type_id around.  -NAF */
typedef struct H5VL_demo_dtype_t {
    H5VL_demo_obj_t obj; /* Must be first */
    hid_t type_id;
    hid_t tcpl_id;
    hid_t tapl_id;
} H5VL_demo_dtype_t;

/* The attribute struct */
typedef struct H5VL_demo_attr_t {
    H5VL_demo_item_t item; /* Must be first */
    H5VL_demo_obj_t *parent;
    char *name;
    hid_t type_id;
    hid_t space_id;
} H5VL_demo_attr_t;

/* The link value struct */
typedef struct H5VL_demo_link_val_t {
    H5L_type_t type;
    union {
        uint64_t hard;
        char *soft;
    } target;
} H5VL_demo_link_val_t;

/* Udata type for H5Dscatter callback */
typedef struct H5VL_demo_scatter_cb_ud_t {
    void *buf;
    size_t len;
} H5VL_demo_scatter_cb_ud_t;

/* Information about a singular selected chunk during a Dataset read/write */
typedef struct H5VL_demo_select_chunk_info_t {
    uint64_t chunk_coords[H5S_MAX_RANK]; /* The starting coordinates ("upper left corner") of the chunk */
    hid_t    mspace_id;                  /* The memory space corresponding to the
                                            selection in the chunk in memory */
    hid_t    fspace_id;                  /* The file space corresponding to the*/
} H5VL_demo_select_chunk_info_t;                            

/* "Management" callbacks */
static herr_t H5VL_demo_init(hid_t vipl_id);
static herr_t H5VL_demo_term(void);

/* VOL info callbacks */
static void *H5VL_demo_info_copy(const void *_old_info);
static herr_t H5VL_demo_info_cmp(int *cmp_value, const void *_info1, const void *_info2);
static herr_t H5VL_demo_info_free(void *_info);

#ifdef __cplusplus
extern "C" {
#endif

H5_DLL hid_t H5VL_demo_register(void);

#ifdef __cplusplus
}
#endif

#endif

