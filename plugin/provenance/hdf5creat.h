/*
 * Please do not edit this file.
 * It was generated using rpcgen.
 */

#ifndef _HDF5CREAT_H_RPCGEN
#define _HDF5CREAT_H_RPCGEN

#define RPCGEN_VERSION	199506

#include <rpc/rpc.h>


#define HDF5SERVER ((rpc_uint)0x2fffffff)
#define HDF5SERVER_V1 ((rpc_uint)1)

#ifdef __cplusplus
#define creat_file ((rpc_uint)1)
extern "C" int * creat_file_1(char **, CLIENT *);
extern "C" int * creat_file_1_svc(char **, struct svc_req *);
#define creat_dataset ((rpc_uint)1)
extern "C" int * creat_dataset_1(char **, CLIENT *);
extern "C" int * creat_dataset_1_svc(char **, struct svc_req *);

#elif __STDC__
#define creat_file ((rpc_uint)1)
extern  int * creat_file_1(char **, CLIENT *);
extern  int * creat_file_1_svc(char **, struct svc_req *);
#define creat_dataset ((rpc_uint)1)
extern  int * creat_dataset_1(char **, CLIENT *);
extern  int * creat_dataset_1_svc(char **, struct svc_req *);

#else /* Old Style C */
#define creat_file ((rpc_uint)1)
extern  int * creat_file_1();
extern  int * creat_file_1_svc();
#define creat_dataset ((rpc_uint)1)
extern  int * creat_dataset_1();
extern  int * creat_dataset_1_svc();
#endif /* Old Style C */

#endif /* !_HDF5CREAT_H_RPCGEN */
