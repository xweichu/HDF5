#include "hdf5creat.h"

int main(int argc, char *argv[]){

char* name = "testttt";

CLIENT *cl;
cl = clnt_create(argv[1], HDF5SERVER, HDF5SERVER_V1, "tcp");
creat_file_1(&name, cl);
return 0;

}

