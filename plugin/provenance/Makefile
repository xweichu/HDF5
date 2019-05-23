CC=mpicc
HDF5_DIR=/usr/local/hdf5/
CFLAGS=-I$(HDF5_DIR)/include -L$(HDF5_DIR)/lib
LIBS=-lhdf5
TARGET=libh5prov.so
makeso:
	gcc -c -o hdf5creat_xdr.o -g -Wall hdf5creat_xdr.c
	gcc -c -o hdf5creat_svc_proc.o -g -Wall hdf5creat_svc_proc.c
	gcc -c -o hdf5creat_svc.o -g -Wall hdf5creat_svc.c
	/usr/local/hdf5/bin/h5cc -o file_create_svc hdf5creat_svc_proc.o hdf5creat_xdr.o hdf5creat_svc.o
	mpicc -c -o hdf5creat_xdr.o -g -fPIC hdf5creat_xdr.c 
	mpicc -c -o hdf5creat_clnt.o -g -fPIC hdf5creat_clnt.c 
	$(CC) -shared $(CFLAGS) $(LIBS) -o $(TARGET) hdf5creat_clnt.o hdf5creat_xdr.o -fPIC H5VLprovnc.c
#mpicc -shared -I../hdf5/include -L../hdf5/lib -lhdf5 -o $(TARGET) -fPIC H5VLprovnc.c
