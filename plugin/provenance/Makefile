CC=mpicc
HDF5_DIR=/usr/local/hdf5
CFLAGS=-I $(HDF5_DIR)/include -L $(HDF5_DIR)/lib
LIBS=-lhdf5
TARGET=libh5prov.so
makeso:
	$(CC) -shared $(CFLAGS) $(LIBS) -o $(TARGET) -fPIC H5VLprovnc.c
#mpicc -shared -I../hdf5/include -L../hdf5/lib -lhdf5 -o $(TARGET) -fPIC H5VLprovnc.c
