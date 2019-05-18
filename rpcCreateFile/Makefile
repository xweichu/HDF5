.SUFFIXES:
.SUFFIXES: .c .o
CLNT = file_create
SRVR = file_create_svc
CFLAGS = -g -Wall

SRVR_OBJ = hdf5creat_svc_proc.o hdf5creat_svc.o
CLNT_OBJ = hdf5creat.o  hdf5creat_clnt.o

.c.o:; gcc -c -o $@ $(CFLAGS) $<

default: $(CLNT) $(SRVR)

$(CLNT): $(CLNT_OBJ) hdf5creat.h
	gcc -o $(CLNT) $(CLNT_OBJ)

$(SRVR): $(SRVR_OBJ) hdf5creat.h
	/usr/local/hdf5/bin/h5cc -o $(SRVR) $(SRVR_OBJ)

clean:
	rm *.o $(CLNT) $(SRVR)
	rm -i *~
	rm core
