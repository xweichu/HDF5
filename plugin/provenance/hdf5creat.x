struct list{
    string name<>;
    string dsname<>;
    uint8_t data<>;

};

program HDF5SERVER{

version HDF5SERVER_V1{

    int creat_file(string) = 1; 
    int creat_dataset(list) = 2;
    int open_dataset(list) =3;
    int open_file(list) =4;
    }=1;

}=0x2fffffff;
