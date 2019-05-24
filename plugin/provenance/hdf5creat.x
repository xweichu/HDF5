struct list{
    string name<>;
    string dsname<>;
    uint8_t data<>;
};
struct dataset{
    int data<>;
}

program HDF5SERVER{

version HDF5SERVER_V1{

    int creat_file(string) = 1; 
    int creat_dataset(list) = 2;
    int open_dataset(list) =3;
    int open_file(string) =4;
    dataset read_dataset(list) = 5;
    }=1;

}=0x2fffffff;
