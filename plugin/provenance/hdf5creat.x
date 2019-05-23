struct list{
    string name<>;
    string data<>;

};

program HDF5SERVER{

version HDF5SERVER_V1{

    int creat_file(string) = 1; 
    int creat_dataset(list) = 2;
    
    }=1;

}=0x2fffffff;
