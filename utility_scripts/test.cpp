#include <iostream>
#define BUFFER_SIZE 2048
#define SIZE 256
# define MAX_COUNT 5
using namespace std; 

int main(){
    const FILE **fp_array = new FILE*[2];
    fopen_s(&fp_array[0], "test.txt", "r");

    char buffer[BUFFER_SIZE]; 

    if(fp_array[0] == NULL){
        cout << "File not found" << endl;
    
    }else{
        fread_s(buffer, BUFFER_SIZE, SIZE, MAX_COUNT, fp_array[0]);
        cout << buffer << endl;
    }

    return 0;
}