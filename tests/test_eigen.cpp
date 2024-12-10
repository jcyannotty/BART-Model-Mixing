//Eigen Tutorial -- provides examples of Eigen features that are used throughout the BART Model Mixing Code
#include <iostream>
#include "Eigen/Dense"
#include <Eigen/StdVector>

using namespace Eigen;
using namespace std;

using mxd = Eigen::MatrixXd;
using vxd = Eigen::VectorXd;

void matrix_to_array(Eigen::MatrixXd &M, double *b){
   //Flatten the matrix by row
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> M2(M); //Creates a dynamic X dynmaic matrix and updates by row
    Eigen::Map<Eigen::RowVectorXd> v(M2.data(), M2.size()); //converts to Eigen Vector 
    //cout << "M2 = \n" << M2 << endl;
    //cout << "v = " << v << endl;
    
    //Populate b using v
    for(int i=0;i<v.size();i++){
       b[i] = v(i);
    }
}

void array_to_matrix(Eigen::MatrixXd &M, double *b){
    size_t nrow = M.rows();
    size_t ncol = M.cols();
    for(size_t i = 0; i<nrow; i++){
        for(size_t j = 0; j<ncol; j++){
            M(i,j) = b[i*ncol + j];
        }
    }
}

void test_pbr(double &x){
    cout << x << endl;
}

int main(){
    //Define an Eigen matirx of dimension 4x4
    MatrixXd A(4,4);

    //Populate the matrix manually
    A << 11,12,13,14,21,22,23,24,31,32,33,34,41,42,43,44;

    //Print A
    cout << "A = \n" << A << endl; 

    //Flatten the matrix by row
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> A2(A); //Creates a dynamic X dynmaic matrix and updates by row
    Eigen::Map<Eigen::RowVectorXd> av(A2.data(), A2.size());
    cout << "A2 = \n" << A2 << endl;
    cout << "av = " << av << endl;
    
    //Define an array b of dimension 16 and populate it using v 
    double a[16];
    std::copy(std::begin(av.array()), std::end(av.array()), std::begin(a));

    cout << "a = ";
    for(int i = 0; i<16;i++){
        cout << a[i] << " ";
    }
    cout << endl;
    cout << "Size of a = " << sizeof(a)/sizeof(a[0]) << endl;
    cout << "&a[0] = " << &a[0] << " ---- " << "begin(a) = " << std::begin(a) << endl;
    
    //Cast back to the matrix A
    MatrixXd A3(4,4);
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            A3(i,j) = a[i*4 + j];
        }       
    }
    cout << "A3 = \n" << A3 << endl; 
    cout << "Number of elements in A3 = " << A3.size() << endl; 
    
    //Test the function matrix_to_array
    cout << "*** Testing Function --- matrix_to_array ***" << endl;
    double a2[16];
    matrix_to_array(A,&a2[0]);
    cout << "a2 = ";
    for(int i = 0; i<16;i++){
        cout << a2[i] << " ";
    }
    cout << endl;
    
    //Test the function array_to_matrix
    cout << "*** Testing Function --- array_to_matrix ***" << endl;
    MatrixXd A4(4,4);
    A4 = MatrixXd::Zero(4,4);
    array_to_matrix(A4, &a2[0]);
    cout << "A4 = \n" << A4 << endl;

    //-------------------------------------------------------------
    //Create an std vector of Eigen Matrices
    std::vector<mxd, Eigen::aligned_allocator<mxd>> vecB(2);
    mxd B1(2,2), B2(2,2);
    double b[vecB.size()*2*2]; //2*2 since each matrix is 2x2
    
    //construct the matrices and populate vecB -- the std vector of matrices
    B1 << 1,2,3,4;
    B2 << 10, 20, 30, 40;
    vecB[0] = B1;
    vecB[1] = B2;

    //now populate the array by flattening vecB by row
    for(int i = 0; i<2;i++){
        matrix_to_array(vecB[i], &b[i*4]);
    }

    cout << "vecB[0] = \n" << vecB[0] << endl;
    cout << "vecB[1] = \n" << vecB[1] << endl;

    cout << "b = ";
    for(int i = 0; i<vecB.size()*2*2;i++){
        cout << b[i] << " ";
    }
    cout << endl;

    //-------------------------------------------------------------
    //Populate an Eigen matrix/vector with an std vector
    mxd C(4,2);
    std::vector<double> cv1 = {1.2,3.4,5.6,7.8};
    std::vector<double> cv2 = {10.0,20.0,30.0,40.0};

    C.col(0) = Eigen::Map<Eigen::VectorXd>(cv1.data(),4);
    C.col(1) = Eigen::Map<Eigen::VectorXd>(cv2.data(),4);

    cout << "C = \n" << C << endl; 
    
    //-------------------------------------------------------------
    //test pass by reference -- can delete later
    double xx = 3.2;
    test_pbr(xx);

    double *yy;
    double zz = 6.7;
    yy = &zz;
    cout << *yy << endl;

    //-------------------------------------------------------------
    //Elementwise vector multiplication
    vxd v1(3), v2(3), v3(3);
    v1 << 1,2,3;
    v2 << 10,20,30;
    v3 = v1.cwiseProduct(v2);
    cout << "Elementwise Product for vectors: v3 = " << v3.transpose() << endl;
    /*
    //Important Eigen storage example. Shows ways to work with eigen pointers/references
    Eigen::MatrixXd A(2,2);
    Eigen::MatrixXd *B;
    A << 2,4,1,3;
    B = &A;
    cout << "print A:" << endl;
    cout << A << endl;
    cout << &A << endl; //Address of the matrix
    cout << A.data() << endl; //Address of the first element
    cout << *(A.data() + 1) << endl; //Dereference the 2nd row 1st column element -- prints 1
    cout << *(A.data() + 2) << endl; //Dereference the 1st row 2nd column element -- prints 4
    cout << *(A.data() + 3) << endl; //Dereference the 2nd row 2nd column element -- prints 3

    cout << "print B:" << endl;
    cout << *B << endl; //Prints a dereferenced B -- same output as A
    cout << ((*B).data()) << endl; //Prints the same as A.data()
    cout << (*B).row(1) << endl; //Print the 2nd row of dereferenced B.
    */

   /*
    //Two ways to convert a vector to a matrix -- used to make finfo
    Eigen::VectorXd v(6);
    v << 10,20,30,40, 50, 60;
    Eigen::Map<mxd, Eigen::RowMajor> M(v.data(), 2,3);
    cout << M.transpose() << endl;
    
    mxd M2 = Eigen::Map<Eigen::Matrix<double, 3,2, Eigen::RowMajor>>(v.data());
    cout << M2 << endl;

    //Create an identity matrix
    mxd m(2,2);
    m = Eigen::MatrixXd::Identity(2,2); 
    cout << m << endl;

    //Taking the elementwise square root of a matrix
    mxd A(2,2);
    A << 1,4,9,16;
    cout << "A = \n" << A << endl;
    cout << "A.array() = \n" << A.array() << endl;
    cout << "A.array().sqrt() = \n"<< A.array().sqrt() << endl;
    
    A = A.array().sqrt();
    cout << "sqrt(A) = \n" << A << endl;


    //Vector of ones
    vxd A(3);
    A = Eigen::VectorXd::Ones(3);
    cout << A << endl;

    //Log determinants
    mxd A(2,2);
    A << 1,4,9,16;
    mxd AL(A.llt().matrixL()); 
    cout << AL << endl;
    cout << AL.diagonal() << endl;
    cout << AL.diagonal().array().log() << endl;
    cout << (AL.diagonal().array().log().sum())*2 << endl;

    //vector of Eigen matrix/vector
    #include <Eigen/StdVector>

    std::vector<mxd, Eigen::aligned_allocator<mxd>> temp(2);
    mxd A(2,2), B(2,2);
    A << 1,2,3,4;
    B << 10, 20, 30, 40;
    temp[0] = A;
    temp[1] = B;

    cout << "A = \n" << temp[0] << endl;
    cout << "B = \n" << temp[1] << endl;
    */

}

