
/**
*   CS6023: GPU Programming 
*   Assignment 2
*   
*   Please don't change any existing code in this file.
*
*   Please add necessary memory APIs for your implementation. Use cudaFree() 
*   to free up memory as soon as you're done with an allocation. 
*   This will ensure that you don't run out of memory while running
*   large test cases. Use the minimum required memory for your 
*   implementation. DO NOT change the kernel configuration parameters.
*/

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <cuda.h>

using namespace std;

using std::cin;
using std::cout;

typedef long long ll;
__global__ void dkernal(long int* inpgpu,long int* gpu_fil,long int m,long int n,long int k,long int* result_mat){
    __shared__ long int hfilter[50*50];
    if(threadIdx.x==0){
        for(int i=0;i<k*k;i++)hfilter[i]=gpu_fil[i];
    }
    __syncthreads();
    long int id=threadIdx.x+blockIdx.x*blockDim.x;
    if(id<m*n){
        long int mainmatrow=id/n;
        long int mainmatcol=id%n;
        long int filmatrow=k/2;
        long int filmatcol=k/2;
        long int mainmat_dup_row_up=mainmatrow;
        long int mainmat_dup_row_down=mainmatrow;
        long int mainmat_dup_col_left=mainmatcol;
        long int mainmat_dup_col_right=mainmatcol;
        long int fil_dup_col_left=filmatcol;
        long int fil_dup_col_right=filmatcol;
        long int fil_dup_row_up=filmatrow;
        long int fil_dup_row_down=filmatrow;
        while(fil_dup_col_left>0 && mainmat_dup_col_left>0)
        {
        fil_dup_col_left--;
        mainmat_dup_col_left--;
        }
        while(fil_dup_col_right<k && mainmat_dup_col_right<n){
        fil_dup_col_right++;
        mainmat_dup_col_right++;
        }
        while(fil_dup_row_up>0 && mainmat_dup_row_up>0){
        fil_dup_row_up--;
        mainmat_dup_row_up--;
        }
        while(fil_dup_row_down<k && mainmat_dup_row_down<m){
        fil_dup_row_down++;
        mainmat_dup_row_down++;
        }
        long int fil_i=fil_dup_row_up;
        for(long int i=mainmat_dup_row_up;i<mainmat_dup_row_down;i++){
        long int fil_j=fil_dup_col_left;
        for(long int j=mainmat_dup_col_left;j<mainmat_dup_col_right;j++){
        result_mat[id]=result_mat[id]+inpgpu[i*n+j]*hfilter[fil_i*k+fil_j];
        fil_j++;
        }
        fil_i++;
        }
        }
}

int main(int argc, char** argv) {

    int m,n,k;
    cin>>m>>n>>k;


    long int* h_mat = new long int[m * n];
    long int* h_filter = new long int[k * k];

    long int* h_ans = new long int[m * n];


    for (long int i = 0; i < m * n; i++) {
        cin>>h_mat[i];
    }

    for (long int i = 0; i < k * k; i++) {
        cin>>h_filter[i];
    }

    /**
     * 
     * DO NOT CHANGE ANYTHING ABOVE THIS LINE
     * 
    **/

    /****************************************************Start Here***********************************************************/
    dim3 grid(1024,1,1);
    dim3 block(1024,1,1);
    long int *gpumat,*gpufil,*result_mat;
    cudaMalloc(&gpumat,m*n*sizeof(long int));
    cudaMalloc(&gpufil,k*k*sizeof(long int));
    cudaMalloc(&result_mat,m*n*sizeof(long int));
    cudaMemcpy(gpumat,h_mat,m*n*sizeof(long int),cudaMemcpyHostToDevice);
    cudaMemcpy(gpufil,h_filter,k*k*sizeof(long int),cudaMemcpyHostToDevice);
    auto start = std::chrono::high_resolution_clock::now();//keep it just before the kernel launch
    dkernal<<<grid,block>>>(gpumat,gpufil,m,n,k,result_mat);
    auto end = std::chrono::high_resolution_clock::now();//keep it just after the kernel launch
    cudaDeviceSynchronize();
    cudaMemcpy(h_ans,result_mat,m*n*sizeof(long int),cudaMemcpyDeviceToHost);
    cudaFree(gpumat);
    cudaFree(gpufil);
    cudaFree(result_mat);
    /*$$$$$$$$$$$$$$$$$$$$$$$$Make sure your final output from the device is stored in h_ans.$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$*/
    std::chrono::duration<double> elapsed1 = end - start;
    /**
     * 
     * DO NOT CHANGE ANYTHING BELOW THIS LINE
     * 
    */


    
    std::ofstream file("cuda.out");
    if (file.is_open()) {
        for (long int i = 0; i < m; i++) {
            for (long int j = 0; j < n; j++) {
                file << h_ans[i * n + j] << " ";
            }
            file << "\n";
        }
        file.close();
    } else {
        std::cout << "Unable to open file";
    }

    std::ofstream file2("cuda_timing.out");
    if(file2.is_open()) {
        file2 << elapsed1.count() << "\n";
        file2.close();
    } else {
        std::cout << "Unable to open file";
    }

    return 0;
}