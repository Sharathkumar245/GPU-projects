#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>

using namespace std;

//*******************************************

// Write down the kernels here

__global__ void updates_alive(int *gpu_alive, int *alive_status, int *d_hp, int T)
{
    unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < T)
    {
        if (d_hp[tid] <= 0 && alive_status[tid] == 1)
        {
            alive_status[tid] = 0;
            atomicSub(gpu_alive, 1);
        }
    }
}
__global__ void dkernal(int k, int T, int *d_score, int *d_hp, int *gpu_alive, int *alive_status, int *dis, int M, int N, int *d_x, int *d_y)
{
    unsigned blkid = blockIdx.x;
    unsigned tid = threadIdx.x;
    unsigned dir = (k + blkid) % T;
    if (alive_status[blkid] == 1)
    {
        long long dir_node = (long long)d_x[dir] * (N + 1) + d_y[dir];
        long long blk_node = (long long)d_x[blkid] * (N + 1) + d_y[blkid];
        long long tid_node = (long long)d_x[tid] * (N + 1) + d_y[tid];
        int dist_i = -1;
        if (alive_status[tid] == 1 && tid != blkid && dir_node > blk_node && tid_node > blk_node && ((long long)(d_y[tid] - d_y[blkid]) * (d_x[dir] - d_x[blkid]) == (long long)(d_y[dir] - d_y[blkid]) * (d_x[tid] - d_x[blkid])))
        {
            dist_i = abs(d_y[tid] - d_y[blkid]) + abs(d_x[tid] - d_x[blkid]);
            atomicMin(&dis[blkid], dist_i);
        }
        else if (alive_status[tid] == 1 && tid != blkid && dir_node < blk_node && tid_node < blk_node && ((long long)(d_y[tid] - d_y[blkid]) * (d_x[dir] - d_x[blkid]) == (long long)(d_y[dir] - d_y[blkid]) * (d_x[tid] - d_x[blkid])))
        {
            dist_i = abs(d_y[tid] - d_y[blkid]) + abs(d_x[tid] - d_x[blkid]);
            atomicMin(&dis[blkid], dist_i);
        }
        __syncthreads();
        if (alive_status[tid] == 1 && tid != blkid && dist_i == dis[blkid] && alive_status[tid] == 1)
        {
            atomicSub(&d_hp[tid], 1);
            atomicAdd(&d_score[blkid], 1);
        }
    }
}
__global__ void allocations(int *alive_status, int *d_score, int *d_hp, int H, int T, int *dis)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < T)
    {
        alive_status[tid] = 1;
        d_score[tid] = 0;
        d_hp[tid] = H;
        dis[tid] = INT_MAX;
    }
}
__global__ void updatedis_ind(int *dis, int T)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < T)
    {
        dis[tid] = INT_MAX;
    }
}
//***********************************************

int main(int argc, char **argv)
{
    // Variable declarations
    int M, N, T, H, *xcoord, *ycoord, *score;

    FILE *inputfilepointer;

    // File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer = fopen(inputfilename, "r");

    if (inputfilepointer == NULL)
    {
        printf("input.txt file failed to open.");
        return 0;
    }

    fscanf(inputfilepointer, "%d", &M);
    fscanf(inputfilepointer, "%d", &N);
    fscanf(inputfilepointer, "%d", &T); // T is number of Tanks
    fscanf(inputfilepointer, "%d", &H); // H is the starting Health point of each Tank

    // Allocate memory on CPU
    xcoord = (int *)malloc(T * sizeof(int)); // X coordinate of each tank
    ycoord = (int *)malloc(T * sizeof(int)); // Y coordinate of each tank
    score = (int *)malloc(T * sizeof(int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for (int i = 0; i < T; i++)
    {
        fscanf(inputfilepointer, "%d", &xcoord[i]);
        fscanf(inputfilepointer, "%d", &ycoord[i]);
    }

    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************

    int alive = T;
    int *alive_status;
    cudaMalloc(&alive_status, sizeof(int) * T);
    int *d_score, *d_hp;
    cudaMalloc(&d_score, sizeof(int) * T);
    cudaMalloc(&d_hp, sizeof(int) * T);
    int *dis;
    cudaMalloc(&dis, sizeof(int) * T);
    allocations<<<1, T>>>(alive_status, d_score, d_hp, H, T, dis);
    cudaDeviceSynchronize();
    int *gpu_alive;
    cudaMalloc(&gpu_alive, sizeof(int));
    cudaMemcpy(gpu_alive, &alive, sizeof(int), cudaMemcpyHostToDevice);
    int *d_x, *d_y;
    cudaMalloc(&d_x, sizeof(int) * T);
    cudaMemcpy(d_x, xcoord, sizeof(int) * T, cudaMemcpyHostToDevice);
    cudaMalloc(&d_y, sizeof(int) * T);
    cudaMemcpy(d_y, ycoord, sizeof(int) * T, cudaMemcpyHostToDevice);
    int k = 1;
    while (alive > 1)
    {
        dkernal<<<T, T>>>(k++, T, d_score, d_hp, gpu_alive, alive_status, dis, M, N, d_x, d_y);
        updates_alive<<<1, T>>>(gpu_alive, alive_status, d_hp, T);
        updatedis_ind<<<1, T>>>(dis, T);
        cudaDeviceSynchronize();
        cudaMemcpy(&alive, gpu_alive, sizeof(int), cudaMemcpyDeviceToHost);
    }
    cudaMemcpy(score, d_score, sizeof(int) * T, cudaMemcpyDeviceToHost);
    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end - start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3];
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename, "w");

    for (int i = 0; i < T; i++)
    {
        fprintf(outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename, "w");
    fprintf(outputfilepointer, "%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}