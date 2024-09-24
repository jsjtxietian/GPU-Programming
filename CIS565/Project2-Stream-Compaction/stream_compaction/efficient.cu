#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

void printArray_cuda(int n, int *a, bool abridged = false)
{
    printf("    [ ");
    for (int i = 0; i < n; i++)
    {
        if (abridged && i + 2 == 15 && n > 16)
        {
            i = n - 2;
            printf("... ");
        }
        printf("%3d ", a[i]);
    }
    printf("]\n");
}

namespace StreamCompaction
{
    namespace Efficient
    {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer &timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void up_sweep(const int n, int d, int *idata)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            int stride = 1 << (d + 1);
            if (index >= n || index % stride != 0)
            {
                return;
            }
            idata[index + stride - 1] += idata[index + (1 << d) - 1];
        }

        __global__ void down_sweep(const int n, int d, int *idata)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            int stride = 1 << (d + 1);
            if (index >= n || index % stride != 0)
            {
                return;
            }
            int left = 1 << d;
            int t = idata[index + left - 1];
            idata[index + left - 1] = idata[index + stride - 1];
            idata[index + stride - 1] += t;
        }

        __global__ void scan_opt(int n, int *odata, int *idata)
        {
            extern __shared__ int temp[]; // allocated on invocation
            int thid = threadIdx.x;
            int offset = 1;
            temp[2 * thid] = idata[2 * thid]; // load input into shared memory
            temp[2 * thid + 1] = idata[2 * thid + 1];
            for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
            {
                __syncthreads();
                if (thid < d)
                {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;
                    temp[bi] += temp[ai];
                }
                offset *= 2;
            }
            if (thid == 0)
            {
                temp[n - 1] = 0;
            } // clear the last element

            for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
            {
                offset >>= 1;
                __syncthreads();
                if (thid < d)
                {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;
                    int t = temp[ai];
                    temp[ai] = temp[bi];
                    temp[bi] += t;
                }
            }
            __syncthreads();
            odata[2 * thid] = temp[2 * thid]; // write results to device memory
            odata[2 * thid + 1] = temp[2 * thid + 1];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata, bool useOpt)
        {
            const int arrLen = n * sizeof(int);
            const int paddedN = 1 << ilog2ceil(n);
            const int paddedLen = paddedN * sizeof(int);

            dim3 blocksPerGrid((paddedN + blockSize - 1) / blockSize);

            int *devData;
            cudaMalloc((void **)&devData, paddedLen);
            cudaMemcpy(devData, idata, arrLen, cudaMemcpyHostToDevice);
            cudaMemset(devData + n, 0, paddedLen - arrLen);

            timer().startGpuTimer();
            if (useOpt)
            {
                scan_opt<<<1, blockSize / 2>>>(paddedN, devData, devData);
            }
            else
            {
                for (int d = 0; d <= log2(paddedN) - 1; d++)
                {
                    up_sweep<<<blocksPerGrid, blockSize>>>(paddedN, d, devData);
                }
                cudaDeviceSynchronize();
                // x[n-1] = 0
                cudaMemset(devData + paddedN - 1, 0, sizeof(int));

                for (int d = log2(paddedN) - 1; d >= 0; d--)
                {
                    down_sweep<<<blocksPerGrid, blockSize>>>(paddedN, d, devData);
                }
            }
            timer().endGpuTimer();

            cudaMemcpy(odata, devData, arrLen, cudaMemcpyDeviceToHost);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata)
        {
            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);

            const int arrSize = n * sizeof(int);

            int *devInData, *devOutData, *bools, *indices;
            cudaMalloc((void **)&devInData, arrSize);
            cudaMalloc((void **)&devOutData, arrSize);
            cudaMalloc((void **)&bools, arrSize);
            cudaMalloc((void **)&indices, arrSize);

            cudaMemcpy(devInData, idata, arrSize, cudaMemcpyHostToDevice);

            // timer().startGpuTimer();
            // TODO
            Common::kernMapToBoolean<<<blocksPerGrid, blockSize>>>(n, bools, devInData);
            scan(n, indices, bools);
            Common::kernScatter<<<blocksPerGrid, blockSize>>>(n, devOutData, devInData, bools, indices);

            // timer().endGpuTimer();

            cudaMemcpy(odata, devOutData, arrSize, cudaMemcpyDeviceToHost);

            int lastBool;
            int lastIndex;
            cudaMemcpy(&lastBool, bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastIndex, indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(bools);
            cudaFree(indices);

            return lastIndex + lastBool;
        }
    }
}
