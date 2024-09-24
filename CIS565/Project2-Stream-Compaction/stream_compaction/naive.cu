#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction
{
    namespace Naive
    {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer &timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // TODO: __global__

        __global__ void scan(const int n, int d, int *idata, int *odata)
        {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n)
            {
                return;
            }
            int offset = 1 << d - 1;
            if (index >= offset)
            {
                odata[index] = idata[index - offset] + idata[index];
            }
            else
            {
                odata[index] = idata[index];
            }
        }

        // This version can handle arrays only as large as can be processed by a single
        // thread block running on one multiprocessor of a GPU.
        __global__ void optScan(int *g_odata, const int *g_idata, int n)
        {
            extern __shared__ int temp[];

            int thid = threadIdx.x;
            if (thid >= n)
            {
                return;
            }
            int pOut = 0, pIn = 1;

            temp[pOut * n + thid] = thid > 0 ? g_idata[thid - 1] : 0;
            __syncthreads();

            for (int offset = 1; offset < n; offset *= 2)
            {
                pOut = 1 - pOut; // swap double buffer indices
                pIn = 1 - pOut;

                if (thid >= offset)
                {
                    temp[pOut * n + thid] = temp[pIn * n + thid] + temp[pIn * n + thid - offset];
                }
                else
                {
                    temp[pOut * n + thid] = temp[pIn * n + thid];
                }
                __syncthreads();
            }

            g_odata[thid] = temp[pOut * n + thid]; // write output
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata, bool useOpt)
        {
            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);
            const int arrLen = n * sizeof(int);

            int *devData1;
            int *devData2;

            cudaMalloc((void **)&devData1, arrLen);
            cudaMalloc((void **)&devData2, arrLen);
            cudaMemcpy(devData1, idata, arrLen, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            if (useOpt)
            {
                optScan<<<1, blockSize>>>(devData2, devData1, n);
                cudaMemcpy(odata, devData2, arrLen, cudaMemcpyDeviceToHost);
            }
            else
            {
                // TODO
                for (int d = 1; d <= ilog2ceil(n); d++)
                {
                    scan<<<blocksPerGrid, blockSize>>>(n, d, devData1, devData2);
                    std::swap(devData1, devData2);
                }
                odata[0] = 0;
                cudaMemcpy(odata + 1, devData1, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
            }
            timer().endGpuTimer();

            cudaFree(devData1);
            cudaFree(devData2);
        }
    }
}
