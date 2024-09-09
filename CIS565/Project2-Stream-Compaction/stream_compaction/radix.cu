#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"
#include "efficient.h"

namespace StreamCompaction
{
    namespace Radix
    {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer &timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void countBits(int n, int bit, const int *input, int *bitCount, int *bitCountNeg)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index < n)
            {
                int bitValue = (input[index] >> bit) & 1;
                bitCount[index] = bitValue;
                bitCountNeg[index] = bitValue == 0 ? 1 : 0;
            }
        }

        __global__ void getT(int n, int totalFalses, int *t, const int *f)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index < n)
            {
                t[index] = index - f[index] + totalFalses;
            }
        }

        __global__ void getD(int n, int *d, const int *b, const int *t, const int *f)
        {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index < n)
            {
                d[index] = b[index] ? t[index] : f[index];
            }
        }

        __global__ void scatter(int n, int *odata, const int *idata, const int *indices)
        {
            // TODO
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n)
            {
                return;
            }

            odata[indices[index]] = idata[index];
        }

        void radixSort(int n, int *odata, const int *idata)
        {
            dim3 blocksPerGrid((n + blockSize - 1) / blockSize);

            int arrLen = n * sizeof(int);
            // devBitCountNeg should be in a temporary buffer in shared memory
            // f array is address for writing false keys.

            int *devInput, *devOutput, *devBitCount, *devBitCountNeg, *devF, *devT, *devD;
            cudaMalloc((void **)&devInput, arrLen);
            cudaMalloc((void **)&devOutput, arrLen);
            cudaMalloc((void **)&devBitCount, arrLen);
            cudaMalloc((void **)&devBitCountNeg, arrLen);
            cudaMalloc((void **)&devF, arrLen);
            cudaMalloc((void **)&devT, arrLen);
            cudaMalloc((void **)&devD, arrLen);

            cudaMemcpy(devInput, idata, arrLen, cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            for (int bit = 0; bit < 32; bit++)
            {
                countBits<<<blocksPerGrid, blockSize>>>(n, bit, devInput, devBitCount, devBitCountNeg);
                Efficient::scan(n, devF, devBitCountNeg);

                int totalFalses = 0;
                {
                    int lastE;
                    int lastF;
                    cudaMemcpy(&lastE, devBitCountNeg + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                    cudaMemcpy(&lastF, devF + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
                    totalFalses = lastE + lastF;
                }

                getT<<<blocksPerGrid, blockSize>>>(n, totalFalses, devT, devF);
                // The address d does not need to be stored in an array, it can be computed when the scatter is executed.
                getD<<<blocksPerGrid, blockSize>>>(n, devD, devBitCount, devT, devF);
                scatter<<<blocksPerGrid, blockSize>>>(n, devOutput, devInput, devD);

                std::swap(devInput, devOutput);
            }

            timer().endGpuTimer();
            cudaMemcpy(odata, devInput, arrLen, cudaMemcpyDeviceToHost);
        }
    }
}
