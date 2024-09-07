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

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata)
        {
            const int blockSize = 128;
            dim3 threadsPerBlock((n + blockSize - 1) / blockSize);

            int *devData1;
            int *devData2;

            cudaMalloc((void **)&devData1, n * sizeof(int));
            cudaMalloc((void **)&devData2, n * sizeof(int));
            cudaMemcpy(devData1, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
            for (int d = 1; d <= ilog2ceil(n); d++)
            {
                scan<<<threadsPerBlock, blockSize>>>(n, d, devData1, devData2);
                std::swap(devData1, devData2);
            }

            timer().endGpuTimer();

            odata[0] = 0;
            cudaMemcpy(odata + 1, devData1, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(devData1);
            cudaFree(devData2);
        }
    }
}
