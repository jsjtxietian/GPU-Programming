#include <cstdio>
#include "cpu.h"

#include "common.h"
#include <vector>

namespace StreamCompaction
{
    namespace CPU
    {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer &timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum). exclusive
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata)
        {
            timer().startCpuTimer();
            odata[0] = 0;
            for (int i = 1; i < n; i++)
            {
                odata[i] = odata[i - 1] + idata[i - 1];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata)
        {
            timer().startCpuTimer();
            int count = 0;
            for (int i = 0; i < n; i++)
            {
                if (idata[i] != 0)
                {
                    odata[count++] = idata[i];
                }
            }
            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata)
        {
            std::vector<int> temp_bool(n, 0);
            std::vector<int> temp_scan(n, 0);

            // timer().startCpuTimer();

            for (int i = 0; i < n; i++)
            {
                temp_bool[i] = idata[i] != 0;
            }

            scan(n, temp_scan.data(), temp_bool.data());
            int count = temp_scan[n - 1] + temp_bool[n - 1];

            for (int i = 0; i < n; i++)
            {
                if (temp_bool[i] != 0)
                {
                    odata[temp_scan[i]] = idata[i];
                }
            }

            // timer().endCpuTimer();
            return count;
        }
    }
}
