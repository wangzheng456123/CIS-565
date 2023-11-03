#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int tmp = idata[0];
            odata[0] = 0;
            for (int i = 1; i < n; i++) {
                int back = idata[i];
                odata[i] = tmp + odata[i - 1];
                tmp = back;
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int curIdx = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0)
                    odata[curIdx++] = idata[i];
            }
            timer().endCpuTimer();
            return curIdx;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int* bools;
            bools = new int[n];
            int* sum;
            sum = new int[n];
            for (int i = 0; i < n; i++) {
                if (idata[i]) bools[i] = 1;
                else bools[i] = 0;
            }
            scan(n, sum, bools);
    
            for (int i = 0; i < n - 1; i++)
                if (bools[i])
                    odata[sum[i]] = idata[i];
            timer().endCpuTimer();

            int res = sum[n - 1];

            delete[] bools;
            delete[] sum;

            return res;
        }
    }
}
