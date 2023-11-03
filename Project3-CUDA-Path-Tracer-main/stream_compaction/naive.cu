#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void kernSumDistance(int n, int *idata, int *odata, int d) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (index >= n) return;

            odata[index] = idata[index];

            if (index < d) return;

            odata[index] = idata[index] + idata[index - d];
        }

        __global__ void kernSubstractInput(int n, int* idata, int* odata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (index >= n) return;

            odata[index] -= idata[index];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            int* dev_idata;
            int* dev_odata;

            cudaMalloc((void**) & dev_idata, n * sizeof(int));
            cudaMalloc((void**) & dev_odata, n * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 128;
            dim3 gridPerBlock = (n + blockSize - 1) / blockSize;

            for (int d = 1; d < n; d *= 2) {
                kernSumDistance<<<gridPerBlock, blockSize>>>(n, dev_idata, dev_odata, d);

                if (d * 2 < n) {

                    int* tmp = dev_odata;
                    dev_odata = dev_idata;

                    dev_idata = tmp;
                }
            }

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            kernSubstractInput<<<gridPerBlock, blockSize>>>(n, dev_idata, dev_odata);

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            
            cudaFree(dev_idata);
            cudaFree(dev_odata);

            timer().endGpuTimer();
        }
    }
}
