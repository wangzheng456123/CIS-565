#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>


namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        const int blockSize = 256;
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        
        __global__ void kernBlockReduce(int *dev_odata, int *dev_idata, int n) {
            __shared__ int sharedMem[blockSize * 2];

            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            int thId = threadIdx.x;

            if (index * 2 + 1 >= n) return;

            sharedMem[2 * thId] = dev_idata[2 * index];
            sharedMem[2 * thId + 1] = dev_idata[2 * index + 1];

            __syncthreads();

            int distance = 1;
            for (int i = 1; i < blockSize * 2; i <<= 1) {
                int left = distance * (2 * thId + 1) - 1;
                int right = distance * (2 * thId + 2) - 1;

                if (right < 2 * blockSize) {

                    sharedMem[right] += sharedMem[left];

                    distance <<= 1;
                }

                __syncthreads();
            }

            dev_odata[blockIdx.x] = sharedMem[blockSize * 2 - 1];
        }

        __global__ void kernEfficientScanPerBlock(int *dev_odata, int *dev_idata, int n) {
            __shared__ int sharedMem[blockSize * 2];

            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            int thId = threadIdx.x;

            if (index * 2 + 1 >= n) return;

            sharedMem[2 * thId] = dev_idata[2 * index];
            sharedMem[2 * thId + 1] = dev_idata[2 * index + 1];

            __syncthreads();

            int distance = 1;
            for (int i = 1; i < blockSize * 2; i <<= 1) {
                int left = distance * (2 * thId + 1) - 1;
                int right = distance * (2 * thId + 2) - 1;

                if (right < 2 * blockSize) {
                    sharedMem[right] += sharedMem[left];
                }

                distance <<= 1;

                __syncthreads();
            }

            distance = blockSize;

            sharedMem[2 * blockSize - 1] = 0;

            for (int i = 1; i < blockSize * 2; i <<= 1) {
                int left = distance * (2 * thId + 1) - 1;
                int right = distance * (2 * thId + 2) - 1;

                if (right < 2 * blockSize) {
                    int t = sharedMem[left];
                    sharedMem[left] = sharedMem[right];
                    sharedMem[right] += t;
                }

                distance >>= 1;

                __syncthreads();
            }

            dev_odata[2 * index] = sharedMem[2 * thId];
            dev_odata[2 * index + 1] = sharedMem[2 * thId + 1];
        }

        __global__ void kernAddArray(int *dev_idata, int *dev_odata, int n, int stride) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (index >= n) return;

            int offsetIndex = index / stride;
            int offset = dev_idata[offsetIndex];

            dev_odata[index] += offset;
        }

        void scan(int n, int *odata, const int *idata) {
            // TODO
            int* dev_data;

            int level = ilog2ceil(n);
            int size = 1;
            while (level) size *= 2, level --;

            size = size < 2 * blockSize ? 2 * blockSize : size;

            cudaMalloc((void**)&dev_data, sizeof(int) * size);
            cudaMemcpy(dev_data, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            dim3 threadsPerBlcok(size / (2  * blockSize));

            int* intermediateSum;
            cudaMalloc((void**)&intermediateSum, sizeof(int) * threadsPerBlcok.x);

            kernBlockReduce<<<threadsPerBlcok, blockSize>>>(intermediateSum, dev_data, size);

            /*debug code
            int* host_intermediateSum = new int[threadsPerBlcok.x];
            int* host_data = new int[size];

            cudaMemcpy(host_intermediateSum, intermediateSum, sizeof(int) * threadsPerBlcok.x, cudaMemcpyDeviceToHost);
            */

            kernEfficientScanPerBlock << <1, threadsPerBlcok.x >> > (intermediateSum, intermediateSum, size);

            /*debug code
            cudaMemcpy(host_intermediateSum, intermediateSum, sizeof(int) * threadsPerBlcok.x, cudaMemcpyDeviceToHost);
            */

            kernEfficientScanPerBlock<<<threadsPerBlcok, blockSize >>>(dev_data, dev_data, size);
            

            /*
            debug code
            cudaMemcpy(host_data, dev_data, sizeof(int) * size, cudaMemcpyDeviceToHost);

            for (int i = 0; i < 10; i++)
                printf("%d\n", host_data[i]);*/

            threadsPerBlcok.x *= 2;

            kernAddArray << <threadsPerBlcok, blockSize >> > (intermediateSum, dev_data, size, blockSize * 2);

            /*
            cudaMemcpy(host_data, dev_data, sizeof(int) * size, cudaMemcpyDeviceToHost);
            */

            cudaMemcpy(odata, dev_data, sizeof(int) * n, cudaMemcpyDeviceToHost);
           

            cudaFree(intermediateSum);
            cudaFree(dev_data);
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
        int compact(int n, int *odata, const int *idata) {
            // TODO
            int* dev_idata;

            cudaMalloc((void**)&dev_idata, sizeof(int) * n);
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            int* bools;
            cudaMalloc((void**)&bools, sizeof(int) * n);

            dim3 threadsPerBlcok((n  + blockSize - 1) / blockSize);

            Common::kernMapToBoolean<<<threadsPerBlcok, blockSize>>>(n, bools, dev_idata);
            
            cudaMemcpy(odata, bools, sizeof(int) * n, cudaMemcpyDeviceToHost);
            
            scan(n, odata, odata);

            int cnt = odata[n - 1];

            int* indices;
            cudaMalloc((void**)&indices, sizeof(int) * n);
            cudaMemcpy(indices, odata, sizeof(int) * n, cudaMemcpyHostToDevice);

            int* dev_odata;
            cudaMalloc((void**)&dev_odata, sizeof(int) * n);

            Common::kernScatter << <threadsPerBlcok, blockSize >> > (n, dev_odata, dev_idata, bools, indices);

            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);

            return cnt;
        }

        __global__ void kernGetBit(int n, const int* idata, int* odata, int nBit) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (index >= n) return;

            int curBit = (idata[index] >> nBit) & 1;

            odata[index] = curBit ? 0 : 1;
        }

        __global__ void kernGetIndices(int n, const int* notZero, int* indices, int totZero) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (index >= n) return;

            indices[index] = notZero[index] ?  indices[index] : totZero + index - indices[index];
        }

        __global__ void kernReshuffle(int n, const int* indices, int* odata, const int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;

            if (index >= n) return;

            odata[indices[index]] = idata[index];
        }

        void radixSort(int n, int* odata, const int* idata) {
            int* dev_idata;
            int* dev_odata;

            int* dev_iszero;
            int* dev_indices;

            cudaMalloc((void**) &dev_idata, sizeof(int) * n);
            cudaMalloc((void**) &dev_odata, sizeof(int) * n);
            cudaMalloc((void**) &dev_iszero, sizeof(int) * n);
            cudaMalloc((void**) &dev_indices, sizeof(int) * n);

            int* host_idata = new int[n];

            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            int bitLen = 30;

            int blockSize = 128;

            dim3 threadPerBlock = (n + blockSize - 1) / blockSize;

            for (int i = 0; i < bitLen; i++) {
                kernGetBit<<<threadPerBlock, blockSize>>>(n, dev_idata, dev_iszero, i);

                int last;
                cudaMemcpy(&last, dev_iszero + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

                cudaMemcpy(host_idata, dev_iszero, sizeof(int) * n, cudaMemcpyDeviceToHost);

                scan(n, odata, host_idata);

                cudaMemcpy(dev_indices, odata, sizeof(int) * n, cudaMemcpyHostToDevice);

                kernGetIndices << <threadPerBlock, blockSize >> > (n, dev_iszero, dev_indices, last + odata[n - 1]);

                

                kernReshuffle << <threadPerBlock, blockSize >> > (n, dev_indices, dev_odata, dev_idata);

                

                if (i < bitLen - 1) {
                    int* tmp = dev_idata;
                    dev_idata = dev_odata;
                    dev_odata = tmp;
                }
            }

            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_iszero);
            cudaFree(dev_indices);
        }

        // scan When both idata and odata are device memory
        void scanGPU(int n, int* odata, int* idata) {
            // TODO
            int level = ilog2ceil(n);
            int size = 1;
            while (level) size *= 2, level--;

            size = size < 2 * blockSize ? 2 * blockSize : size;

            dim3 threadsPerBlcok(size / (2 * blockSize));

            int* intermediateSum;
            cudaMalloc((void**)&intermediateSum, sizeof(int) * size);

            kernBlockReduce << <threadsPerBlcok, blockSize >> > (intermediateSum, idata, size);

            // kernEfficientScanPerBlock << <1, threadsPerBlcok.x >> > (intermediateSum, intermediateSum, size);
            thrust::device_ptr<int> thrust_sum(intermediateSum);
            thrust::exclusive_scan(thrust_sum, thrust_sum + size, thrust_sum);

            kernEfficientScanPerBlock << <threadsPerBlcok, blockSize >> > (idata, idata, size);

            threadsPerBlcok.x *= 2;

            kernAddArray << <threadsPerBlcok, blockSize >> > (intermediateSum, idata, size, blockSize * 2);

            cudaMemcpy(odata, idata, sizeof(int) * n, cudaMemcpyHostToHost);

            cudaFree(intermediateSum);
        }

        // compact when both idata and odata are device memory
        int compactGPU(int n, int* odata, const int* idata) {
            int* bools;
            int* tmpOdata;
            cudaMalloc((void**)&bools, sizeof(int) * n);
            cudaMalloc((void**)&tmpOdata, sizeof(int) * n);

            dim3 threadsPerBlcok((n + blockSize - 1) / blockSize);

            Common::kernMapToBoolean << <threadsPerBlcok, blockSize >> > (n, bools, idata);

            cudaMemcpy(tmpOdata, bools, sizeof(int) * n, cudaMemcpyDeviceToDevice);

            scanGPU(n, tmpOdata, tmpOdata);
            // thrust::device_ptr<int> thrust_odata(tmpOdata);

            // thrust::exclusive_scan(thrust_odata, thrust_odata + n, thrust_odata);

            int cnt;

            cudaMemcpy(&cnt, &tmpOdata[n - 1], sizeof(int), cudaMemcpyDeviceToHost);

            Common::kernScatter << <threadsPerBlcok, blockSize >> > (n, odata, idata, bools, tmpOdata);

            cudaFree(bools);
            cudaFree(tmpOdata);

            return cnt;
        }

        void dummpyTest() {
            return;
        }
    }
}
