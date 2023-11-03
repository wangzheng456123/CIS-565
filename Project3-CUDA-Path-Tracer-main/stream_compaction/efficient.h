#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);

        void scanGPU(int n, int* odata, int* idata);

        int compactGPU(int n, int* odata, const int* idata);

        void radixSort(int n, int* odata, const int *idata);

        void dummpyTest();
    }
}
