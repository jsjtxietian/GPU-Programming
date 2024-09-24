/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include <cassert>
#include <algorithm>
#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>
#include <stream_compaction/radix.h>
#include "testing_helpers.hpp"

const int SIZE = 1 << 22; // feel free to change the size of array
const int NPOT = SIZE - 3; // Non-Power-Of-Two
int *a = new int[SIZE];
int *b = new int[SIZE];
int *c = new int[SIZE];

using namespace StreamCompaction::CPU;

void testGPUscan() {
    int idata[8] = { 0,1,2,3,4,5,6,7};
    int odata[8];
    printArray(8, idata, false);
    StreamCompaction::Naive::scan(8, odata, idata, true);
    printArray(8, odata, false);
}

void testRadixSort() {
    int idata[] = {4, 7, 2, 6, 3, 5, 1, 0};
    int odata[8];
    StreamCompaction::Radix::radixSort(8, odata, idata);
    //printArray(8, odata, false);
}

int main(int argc, char* argv[]) {
    
    // DIY test
    testGPUscan();

    // Scan tests
    printf("\n");
    printf("****************\n");
    printf("** SCAN TESTS **\n");
    printf("****************\n");

    genArray(SIZE - 1, a, 50);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;

    printf("Source Array A:\n");
    printArray(SIZE, a, true);

    // initialize b using StreamCompaction::CPU::scan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::scan is correct.
    // At first all cases passed because b && c are all zeroes.
    zeroArray(SIZE, b);
    printDesc("cpu scan, power-of-two");
    StreamCompaction::CPU::scan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(SIZE, b, true);

    zeroArray(SIZE, c);
    printDesc("cpu scan, non-power-of-two");
    StreamCompaction::CPU::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    //printArray(NPOT, b, true);
    printCmpResult(NPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("naive scan, power-of-two");
    StreamCompaction::Naive::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    if (SIZE <= blockSize) {
        zeroArray(SIZE, c);
        printDesc("naive scan opt, power-of-two");
        StreamCompaction::Naive::scan(SIZE, c, a, true);
        printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
        printArray(SIZE, c, true);
        printCmpResult(SIZE, b, c);
    }

    // For bug-finding only: Array of 1s to help find bugs in stream compaction or scan
    // onesArray(SIZE, c);
    // printDesc("1s array for finding bugs");
    // StreamCompaction::Naive::scan(SIZE, c, a);
    // printArray(SIZE, c, true);
    // printf("\n");

    zeroArray(SIZE, c);
    printDesc("naive scan, non-power-of-two");
    StreamCompaction::Naive::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(SIZE, c, true);
    printCmpResult(NPOT, b, c);

    if (SIZE <= blockSize) {
        zeroArray(SIZE, c);
        printDesc("naive scan opt, non-power-of-two");
        StreamCompaction::Naive::scan(NPOT, c, a, true);
        printElapsedTime(StreamCompaction::Naive::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
        //printArray(SIZE, c, true);
        printCmpResult(NPOT, b, c);
    }

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, power-of-two");
    StreamCompaction::Efficient::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient scan, non-power-of-two");
    StreamCompaction::Efficient::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);


    zeroArray(SIZE, c);
    printDesc("thrust scan, power-of-two");
    StreamCompaction::Thrust::scan(SIZE, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(SIZE, c, true);
    printCmpResult(SIZE, b, c);

    zeroArray(SIZE, c);
    printDesc("thrust scan, non-power-of-two");
    StreamCompaction::Thrust::scan(NPOT, c, a);
    printElapsedTime(StreamCompaction::Thrust::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(NPOT, c, true);
    printCmpResult(NPOT, b, c);

    int *sortedA = new int[SIZE];
    std::memcpy(sortedA, a, SIZE * sizeof(int)); 
    printDesc("quick sort");
    StreamCompaction::Common::PerformanceTimer quickSortTimer;
    quickSortTimer.startCpuTimer();
    std::sort(sortedA, sortedA + SIZE);
    quickSortTimer.endCpuTimer();
    printElapsedTime(quickSortTimer.getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");

    zeroArray(SIZE, c);
    printDesc("radix sort");
    StreamCompaction::Radix::radixSort(SIZE, c, a);
    printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printCmpResult(SIZE, sortedA, c);

    zeroArray(SIZE, c);
    printDesc("radix sort, non-power-of-two");
    StreamCompaction::Radix::radixSort(SIZE, c, a);
    printElapsedTime(StreamCompaction::Radix::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printCmpResult(NPOT, sortedA, c);

    printf("\n");
    printf("*****************************\n");
    printf("** STREAM COMPACTION TESTS **\n");
    printf("*****************************\n");

    // Compaction tests

    genArray(SIZE - 1, a, 4);  // Leave a 0 at the end to test that edge case
    a[SIZE - 1] = 0;
    printArray(SIZE, a, true);

    int count, expectedCount, expectedNPOT;

    // initialize b using StreamCompaction::CPU::compactWithoutScan you implement
    // We use b for further comparison. Make sure your StreamCompaction::CPU::compactWithoutScan is correct.
    zeroArray(SIZE, b);
    printDesc("cpu compact without scan, power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(SIZE, b, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedCount = count;
    printArray(count, b, true);
    printCmpLenResult(count, expectedCount, b, b);

    zeroArray(SIZE, c);
    printDesc("cpu compact without scan, non-power-of-two");
    count = StreamCompaction::CPU::compactWithoutScan(NPOT, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    expectedNPOT = count;
    printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    zeroArray(SIZE, c);
    printDesc("cpu compact with scan");
    count = StreamCompaction::CPU::compactWithScan(SIZE, c, a);
    printElapsedTime(StreamCompaction::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "(std::chrono Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, power-of-two");
    count = StreamCompaction::Efficient::compact(SIZE, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    printArray(count, c, true);
    printCmpLenResult(count, expectedCount, b, c);

    zeroArray(SIZE, c);
    printDesc("work-efficient compact, non-power-of-two");
    count = StreamCompaction::Efficient::compact(NPOT, c, a);
    printElapsedTime(StreamCompaction::Efficient::timer().getGpuElapsedTimeForPreviousOperation(), "(CUDA Measured)");
    //printArray(count, c, true);
    printCmpLenResult(count, expectedNPOT, b, c);

    system("pause"); // stop Win32 console from closing on exit
    delete[] a;
    delete[] b;
    delete[] c;
}
