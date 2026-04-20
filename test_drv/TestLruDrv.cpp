/*
* Apache LRU Hash Table Kernel Sample/Test
* Copyright 2026 Alexander Danileiko
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at:
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* This software is provided on an "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS
* OF ANY KIND, either express or implied.
*/

// ----------------------------------------------------------------------------
// This file defines the master testing, validation, and benchmarking 
// orchestrator for the Windows Kernel-Mode CLruHashTable, evaluating its 
// stability and performance within the strict constraints of the NT executive.
//
// CORRECTNESS & INTEGRITY SUITE:
// A comprehensive battery of tests validating fundamental correctness, memory
// safety (BSOD prevention), and concurrency bounds. It covers basic 
// Add/Lookup/Remove operations, forced hash collisions, Free-List integrity, 
// and Lookaside List memory tracking. It also includes extreme edge-case 
// simulations, such as zero-capacity clamping, tiny-table eviction constraints, 
// and aggressive system-thread thrashing to verify safety against ABA hazards, 
// TOCTOU races, and paged pool memory leaks.
//
// PERFORMANCE & CONTENTION BENCHMARKS:
// A high-throughput benchmarking engine that measures operations per second
// across simulated kernel-mode workloads. It evaluates sequential versus
// multithreaded scaling, isolating PushLock contention, eviction thrashing, 
// and oversubscribed logical processors. It tests mixed read/write ratios 
// (e.g., read-heavy skewed distributions) and measures the efficiency of 
// background yielding trims.
//
// TAIL LATENCY & SCALING ANALYSIS:
// Captures precise nanosecond-level execution times via hardware performance 
// counters across millions of operations. It calculates percentile latency 
// bounds (P50 to P99.99) to evaluate algorithmic stability and OS scheduler 
// jitter. Operating without a standard library baseline, this suite focuses 
// strictly on proving absolute throughput, linear scalability, and robust 
// driver unload/rundown safety under massive concurrent load.
// 
// TEST RESULTS OUTPUT:
// DbgPrintEx(DPFLTR_IHVDRIVER_ID, ...) is used for all logging and output, 
// allowing real-time monitoring
// ----------------------------------------------------------------------------

#define TEST_IS_KM 1

#include <ntifs.h>
#include <ntstrsafe.h>
#include "LRUHashTable.h"

#define DRIVER_TAG 'TseT'

// ----------------------------------------------------------------------------
// Global Synchronization & Protection
// ----------------------------------------------------------------------------
PETHREAD             g_pMasterBenchmarkThread = NULL;
EX_RUNDOWN_REF       g_TestRundown;
volatile LONG        g_lAbortTests = 0;
PAGED_LOOKASIDE_LIST g_PayloadLookasideList;

// ----------------------------------------------------------------------------
// Kernel-Mode Global new/delete Overloads
// ----------------------------------------------------------------------------
void* __cdecl operator new(_In_ SIZE_T     uSize,
                           _In_ POOL_FLAGS ullPoolFlags,
                           _In_ ULONG      ulTag) noexcept
{
    return ExAllocatePool2(ullPoolFlags, uSize, ulTag);
}

void* __cdecl operator new[](_In_ SIZE_T     uSize,
                             _In_ POOL_FLAGS ullPoolFlags,
                             _In_ ULONG      ulTag) noexcept
{
    return ExAllocatePool2(ullPoolFlags, uSize, ulTag);
}

void __cdecl operator delete(_In_opt_ void* pMemory)
{
    if (pMemory)
    {
        ExFreePool(pMemory);
    }
}

void __cdecl operator delete(_In_opt_ void* pMemory,
                             _In_     SIZE_T uSize)
{
    UNREFERENCED_PARAMETER(uSize);

    if (pMemory)
    {
        ExFreePool(pMemory);
    }
}

void __cdecl operator delete[](_In_opt_ void* pMemory)
{
    if (pMemory)
    {
        ExFreePool(pMemory);
    }
}

void __cdecl operator delete[](_In_opt_ void* pMemory,
                               _In_     SIZE_T uSize)
{
    UNREFERENCED_PARAMETER(uSize);

    if (pMemory)
    {
        ExFreePool(pMemory);
    }
}

// ----------------------------------------------------------------------------
// Test Logging & Validation Macros
// ----------------------------------------------------------------------------
#define LOG_INFO(...) DbgPrintEx(DPFLTR_IHVDRIVER_ID, DPFLTR_INFO_LEVEL, __VA_ARGS__)
#define LOG_ERR(...)  DbgPrintEx(DPFLTR_IHVDRIVER_ID, DPFLTR_ERROR_LEVEL, __VA_ARGS__)

#define TEST_REQUIRE(condition, msg, retVal) \
    do \
    { \
        if (InterlockedCompareExchange(&g_lAbortTests, 0, 0)) \
        { \
            return retVal; \
        } \
        if (!(condition)) \
        { \
            LOG_ERR("[LRU]      [!] TEST FAILED: %s (Line %d)\n", msg, __LINE__); \
            return retVal; \
        } \
    } \
    while(0)

// ----------------------------------------------------------------------------
// Test Data Structures
// ----------------------------------------------------------------------------
volatile LONG g_lLiveObjectsCount = 0;

class RefCountedPayload
{
private:
    volatile LONG m_lRefCount;

public:
    UINT64 ullData;

    RefCountedPayload(_In_ UINT64 ullValue = 0) : m_lRefCount(1),
                                                  ullData(ullValue)
    {
        InterlockedIncrement(&g_lLiveObjectsCount);
    }

    ~RefCountedPayload()
    {
        InterlockedDecrement(&g_lLiveObjectsCount);
    }

    void AddRef() noexcept
    {
        InterlockedIncrement(&m_lRefCount);
    }

    void Release() noexcept
    {
        if (InterlockedDecrement(&m_lRefCount) == 0)
        {
            delete this;
        }
    }

    static void* __cdecl operator new(_In_ SIZE_T uSize) noexcept
    {
        UNREFERENCED_PARAMETER(uSize);
        return ExAllocateFromPagedLookasideList(&g_PayloadLookasideList);
    }

    static void __cdecl operator delete(_In_opt_ void* pMemory)
    {
        if (pMemory)
        {
            ExFreeToPagedLookasideList(&g_PayloadLookasideList, pMemory);
        }
    }
};

struct NtfsIdHasher
{
    static inline UINT64 ComputeHash(_In_ const UINT64& ullKey) noexcept
    {
        UINT64 ullZ = ullKey + 0x9e3779b97f4a7c15ULL;
        ullZ ^= (ullZ >> 33);

        ullZ *= 0xff51afd7ed558ccdULL;
        ullZ ^= (ullZ >> 33);

        ullZ *= 0xc4ceb9fe1a85ec53ULL;
        ullZ ^= (ullZ >> 33);

        return ullZ;
    }
};

struct DegradedHasher
{
    static inline UINT64 ComputeHash(_In_ const UINT64& ullKey) noexcept
    {
        UNREFERENCED_PARAMETER(ullKey);
        return 0xBADF00D;
    }
};

class FastRng
{
private:
    UINT32 m_u32State;

public:
    FastRng(_In_ UINT32 u32Seed) : m_u32State(u32Seed ? u32Seed : 0xBADF00D)
    {
    }

    UINT32 Next() noexcept
    {
        m_u32State ^= m_u32State << 13;
        m_u32State ^= m_u32State >> 17;
        m_u32State ^= m_u32State << 5;
        return m_u32State;
    }
};

using TestTable = CLruHashTable<UINT64, RefCountedPayload, NtfsIdHasher>;
using CollisionTable = CLruHashTable<UINT64, RefCountedPayload, DegradedHasher>;

struct PerformanceMetrics
{
    UINT64 ullAddThroughput;
    UINT64 ullLookupThroughput;
    UINT64 ullRemoveThroughput;
    UINT64 ullContentionThroughput;
    UINT64 ullReadHeavyThroughput0;
    UINT64 ullReadHeavyThroughput25;
    UINT64 ullReadHeavyThroughput50;
    UINT64 ullReadHeavyThroughput75;
    UINT64 ullReadHeavyThroughput100;
};

// ----------------------------------------------------------------------------
// Thread Management Utilities
// ----------------------------------------------------------------------------
// Increased to allow massive oversubscription on >64 core systems
#define MAX_TEST_THREADS 2048 

struct TEST_WORKER_CONTEXT
{
    ULONG          ulThreadId;
    PKEVENT        pStartEvent;
    volatile LONG* plStopFlag;
    PVOID          pUserContext;
};

typedef VOID(*PTEST_WORKER_FUNC)(_Inout_ TEST_WORKER_CONTEXT* pContext);

struct TEST_THREAD_MANAGER
{
    PETHREAD            pThreads[MAX_TEST_THREADS];
    TEST_WORKER_CONTEXT Contexts[MAX_TEST_THREADS];
    ULONG               ulThreadCount;
    KEVENT              StartEvent;
    volatile LONG       lStopFlag;
};

VOID StartThreads(_Inout_  TEST_THREAD_MANAGER* pMgr,
                  _In_     ULONG                ulCount,
                  _In_     PTEST_WORKER_FUNC    pFunc,
                  _In_opt_ PVOID                pUserContext)
{
    PAGED_CODE();

    if (ulCount > MAX_TEST_THREADS)
    {
        ulCount = MAX_TEST_THREADS;
    }

    pMgr->ulThreadCount = ulCount;
    pMgr->lStopFlag     = 0;

    KeInitializeEvent(&pMgr->StartEvent, NotificationEvent, FALSE);

    for (ULONG ulIndex = 0; ulIndex < ulCount; ++ulIndex)
    {
        pMgr->Contexts[ulIndex].ulThreadId = ulIndex;
        pMgr->Contexts[ulIndex].pStartEvent = &pMgr->StartEvent;
        pMgr->Contexts[ulIndex].plStopFlag = &pMgr->lStopFlag;
        pMgr->Contexts[ulIndex].pUserContext = pUserContext;

        HANDLE hThread;
        NTSTATUS ntStatus = PsCreateSystemThread(&hThread,
                                                 THREAD_ALL_ACCESS,
                                                 NULL,
                                                 NULL,
                                                 NULL,
                                                 (PKSTART_ROUTINE)pFunc,
                                                 &pMgr->Contexts[ulIndex]);

        if (NT_SUCCESS(ntStatus))
        {
            ntStatus = ObReferenceObjectByHandle(hThread,
                                                 THREAD_ALL_ACCESS,
                                                 NULL,
                                                 KernelMode,
                                                 (PVOID*)&pMgr->pThreads[ulIndex],
                                                 NULL);
            if (NT_SUCCESS(ntStatus))
            {                
                KeSetPriorityThread((PKTHREAD)pMgr->pThreads[ulIndex], 15);
            }
            else
            {
                // We created a thread but failed to secure an object reference to it.
                // We MUST signal it to stop immediately and wait for its handle, otherwise 
                // it becomes an orphaned runaway thread that bypasses StopAndWaitThreads.
                LOG_ERR("[LRU] [!] Failed to reference thread %u. Aborting test.\n", ulIndex);

                pMgr->pThreads[ulIndex] = NULL;

                InterlockedExchange(&pMgr->lStopFlag, 1);
                InterlockedExchange(&g_lAbortTests, 1);
                KeSetEvent(&pMgr->StartEvent, IO_NO_INCREMENT, FALSE);

                ZwWaitForSingleObject(hThread, FALSE, NULL);
            }

            ZwClose(hThread);
        }
        else
        {
            pMgr->pThreads[ulIndex] = NULL;
        }
    }

    KeSetEvent(&pMgr->StartEvent, IO_NO_INCREMENT, FALSE);
}

VOID StopAndWaitThreads(_Inout_ TEST_THREAD_MANAGER* pMgr,
                        _In_    int                  nSleepSeconds = 0)
{
    PAGED_CODE();

    if (nSleepSeconds > 0)
    {
        LARGE_INTEGER liDelay;
        liDelay.QuadPart = -1000000ll; // 100ms per iteration to allow abort checking
        int nIterations  = nSleepSeconds * 10;

        for (int nIndex = 0; nIndex < nIterations; ++nIndex)
        {
            if (InterlockedCompareExchange(&g_lAbortTests, 0, 0))
            {
                break;
            }

            KeDelayExecutionThread(KernelMode, FALSE, &liDelay);
        }
    }

    // Always ensure workers get the stop signal
    InterlockedExchange(&pMgr->lStopFlag, 1);

    // Chunked Waiting Logic
    PVOID* pValidThreads = new (POOL_FLAG_NON_PAGED, DRIVER_TAG) PVOID[pMgr->ulThreadCount];
    if (pValidThreads != NULL)
    {
        ULONG ulValidCount = 0;

        for (ULONG ulIndex = 0; ulIndex < pMgr->ulThreadCount; ++ulIndex)
        {
            if (pMgr->pThreads[ulIndex] != NULL)
            {
                pValidThreads[ulValidCount++] = pMgr->pThreads[ulIndex];
            }
        }

        ULONG ulRemaining = ulValidCount;
        ULONG ulOffset = 0;

        while (ulRemaining > 0)
        {
            ULONG ulWaitCount = (ulRemaining > MAXIMUM_WAIT_OBJECTS) ? MAXIMUM_WAIT_OBJECTS : ulRemaining;
            KWAIT_BLOCK WaitBlocks[MAXIMUM_WAIT_OBJECTS];

            KeWaitForMultipleObjects(ulWaitCount,
                                     &pValidThreads[ulOffset],
                                     WaitAll,
                                     Executive,
                                     KernelMode,
                                     FALSE,
                                     NULL,
                                     WaitBlocks);

            ulRemaining -= ulWaitCount;
            ulOffset    += ulWaitCount;
        }

        delete[] pValidThreads;
    }
    else
    {
        // If allocation fails, we MUST still wait for threads to prevent a BSOD on unload.
        LOG_ERR("[LRU] [!] Failed to allocate wait array. Falling back to single waits.\n");

        for (ULONG ulIndex = 0; ulIndex < pMgr->ulThreadCount; ++ulIndex)
        {
            if (pMgr->pThreads[ulIndex] != NULL)
            {
                KeWaitForSingleObject(pMgr->pThreads[ulIndex],
                                      Executive,
                                      KernelMode,
                                      FALSE,
                                      NULL);
            }
        }
    }

    // Dereference all thread objects now that they have terminated
    for (ULONG ulIndex = 0; ulIndex < pMgr->ulThreadCount; ++ulIndex)
    {
        if (pMgr->pThreads[ulIndex] != NULL)
        {
            ObDereferenceObject(pMgr->pThreads[ulIndex]);
        }
    }
}

VOID SetWorkerThreadAffinity(_In_ ULONG ulThreadId)
{
    ULONG ulTotalProcessors = KeQueryActiveProcessorCountEx(ALL_PROCESSOR_GROUPS);
    if (ulTotalProcessors == 0)
    {
        ulTotalProcessors = 1;
    }

    PROCESSOR_NUMBER procNum;
    if (NT_SUCCESS(KeGetProcessorNumberFromIndex(ulThreadId % ulTotalProcessors, &procNum)))
    {
        GROUP_AFFINITY affinity = { 0 };
        affinity.Group = procNum.Group;
        affinity.Mask  = (KAFFINITY)-1; // Allow execution on any core within this group
        
        // Correctly set the group affinity for the CURRENT thread
        KeSetSystemGroupAffinityThread(&affinity, NULL);
    }
}

// ----------------------------------------------------------------------------
// Custom C++ templated Kernel QuickSort Implementation
// ----------------------------------------------------------------------------
template<typename T, typename Compare>
void QuickSort(_Inout_updates_(uCount) T* pBase,
               _In_                       SIZE_T  uCount,
               _In_                       Compare Comp)
{
    if (pBase == nullptr || uCount <= 1)
    {
        return;
    }

    struct Range
    {
        SIZE_T uLow;
        SIZE_T uHigh;
    };

    Range Stack[64];
    SIZE_T uSp = 0;

    Stack[uSp].uLow = 0;
    Stack[uSp].uHigh = uCount - 1;
    ++uSp;

    while (uSp > 0)
    {
        --uSp;
        Range CurrentRange = Stack[uSp];

        SIZE_T uLow = CurrentRange.uLow;
        SIZE_T uHigh = CurrentRange.uHigh;

        while (uLow < uHigh)
        {
            SIZE_T uI = uLow;
            SIZE_T uJ = uHigh;
            SIZE_T uPivot = uLow + ((uHigh - uLow) >> 1);

            for (;;)
            {
                while (Comp(pBase[uI], pBase[uPivot]) < 0)
                {
                    ++uI;
                }

                while (Comp(pBase[uJ], pBase[uPivot]) > 0)
                {
                    if (uJ == 0)
                    {
                        break;
                    }

                    --uJ;
                }

                if (uI >= uJ)
                {
                    break;
                }

                T Tmp = pBase[uI];
                pBase[uI] = pBase[uJ];
                pBase[uJ] = Tmp;

                if (uI == uPivot)
                {
                    uPivot = uJ;
                }
                else if (uJ == uPivot)
                {
                    uPivot = uI;
                }

                ++uI;

                if (uJ == 0)
                {
                    break;
                }

                --uJ;
            }

            SIZE_T uLeftLow = uLow;
            SIZE_T uLeftHigh = (uJ > 0) ? (uJ - 1) : 0;
            SIZE_T uRightLow = uJ + 1;
            SIZE_T uRightHigh = uHigh;

            SIZE_T uLeftSize  = (uLeftHigh >= uLeftLow) ? (uLeftHigh - uLeftLow + 1) : 0;
            SIZE_T uRightSize = (uRightHigh >= uRightLow) ? (uRightHigh - uRightLow + 1) : 0;

            if (uLeftSize > 1 && uRightSize > 1)
            {
                if (uLeftSize < uRightSize)
                {
                    Stack[uSp].uLow = uRightLow;
                    Stack[uSp].uHigh = uRightHigh;
                    ++uSp;

                    uHigh = uLeftHigh;
                    uLow = uLeftLow;
                }
                else
                {
                    Stack[uSp].uLow = uLeftLow;
                    Stack[uSp].uHigh = uLeftHigh;
                    ++uSp;

                    uLow = uRightLow;
                    uHigh = uRightHigh;
                }
            }
            else if (uLeftSize > 1)
            {
                uHigh = uLeftHigh;
                uLow = uLeftLow;
            }
            else if (uRightSize > 1)
            {
                uLow = uRightLow;
                uHigh = uRightHigh;
            }
            else
            {
                break;
            }

            if (uSp >= RTL_NUMBER_OF(Stack))
            {
                for (SIZE_T uM = uLow + 1; uM <= uHigh; ++uM)
                {
                    SIZE_T uN = uM;

                    while (uN > uLow && Comp(pBase[uN], pBase[uN - 1]) < 0)
                    {
                        T Tmp = pBase[uN];
                        pBase[uN] = pBase[uN - 1];
                        pBase[uN - 1] = Tmp;
                        --uN;
                    }
                }

                uSp = 0;
                break;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Test Suites
// ----------------------------------------------------------------------------

/*
* Validates fundamental hash table correctness, including basic Add/Lookup/Remove,
* overwrite behaviors (Get-Or-Add), and natural LRU eviction bounds.
*/
BOOLEAN RunCorrectnessTests()
{
    PAGED_CODE();

    LOG_INFO("\n[LRU] [*] Running Correctness Tests...\n");

    TestTable Table;
    TEST_REQUIRE(NT_SUCCESS(Table.Initialize(1024)), "Failed to initialize table", FALSE);

    RefCountedPayload* pOut = NULL;

    TEST_REQUIRE(!Table.Lookup(999, pOut), "Lookup succeeded on an empty table", FALSE);

    RefCountedPayload* p1 = new RefCountedPayload(100);
    TEST_REQUIRE(p1 != NULL, "Allocation failed", FALSE);

    BOOLEAN bIsAdded = Table.Add(1, p1);
    TEST_REQUIRE(bIsAdded, "Initial Add failed", FALSE);
    p1->Release();

    if (Table.Lookup(1, pOut))
    {
        TEST_REQUIRE(pOut->ullData == 100, "Data mismatch", FALSE);
        pOut->Release();
    }
    else
    {
        TEST_REQUIRE(FALSE, "Lookup failed", FALSE);
    }

    // 1. Verify Get-Or-Add correctly returns the existing object pointer
    RefCountedPayload* pGetOrAdd = new RefCountedPayload(300);
    TEST_REQUIRE(pGetOrAdd != NULL, "Allocation failed", FALSE);

    RefCountedPayload* pExisting = NULL;
    // Because default is KeepIfExists, it should return FALSE and populate pExisting
    BOOLEAN bAdded = Table.Add(1, pGetOrAdd, &pExisting, AddAction::KeepIfExists);
    TEST_REQUIRE(!bAdded, "Get-Or-Add should return FALSE when key exists", FALSE);
    TEST_REQUIRE(pExisting != NULL, "Get-Or-Add out parameter is NULL", FALSE);
    TEST_REQUIRE(pExisting->ullData == 100, "Get-Or-Add out parameter data mismatch", FALSE);

    pGetOrAdd->Release(); // Safely releases the loser payload (caller retained ownership)
    pExisting->Release(); // Safely releases the returned reference from the table

    // 2. Verify Explicit Overwrite works
    RefCountedPayload* p2 = new RefCountedPayload(200);
    TEST_REQUIRE(p2 != NULL, "Allocation failed", FALSE);

    BOOLEAN bIsOverwritten = Table.Add(1, p2, NULL, AddAction::ReplaceIfExists);
    TEST_REQUIRE(bIsOverwritten, "Explicit Overwrite Add failed", FALSE);
    p2->Release();

    if (Table.Lookup(1, pOut))
    {
        TEST_REQUIRE(pOut->ullData == 200, "Lookup failed on overwrite", FALSE);
        pOut->Release();
    }
    else
    {
        TEST_REQUIRE(FALSE, "Lookup failed on overwrite", FALSE);
    }

    BOOLEAN bIsRemoved = Table.Remove(1);
    TEST_REQUIRE(bIsRemoved, "Remove failed", FALSE);
    TEST_REQUIRE(!Table.Lookup(1, pOut), "Item found after remove", FALSE);

    BOOLEAN bIsRemovedNonExistent = Table.Remove(999);
    TEST_REQUIRE(!bIsRemovedNonExistent, "Remove succeeded on non-existent key", FALSE);

    LOG_INFO("[LRU]      [-] Running bulk insert and eviction...\n");

    for (int nIndex = 1000; nIndex < 101000; ++nIndex)
    {
        if ((nIndex % 5000) == 0 && InterlockedCompareExchange(&g_lAbortTests, 0, 0))
        {
            return FALSE;
        }

        RefCountedPayload* pBulk = new RefCountedPayload(nIndex);

        if (pBulk)
        {
            (void)Table.Add(nIndex, pBulk);
            pBulk->Release();
        }
    }

    SIZE_T uTotalItems = Table.GetTotalItemCount();
    TEST_REQUIRE(uTotalItems < 100000 && uTotalItems > 0, "Natural LRU eviction failed to cap size", FALSE);

    LOG_INFO("[LRU]      [-] Running active trim...\n");
    SIZE_T uTrimmed = Table.Trim(500);

    TEST_REQUIRE(uTrimmed > 0, "Trim failed to evict any items from full shards", FALSE);
    TEST_REQUIRE(Table.GetTotalItemCount() <= uTotalItems - uTrimmed, "Item count mismatch after trim", FALSE);

    if (Table.Lookup(100999, pOut))
    {
        TEST_REQUIRE(pOut->ullData == 100999, "Failed to find MRU item data mismatch", FALSE);
        pOut->Release();
    }
    else
    {
        TEST_REQUIRE(FALSE, "Failed to find MRU item after bulk insert", FALSE);
    }

    LOG_INFO("[LRU] [+] Correctness Tests Passed.\n");
    return TRUE;
}

/*
* Tests edge cases on a highly constrained table (e.g., capacity of 10),
* ensuring strict LRU ordering and correct eviction of the oldest items under pressure.
*/
BOOLEAN RunTinyTableTest()
{
    PAGED_CODE();

    LOG_INFO("\n[LRU] [*] Running Tiny Table Tests...\n");

    TestTable Table;
    TEST_REQUIRE(NT_SUCCESS(Table.Initialize(10, 0)), "Failed to initialize table", FALSE);

    SIZE_T uMemUsage = Table.GetTotalMemoryUsage();
    LOG_INFO("[LRU]      - Tiny Table Memory Footprint: %llu bytes\n", uMemUsage);

    for (int nIndex = 0; nIndex < 25; ++nIndex)
    {
        if (InterlockedCompareExchange(&g_lAbortTests, 0, 0))
        {
            return FALSE;
        }

        RefCountedPayload* pPayload = new RefCountedPayload(nIndex);
        TEST_REQUIRE(pPayload != NULL, "Allocation failed", FALSE);

        BOOLEAN bRes = Table.Add(nIndex, pPayload);
        pPayload->Release();
        TEST_REQUIRE(bRes, "Add failed on tiny table", FALSE);
    }

    SIZE_T uItemCount = Table.GetTotalItemCount();
    LOG_INFO("[LRU]      - Item Count after 25 insertions: %llu\n", uItemCount);
    TEST_REQUIRE(uItemCount == 10, "Failed to keep expected number of items", FALSE);

    RefCountedPayload* pOut = NULL;

    for (int nIndex = 0; nIndex < 15; ++nIndex)
    {
        if (Table.Lookup(nIndex, pOut))
        {
            pOut->Release();
            TEST_REQUIRE(FALSE, "Old item was not evicted", FALSE);
        }
    }

    for (int nIndex = 15; nIndex < 25; ++nIndex)
    {
        if (Table.Lookup(nIndex, pOut))
        {
            TEST_REQUIRE(pOut->ullData == nIndex, "Data mismatch in tiny table", FALSE);
            pOut->Release();
        }
        else
        {
            TEST_REQUIRE(FALSE, "Recent item was improperly evicted", FALSE);
        }
    }

    SIZE_T uTrimmed = Table.Trim(3);

    if (uTrimmed > 0)
    {
        LOG_INFO("[LRU]      - Trimmed %llu items from tiny table.\n", uTrimmed);
    }

    LOG_INFO("[LRU] [+] Tiny Table Tests Passed.\n");
    return TRUE;
}

/*
* Validates the internal Free-List mechanics by completely filling, draining,
* and refilling the table, ensuring no nodes are leaked or orphaned.
*/
BOOLEAN RunFreeListIntegrityTest()
{
    PAGED_CODE();

    LOG_INFO("\n[LRU] [*] Running Free-List Integrity Test...\n");

    TestTable Table;
    TEST_REQUIRE(NT_SUCCESS(Table.Initialize(32, 0)), "Failed to initialize table", FALSE);

    for (int nIndex = 0; nIndex < 32; ++nIndex)
    {
        if (InterlockedCompareExchange(&g_lAbortTests, 0, 0))
        {
            return FALSE;
        }

        RefCountedPayload* pPayload = new RefCountedPayload(nIndex);

        if (pPayload)
        {
            (void)Table.Add(nIndex, pPayload);
            pPayload->Release();
        }
    }

    TEST_REQUIRE(Table.GetTotalItemCount() == 32, "Table failed to reach capacity", FALSE);

    for (int nIndex = 0; nIndex < 32; ++nIndex)
    {
        BOOLEAN bRes = Table.Remove(nIndex);
        TEST_REQUIRE(bRes, "Remove failed during drain phase", FALSE);
    }

    TEST_REQUIRE(Table.GetTotalItemCount() == 0, "Table failed to drain completely", FALSE);

    for (int nIndex = 100; nIndex < 132; ++nIndex)
    {
        if (InterlockedCompareExchange(&g_lAbortTests, 0, 0))
        {
            return FALSE;
        }

        RefCountedPayload* pPayload = new RefCountedPayload(nIndex);

        if (pPayload)
        {
            BOOLEAN bRes = Table.Add(nIndex, pPayload);
            pPayload->Release();
            TEST_REQUIRE(bRes, "Add failed during refill phase", FALSE);
        }
    }

    TEST_REQUIRE(Table.GetTotalItemCount() == 32, "Table failed to refill to capacity", FALSE);

    LOG_INFO("[LRU] [+] Free-List Integrity Test Passed.\n");
    return TRUE;
}

/*
* Ensures the table correctly handles degenerate initialization requests
* by clamping zero-capacity to a safe minimum bound.
*/
BOOLEAN RunZeroCapacitySanityTest()
{
    PAGED_CODE();

    LOG_INFO("\n[LRU] [*] Running Zero Capacity Sanity Test...\n");

    TestTable Table;
    TEST_REQUIRE(NT_SUCCESS(Table.Initialize(0, 0)), "Initialize failed", FALSE);

    SIZE_T uMemUsage = Table.GetTotalMemoryUsage();
    TEST_REQUIRE(uMemUsage > 0, "Memory usage is zero, clamping failed", FALSE);

    RefCountedPayload* pPayload = new RefCountedPayload(99);

    if (pPayload)
    {
        BOOLEAN bRes = Table.Add(99, pPayload);
        pPayload->Release();
        TEST_REQUIRE(bRes, "Add failed on clamped table", FALSE);
    }

    RefCountedPayload* pOut = NULL;

    if (Table.Lookup(99, pOut))
    {
        pOut->Release();
    }
    else
    {
        TEST_REQUIRE(FALSE, "Lookup failed on clamped zero-capacity table", FALSE);
    }

    LOG_INFO("[LRU] [+] Zero Capacity Sanity Test Passed.\n");
    return TRUE;
}

/*
* Forces artificial hash collisions to verify the integrity and correct traversal
* of the intra-array singly-linked collision chains.
*/
BOOLEAN RunHashCollisionTest()
{
    PAGED_CODE();

    LOG_INFO("\n[LRU] [*] Running Severe Hash Collision Test...\n");

    CollisionTable Table;
    TEST_REQUIRE(NT_SUCCESS(Table.Initialize(256, 0)), "Initialize failed", FALSE);

    for (int nIndex = 0; nIndex < 50; ++nIndex)
    {
        if (InterlockedCompareExchange(&g_lAbortTests, 0, 0))
        {
            return FALSE;
        }

        RefCountedPayload* pPayload = new RefCountedPayload(nIndex);

        if (pPayload)
        {
            BOOLEAN bRes = Table.Add(nIndex, pPayload);
            pPayload->Release();
            TEST_REQUIRE(bRes, "Add failed during collision test", FALSE);
        }
    }

    RefCountedPayload* pOut = NULL;

    for (int nIndex = 0; nIndex < 50; ++nIndex)
    {
        if (InterlockedCompareExchange(&g_lAbortTests, 0, 0))
        {
            return FALSE;
        }

        if (Table.Lookup(nIndex, pOut))
        {
            TEST_REQUIRE(pOut->ullData == nIndex, "Data mismatch during traversal", FALSE);
            pOut->Release();
        }
        else
        {
            TEST_REQUIRE(FALSE, "Lookup failed to traverse collision chain", FALSE);
        }
    }

    TEST_REQUIRE(Table.Remove(49), "Failed to remove head of chain", FALSE);
    TEST_REQUIRE(Table.Remove(25), "Failed to remove middle of chain", FALSE);
    TEST_REQUIRE(Table.Remove(0), "Failed to remove tail of chain", FALSE);

    TEST_REQUIRE(!Table.Lookup(49, pOut), "Head ghost item found", FALSE);
    TEST_REQUIRE(!Table.Lookup(25, pOut), "Middle ghost item found", FALSE);
    TEST_REQUIRE(!Table.Lookup(0, pOut), "Tail ghost item found", FALSE);

    LOG_INFO("[LRU] [+] Severe Hash Collision Test Passed.\n");
    return TRUE;
}

struct EvictionExhaustionCtx
{
    TestTable* pTable;
};

VOID EvictionWorker(_Inout_ TEST_WORKER_CONTEXT* pCtx)
{
    PAGED_CODE();

    SetWorkerThreadAffinity(pCtx->ulThreadId);

    EvictionExhaustionCtx* pEvictionCtx = (EvictionExhaustionCtx*)pCtx->pUserContext;
    KeWaitForSingleObject(pCtx->pStartEvent, Executive, KernelMode, FALSE, NULL);

    if (ExAcquireRundownProtection(&g_TestRundown))
    {
        for (int nIndex = 0; nIndex < 50000; ++nIndex)
        {
            if (InterlockedCompareExchange(&g_lAbortTests, 0, 0))
            {
                break;
            }

            UINT64 ullKey = (pCtx->ulThreadId * 50000) + nIndex;
            RefCountedPayload* pPayload = new RefCountedPayload(ullKey);

            if (pPayload)
            {
                (void)pEvictionCtx->pTable->Add(ullKey, pPayload);
                pPayload->Release();
            }

            if (nIndex % 10 == 0)
            {
                RefCountedPayload* pOut = NULL;

                if (pEvictionCtx->pTable->Lookup(ullKey, pOut))
                {
                    pOut->Release();
                }
            }
        }

        ExReleaseRundownProtection(&g_TestRundown);
    }

    PsTerminateSystemThread(STATUS_SUCCESS);
}

/*
* A stress test designed to rapidly cycle through a small cache capacity,
* ensuring memory is reclaimed properly without leaks during heavy multi-threaded turnover.
*/
BOOLEAN RunEvictionAndExhaustionTest()
{
    PAGED_CODE();

    LOG_INFO("\n[LRU] [*] Running Eviction & Exhaustion Tests...\n");

    TestTable Table;
    TEST_REQUIRE(NT_SUCCESS(Table.Initialize(256)), "Initialize failed", FALSE);

    EvictionExhaustionCtx Ctx;
    Ctx.pTable = &Table;

    TEST_THREAD_MANAGER* pMgr = new (POOL_FLAG_NON_PAGED, DRIVER_TAG) TEST_THREAD_MANAGER();

    if (!pMgr)
    {
        LOG_ERR("[LRU]      [!] Failed to allocate TEST_THREAD_MANAGER.\n");
        return FALSE;
    }

    StartThreads(pMgr, 8, EvictionWorker, &Ctx);
    StopAndWaitThreads(pMgr, 0);

    delete pMgr;
    Table.Cleanup();

    TEST_REQUIRE(g_lLiveObjectsCount == 0, "MEMORY LEAK DETECTED: Objects stranded.", FALSE);

    LOG_INFO("[LRU] [+] Eviction & Exhaust Tests Passed.\n");
    return TRUE;
}

struct MtCorrectnessCtx
{
    TestTable* pTable;
    SIZE_T        uCacheCapacity;
    UINT64        ullKeySpace;
    volatile LONG lDataCorruption;
};

VOID MtCorrectnessWorker(_Inout_ TEST_WORKER_CONTEXT* pCtx)
{
    PAGED_CODE();

    SetWorkerThreadAffinity(pCtx->ulThreadId);

    MtCorrectnessCtx* pMtCtx = (MtCorrectnessCtx*)pCtx->pUserContext;
    FastRng Rng(pCtx->ulThreadId + 9999);

    KeWaitForSingleObject(pCtx->pStartEvent, Executive, KernelMode, FALSE, NULL);

    if (ExAcquireRundownProtection(&g_TestRundown))
    {
        const int OPS_PER_THREAD = 100000;

        for (int i = 0; i < OPS_PER_THREAD; ++i)
        {
            if (InterlockedCompareExchange(pCtx->plStopFlag, 0, 0) || InterlockedCompareExchange(&g_lAbortTests, 0, 0))
            {
                break;
            }

            UINT64 ullKey = Rng.Next() % pMtCtx->ullKeySpace;
            UINT32 u32OpType = Rng.Next() % 100;

            if (u32OpType < 50)
            {
                // 50% Lookup
                RefCountedPayload* pOut = NULL;

                if (pMtCtx->pTable->Lookup(ullKey, pOut))
                {
                    // Strict Data Integrity Check
                    if (pOut->ullData != ullKey)
                    {
                        InterlockedExchange(&pMtCtx->lDataCorruption, 1);
                    }

                    pOut->Release();
                }
            }
            else if (u32OpType < 75)
            {
                // 25% Add
                RefCountedPayload* pPayload = new RefCountedPayload(ullKey);

                if (pPayload)
                {
                    (void)pMtCtx->pTable->Add(ullKey, pPayload);
                    pPayload->Release();
                }
            }
            else
            {
                // 25% Remove
                (void)pMtCtx->pTable->Remove(ullKey);
            }
        }

        ExReleaseRundownProtection(&g_TestRundown);
    }

    PsTerminateSystemThread(STATUS_SUCCESS);
}

/*
* A highly aggressive multi-threaded test running on a tiny cache capacity to force
* simultaneous ABA hazards, TOCTOU races, and heavy eviction thrashing.
*/
BOOLEAN RunMultiThreadedCorrectnessTest()
{
    PAGED_CODE();

    LOG_INFO("\n[LRU] [*] Running Multi-Threaded Correctness Test (High Contention & Eviction)...\n");

    TestTable Table;
    const SIZE_T CACHE_CAP = 64;
    const UINT64 KEY_SPACE = 200;

    TEST_REQUIRE(NT_SUCCESS(Table.Initialize(CACHE_CAP, 0)), "Failed to initialize table", FALSE);

    ULONG ulThreadCount = KeQueryActiveProcessorCountEx(ALL_PROCESSOR_GROUPS);

    if (ulThreadCount == 0)
    {
        ulThreadCount = 4;
    }

    if (ulThreadCount > MAX_TEST_THREADS)
    {
        ulThreadCount = MAX_TEST_THREADS;
    }

    MtCorrectnessCtx Ctx = { 0 };
    Ctx.pTable = &Table;
    Ctx.uCacheCapacity = CACHE_CAP;
    Ctx.ullKeySpace = KEY_SPACE;

    TEST_THREAD_MANAGER* pMgr = new (POOL_FLAG_NON_PAGED, DRIVER_TAG) TEST_THREAD_MANAGER();

    if (!pMgr)
    {
        LOG_ERR("[LRU]      [!] Failed to allocate TEST_THREAD_MANAGER.\n");
        return FALSE;
    }

    StartThreads(pMgr, ulThreadCount, MtCorrectnessWorker, &Ctx);
    StopAndWaitThreads(pMgr, 0);

    delete pMgr;

    TEST_REQUIRE(Ctx.lDataCorruption == 0, "Data corruption detected! (Value data mismatched the mapped Key)", FALSE);

    SIZE_T uFinalCount = Table.GetTotalItemCount();
    TEST_REQUIRE(uFinalCount <= CACHE_CAP, "Item count exceeds cache capacity bounds!", FALSE);
    TEST_REQUIRE(uFinalCount <= KEY_SPACE, "Item count exceeds unique keys inserted!", FALSE);

    SIZE_T uEnumCount = 0;

    Table.Enumerate([&](const UINT64& ullKey,
        RefCountedPayload* pVal) -> BOOLEAN
        {
            if (pVal)
            {
                if (pVal->ullData != ullKey)
                {
                    InterlockedExchange(&Ctx.lDataCorruption, 1);
                }
                uEnumCount++;
            }

            return TRUE;
        });

    TEST_REQUIRE(Ctx.lDataCorruption == 0, "Data corruption detected during Enumerate pass!", FALSE);
    TEST_REQUIRE(uEnumCount == uFinalCount, "Enumerate count does not match GetTotalItemCount! (Structural corruption)", FALSE);

    LOG_INFO("[LRU] [+] Multi-Threaded Correctness Test Passed.\n");
    return TRUE;
}

/*
* Measures uncontended, single-threaded sequential throughput for Add, Lookup,
* and Remove operations. Establishes the baseline performance metrics.
*/
PerformanceMetrics RunPerformanceTest(_In_   SIZE_T      uCacheCapacity,
                                      _In_   SIZE_T      uPrePopulateCount,
                                      _In_   SIZE_T      uOperationCount,
                                      _In_z_ const char* szTestName,
                                      _In_   UINT32      u32PromotionThreshold = 0)
{
    PAGED_CODE();

    LOG_INFO("\n[LRU] [*] Running Performance Test: %s (Cap: %llu, Threshold: %u%%)...\n", szTestName, uCacheCapacity, u32PromotionThreshold);

    PerformanceMetrics Metrics = { 0 };
    TestTable Table;

    if (!NT_SUCCESS(Table.Initialize(uCacheCapacity, u32PromotionThreshold)))
    {
        LOG_ERR("[LRU]      [!] Initialize failed\n");
        return Metrics;
    }

    SIZE_T uMemoryBytes = Table.GetTotalMemoryUsage();
    UINT64 ullMbWhole = uMemoryBytes / (1024 * 1024);
    UINT64 ullMbFrac = ((uMemoryBytes % (1024 * 1024)) * 100) / (1024 * 1024);

    LOG_INFO("[LRU]      - Estimated Memory Footprint: %llu bytes (%llu.%02llu MB)\n", uMemoryBytes, ullMbWhole, ullMbFrac);

    for (SIZE_T uI = 0; uI < uPrePopulateCount; ++uI)
    {
        if ((uI % 10000) == 0 && InterlockedCompareExchange(&g_lAbortTests, 0, 0))
        {
            return Metrics;
        }

        RefCountedPayload* pPayload = new RefCountedPayload(uI);

        if (pPayload)
        {
            (void)Table.Add(uI, pPayload);
            pPayload->Release();
        }
        else
        {
            LOG_ERR("[LRU]      [!] Pre-populate allocation failed at index %llu\n", uI);
        }
    }

    const int nPasses = 4;
    LARGE_INTEGER liFreq, liStart, liEnd;
    KeQueryPerformanceCounter(&liFreq);

    UINT64 ullAddTicks = 0;
    UINT64 ullLookupTicks = 0;
    UINT64 ullRemoveTicks = 0;

    for (int nPass = 0; nPass < nPasses; ++nPass)
    {
        if (InterlockedCompareExchange(&g_lAbortTests, 0, 0))
        {
            return Metrics;
        }

        liStart = KeQueryPerformanceCounter(NULL);

        for (SIZE_T uI = uPrePopulateCount; uI < uPrePopulateCount + uOperationCount; ++uI)
        {
            if ((uI % 10000) == 0 && InterlockedCompareExchange(&g_lAbortTests, 0, 0))
            {
                return Metrics;
            }

            RefCountedPayload* pPayload = new RefCountedPayload(uI);

            if (pPayload)
            {
                (void)Table.Add(uI, pPayload);
                pPayload->Release();
            }
        }

        liEnd = KeQueryPerformanceCounter(NULL);
        ullAddTicks += (liEnd.QuadPart - liStart.QuadPart);

        liStart = KeQueryPerformanceCounter(NULL);

        for (SIZE_T uI = uPrePopulateCount; uI < uPrePopulateCount + uOperationCount; ++uI)
        {
            if ((uI % 10000) == 0 && InterlockedCompareExchange(&g_lAbortTests, 0, 0))
            {
                return Metrics;
            }

            RefCountedPayload* pOut = NULL;

            if (Table.Lookup(uI, pOut))
            {
                pOut->Release();
            }
        }

        liEnd = KeQueryPerformanceCounter(NULL);
        ullLookupTicks += (liEnd.QuadPart - liStart.QuadPart);

        liStart = KeQueryPerformanceCounter(NULL);

        for (SIZE_T uI = uPrePopulateCount; uI < uPrePopulateCount + uOperationCount; ++uI)
        {
            if ((uI % 10000) == 0 && InterlockedCompareExchange(&g_lAbortTests, 0, 0))
            {
                return Metrics;
            }

            (void)Table.Remove(uI);
        }

        liEnd = KeQueryPerformanceCounter(NULL);
        ullRemoveTicks += (liEnd.QuadPart - liStart.QuadPart);
    }

    auto printMetrics = [&](const char* szOpName,
                            SIZE_T      uOps,
                            UINT64      ullTicks) -> UINT64
        {
            if (ullTicks == 0)
            {
                ullTicks = 1;
            }

            UINT64 ullElapsedMs = (ullTicks * 1000) / liFreq.QuadPart;
            UINT64 ullTimePerOpNs = (ullTicks * 1000000000ULL) / (uOps * liFreq.QuadPart);
            UINT64 ullThroughput = (uOps * liFreq.QuadPart) / ullTicks;

            LOG_INFO("[LRU]      - %s %llu items:\n", szOpName, uOps);
            LOG_INFO("[LRU]        Total Time  : %llu ms\n", ullElapsedMs);
            LOG_INFO("[LRU]        Time per Op : %llu ns\n", ullTimePerOpNs);
            LOG_INFO("[LRU]        Throughput  : %llu Ops/sec\n", ullThroughput);

            return ullThroughput;
        };

    SIZE_T uTotalOps = uOperationCount * nPasses;

    Metrics.ullAddThroughput = printMetrics("Add", uTotalOps, ullAddTicks);
    Metrics.ullLookupThroughput = printMetrics("Lookup", uTotalOps, ullLookupTicks);
    Metrics.ullRemoveThroughput = printMetrics("Remove", uTotalOps, ullRemoveTicks);

    LOG_INFO("[LRU] [+] %s Complete.\n", szTestName);
    return Metrics;
}

struct ContentionCtx
{
    TestTable* pTable;
    volatile LONG64 llTotalOps;
    volatile LONG64 llTotalTrimmed;
    SIZE_T          uCacheCapacity;
    UINT32          u32ReadPercent;
    volatile LONG   lWarmUpFlag;
};

VOID ContentionWorker(_Inout_ TEST_WORKER_CONTEXT* pCtx)
{
    PAGED_CODE();

    SetWorkerThreadAffinity(pCtx->ulThreadId);

    ContentionCtx* pContentionCtx = (ContentionCtx*)pCtx->pUserContext;
    FastRng Rng(pCtx->ulThreadId + 1000);

    KeWaitForSingleObject(pCtx->pStartEvent, Executive, KernelMode, FALSE, NULL);

    if (ExAcquireRundownProtection(&g_TestRundown))
    {
        UINT64 ullLocalOps = 0;
        UINT64 ullMaxKey = pContentionCtx->uCacheCapacity + (pContentionCtx->uCacheCapacity / 2);
        UINT64 ullHotSet = ullMaxKey / 5;

        // Warm-up phase (No artificial yielding needed at Priority 15)
        while (InterlockedCompareExchange(&pContentionCtx->lWarmUpFlag, 0, 0) != 0)
        {
            if (InterlockedCompareExchange(&g_lAbortTests, 0, 0))
            {
                break;
            }

            UINT64 ullKey = Rng.Next() % ullMaxKey;
            RefCountedPayload* pOut = NULL;

            if (pContentionCtx->pTable->Lookup(ullKey, pOut))
            {
                pOut->Release();
            }
        }

        // Timed benchmark phase
        while (!InterlockedCompareExchange(pCtx->plStopFlag, 0, 0))
        {
            if (InterlockedCompareExchange(&g_lAbortTests, 0, 0))
            {
                break;
            }

            UINT64 ullKey;

            if (pContentionCtx->u32ReadPercent >= 80)
            {
                if ((Rng.Next() % 100) < 80)
                {
                    ullKey = Rng.Next() % ullHotSet;
                }
                else
                {
                    ullKey = Rng.Next() % ullMaxKey;
                }
            }
            else
            {
                ullKey = Rng.Next() % ullMaxKey;
            }

            UINT32 u32OpType = Rng.Next() % 100;

            if (u32OpType < pContentionCtx->u32ReadPercent)
            {
                RefCountedPayload* pOut = NULL;

                if (pContentionCtx->pTable->Lookup(ullKey, pOut))
                {
                    pOut->Release();
                }
            }
            else if (u32OpType < pContentionCtx->u32ReadPercent + ((100 - pContentionCtx->u32ReadPercent) / 2))
            {
                RefCountedPayload* pPayload = new RefCountedPayload(ullKey);
                if (pPayload)
                {
                    (void)pContentionCtx->pTable->Add(ullKey, pPayload);
                    pPayload->Release();
                }
            }
            else
            {
                (void)pContentionCtx->pTable->Remove(ullKey);
            }

            ullLocalOps++;
        }

        InterlockedExchangeAdd64(&pContentionCtx->llTotalOps, ullLocalOps);
        ExReleaseRundownProtection(&g_TestRundown);
    }

    PsTerminateSystemThread(STATUS_SUCCESS);
}

VOID ContentionTrimmer(_Inout_ TEST_WORKER_CONTEXT* pCtx)
{
    PAGED_CODE();

    SetWorkerThreadAffinity(pCtx->ulThreadId);

    ContentionCtx* pContentionCtx = (ContentionCtx*)pCtx->pUserContext;
    KeWaitForSingleObject(pCtx->pStartEvent, Executive, KernelMode, FALSE, NULL);

    if (ExAcquireRundownProtection(&g_TestRundown))
    {
        SIZE_T uTrimTarget = pContentionCtx->uCacheCapacity / 20;
        UINT64 ullLocalTrimmed = 0;

        LARGE_INTEGER liDelay;
        liDelay.QuadPart = -500000ll; // 50ms

        // Trimmer also waits out the warm-up phase
        while (InterlockedCompareExchange(&pContentionCtx->lWarmUpFlag, 0, 0) != 0)
        {
            if (InterlockedCompareExchange(&g_lAbortTests, 0, 0))
            {
                break;
            }

            KeDelayExecutionThread(KernelMode, FALSE, &liDelay);
        }

        while (!InterlockedCompareExchange(pCtx->plStopFlag, 0, 0))
        {
            if (InterlockedCompareExchange(&g_lAbortTests, 0, 0))
            {
                break;
            }

            KeDelayExecutionThread(KernelMode, FALSE, &liDelay);
            ullLocalTrimmed += pContentionCtx->pTable->Trim(uTrimTarget);
        }

        InterlockedExchangeAdd64(&pContentionCtx->llTotalTrimmed, ullLocalTrimmed);
        ExReleaseRundownProtection(&g_TestRundown);
    }

    PsTerminateSystemThread(STATUS_SUCCESS);
}

VOID MixedWorkloadSelector(_Inout_ TEST_WORKER_CONTEXT* pCtx)
{
    PAGED_CODE();

    if (pCtx->ulThreadId == 0)
    {
        ContentionTrimmer(pCtx);
    }
    else
    {
        ContentionWorker(pCtx);
    }
}

/*
* Measures highly concurrent multi-threaded throughput on a massive capacity table
* where no evictions occur. Isolates lock contention and routing efficiency.
* Includes a 1-second warm-up phase.
*/
BOOLEAN RunContentionTest_NoEviction(_In_ int nSecondsToRun)
{
    PAGED_CODE();

    LOG_INFO("\n[LRU] [*] Running Contention Test: Lightly Populated (No Evictions) for %d seconds...\n", nSecondsToRun);

    TestTable Table;
    TEST_REQUIRE(NT_SUCCESS(Table.Initialize(10000000)), "Failed to init", FALSE);

    ContentionCtx Ctx = { 0 };
    Ctx.pTable         = &Table;
    Ctx.uCacheCapacity = 50000;
    Ctx.u32ReadPercent = 70;
    Ctx.lWarmUpFlag    = 1;

    ULONG ulThreadCount = KeQueryActiveProcessorCountEx(ALL_PROCESSOR_GROUPS);

    if (ulThreadCount == 0)
    {
        ulThreadCount = 4;
    }

    if (ulThreadCount > MAX_TEST_THREADS)
    {
        ulThreadCount = MAX_TEST_THREADS;
    }

    LARGE_INTEGER liFreq, liStart, liEnd;
    KeQueryPerformanceCounter(&liFreq);

    TEST_THREAD_MANAGER* pMgr = new (POOL_FLAG_NON_PAGED, DRIVER_TAG) TEST_THREAD_MANAGER();

    if (!pMgr)
    {
        LOG_ERR("[LRU]      [!] Failed to allocate TEST_THREAD_MANAGER.\n");
        return FALSE;
    }

    StartThreads(pMgr, ulThreadCount, ContentionWorker, &Ctx);

    LOG_INFO("[LRU]      - Warming up for 1 second...\n");
    LARGE_INTEGER liWarmup;
    liWarmup.QuadPart = -10000000ll;
    KeDelayExecutionThread(KernelMode, FALSE, &liWarmup);

    LOG_INFO("[LRU]      - Warm-up complete. Running benchmark for %d seconds...\n", nSecondsToRun);

    InterlockedExchange(&Ctx.lWarmUpFlag, 0);
    liStart = KeQueryPerformanceCounter(NULL);

    StopAndWaitThreads(pMgr, nSecondsToRun);
    liEnd = KeQueryPerformanceCounter(NULL);

    delete pMgr;

    UINT64 ullTicks = liEnd.QuadPart - liStart.QuadPart;

    if (ullTicks == 0)
    {
        ullTicks = 1;
    }

    UINT64 ullOps = Ctx.llTotalOps;

    if (ullOps == 0)
    {
        ullOps = 1;
    }

    UINT64 ullTimePerOpNs = (ullTicks * 1000000000ULL) / (ullOps * liFreq.QuadPart);
    UINT64 ullThroughput = (Ctx.llTotalOps * liFreq.QuadPart) / ullTicks;

    LOG_INFO("[LRU]      - Threads: %u\n", ulThreadCount);
    LOG_INFO("[LRU]      - Total Operations: %llu\n", Ctx.llTotalOps);
    LOG_INFO("[LRU]      - Time per Op: %llu ns\n", ullTimePerOpNs);
    LOG_INFO("[LRU]      - Ops/Second: %llu\n", ullThroughput);
    LOG_INFO("[LRU] [+] Contention Test Complete.\n");

    return TRUE;
}

/*
* Measures multi-threaded throughput on a severely constrained table, forcing constant
* out-of-lock destruction and background yielding under heavy lock pressure.
* Includes a 1-second warm-up phase.
*/
BOOLEAN RunContentionTest_EvictionThrashing(_In_ int nSecondsToRun)
{
    PAGED_CODE();

    LOG_INFO("\n[LRU] [*] Running Contention Test: Full Table (Thrashing) for %d seconds...\n", nSecondsToRun);

    TestTable Table;
    TEST_REQUIRE(NT_SUCCESS(Table.Initialize(1000)), "Failed to init", FALSE);

    ContentionCtx Ctx = { 0 };
    Ctx.pTable         = &Table;
    Ctx.uCacheCapacity = 5000000;
    Ctx.u32ReadPercent = 20;
    Ctx.lWarmUpFlag    = 1;

    ULONG ulThreadCount = KeQueryActiveProcessorCountEx(ALL_PROCESSOR_GROUPS);

    if (ulThreadCount == 0)
    {
        ulThreadCount = 4;
    }

    if (ulThreadCount > MAX_TEST_THREADS)
    {
        ulThreadCount = MAX_TEST_THREADS;
    }

    LARGE_INTEGER liFreq, liStart, liEnd;
    KeQueryPerformanceCounter(&liFreq);

    TEST_THREAD_MANAGER* pMgr = new (POOL_FLAG_NON_PAGED, DRIVER_TAG) TEST_THREAD_MANAGER();

    if (!pMgr)
    {
        LOG_ERR("[LRU]      [!] Failed to allocate TEST_THREAD_MANAGER.\n");
        return FALSE;
    }

    StartThreads(pMgr, ulThreadCount, ContentionWorker, &Ctx);

    LOG_INFO("[LRU]      - Warming up for 1 second...\n");
    LARGE_INTEGER liWarmup;
    liWarmup.QuadPart = -10000000ll;
    KeDelayExecutionThread(KernelMode, FALSE, &liWarmup);

    LOG_INFO("[LRU]      - Warm-up complete. Running benchmark for %d seconds...\n", nSecondsToRun);

    InterlockedExchange(&Ctx.lWarmUpFlag, 0);
    liStart = KeQueryPerformanceCounter(NULL);

    StopAndWaitThreads(pMgr, nSecondsToRun);
    liEnd = KeQueryPerformanceCounter(NULL);

    delete pMgr;
    Table.Cleanup();

    UINT64 ullTicks = liEnd.QuadPart - liStart.QuadPart;

    if (ullTicks == 0)
    {
        ullTicks = 1;
    }

    UINT64 ullOps = Ctx.llTotalOps;

    if (ullOps == 0)
    {
        ullOps = 1;
    }

    UINT64 ullTimePerOpNs = (ullTicks * 1000000000ULL) / (ullOps * liFreq.QuadPart);
    UINT64 ullThroughput = (Ctx.llTotalOps * liFreq.QuadPart) / ullTicks;

    LOG_INFO("[LRU]      - Threads: %u\n", ulThreadCount);
    LOG_INFO("[LRU]      - Total Operations: %llu\n", Ctx.llTotalOps);
    LOG_INFO("[LRU]      - Time per Op: %llu ns\n", ullTimePerOpNs);
    LOG_INFO("[LRU]      - Ops/Second: %llu\n", ullThroughput);
    LOG_INFO("[LRU] [+] Contention Test Complete.\n");

    return TRUE;
}

/*
* Simulates a realistic production environment with a mix of Add, Lookup, Remove,
* and a dedicated background Trimmer thread. Includes a 1-second warm-up phase.
*/
UINT64 RunContentionTest_MixedWorkload(_In_ int    nSecondsToRun,
                                       _In_ SIZE_T uCacheCapacity,
                                       _In_ UINT32 u32PromotionThreshold = 100)
{
    PAGED_CODE();

    LOG_INFO("\n[LRU] [*] Running Mixed Workload for %d seconds (Cap: %llu, Thresh: %u%%)...\n", nSecondsToRun, uCacheCapacity, u32PromotionThreshold);

    TestTable Table;

    if (!NT_SUCCESS(Table.Initialize(uCacheCapacity, u32PromotionThreshold)))
    {
        return 0;
    }

    for (SIZE_T uI = 0; uI < uCacheCapacity; ++uI)
    {
        if (InterlockedCompareExchange(&g_lAbortTests, 0, 0))
        {
            return 0;
        }

        RefCountedPayload* pPayload = new RefCountedPayload(uI);

        if (pPayload)
        {
            (void)Table.Add(uI, pPayload);
            pPayload->Release();
        }
    }

    ContentionCtx Ctx = { 0 };
    Ctx.pTable = &Table;
    Ctx.uCacheCapacity = uCacheCapacity;
    Ctx.u32ReadPercent = 60;
    Ctx.lWarmUpFlag = 1;

    ULONG ulThreadCount = KeQueryActiveProcessorCountEx(ALL_PROCESSOR_GROUPS);
    if (ulThreadCount == 0)
    {
        ulThreadCount = 4;
    }

    if (ulThreadCount > MAX_TEST_THREADS)
    {
        ulThreadCount = MAX_TEST_THREADS;
    }

    LARGE_INTEGER liFreq, liStart, liEnd;
    KeQueryPerformanceCounter(&liFreq);

    TEST_THREAD_MANAGER* pMgr = new (POOL_FLAG_NON_PAGED, DRIVER_TAG) TEST_THREAD_MANAGER();

    if (!pMgr)
    {
        LOG_ERR("[LRU]      [!] Failed to allocate TEST_THREAD_MANAGER.\n");
        return 0;
    }

    StartThreads(pMgr, ulThreadCount, MixedWorkloadSelector, &Ctx);

    LOG_INFO("[LRU]      - Warming up for 1 second...\n");
    LARGE_INTEGER liWarmup;
    liWarmup.QuadPart = -10000000ll;
    KeDelayExecutionThread(KernelMode, FALSE, &liWarmup);

    LOG_INFO("[LRU]      - Warm-up complete. Running benchmark for %d seconds...\n", nSecondsToRun);

    InterlockedExchange(&Ctx.lWarmUpFlag, 0);
    liStart = KeQueryPerformanceCounter(NULL);

    StopAndWaitThreads(pMgr, nSecondsToRun);
    liEnd = KeQueryPerformanceCounter(NULL);

    delete pMgr;
    Table.Cleanup();

    UINT64 ullTicks = liEnd.QuadPart - liStart.QuadPart;

    if (ullTicks == 0)
    {
        ullTicks = 1;
    }

    UINT64 ullOps = Ctx.llTotalOps;

    if (ullOps == 0)
    {
        ullOps = 1;
    }

    UINT64 ullTimePerOpNs = (ullTicks * 1000000000ULL) / (ullOps * liFreq.QuadPart);
    UINT64 ullThroughput = (Ctx.llTotalOps * liFreq.QuadPart) / ullTicks;

    LOG_INFO("[LRU]      - Threads: %u (1 Trimmer, %u Workers)\n", ulThreadCount, ulThreadCount - 1);
    LOG_INFO("[LRU]      - Total Operations: %llu\n", Ctx.llTotalOps);
    LOG_INFO("[LRU]      - Total Trimmed: %llu\n", Ctx.llTotalTrimmed);
    LOG_INFO("[LRU]      - Time per Op: %llu ns\n", ullTimePerOpNs);
    LOG_INFO("[LRU]      - Ops/Second: %llu\n", ullThroughput);
    LOG_INFO("[LRU] [+] Mixed Workload Complete.\n");

    return ullThroughput;
}

/*
* Simulates a read-heavy workload (95% Lookup) with an uneven key distribution
* (Pareto/Zipfian approximation). Evaluates the impact of Lazy LRU Promotion.
* Includes a 1-second warm-up phase.
*/
UINT64 RunContentionTest_ReadHeavySkewed(_In_ int    nSecondsToRun,
                                         _In_ SIZE_T uCacheCapacity,
                                         _In_ UINT32 u32PromotionThreshold = 100)
{
    PAGED_CODE();

    LOG_INFO("\n[LRU] [*] Running Read-Heavy Skewed (95%% Read) for %d seconds (Cap: %llu, Thresh: %u%%)...\n", nSecondsToRun, uCacheCapacity, u32PromotionThreshold);

    TestTable Table;

    if (!NT_SUCCESS(Table.Initialize(uCacheCapacity, u32PromotionThreshold)))
    {
        return 0;
    }

    for (SIZE_T uI = 0; uI < uCacheCapacity; ++uI)
    {
        if (InterlockedCompareExchange(&g_lAbortTests, 0, 0))
        {
            return 0;
        }

        RefCountedPayload* pPayload = new RefCountedPayload(uI);

        if (pPayload)
        {
            (void)Table.Add(uI, pPayload);
            pPayload->Release();
        }
    }

    ContentionCtx Ctx = { 0 };
    Ctx.pTable         = &Table;
    Ctx.uCacheCapacity = uCacheCapacity;
    Ctx.u32ReadPercent = 95;
    Ctx.lWarmUpFlag    = 1;

    ULONG ulThreadCount = KeQueryActiveProcessorCountEx(ALL_PROCESSOR_GROUPS);

    if (ulThreadCount == 0)
    {
        ulThreadCount = 4;
    }

    if (ulThreadCount > MAX_TEST_THREADS)
    {
        ulThreadCount = MAX_TEST_THREADS;
    }

    LARGE_INTEGER liFreq, liStart, liEnd;
    KeQueryPerformanceCounter(&liFreq);

    TEST_THREAD_MANAGER* pMgr = new (POOL_FLAG_NON_PAGED, DRIVER_TAG) TEST_THREAD_MANAGER();

    if (!pMgr)
    {
        LOG_ERR("[LRU]      [!] Failed to allocate TEST_THREAD_MANAGER.\n");
        return 0;
    }

    StartThreads(pMgr, ulThreadCount, MixedWorkloadSelector, &Ctx);

    LOG_INFO("[LRU]      - Warming up for 1 second...\n");
    LARGE_INTEGER liWarmup;
    liWarmup.QuadPart = -10000000ll;
    KeDelayExecutionThread(KernelMode, FALSE, &liWarmup);

    LOG_INFO("[LRU]      - Warm-up complete. Running benchmark for %d seconds...\n", nSecondsToRun);

    InterlockedExchange(&Ctx.lWarmUpFlag, 0);
    liStart = KeQueryPerformanceCounter(NULL);

    StopAndWaitThreads(pMgr, nSecondsToRun);
    liEnd = KeQueryPerformanceCounter(NULL);

    delete pMgr;
    Table.Cleanup();

    UINT64 ullTicks = liEnd.QuadPart - liStart.QuadPart;

    if (ullTicks == 0)
    {
        ullTicks = 1;
    }

    UINT64 ullOps = Ctx.llTotalOps;

    if (ullOps == 0)
    {
        ullOps = 1;
    }

    UINT64 ullTimePerOpNs = (ullTicks * 1000000000ULL) / (ullOps * liFreq.QuadPart);
    UINT64 ullThroughput = (Ctx.llTotalOps * liFreq.QuadPart) / ullTicks;

    LOG_INFO("[LRU]      - Threads: %u (1 Trimmer, %u Workers)\n", ulThreadCount, ulThreadCount - 1);
    LOG_INFO("[LRU]      - Total Operations: %llu\n", Ctx.llTotalOps);
    LOG_INFO("[LRU]      - Total Trimmed: %llu\n", Ctx.llTotalTrimmed);
    LOG_INFO("[LRU]      - Time per Op: %llu ns\n", ullTimePerOpNs);
    LOG_INFO("[LRU]      - Ops/Second: %llu\n", ullThroughput);
    LOG_INFO("[LRU] [+] Read-Heavy Skewed Complete.\n");

    return ullThroughput;
}

// ----------------------------------------------------------------------------
// Trim Performance Test - Pure Integer Math
// ----------------------------------------------------------------------------
/*
* Measures the sheer speed and efficiency of the background yielding trim operation
* when instructed to remove a specific percentage of the table.
*/
BOOLEAN RunTrimPerformanceTest(_In_ SIZE_T uCacheCapacity,
                               _In_ ULONG  ulTrimPercentage)
{
    PAGED_CODE();

    LOG_INFO("\n[LRU] [*] Running Trim Performance Test (Cap: %llu, Trim: %u%%)...\n",
             uCacheCapacity,
             ulTrimPercentage);

    TestTable Table;
    TEST_REQUIRE(NT_SUCCESS(Table.Initialize(uCacheCapacity)), "Failed to init", FALSE);

    for (SIZE_T uI = 0; uI < uCacheCapacity; ++uI)
    {
        if ((uI % 10000) == 0 && InterlockedCompareExchange(&g_lAbortTests, 0, 0))
        {
            return FALSE;
        }

        RefCountedPayload* pPayload = new RefCountedPayload(uI);

        if (pPayload)
        {
            (void)Table.Add(uI, pPayload);
            pPayload->Release();
        }
        else
        {
            LOG_ERR("[LRU]      [!] Trim-populate allocation failed at %llu\n", uI);
        }
    }

    // Calculate items to trim using integer math: (Cap * Percent) / 100
    SIZE_T uItemsToTrim = (uCacheCapacity * ulTrimPercentage) / 100;

    LARGE_INTEGER liFreq, liStart, liEnd;
    KeQueryPerformanceCounter(&liFreq);

    liStart = KeQueryPerformanceCounter(NULL);
    SIZE_T uItemsActuallyTrimmed = Table.Trim(uItemsToTrim);
    liEnd = KeQueryPerformanceCounter(NULL);

    UINT64 ullTicks = liEnd.QuadPart - liStart.QuadPart;

    if (ullTicks == 0)
    {
        ullTicks = 1;
    }

    UINT64 ullElapsedMs = (ullTicks * 1000) / liFreq.QuadPart;
    UINT64 ullTimePerOpNs = 0;
    UINT64 ullThroughput = 0;

    if (uItemsActuallyTrimmed > 0)
    {
        ullTimePerOpNs = (ullTicks * 1000000000ULL) / (uItemsActuallyTrimmed * liFreq.QuadPart);
        ullThroughput = (uItemsActuallyTrimmed * liFreq.QuadPart) / ullTicks;
    }

    LOG_INFO("[LRU]      - Target Trim Count : %llu items\n", uItemsToTrim);
    LOG_INFO("[LRU]      - Actual Trim Count : %llu items\n", uItemsActuallyTrimmed);
    LOG_INFO("[LRU]      - Total Trim Time   : %llu ms\n", ullElapsedMs);

    if (uItemsActuallyTrimmed > 0)
    {
        LOG_INFO("[LRU]      - Time per Item     : %llu ns\n", ullTimePerOpNs);
        LOG_INFO("[LRU]      - Trim Throughput   : %llu Ops/sec\n", ullThroughput);
    }

    LOG_INFO("[LRU] [+] Trim Performance Test Complete.\n");
    return TRUE;
}

/*
* Validates the high/low watermark background trimming logic by overfilling
* the table and calling Trim(0) to force stabilization.
*/
BOOLEAN RunTrimToWatermarkTest(_In_ SIZE_T uCacheCapacity)
{
    PAGED_CODE();

    LOG_INFO("\n[LRU] [*] Running Trim-to-Watermark Test (Trim(0)) (Cap: %llu)...\n", uCacheCapacity);

    TestTable Table;
    TEST_REQUIRE(NT_SUCCESS(Table.Initialize(uCacheCapacity)), "Failed to init", FALSE);

    for (SIZE_T uI = 0; uI < uCacheCapacity; ++uI)
    {
        if ((uI % 10000) == 0 && InterlockedCompareExchange(&g_lAbortTests, 0, 0))
        {
            return FALSE;
        }

        RefCountedPayload* pPayload = new RefCountedPayload(uI);

        if (pPayload)
        {
            (void)Table.Add(uI, pPayload);
            pPayload->Release();
        }
        else
        {
            LOG_ERR("[LRU]      [!] Trim-populate allocation failed at %llu\n", uI);
        }
    }

    SIZE_T uItemsBeforeTrim = Table.GetTotalItemCount();
    LOG_INFO("[LRU]      - Items before Trim(0) : %llu\n", uItemsBeforeTrim);

    LARGE_INTEGER liFreq, liStart, liEnd;
    KeQueryPerformanceCounter(&liFreq);

    liStart = KeQueryPerformanceCounter(NULL);
    SIZE_T uItemsActuallyTrimmed = Table.Trim(0);
    liEnd = KeQueryPerformanceCounter(NULL);

    SIZE_T uItemsAfterTrim = Table.GetTotalItemCount();
    LOG_INFO("[LRU]      - Items after Trim(0)  : %llu\n", uItemsAfterTrim);
    LOG_INFO("[LRU]      - Actual Trim Count    : %llu items\n", uItemsActuallyTrimmed);

    UINT64 ullTicks = liEnd.QuadPart - liStart.QuadPart;
    if (ullTicks == 0)
    {
        ullTicks = 1;
    }

    UINT64 ullElapsedMs = (ullTicks * 1000) / liFreq.QuadPart;

    LOG_INFO("[LRU]      - Total Trim(0) Time   : %llu ms\n", ullElapsedMs);

    if (uItemsActuallyTrimmed > 0)
    {
        UINT64 ullTimePerOpNs = (ullTicks * 1000000000ULL) / (uItemsActuallyTrimmed * liFreq.QuadPart);
        UINT64 ullThroughput = (uItemsActuallyTrimmed * liFreq.QuadPart) / ullTicks;

        LOG_INFO("[LRU]      - Time per Item         : %llu ns\n", ullTimePerOpNs);
        LOG_INFO("[LRU]      - Trim Throughput       : %llu Ops/sec\n", ullThroughput);
    }

    TEST_REQUIRE(uItemsActuallyTrimmed > 0, "Trim(0) did not remove any items", FALSE);
    TEST_REQUIRE(uItemsBeforeTrim == (uItemsAfterTrim + uItemsActuallyTrimmed), "Trim count mismatch", FALSE);

    SIZE_T uExpectedMaxItems = (uCacheCapacity * 90) / 100;

    TEST_REQUIRE(uItemsAfterTrim <= (uExpectedMaxItems + 256), "Trim(0) failed to reach watermark bounds", FALSE);

    LOG_INFO("[LRU] [+] Trim-to-Watermark Test Complete.\n");
    return TRUE;
}

/*
* Validates the forced-trim override, ensuring that Trim(N, TRUE) bypasses
* capacity watermarks and strictly evicts the requested number of items.
*/
BOOLEAN RunForcedTrimTest(_In_ SIZE_T uCacheCapacity)
{
    PAGED_CODE();

    LOG_INFO("\n[LRU] [*] Running Forced Trim Test (Trim(Count, TRUE)) (Cap: %llu)...\n", uCacheCapacity);

    TestTable Table;

    TEST_REQUIRE(NT_SUCCESS(Table.Initialize(uCacheCapacity)), "Failed to init", FALSE);

    // ------------------------------------------------------------------------
    // Phase 1: Test Complete Eviction via Trim(0, TRUE)
    // ------------------------------------------------------------------------
    for (SIZE_T uI = 0; uI < uCacheCapacity; ++uI)
    {
        if ((uI % 10000) == 0 && InterlockedCompareExchange(&g_lAbortTests, 0, 0))
        {
            return FALSE;
        }

        RefCountedPayload* pPayload = new RefCountedPayload(uI);

        if (pPayload)
        {
            (void)Table.Add(uI, pPayload);
            pPayload->Release();
        }
    }

    SIZE_T uFullCount = Table.GetTotalItemCount();

    TEST_REQUIRE(uFullCount > 0, "Table failed to populate", FALSE);

    SIZE_T uTrimmedAll      = Table.Trim(0, TRUE);
    SIZE_T uCountAfterClear = Table.GetTotalItemCount();

    LOG_INFO("[LRU]      - Trim(0, TRUE) removed %llu items.\n", uTrimmedAll);

    TEST_REQUIRE(uCountAfterClear == 0, "Trim(0, TRUE) failed to empty table", FALSE);
    TEST_REQUIRE(uTrimmedAll == uFullCount, "Trim(0, TRUE) return count mismatch", FALSE);

    // ------------------------------------------------------------------------
    // Phase 2: Test Targeted Forced Eviction Below Watermarks
    // ------------------------------------------------------------------------
    for (SIZE_T uI = 0; uI < 10; ++uI)
    {
        RefCountedPayload* pPayload = new RefCountedPayload(uI);

        if (pPayload)
        {
            (void)Table.Add(uI, pPayload);
            pPayload->Release();
        }
    }

    SIZE_T uCountBeforeForced = Table.GetTotalItemCount();

    TEST_REQUIRE(uCountBeforeForced == 10, "Failed to repopulate 10 items", FALSE);

    // Verify normal trim respects the watermark and does nothing
    SIZE_T uNormalTrim = Table.Trim(5, FALSE);

    TEST_REQUIRE(uNormalTrim == 0, "Normal trim incorrectly removed items below watermark", FALSE);

    // Verify forced trim ignores the watermark and removes exactly 5
    SIZE_T uForcedTrim = Table.Trim(5, TRUE);
    SIZE_T uCountAfterForced = Table.GetTotalItemCount();

    LOG_INFO("[LRU]      - Trim(5, TRUE) removed %llu items from a 10-item table.\n", uForcedTrim);

    TEST_REQUIRE(uForcedTrim == 5, "Trim(5, TRUE) failed to remove exactly 5 items", FALSE);
    TEST_REQUIRE(uCountAfterForced == 5, "Table count is incorrect after forced trim", FALSE);

    LOG_INFO("[LRU] [+] Forced Trim Test Complete.\n");

    return TRUE;
}

/*
* Tests the safe traversal of all active nodes in the table via the
* Enumerate callback, including validation of early-abort mechanics.
*/
BOOLEAN RunEnumerateTest(_In_ SIZE_T uCacheCapacity)
{
    PAGED_CODE();

    LOG_INFO("\n[LRU] [*] Running Enumerate Test (Cap: %llu)...\n", uCacheCapacity);

    TestTable Table;

    // Initialize with 2x capacity to guarantee no premature evictions 
    // due to imperfect hash distribution across the physical shards.
    TEST_REQUIRE(NT_SUCCESS(Table.Initialize(uCacheCapacity * 2)), "Failed to init", FALSE);

    UINT64 ullExpectedSum = 0;

    for (SIZE_T uI = 0; uI < uCacheCapacity; ++uI)
    {
        if ((uI % 10000) == 0 && InterlockedCompareExchange(&g_lAbortTests, 0, 0))
        {
            return FALSE;
        }

        RefCountedPayload* pPayload = new RefCountedPayload(uI);

        if (pPayload)
        {
            (void)Table.Add(uI, pPayload);
            pPayload->Release();
            ullExpectedSum += uI;
        }
    }

    SIZE_T uCount = 0;
    UINT64 ullActualSum = 0;

    // ------------------------------------------------------------------------
    // Phase 1: Test Complete Enumeration and Data Integrity
    // ------------------------------------------------------------------------
    Table.Enumerate([&](const UINT64&      ullKey,
                        RefCountedPayload* pValue) -> BOOLEAN
        {
            if (pValue)
            {
                uCount++;
                ullActualSum += ullKey;
            }

            return TRUE;
        });

    LOG_INFO("[LRU]      - Enumerated %llu items.\n", uCount);

    TEST_REQUIRE(uCount == uCacheCapacity, "Enumerate did not visit all items", FALSE);
    TEST_REQUIRE(ullActualSum == ullExpectedSum, "Enumerate data mismatch (checksum failed)", FALSE);

    // ------------------------------------------------------------------------
    // Phase 2: Test Early Abort
    // ------------------------------------------------------------------------
    SIZE_T uAbortCount = 0;
    const SIZE_T uTargetAbort = uCacheCapacity / 2;

    Table.Enumerate([&](const UINT64&      ullKey,
                        RefCountedPayload* pValue) -> BOOLEAN
        {
            UNREFERENCED_PARAMETER(ullKey);
            UNREFERENCED_PARAMETER(pValue);

            uAbortCount++;

            if (uAbortCount >= uTargetAbort)
            {
                return FALSE;
            }

            return TRUE;
        });

    LOG_INFO("[LRU]      - Early abort stopped at %llu items.\n", uAbortCount);

    TEST_REQUIRE(uAbortCount == uTargetAbort, "Enumerate failed to abort early", FALSE);

    LOG_INFO("[LRU] [+] Enumerate Test Complete.\n");

    return TRUE;
}

struct ScalingCtx
{
    TestTable* pTable;
    volatile LONG64 llTotalOps;
    SIZE_T          uCacheCapacity;
    volatile LONG   lWarmUpFlag;
};

VOID ScalingWorker(_Inout_ TEST_WORKER_CONTEXT* pCtx)
{
    PAGED_CODE();

    SetWorkerThreadAffinity(pCtx->ulThreadId);

    ScalingCtx* pScalingCtx = (ScalingCtx*)pCtx->pUserContext;
    FastRng Rng(pCtx->ulThreadId + 5000);

    KeWaitForSingleObject(pCtx->pStartEvent, Executive, KernelMode, FALSE, NULL);

    if (ExAcquireRundownProtection(&g_TestRundown))
    {
        UINT64 ullLocalOps = 0;
        UINT64 ullMaxKey = pScalingCtx->uCacheCapacity + (pScalingCtx->uCacheCapacity / 2);

        // Warm-up phase
        while (InterlockedCompareExchange(&pScalingCtx->lWarmUpFlag, 0, 0) != 0)
        {
            if (InterlockedCompareExchange(&g_lAbortTests, 0, 0))
            {
                break;
            }

            UINT64 ullKey = Rng.Next() % ullMaxKey;
            RefCountedPayload* pOut = NULL;

            if (pScalingCtx->pTable->Lookup(ullKey, pOut))
            {
                pOut->Release();
            }
        }

        // Timed benchmark phase
        while (!InterlockedCompareExchange(pCtx->plStopFlag, 0, 0))
        {
            if (InterlockedCompareExchange(&g_lAbortTests, 0, 0))
            {
                break;
            }

            UINT64 ullKey = Rng.Next() % ullMaxKey;
            UINT32 u32OpType = Rng.Next() % 100;

            if (u32OpType < 80)
            {
                RefCountedPayload* pOut = NULL;

                if (pScalingCtx->pTable->Lookup(ullKey, pOut))
                {
                    pOut->Release();
                }
            }
            else
            {
                RefCountedPayload* pPayload = new RefCountedPayload(ullKey);

                if (pPayload)
                {
                    (void)pScalingCtx->pTable->Add(ullKey, pPayload);
                    pPayload->Release();
                }
            }

            ullLocalOps++;
        }

        InterlockedExchangeAdd64(&pScalingCtx->llTotalOps, ullLocalOps);
        ExReleaseRundownProtection(&g_TestRundown);
    }

    PsTerminateSystemThread(STATUS_SUCCESS);
}

/*
* Iteratively ramps up concurrent thread counts to map the scalability curve.
* Highlights linear scaling for sharded designs versus negative scaling for global locks.
* Includes a 1-second warm-up phase per step.
*/
BOOLEAN RunThreadScalingSweep(_In_ SIZE_T uCacheCapacity,
                              _In_ int    nSecondsPerStep)
{
    PAGED_CODE();

    LOG_INFO("\n[LRU] [*] Running Thread Scaling Sweep\n");
    LOG_INFO("[LRU]      %-10s| %-15s| %s\n", "Threads", "Ops/Second", "Scaling Factor");
    LOG_INFO("[LRU]      ------------------------------------------\n");

    ULONG ulMaxThreads = KeQueryActiveProcessorCountEx(ALL_PROCESSOR_GROUPS);

    if (ulMaxThreads == 0)
    {
        ulMaxThreads = 4;
    }

    if (ulMaxThreads > MAX_TEST_THREADS)
    {
        ulMaxThreads = MAX_TEST_THREADS;
    }

    UINT64 ullBaselineThroughput = 0;
    LARGE_INTEGER liFreq, liStart, liEnd;
    KeQueryPerformanceCounter(&liFreq);

    ULONG   ulThreadCount = 1;
    BOOLEAN bDidMax = FALSE;

    while (!bDidMax)
    {
        if (ulThreadCount >= ulMaxThreads)
        {
            ulThreadCount = ulMaxThreads;
            bDidMax = TRUE;
        }

        if (InterlockedCompareExchange(&g_lAbortTests, 0, 0))
        {
            return FALSE;
        }

        TestTable Table;

        TEST_REQUIRE(NT_SUCCESS(Table.Initialize(uCacheCapacity, 0)), "Init failed", FALSE);

        for (SIZE_T uI = 0; uI < uCacheCapacity; ++uI)
        {
            if ((uI % 10000) == 0 && InterlockedCompareExchange(&g_lAbortTests, 0, 0))
            {
                return FALSE;
            }

            RefCountedPayload* pPayload = new RefCountedPayload(uI);

            if (pPayload)
            {
                (void)Table.Add(uI, pPayload);
                pPayload->Release();
            }
        }

        ScalingCtx Ctx = { 0 };
        Ctx.pTable = &Table;
        Ctx.uCacheCapacity = uCacheCapacity;
        Ctx.lWarmUpFlag = 1;

        TEST_THREAD_MANAGER* pMgr = new (POOL_FLAG_NON_PAGED, DRIVER_TAG) TEST_THREAD_MANAGER();

        if (pMgr == NULL)
        {
            LOG_ERR("[LRU]      [!] Failed to allocate TEST_THREAD_MANAGER.\n");
            return FALSE;
        }

        StartThreads(pMgr, ulThreadCount, ScalingWorker, &Ctx);

        LARGE_INTEGER liWarmup;
        liWarmup.QuadPart = -10000000ll;
        KeDelayExecutionThread(KernelMode, FALSE, &liWarmup);

        InterlockedExchange(&Ctx.lWarmUpFlag, 0);
        liStart = KeQueryPerformanceCounter(NULL);

        StopAndWaitThreads(pMgr, nSecondsPerStep);
        liEnd = KeQueryPerformanceCounter(NULL);

        delete pMgr;
        Table.Cleanup();

        UINT64 ullTicks = liEnd.QuadPart - liStart.QuadPart;
        if (ullTicks == 0)
        {
            ullTicks = 1;
        }

        UINT64 ullOps = Ctx.llTotalOps;
        if (ullOps == 0)
        {
            ullOps = 1;
        }

        UINT64 ullThroughput = (ullOps * liFreq.QuadPart) / ullTicks;

        if (ulThreadCount == 1)
        {
            ullBaselineThroughput = ullThroughput;
        }

        UINT64 ullScaleWhole = 0;
        UINT64 ullScaleFrac = 0;

        if (ullBaselineThroughput > 0)
        {
            UINT64 ullScaleFactor = (ullThroughput * 100) / ullBaselineThroughput;
            ullScaleWhole = ullScaleFactor / 100;
            ullScaleFrac = ullScaleFactor % 100;
        }

        LOG_INFO("[LRU]      %-10u| %-15llu| %llu.%02llux\n",
                 ulThreadCount,
                 ullThroughput,
                 ullScaleWhole,
                 ullScaleFrac);

        if (!bDidMax)
        {
            ulThreadCount *= 2;
        }
    }

    LOG_INFO("[LRU] [+] Thread Scaling Sweep Complete.\n");
    return TRUE;
}

struct TailLatencyCtx
{
    TestTable* pTable;
    UINT64* pGlobalSamples;
    SIZE_T        uSamplesPerThread;
    SIZE_T        uCacheCapacity;
    LARGE_INTEGER liFreq;
    volatile LONG lWarmUpFlag;
};

VOID TailLatencyWorker(_Inout_ TEST_WORKER_CONTEXT* pCtx)
{
    PAGED_CODE();

    SetWorkerThreadAffinity(pCtx->ulThreadId);

    TailLatencyCtx* pTailLatencyCtx = (TailLatencyCtx*)pCtx->pUserContext;
    FastRng Rng(pCtx->ulThreadId + 8000);

    KeWaitForSingleObject(pCtx->pStartEvent, Executive, KernelMode, FALSE, NULL);

    if (ExAcquireRundownProtection(&g_TestRundown))
    {
        UINT64  ullMaxKey = pTailLatencyCtx->uCacheCapacity + (pTailLatencyCtx->uCacheCapacity / 2);
        UINT32  u32OpCounter = 0;
        SIZE_T  uSampleIdx = 0;
        UINT64* pMySamples = pTailLatencyCtx->pGlobalSamples + (pCtx->ulThreadId * pTailLatencyCtx->uSamplesPerThread);

        // Warm-up phase
        while (InterlockedCompareExchange(&pTailLatencyCtx->lWarmUpFlag, 0, 0) != 0)
        {
            if (InterlockedCompareExchange(&g_lAbortTests, 0, 0))
            {
                break;
            }

            UINT64 ullKey = Rng.Next() % ullMaxKey;
            RefCountedPayload* pOut = NULL;

            if (pTailLatencyCtx->pTable->Lookup(ullKey, pOut))
            {
                pOut->Release();
            }
        }

        // Sampling phase
        while (uSampleIdx < pTailLatencyCtx->uSamplesPerThread)
        {
            if (InterlockedCompareExchange(&g_lAbortTests, 0, 0))
            {
                break;
            }

            UINT64 ullKey = Rng.Next() % ullMaxKey;
            UINT32 u32OpType = Rng.Next() % 100;

            u32OpCounter++;
            BOOLEAN bShouldSample = (u32OpCounter % 100 == 0);

            LARGE_INTEGER liT0 = { 0 };
            LARGE_INTEGER liT1 = { 0 };

            if (bShouldSample)
            {
                liT0 = KeQueryPerformanceCounter(NULL);
            }

            if (u32OpType < 90)
            {
                RefCountedPayload* pOut = NULL;

                if (pTailLatencyCtx->pTable->Lookup(ullKey, pOut))
                {
                    pOut->Release();
                }
            }
            else
            {
                RefCountedPayload* pPayload = new RefCountedPayload(ullKey);

                if (pPayload)
                {
                    (void)pTailLatencyCtx->pTable->Add(ullKey, pPayload);
                    pPayload->Release();
                }
            }

            if (bShouldSample)
            {
                liT1 = KeQueryPerformanceCounter(NULL);
                UINT64 ullElapsedNs = ((liT1.QuadPart - liT0.QuadPart) * 1000000000ULL) / pTailLatencyCtx->liFreq.QuadPart;
                pMySamples[uSampleIdx++] = ullElapsedNs;
            }
        }

        ExReleaseRundownProtection(&g_TestRundown);
    }

    PsTerminateSystemThread(STATUS_SUCCESS);
}

/*
* Captures precise nanosecond-level execution times for millions of operations.
* Analyzes algorithmic stability (P50/P90) and hardware/OS jitter (P99.99).
* Includes a 1-second warm-up phase.
*/
BOOLEAN RunTailLatencyTest(_In_ SIZE_T uCacheCapacity)
{
    PAGED_CODE();

    LOG_INFO("\n[LRU] [*] Running Tail Latency Test:\n");

    TestTable Table;
    TEST_REQUIRE(NT_SUCCESS(Table.Initialize(uCacheCapacity, 0)), "Init failed", FALSE);

    for (SIZE_T uI = 0; uI < uCacheCapacity; ++uI)
    {
        if ((uI % 10000) == 0 && InterlockedCompareExchange(&g_lAbortTests, 0, 0))
        {
            return FALSE;
        }

        RefCountedPayload* pPayload = new RefCountedPayload(uI);
        if (pPayload)
        {
            (void)Table.Add(uI, pPayload);
            pPayload->Release();
        }
    }

    ULONG ulThreadCount = KeQueryActiveProcessorCountEx(ALL_PROCESSOR_GROUPS);

    if (ulThreadCount == 0)
    {
        ulThreadCount = 4;
    }

    if (ulThreadCount > MAX_TEST_THREADS)
    {
        ulThreadCount = MAX_TEST_THREADS;
    }

    const SIZE_T uSamplesPerThread = 50000;
    SIZE_T uTotalSamples = uSamplesPerThread * ulThreadCount;

    UINT64* pGlobalSamples = (UINT64*)ExAllocatePool2(POOL_FLAG_PAGED, uTotalSamples * sizeof(UINT64), DRIVER_TAG);
    if (!pGlobalSamples)
    {
        LOG_ERR("[LRU] [!] Failed to allocate latency samples buffer\n");
        return FALSE;
    }

    TailLatencyCtx Ctx = { 0 };
    Ctx.pTable            = &Table;
    Ctx.pGlobalSamples    = pGlobalSamples;
    Ctx.uSamplesPerThread = uSamplesPerThread;
    Ctx.uCacheCapacity    = uCacheCapacity;
    Ctx.lWarmUpFlag       = 1;
    KeQueryPerformanceCounter(&Ctx.liFreq);

    TEST_THREAD_MANAGER* pMgr = new (POOL_FLAG_NON_PAGED, DRIVER_TAG) TEST_THREAD_MANAGER();

    if (!pMgr)
    {
        ExFreePoolWithTag(pGlobalSamples, DRIVER_TAG);
        LOG_ERR("[LRU]      [!] Failed to allocate TEST_THREAD_MANAGER.\n");
        return FALSE;
    }

    StartThreads(pMgr, ulThreadCount, TailLatencyWorker, &Ctx);

    LOG_INFO("[LRU]      - Warming up for 1 second...\n");
    LARGE_INTEGER liWarmup;
    liWarmup.QuadPart = -10000000ll;
    KeDelayExecutionThread(KernelMode, FALSE, &liWarmup);

    LOG_INFO("[LRU]      - Warm-up complete. Collecting tail latency samples...\n");

    InterlockedExchange(&Ctx.lWarmUpFlag, 0);

    StopAndWaitThreads(pMgr, 0);

    delete pMgr;
    Table.Cleanup();

    QuickSort(pGlobalSamples, uTotalSamples, [](const UINT64& ullA,
        const UINT64& ullB) -> int
        {
            if (ullA < ullB)
            {
                return -1;
            }

            if (ullA > ullB)
            {
                return 1;
            }

            return 0;
        });

    auto getPercentile = [&](UINT32 u32Permille) -> UINT64
        {
            SIZE_T uIndex = (uTotalSamples * u32Permille) / 10000;

            if (uIndex >= uTotalSamples)
            {
                uIndex = uTotalSamples - 1;
            }

            return pGlobalSamples[uIndex];
        };

    LOG_INFO("[LRU]      - Total Samples Collected : %llu\n", uTotalSamples);
    LOG_INFO("[LRU]      - P50 (Median) Latency    : %llu ns\n", getPercentile(5000));
    LOG_INFO("[LRU]      - P90 Latency             : %llu ns\n", getPercentile(9000));
    LOG_INFO("[LRU]      - P99 Latency             : %llu ns\n", getPercentile(9900));
    LOG_INFO("[LRU]      - P99.9 Latency           : %llu ns\n", getPercentile(9990));
    LOG_INFO("[LRU]      - P99.99 Latency          : %llu ns\n", getPercentile(9999));
    LOG_INFO("[LRU] [+] Tail Latency Test Complete.\n");

    ExFreePoolWithTag(pGlobalSamples, DRIVER_TAG);
    return TRUE;
}

VOID RunBenchmarks(_In_opt_ PVOID pContext)
{
    PAGED_CODE();

    UNREFERENCED_PARAMETER(pContext);
    KeSetPriorityThread(KeGetCurrentThread(), LOW_REALTIME_PRIORITY);

    LOG_INFO("[LRU] \n--- LruHashTable KM Test Suite ---\n");

    const SIZE_T uCacheCap = 1000000;
    const SIZE_T uPrePopulate = 900000;
    const SIZE_T uOperations = 5000000;
    const int    nContentionSec = 15;
    const SIZE_T uContentionCap = 250000;

    PerformanceMetrics CustomPerf = { 0 };

    auto RunAllTests = [&]() -> BOOLEAN
        {
    #define ABORT_CHECK() \
        if (InterlockedCompareExchange(&g_lAbortTests, 0, 0)) \
        { \
            LOG_INFO("[LRU] Abort requested. Stopping test suite execution.\n"); \
            return FALSE; \
        }

            ABORT_CHECK();
            if (!RunCorrectnessTests())
            {
                return FALSE;
            }

            ABORT_CHECK();
            if (!RunTinyTableTest())
            {
                return FALSE;
            }

            ABORT_CHECK();
            if (!RunFreeListIntegrityTest())
            {
                return FALSE;
            }

            ABORT_CHECK();
            if (!RunZeroCapacitySanityTest())
            {
                return FALSE;
            }

            ABORT_CHECK();
            if (!RunEvictionAndExhaustionTest())
            {
                return FALSE;
            }

            ABORT_CHECK();
            if (!RunMultiThreadedCorrectnessTest())
            {
                return FALSE;
            }

            ABORT_CHECK();
            if (!RunHashCollisionTest())
            {
                return FALSE;
            }

            ABORT_CHECK();
            (void)RunPerformanceTest(10000, 5000, 5000000, "Lightly Populated");

            ABORT_CHECK();
            CustomPerf = RunPerformanceTest(uCacheCap,
                                            uPrePopulate,
                                            uOperations,
                                            "Heavily Populated (0% Safe Zone / Strict LRU)",
                                            0);

            ABORT_CHECK();
            (void)RunPerformanceTest(uCacheCap,
                                     uPrePopulate,
                                     uOperations,
                                     "Heavily Populated (25% Safe Zone)",
                                     25);

            ABORT_CHECK();
            (void)RunPerformanceTest(uCacheCap,
                                     uPrePopulate,
                                     uOperations,
                                     "Heavily Populated (50% Safe Zone)",
                                     50);

            ABORT_CHECK();
            (void)RunPerformanceTest(uCacheCap,
                                     uPrePopulate,
                                     uOperations,
                                     "Heavily Populated (75% Safe Zone)",
                                     75);

            ABORT_CHECK();
            if (!RunContentionTest_NoEviction(5))
            {
                return FALSE;
            }

            ABORT_CHECK();
            if (!RunContentionTest_EvictionThrashing(5))
            {
                return FALSE;
            }

            ABORT_CHECK();
            CustomPerf.ullContentionThroughput = RunContentionTest_MixedWorkload(nContentionSec,
                                                                                 uContentionCap,
                                                                                 0);

            ABORT_CHECK();
            CustomPerf.ullReadHeavyThroughput0 = RunContentionTest_ReadHeavySkewed(nContentionSec,
                                                                                   uContentionCap,
                                                                                   0);

            ABORT_CHECK();
            CustomPerf.ullReadHeavyThroughput25 = RunContentionTest_ReadHeavySkewed(nContentionSec,
                                                                                    uContentionCap,
                                                                                    25);

            ABORT_CHECK();
            CustomPerf.ullReadHeavyThroughput50 = RunContentionTest_ReadHeavySkewed(nContentionSec,
                                                                                    uContentionCap,
                                                                                    50);

            ABORT_CHECK();
            CustomPerf.ullReadHeavyThroughput75 = RunContentionTest_ReadHeavySkewed(nContentionSec,
                                                                                    uContentionCap,
                                                                                    75);

            ABORT_CHECK();
            CustomPerf.ullReadHeavyThroughput100 = RunContentionTest_ReadHeavySkewed(nContentionSec,
                                                                                     uContentionCap,
                                                                                     100);

            ABORT_CHECK();
            if (!RunTrimPerformanceTest(120000, 2))
            {
                return FALSE;
            }

            ABORT_CHECK();
            if (!RunTrimToWatermarkTest(120000))
            {
                return FALSE;
            }


            ABORT_CHECK();
            if (!RunForcedTrimTest(120000))
            {
                return FALSE;
            }

            ABORT_CHECK();
            if (!RunEnumerateTest(10000))
            {
                return FALSE;
            }

            if (!RunThreadScalingSweep(uCacheCap, 2))
            {
                return false;
            }

            ABORT_CHECK();
            if (!RunTailLatencyTest(uCacheCap))
            {
                return FALSE;
            }

            return TRUE;

    #undef ABORT_CHECK
        };

    BOOLEAN bTestsPassed = RunAllTests();

    if (g_lLiveObjectsCount != 0)
    {
        LOG_ERR("[LRU] [-] MASTER MEMORY LEAK CHECK: %d objects stranded.\n", g_lLiveObjectsCount);
    }
    else if (!bTestsPassed && InterlockedCompareExchange(&g_lAbortTests, 0, 0) == 0)
    {
        LOG_ERR("[LRU] [-] Test suite aborted due to an error.\n");
    }
    else if (!bTestsPassed && InterlockedCompareExchange(&g_lAbortTests, 0, 0) != 0)
    {
        LOG_INFO("[LRU] [-] Test suite aborted by driver unload.\n");
    }
    else
    {
        LOG_INFO("[LRU] \n--- All Tests Finished Successfully ---\n");
    }

    PsTerminateSystemThread(bTestsPassed ? STATUS_SUCCESS : STATUS_UNSUCCESSFUL);
}

VOID DriverUnload(_In_ PDRIVER_OBJECT pDriverObject)
{
    PAGED_CODE();

    UNREFERENCED_PARAMETER(pDriverObject);

    LOG_INFO("\n[LRU] ***Unload requested. Signaling abort...\n");

    // Signal all long-running test loops and worker threads to exit
    InterlockedExchange(&g_lAbortTests, 1);

    // Block until all threads that took a reference have aborted and released it
    ExWaitForRundownProtectionRelease(&g_TestRundown);

    if (g_pMasterBenchmarkThread != NULL)
    {
        LOG_INFO("[LRU] Waiting for master benchmark thread to terminate...\n");

        KeWaitForSingleObject(g_pMasterBenchmarkThread,
                              Executive,
                              KernelMode,
                              FALSE,
                              NULL);

        ObDereferenceObject(g_pMasterBenchmarkThread);
        g_pMasterBenchmarkThread = NULL;
    }

    ExDeletePagedLookasideList(&g_PayloadLookasideList);

    LOG_INFO("[LRU] ***Driver Unloaded.\n");
}

extern "C" NTSTATUS DriverEntry(_In_ PDRIVER_OBJECT  pDriverObject,
                                _In_ PUNICODE_STRING pRegistryPath);

#pragma alloc_text(INIT, DriverEntry)
extern "C" NTSTATUS DriverEntry(_In_ PDRIVER_OBJECT  pDriverObject,
                                _In_ PUNICODE_STRING pRegistryPath)
{
    PAGED_CODE();

    UNREFERENCED_PARAMETER(pRegistryPath);

    LOG_INFO("[LRU] ***DriverEntry Called.\n");

    ExInitializePagedLookasideList(&g_PayloadLookasideList,
                                   NULL,
                                   NULL,
                                   0,
                                   sizeof(RefCountedPayload),
                                   DRIVER_TAG,
                                   0);

    ExInitializeRundownProtection(&g_TestRundown);

    pDriverObject->DriverUnload = DriverUnload;

    HANDLE hThread;
    NTSTATUS ntStatus = PsCreateSystemThread(&hThread,
                                             THREAD_ALL_ACCESS,
                                             NULL,
                                             NULL,
                                             NULL,
                                             (PKSTART_ROUTINE)RunBenchmarks,
                                             NULL);

    if (NT_SUCCESS(ntStatus))
    {
        NTSTATUS refStatus = ObReferenceObjectByHandle(hThread,
                                                       THREAD_ALL_ACCESS,
                                                       NULL,
                                                       KernelMode,
                                                       (PVOID*)&g_pMasterBenchmarkThread,
                                                       NULL);

        if (!NT_SUCCESS(refStatus))
        {
            LOG_ERR("[LRU] [!] Failed to reference master thread handle. Aborting.\n");
            
            // Fast abort the thread we just spawned
            InterlockedExchange(&g_lAbortTests, 1);
            
            // Wait for it to safely terminate to prevent the BSOD condition
            ZwWaitForSingleObject(hThread, FALSE, NULL);
            ZwClose(hThread);
            
            return refStatus;
        }

        ZwClose(hThread);
    }
    else
    {
        LOG_ERR("[LRU] [!] Failed to create master system thread.\n");
        return ntStatus;
    }

    return STATUS_SUCCESS;
}