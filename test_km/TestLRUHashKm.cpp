/*
* Apache LRU Hash Table Sample/Test
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

#define TEST_IS_KM 1
#include <windows.h>
#include <iostream>
#include <atomic>
#include <thread>
#include <chrono>
#include <intrin.h>
#include <string>
#include <new>

// ----------------------------------------------------------------------------
// Kernel-Mode API Stubs for User-Mode Testing
// ----------------------------------------------------------------------------
#ifndef POOL_FLAG_NON_PAGED
#define POOL_FLAG_NON_PAGED 0
#endif
#ifndef POOL_FLAG_PAGED
#define POOL_FLAG_PAGED 1
#endif
#ifndef POOL_FLAG_UNINITIALIZED
#define POOL_FLAG_UNINITIALIZED 0x0000000000000002UI64
#endif
#ifndef ALL_PROCESSOR_GROUPS
#define ALL_PROCESSOR_GROUPS 0xFFFF
#endif
#ifndef STATUS_SUCCESS
#define STATUS_SUCCESS 0
#endif
#ifndef STATUS_INSUFFICIENT_RESOURCES
#define STATUS_INSUFFICIENT_RESOURCES 0xC000009A
#endif
#ifndef PAGED_CODE
#define PAGED_CODE()
#endif

// ----------------------------------------------------------------------------
// Hardware Pause Macro
// ----------------------------------------------------------------------------
#ifndef CPU_PAUSE
#if defined(_M_ARM64) || defined(_M_ARM)
#define CPU_PAUSE() __yield()
#else
#define CPU_PAUSE() _mm_pause()
#endif
#endif

typedef unsigned __int64      ULONG64;
typedef ULONG64               POOL_FLAGS;
typedef unsigned long         ULONG;
typedef void* PVOID;
typedef LONG                  NTSTATUS;
typedef std::atomic<uint32_t> EX_PUSH_LOCK;

#ifndef POOL_NODE_REQUIREMENT
typedef ULONG POOL_NODE_REQUIREMENT;
#endif

#ifndef EX_POOL_PRIORITY
typedef enum _EX_POOL_PRIORITY
{
    LowPoolPriority,
    LowPoolPrioritySpecialPoolOverrun,
    LowPoolPrioritySpecialPoolUnderrun,
    NormalPoolPriority,
    NormalPoolPrioritySpecialPoolOverrun,
    NormalPoolPrioritySpecialPoolUnderrun,
    HighPoolPriority,
    HighPoolPrioritySpecialPoolOverrun,
    HighPoolPrioritySpecialPoolUnderrun
} EX_POOL_PRIORITY;
#endif

#ifndef POOL_EXTENDED_PARAMETER_TYPE
typedef enum _POOL_EXTENDED_PARAMETER_TYPE
{
    PoolExtendedParameterInvalidType = 0,
    PoolExtendedParameterPriority    = 1,
    PoolExtendedParameterNumaNode    = 2
} POOL_EXTENDED_PARAMETER_TYPE;
#endif

#ifndef POOL_EXTENDED_PARAMETER
typedef struct _POOL_EXTENDED_PARAMS_SECURE_POOL POOL_EXTENDED_PARAMS_SECURE_POOL;
typedef struct _POOL_EXTENDED_PARAMETER
{
    struct
    {
        ULONG64 Type     : 8;
        ULONG64 Optional : 1;
        ULONG64 Reserved : 55;
    };
    union
    {
        ULONG64                           Reserved2;
        PVOID                             Reserved3;
        EX_POOL_PRIORITY                  Priority;
        POOL_EXTENDED_PARAMS_SECURE_POOL* SecurePoolParams;
        POOL_NODE_REQUIREMENT             PreferredNode;
    };
} POOL_EXTENDED_PARAMETER, * PPOOL_EXTENDED_PARAMETER;
#endif

// ----------------------------------------------------------------------------
// Memory API
// ----------------------------------------------------------------------------

inline PVOID ExAllocatePool2(_In_ ULONG  Flags,
                             _In_ SIZE_T NumberOfBytes,
                             _In_ ULONG  Tag)
{
    UNREFERENCED_PARAMETER(Flags);
    UNREFERENCED_PARAMETER(Tag);

    return ::operator new[](NumberOfBytes, std::align_val_t{ 64 }, std::nothrow);
}

inline PVOID ExAllocatePool3(_In_ ULONG                                                       Flags,
                             _In_ SIZE_T                                                      NumberOfBytes,
                             _In_ ULONG                                                       Tag,
                             _In_reads_opt_(ExtendedParametersCount) PPOOL_EXTENDED_PARAMETER ExtendedParameters,
                             _In_ ULONG                                                       ExtendedParametersCount)
{
    UNREFERENCED_PARAMETER(Flags);
    UNREFERENCED_PARAMETER(Tag);
    UNREFERENCED_PARAMETER(ExtendedParameters);
    UNREFERENCED_PARAMETER(ExtendedParametersCount);

    return ::operator new[](NumberOfBytes, std::align_val_t{ 64 }, std::nothrow);
}

inline void ExFreePoolWithTag(_In_opt_ PVOID P,
                              _In_     ULONG Tag
)
{
    UNREFERENCED_PARAMETER(Tag);
    if (P)
    {
        ::operator delete[](P, std::align_val_t{ 64 });
    }
}

// ----------------------------------------------------------------------------
// Processor/Topology API
// ----------------------------------------------------------------------------
inline ULONG KeQueryActiveProcessorCountEx(_In_ USHORT GroupNumber) noexcept
{
    DWORD dwProcessorCount = GetActiveProcessorCount(GroupNumber);
    if (dwProcessorCount == 0)
    {
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        return sysInfo.dwNumberOfProcessors > 0 ? sysInfo.dwNumberOfProcessors : 1;
    }
    return dwProcessorCount;
}

inline USHORT KeQueryHighestNodeNumber() noexcept
{
    ULONG ulHighestNodeNumber = 0;
    if (GetNumaHighestNodeNumber(&ulHighestNodeNumber))
    {
        return static_cast<USHORT>(ulHighestNodeNumber);
    }

    return 0;
}

inline USHORT KeGetCurrentNodeNumber() noexcept
{
    PROCESSOR_NUMBER procNumber;
    GetCurrentProcessorNumberEx(&procNumber);

    USHORT usNodeNumber = 0;
    if (GetNumaProcessorNodeEx(&procNumber, &usNodeNumber))
    {
        return usNodeNumber;
    }

    return 0;
}

#ifndef KernelMode
typedef char KPROCESSOR_MODE;
#define KernelMode 0
#endif

#ifndef _WIN32
typedef union _LARGE_INTEGER
{
    struct
    {
        uint32_t LowPart;
        int32_t  HighPart;
    } DUMMYSTRUCTNAME;
    struct
    {
        uint32_t LowPart;
        int32_t  HighPart;
    } u;
    int64_t QuadPart;
} LARGE_INTEGER, * PLARGE_INTEGER;
#endif

inline NTSTATUS KeDelayExecutionThread(_In_     KPROCESSOR_MODE WaitMode,
                                       _In_     BOOLEAN         Alertable,
                                       _In_opt_ PLARGE_INTEGER  Interval) noexcept
{
    (void)WaitMode;
    (void)Alertable;

    if (Interval)
    {
        int64_t time100ns = Interval->QuadPart;
        if (time100ns < 0) time100ns = -time100ns;

        int64_t milliseconds = time100ns / 10000;
        if (milliseconds > 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
        }
        else
        {
            std::this_thread::yield();
        }
    }
    return STATUS_SUCCESS;
}

// ----------------------------------------------------------------------------
// EX_PUSH_LOCK Stubs
// ----------------------------------------------------------------------------
using EX_PUSH_LOCK           = std::atomic<uint32_t>;
constexpr uint32_t WRITE_BIT = 0x80000000;

struct AdaptiveSpinPolicy
{
    // Industry standard sweet-spot for user-mode spinlocks.
    static constexpr uint32_t PURE_SPIN_LIMIT = 4000;

    static inline void SpinWait(_Inout_ uint64_t& SpinPhase) noexcept
    {
        if (SpinPhase < PURE_SPIN_LIMIT)
        {
            // FAST PATH: Pure hardware pause. 
            CPU_PAUSE();
            SpinPhase++;
        }
        else
        {
            // SLOW PATH: Lock is severely contended.            
            SwitchToThread();

            // Reset the phase to resume pure spinning when we wake up.
            SpinPhase = 0;
        }
    }
};

__forceinline void ExInitializePushLock(_Out_ EX_PUSH_LOCK* Lock)
{
    Lock->store(0, std::memory_order_relaxed);
}

__forceinline void ExAcquirePushLockShared(_Inout_ EX_PUSH_LOCK* Lock)
{
    uint64_t spinPhase = 0;

    while (true)
    {
        uint32_t current = Lock->load(std::memory_order_relaxed);

        // If the write bit is not set, we can attempt to add a reader
        if (!(current & WRITE_BIT))
        {
            if (Lock->compare_exchange_weak(current, current + 1,
                                            std::memory_order_acquire,
                                            std::memory_order_relaxed))
            {
                return; // Successfully acquired shared
            }
        }

        // Backoff and adaptively spin/yield
        AdaptiveSpinPolicy::SpinWait(spinPhase);
    }
}

__forceinline void ExReleasePushLockShared(_Inout_ EX_PUSH_LOCK* Lock)
{
    Lock->fetch_sub(1, std::memory_order_release);
}

__forceinline void ExAcquirePushLockExclusive(_Inout_ EX_PUSH_LOCK* Lock)
{
    uint64_t spinPhase = 0;

    while (true)
    {
        uint32_t current = Lock->load(std::memory_order_relaxed);

        // PushLocks can only be acquired exclusively if there are NO readers AND NO writers
        if (current == 0)
        {
            if (Lock->compare_exchange_weak(current, WRITE_BIT,
                                            std::memory_order_acquire,
                                            std::memory_order_relaxed))
            {
                return; // Successfully acquired exclusive
            }
        }

        // Backoff and adaptively spin/yield
        AdaptiveSpinPolicy::SpinWait(spinPhase);
    }
}

__forceinline void ExReleasePushLockExclusive(_Inout_ EX_PUSH_LOCK* Lock)
{
    Lock->store(0, std::memory_order_release);
}

__forceinline BOOLEAN ExTryAcquirePushLockExclusive(_Inout_ EX_PUSH_LOCK* Lock)
{
    uint32_t current = Lock->load(std::memory_order_relaxed);

    if (current == 0)
    {
        return Lock->compare_exchange_strong(current, WRITE_BIT,
                                             std::memory_order_acquire,
                                             std::memory_order_relaxed) ? TRUE : FALSE;
    }

    return FALSE;
}

__forceinline void KeEnterCriticalRegion() {}

__forceinline void KeLeaveCriticalRegion() {}

#include <windows.h>
#include <intrin.h>

// ----------------------------------------------------------------------------
// KeQueryNodeActiveAffinity stub
// Maps directly to the Win32 NUMA APIs to simulate realistic kernel topology.
// ----------------------------------------------------------------------------
VOID KeQueryNodeActiveAffinity(_In_      USHORT          NodeNumber,
                               _Out_opt_ PGROUP_AFFINITY Affinity,
                               _Out_opt_ PUSHORT         Count)
{
    GROUP_AFFINITY LocalAffinity = { 0 };

    // Query the OS for the actual user-mode processor mask for this NUMA node
    BOOL bSuccess = GetNumaNodeProcessorMaskEx(NodeNumber, &LocalAffinity);
    if (!bSuccess)
    {
        // If the node doesn't exist or the call fails, zero out the mask 
        // to mimic a node with no active processors.
        LocalAffinity.Mask = 0;
        LocalAffinity.Group = 0;
    }

    if (Affinity != NULL)
    {
        *Affinity = LocalAffinity;
    }

    if (Count != NULL)
    {
        if (LocalAffinity.Mask == 0)
        {
            *Count = 0;
        }
        else
        {
            // Use hardware popcnt to count the number of active logical processors
#if defined(_M_AMD64) || defined(_M_ARM64)
            *Count = (USHORT)__popcnt64(LocalAffinity.Mask);
#else
            // Fallback for 32-bit
            // GROUP_AFFINITY.Mask is a KAFFINITY (ULONG_PTR)
            *Count = (USHORT)__popcnt((unsigned int)LocalAffinity.Mask);
#endif
        }
    }
}

// ----------------------------------------------------------------------------
// Includes for Targets and Shared Tests
// ----------------------------------------------------------------------------
#include "LRUHashTable.h"
#include "test_lru_hash_common.h"

int main()
{
    std::cout << "\n=========================================================\n";
    std::cout << "       KERNEL-MODE LRU HASH TABLE TEST SUITE             \n";
    std::cout << "=========================================================\n";

    using CustomTestTable    = CLruHashTable<uint64_t, RefCountedPayload, Hasher64Bit>;
    using CollisionTestTable = CLruHashTable<uint64_t, RefCountedPayload, DegradedHasher>;

    return RunAllTests<CustomTestTable, CollisionTestTable>("KM Custom Array Table");
}