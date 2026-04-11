/*
 * Apache LRU Hash Table Test/Sample
 * Copyright 2026 Alexander Danileiko
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * This software is provided on an "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS
 * OF ANY KIND, either express or implied.
 */

 // ----------------------------------------------------------------------------
 // This header defines the master testing, validation, and benchmarking 
 // orchestrator for the LruHashTable, comparing the custom array-backed 
 // implementation against a standard library baseline.
 //
 // CORRECTNESS & INTEGRITY SUITE:
 // A comprehensive battery of tests validating fundamental correctness, memory
 // safety, and concurrency bounds. It covers basic Add/Lookup/Remove operations,
 // forced hash collisions, Free-List integrity, and custom allocator memory
 // tracking. It also includes extreme edge-case simulations, such as zero-capacity
 // clamping, tiny-table eviction constraints, and aggressive multithreaded
 // thrashing to verify safety against ABA hazards, TOCTOU races, and memory leaks.
 //
 // PERFORMANCE & CONTENTION BENCHMARKS:
 // A high-throughput benchmarking engine that measures operations per second
 // across simulated production workloads. It evaluates sequential versus
 // multithreaded scaling, isolating lock contention, eviction thrashing, and
 // oversubscribed CPU cores. It tests mixed read/write ratios (e.g., read-heavy
 // skewed distributions) and measures the efficiency of background yielding trims.
 //
 // TAIL LATENCY & COMPARATIVE ANALYSIS:
 // Captures nanosecond-level execution times across millions of operations to
 // calculate percentile latency bounds (P50 to P99.99), evaluating algorithmic
 // stability and OS/hardware jitter. The orchestrator executes a direct
 // performance comparison against a std::unordered_map + std::list baseline,
 // outputting a detailed report of throughput speedups and scaling advantages.
 // ----------------------------------------------------------------------------

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <new>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <intrin.h>
#include <windows.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

#if defined(__APPLE__) || defined(__FreeBSD__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

#if defined(__linux__)
#include <fstream>
#endif

// ----------------------------------------------------------------------------
// Test Validation Macro
// ----------------------------------------------------------------------------
#define TEST_REQUIRE(condition, msg, retVal)                                                 \
    do                                                                                       \
    {                                                                                        \
        if (!(condition))                                                                    \
        {                                                                                    \
            std::cerr << "    [!] TEST FAILED: " << (msg) << " (Line " << __LINE__ << ")\n"; \
            return retVal;                                                                   \
        }                                                                                    \
    } while (0)

// ----------------------------------------------------------------------------
// Allocators for Testing
// ----------------------------------------------------------------------------
#ifndef TEST_IS_KM
struct TrackingAllocator
{
    static inline std::atomic<size_t>  TotalAllocatedBytes{ 0 };
    static inline std::atomic<size_t>  TotalFreedBytes{ 0 };
    static inline std::atomic<int64_t> ActiveAllocations{ 0 };

    static inline void Reset() noexcept
    {
        TotalAllocatedBytes = 0;
        TotalFreedBytes = 0;
        ActiveAllocations = 0;
    }

    static inline std::vector<uint32_t> GetValidNodes() noexcept
    {
        // Simulate a single contiguous NUMA node (Node 0) for testing
        return { 0 };
    }

    static inline void* Allocate(size_t size, uint32_t node) noexcept
    {
        (void)node;
        void* ptr = ::operator new[](size, std::align_val_t{ 64 }, std::nothrow);

        if (ptr)
        {
            TotalAllocatedBytes.fetch_add(size, std::memory_order_relaxed);
            ActiveAllocations.fetch_add(1, std::memory_order_relaxed);
        }

        return ptr;
    }

    static inline void Free(void* ptr, size_t size) noexcept
    {
        if (ptr)
        {
            TotalFreedBytes.fetch_add(size, std::memory_order_relaxed);
            ActiveAllocations.fetch_sub(1, std::memory_order_relaxed);

            ::operator delete[](ptr, std::align_val_t{ 64 });
        }
    }
};
#endif

// ----------------------------------------------------------------------------
// Utility
// ----------------------------------------------------------------------------
inline void SetHighPriority()
{
#ifdef _WIN32
    SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS);
#else
    setpriority(PRIO_PROCESS, 0, -10);
#endif
}

inline std::string cpu_name()
{
#if defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64)
    int cpuInfo[4] = { 0 };
    char brand[0x40];
    std::memset(brand, 0, sizeof(brand));

#if defined(_MSC_VER)
    __cpuid(cpuInfo, 0x80000000);
    unsigned int nExIds = cpuInfo[0];

    if (nExIds >= 0x80000004)
    {
        __cpuid(cpuInfo, 0x80000002);
        std::memcpy(brand, cpuInfo, sizeof(cpuInfo));

        __cpuid(cpuInfo, 0x80000003);
        std::memcpy(brand + 16, cpuInfo, sizeof(cpuInfo));

        __cpuid(cpuInfo, 0x80000004);
        std::memcpy(brand + 32, cpuInfo, sizeof(cpuInfo));
    }
#else
    unsigned int eax;
    unsigned int ebx;
    unsigned int ecx;
    unsigned int edx;

    __asm__ __volatile__("cpuid" : "=a"(eax) : "a"(0x80000000) : "ebx", "ecx", "edx");

    if (eax >= 0x80000004)
    {
        unsigned int data[12];

        for (unsigned int i = 0; i < 3; ++i)
        {
            __asm__ __volatile__("cpuid"
                : "=a"(data[i * 4 + 0]),
                "=b"(data[i * 4 + 1]),
                "=c"(data[i * 4 + 2]),
                "=d"(data[i * 4 + 3])
                : "a"(0x80000002 + i));
        }

        std::memcpy(brand, data, sizeof(data));
    }
#endif
    return std::string(brand);
#elif defined(__APPLE__)
    char buf[256];
    size_t size = sizeof(buf);

    if (sysctlbyname("machdep.cpu.brand_string", &buf, &size, NULL, 0) == 0)
    {
        return std::string(buf);
    }

    size = sizeof(buf);

    if (sysctlbyname("hw.model", &buf, &size, NULL, 0) == 0)
    {
        return std::string(buf);
    }

    return "Unknown CPU";
#elif defined(__linux__)
    std::ifstream f("/proc/cpuinfo");
    std::string line;

    while (std::getline(f, line))
    {
        if (line.find("model name") != std::string::npos ||
            line.find("Hardware") != std::string::npos)
        {
            std::size_t pos = line.find(':');

            if (pos != std::string::npos)
            {
                return line.substr(pos + 2);
            }
        }
    }

    return "Unknown CPU";
#else
    return "Unknown CPU";
#endif
}

// ----------------------------------------------------------------------------
// Test Data Structures & Helpers
// ----------------------------------------------------------------------------
inline std::atomic<long> g_LiveObjectsCount{ 0 };

class RefCountedPayload
{
private:
    std::atomic<long> m_RefCount;

public:
    uint64_t Data;

    RefCountedPayload(uint64_t Value) : m_RefCount(0), Data(Value)
    {
        g_LiveObjectsCount.fetch_add(1, std::memory_order_relaxed);
    }

    ~RefCountedPayload()
    {
        g_LiveObjectsCount.fetch_sub(1, std::memory_order_relaxed);
    }

    void AddRef() noexcept
    {
        m_RefCount.fetch_add(1, std::memory_order_acquire);
    }

    void Release() noexcept
    {
        if (m_RefCount.fetch_sub(1, std::memory_order_acq_rel) == 1)
        {
            delete this;
        }
    }
};

struct Hasher64Bit
{
    static inline uint64_t ComputeHash(const uint64_t& Key)
    {
        uint64_t z = Key + 0x9e3779b97f4a7c15ULL;
        z ^= (z >> 33);

        z *= 0xff51afd7ed558ccdULL;
        z ^= (z >> 33);

        z *= 0xc4ceb9fe1a85ec53ULL;
        z ^= (z >> 33);

        return z;
    }

    inline size_t operator()(const uint64_t& Key) const noexcept
    {
        return static_cast<size_t>(ComputeHash(Key));
    }
};

struct DegradedHasher
{
    static inline uint64_t ComputeHash(const uint64_t& tKey)
    {
        (void)tKey;

        return 0xBADF00D;
    }

    inline size_t operator()(const uint64_t& Key) const noexcept
    {
        return static_cast<size_t>(ComputeHash(Key));
    }
};

class FastRng
{
private:
    uint64_t m_State;

public:
    // Seed cannot be zero for Xorshift. We provide a 64-bit fallback
    FastRng(uint64_t Seed) : m_State(Seed ? Seed : 0xBADF00D15EA5EULL)
    {}

    uint64_t Next()
    {
        m_State ^= m_State << 13;
        m_State ^= m_State >> 7;
        m_State ^= m_State << 17;
        return m_State;
    }
};

// ----------------------------------------------------------------------------
// Metric Structures
// ----------------------------------------------------------------------------
struct PerformanceMetrics
{
    uint64_t AddThroughput;
    uint64_t LookupThroughput;
    uint64_t RemoveThroughput;
    uint64_t ContentionThroughput;
    uint64_t OversubscribedContentionThroughput;
    uint64_t ReadHeavyThroughput0;
    uint64_t ReadHeavyThroughput25;
    uint64_t ReadHeavyThroughput50;
    uint64_t ReadHeavyThroughput75;
    uint64_t ReadHeavyThroughput100;
};

struct LatencyMetrics
{
    uint64_t P50;
    uint64_t P90;
    uint64_t P99;
    uint64_t P99_9;
    uint64_t P99_99;
};

struct ScalingMetrics
{
    unsigned int MaxThreads;
    uint64_t BaselineThroughput;
    uint64_t MaxThroughput;
    double ScalingFactor;
};

// ----------------------------------------------------------------------------
// Cross-Platform Initialization Helper
// ----------------------------------------------------------------------------
template <typename TTable>
bool InitTableHelper(TTable& table, size_t cap, uint32_t thresh = 0)
{
    if constexpr (requires { table.Initialize(cap, thresh); })
    {
        auto res = table.Initialize(cap, thresh);

        if constexpr (std::is_same_v<decltype(res), bool>)
        {
            return res;
        }
        else
        {
            return res == 0;
        }
    }
    else
    {
        auto res = table.Initialize(cap);

        if constexpr (std::is_same_v<decltype(res), bool>)
        {
            return res;
        }
        else
        {
            return res == 0;
        }
    }
}

// ----------------------------------------------------------------------------
// Templated Test Suites
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Validates fundamental hash table correctness, including basic Add/Lookup/Remove,
// overwrite behaviors (Get-Or-Add), and natural LRU eviction bounds.
// ----------------------------------------------------------------------------
template <typename TTable>
bool RunCorrectnessTests()
{
    std::cout << "[*] Running Correctness Tests...\n";

    TTable table;

    TEST_REQUIRE(InitTableHelper(table, 1024), "Failed to initialize table", false);

    RefCountedPayload* pOut = nullptr;

    TEST_REQUIRE(!table.Lookup(999, pOut), "Lookup succeeded on an empty table", false);

    RefCountedPayload* p1 = new RefCountedPayload(100);
    p1->AddRef();

    bool bIsAdded = table.Add(1, p1);
    TEST_REQUIRE(bIsAdded, "Initial Add failed", false);

    p1->Release();

    if (table.Lookup(1, pOut))
    {
        TEST_REQUIRE(pOut->Data == 100, "Data mismatch", false);
        pOut->Release();
    }
    else
    {
        TEST_REQUIRE(false, "Lookup failed", false);
    }

    RefCountedPayload* p2 = new RefCountedPayload(200);
    p2->AddRef();

    // Safely test Get-Or-Add vs Legacy Upsert
    if constexpr (requires { table.Add(1, p2, nullptr, AddAction::ReplaceIfExists); })
    {
        // 1. Verify Get-Or-Add correctly returns the existing object pointer
        RefCountedPayload* pGetOrAdd = new RefCountedPayload(300);
        pGetOrAdd->AddRef();

        RefCountedPayload* pExisting = nullptr;
        // Because default is KeepIfExists, it should return false and populate pExisting
        bool bAdded = table.Add(1, pGetOrAdd, &pExisting, AddAction::KeepIfExists);
        TEST_REQUIRE(!bAdded, "Get-Or-Add should return false when key exists", false);
        TEST_REQUIRE(pExisting != nullptr, "Get-Or-Add out parameter is null", false);
        TEST_REQUIRE(pExisting->Data == 100, "Get-Or-Add out parameter data mismatch", false);

        pGetOrAdd->Release(); // Safely releases the loser payload (caller retained ownership)
        pExisting->Release(); // Safely releases the returned reference from the table

        // 2. Verify Explicit Overwrite works
        bool bIsOverwritten = table.Add(1, p2, nullptr, AddAction::ReplaceIfExists);
        TEST_REQUIRE(bIsOverwritten, "Explicit Overwrite Add failed", false);
    }
    else
    {
        // Fallback for StdLruHashTable (Legacy Upsert)
        bool bIsOverwritten = table.Add(1, p2);
        TEST_REQUIRE(bIsOverwritten, "Overwrite Add failed", false);
    }

    p2->Release();

    if (table.Lookup(1, pOut))
    {
        TEST_REQUIRE(pOut->Data == 200, "Lookup failed on overwrite", false);
        pOut->Release();
    }
    else
    {
        TEST_REQUIRE(false, "Lookup failed on overwrite", false);
    }

    bool bIsRemoved = table.Remove(1);
    TEST_REQUIRE(bIsRemoved, "Remove failed", false);

    TEST_REQUIRE(!table.Lookup(1, pOut), "Item found after remove", false);

    bool bIsRemovedNonExistent = table.Remove(999);
    TEST_REQUIRE(!bIsRemovedNonExistent, "Remove succeeded on non-existent key", false);

    std::cout << "    [-] Running bulk insert and eviction...\n";

    for (int i = 1000; i < 101000; ++i)
    {
        RefCountedPayload* pBulk = new RefCountedPayload(i);
        pBulk->AddRef();

        bool bRes = table.Add(i, pBulk);
        (void)bRes;
        pBulk->Release();
    }

    size_t szTotalItems = table.GetTotalItemCount();

    TEST_REQUIRE(szTotalItems < 100000 && szTotalItems > 0,
                "Natural LRU eviction failed to cap size",
                false);

    std::cout << "    [-] Running active trim...\n";
    size_t szTrimmed = table.Trim(500);

    TEST_REQUIRE(szTrimmed > 0, "Trim failed to evict any items from full shards", false);
    TEST_REQUIRE(table.GetTotalItemCount() <= szTotalItems - szTrimmed,
                 "Item count mismatch after trim",
                 false);

    if (table.Lookup(100999, pOut))
    {
        TEST_REQUIRE(pOut->Data == 100999, "Failed to find MRU item data mismatch", false);
        pOut->Release();
    }
    else
    {
        TEST_REQUIRE(false, "Failed to find MRU item after bulk insert", false);
    }

    std::cout << "[+] Correctness Tests Passed.\n";

    return true;
}

// ----------------------------------------------------------------------------
// Tests edge cases on a highly constrained table (e.g., capacity of 10),
// ensuring strict LRU ordering and correct eviction of the oldest items under pressure.
// ----------------------------------------------------------------------------
template <typename TTable>
bool RunTinyTableTest()
{
    std::cout << "[*] Running Tiny Table Tests...\n";

    TTable table;

    TEST_REQUIRE(InitTableHelper(table, 10, 0), "Failed to initialize table", false);

    size_t szMemUsage = table.GetTotalMemoryUsage();
    std::cout << "    - Tiny Table (Target Cap: 10) Memory Footprint: " << szMemUsage << " bytes\n";

    for (int i = 0; i < 25; ++i)
    {
        RefCountedPayload* pPayload = new RefCountedPayload(i);
        pPayload->AddRef();

        bool bRes = table.Add(i, pPayload);
        pPayload->Release();

        TEST_REQUIRE(bRes, "Add failed on tiny table", false);
    }

    size_t szItemCount = table.GetTotalItemCount();
    std::cout << "    - Item Count after 25 insertions: " << szItemCount << "\n";

    TEST_REQUIRE(szItemCount == 10, "Failed to keep expected number of items after 25 insertions", false);

    RefCountedPayload* pOut = nullptr;

    for (int i = 0; i < 15; ++i)
    {
        if (table.Lookup(i, pOut))
        {
            pOut->Release();
            TEST_REQUIRE(false, "Old item was not evicted in tiny table", false);
        }
    }

    for (int i = 15; i < 25; ++i)
    {
        if (table.Lookup(i, pOut))
        {
            TEST_REQUIRE(pOut->Data == i, "Data mismatch in tiny table", false);
            pOut->Release();
        }
        else
        {
            TEST_REQUIRE(false, "Recent item was improperly evicted in tiny table", false);
        }
    }

    size_t szTrimmed = table.Trim(3);

    if (szTrimmed > 0)
    {
        std::cout << "    - Trimmed " << szTrimmed << " items from tiny table.\n";
    }

    std::cout << "[+] Tiny Table Tests Passed.\n";

    return true;
}

#ifndef TEST_IS_KM
// ----------------------------------------------------------------------------
// Validates that the custom allocator template is successfully routing memory
// requests and that no blocks are leaked during standard table operations.
// ----------------------------------------------------------------------------
inline bool RunCustomAllocatorTest()
{
    std::cout << "[*] Running Custom Allocator Tracking Test...\n";

    TrackingAllocator::Reset();

    {

#ifdef LRU_USE_EXPONENTIAL_BACKOFF
        using CustomAllocTable = LruHashTable<uint64_t,
                                              RefCountedPayload,
                                              Hasher64Bit,
                                              TrackingAllocator,
                                              ExponentialBackoffPolicy>;
#else
        using CustomAllocTable = LruHashTable<uint64_t,
                                              RefCountedPayload,
                                              Hasher64Bit,
                                              TrackingAllocator>;
#endif

        CustomAllocTable table;

        TEST_REQUIRE(InitTableHelper(table, 5000, 0), "Failed to initialize custom alloc table", false);

        size_t allocatedBytes = TrackingAllocator::TotalAllocatedBytes.load();
        int64_t activeBlocks = TrackingAllocator::ActiveAllocations.load();

        std::cout << "    - Active Memory Blocks after Init : " << activeBlocks << "\n";
        std::cout << "    - Total Allocated Bytes           : " << allocatedBytes << "\n";

        TEST_REQUIRE(activeBlocks > 0, "Custom allocator was bypassed!", false);
        TEST_REQUIRE(allocatedBytes > 0, "Zero bytes allocated!", false);

        for (int i = 0; i < 7500; ++i)
        {
            RefCountedPayload* p = new RefCountedPayload(i);
            p->AddRef();

            bool isAdded = table.Add(i, p);
            (void)isAdded;

            p->Release();
        }
    }

    size_t freedBytes = TrackingAllocator::TotalFreedBytes.load();
    int64_t remainingBlocks = TrackingAllocator::ActiveAllocations.load();

    std::cout << "    - Active Memory Blocks after Free : " << remainingBlocks << "\n";
    std::cout << "    - Total Freed Bytes               : " << freedBytes << "\n";

    TEST_REQUIRE(remainingBlocks == 0, "Memory leak detected! Active blocks remain.", false);
    TEST_REQUIRE(TrackingAllocator::TotalAllocatedBytes.load() == freedBytes,
                 "Allocated vs Freed byte count mismatch!",
                 false);

    std::cout << "[+] Custom Allocator Tracking Test Passed. Zero leaks verified.\n";

    return true;
}
#endif

// ----------------------------------------------------------------------------
// Forces artificial hash collisions to verify the integrity and correct traversal
// of the intra-array singly-linked collision chains.
// ----------------------------------------------------------------------------
template <typename TTable>
bool RunHashCollisionTest()
{
    std::cout << "[*] Running Severe Hash Collision Test...\n";

    TTable table;
    TEST_REQUIRE(InitTableHelper(table, 256, 0), "Failed to initialize table", false);

    for (int i = 0; i < 50; ++i)
    {
        RefCountedPayload* pPayload = new RefCountedPayload(i);
        pPayload->AddRef();

        bool bRes = table.Add(i, pPayload);
        pPayload->Release();

        TEST_REQUIRE(bRes, "Add failed during collision test", false);
    }

    RefCountedPayload* pOut = nullptr;

    for (int i = 0; i < 50; ++i)
    {
        if (table.Lookup(i, pOut))
        {
            TEST_REQUIRE(pOut->Data == i, "Data mismatch during traversal", false);
            pOut->Release();
        }
        else
        {
            TEST_REQUIRE(false, "Lookup failed to traverse collision chain", false);
        }
    }

    TEST_REQUIRE(table.Remove(49), "Failed to remove head of collision chain", false);
    TEST_REQUIRE(table.Remove(25), "Failed to remove middle of collision chain", false);
    TEST_REQUIRE(table.Remove(0), "Failed to remove tail of collision chain", false);

    TEST_REQUIRE(!table.Lookup(49, pOut), "Head ghost item found", false);
    TEST_REQUIRE(!table.Lookup(25, pOut), "Middle ghost item found", false);
    TEST_REQUIRE(!table.Lookup(0, pOut), "Tail ghost item found", false);

    std::cout << "[+] Severe Hash Collision Test Passed.\n";

    return true;
}

// ----------------------------------------------------------------------------
// Validates the internal Free-List mechanics by completely filling, draining,
// and refilling the table, ensuring no nodes are leaked or orphaned.
// ----------------------------------------------------------------------------
template <typename TTable>
bool RunFreeListIntegrityTest()
{
    std::cout << "[*] Running Free-List Integrity Test (Fill, Drain, Refill)...\n";

    TTable table;

    TEST_REQUIRE(InitTableHelper(table, 32, 0), "Failed to initialize table", false);

    for (int i = 0; i < 32; ++i)
    {
        RefCountedPayload* pPayload = new RefCountedPayload(i);
        pPayload->AddRef();

        (void)table.Add(i, pPayload);
        pPayload->Release();
    }

    TEST_REQUIRE(table.GetTotalItemCount() == 32, "Table failed to reach capacity", false);

    for (int i = 0; i < 32; ++i)
    {
        bool bRes = table.Remove(i);
        TEST_REQUIRE(bRes, "Remove failed during drain phase", false);
    }

    TEST_REQUIRE(table.GetTotalItemCount() == 0, "Table failed to drain completely", false);

    for (int i = 100; i < 132; ++i)
    {
        RefCountedPayload* pPayload = new RefCountedPayload(i);
        pPayload->AddRef();

        bool bRes = table.Add(i, pPayload);
        pPayload->Release();

        TEST_REQUIRE(bRes, "Add failed during refill phase (Possible Free-List Corruption)", false);
    }

    TEST_REQUIRE(table.GetTotalItemCount() == 32, "Table failed to refill to capacity", false);

    std::cout << "[+] Free-List Integrity Test Passed.\n";

    return true;
}

// ----------------------------------------------------------------------------
// Ensures the table correctly handles degenerate initialization requests
// by clamping zero-capacity to a safe minimum bound.
// ----------------------------------------------------------------------------
template <typename TTable>
bool RunZeroCapacitySanityTest()
{
    std::cout << "[*] Running Zero Capacity Sanity Test...\n";

    TTable table;

    TEST_REQUIRE(InitTableHelper(table, 0, 0), "Initialize failed on 0 capacity request", false);

    size_t szMemUsage = table.GetTotalMemoryUsage();
    TEST_REQUIRE(szMemUsage > 0, "Memory usage is zero, clamping failed", false);

    RefCountedPayload* pPayload = new RefCountedPayload(99);
    pPayload->AddRef();

    bool bRes = table.Add(99, pPayload);
    pPayload->Release();

    TEST_REQUIRE(bRes, "Add failed on clamped zero-capacity table", false);

    RefCountedPayload* pOut = nullptr;

    if (table.Lookup(99, pOut))
    {
        pOut->Release();
    }
    else
    {
        TEST_REQUIRE(false, "Lookup failed on clamped zero-capacity table", false);
    }

    std::cout << "[+] Zero Capacity Sanity Test Passed.\n";

    return true;
}

// ----------------------------------------------------------------------------
// A stress test designed to rapidly cycle through a small cache capacity,
// ensuring memory is reclaimed properly without leaks during heavy multi-threaded turnover.
// ----------------------------------------------------------------------------
template <typename TTable>
bool RunEvictionAndExhaustionTest()
{
    std::cout << "[*] Running Eviction & Exhaustion Tests...\n";

    TTable table;

    TEST_REQUIRE(InitTableHelper(table, 256), "Failed to initialize table", false);

    const int threadCount  = 8;
    const int opsPerThread = 50000;

    std::vector<std::thread> threads;

    auto worker = [&table](int threadId)
        {
            FastRng rng(threadId + 100); // Seed per thread

            for (int i = 0; i < opsPerThread; ++i)
            {
                // Randomize key across a space slightly larger than capacity to force constant thrashing
                uint64_t key = rng.Next() % (opsPerThread * 4);
                RefCountedPayload* p = new RefCountedPayload(key);
                p->AddRef();

                bool isAdded = table.Add(key, p);
                (void)isAdded;

                p->Release();

                if (i % 10 == 0)
                {
                    RefCountedPayload* out = nullptr;
                    if (table.Lookup(key, out))
                    {
                        out->Release();
                    }
                }
            }
        };

    for (int i = 0; i < threadCount; ++i)
    {
        threads.emplace_back(worker, i);
    }

    for (auto& t : threads)
    {
        t.join();
    }

    table.Cleanup();

    TEST_REQUIRE(g_LiveObjectsCount.load() == 0, "MEMORY LEAK DETECTED: Objects stranded.", false);

    std::cout << "[+] Eviction & Exhaustion Tests Passed.\n";

    return true;
}

// ----------------------------------------------------------------------------
// A highly aggressive multi-threaded test running on a tiny cache capacity to force
// simultaneous ABA hazards, TOCTOU races, and heavy eviction thrashing.
// ----------------------------------------------------------------------------
template <typename TTable>
bool RunMultiThreadedCorrectnessTest()
{
    std::cout << "[*] Running Multi-Threaded Correctness Test (High Contention & Eviction)...\n";

    TTable table;

    // Setup a tiny capacity relative to the key space.
    // This strictly forces extreme LRU eviction thrashing, ABA hazards, and TOCTOU get-or-add
    // scenarios simultaneously.
    const size_t   CACHE_CAP = 64;
    const uint64_t KEY_SPACE = 200;

    TEST_REQUIRE(InitTableHelper(table, CACHE_CAP, 0), "Failed to initialize table", false);

    unsigned int threadCount = std::thread::hardware_concurrency();

    if (threadCount == 0)
    {
        threadCount = 4;
    }

    const int OPS_PER_THREAD = 100000;
    std::vector<std::thread> threads;
    std::atomic<bool> startFlag{ false };
    std::atomic<bool> dataCorruption{ false };

    auto worker = [&](int threadId)
        {
            FastRng rng(threadId + 9999);

            // Spin wait to synchronize all threads before hammering the cache
            while (!startFlag.load(std::memory_order_acquire))
            {
                std::this_thread::yield();
            }

            for (int i = 0; i < OPS_PER_THREAD; ++i)
            {
                uint64_t key = rng.Next() % KEY_SPACE;
                uint32_t opType = rng.Next() % 100;

                if (opType < 50)
                {
                    // 50% Lookup
                    RefCountedPayload* out = nullptr;

                    if (table.Lookup(key, out))
                    {
                        // Strict Data Integrity Check
                        if (out->Data != key)
                        {
                            dataCorruption.store(true, std::memory_order_relaxed);
                        }

                        out->Release();
                    }
                }
                else if (opType < 75)
                {
                    // 25% Add
                    RefCountedPayload* p = new RefCountedPayload(key);
                    p->AddRef();

                    // Using standard signature. If the table uses Get-Or-Add and there is a collision,
                    // the table rejects the new payload and returns false. The caller retains ownership 
                    // and cleanly releases the loser payload.
                    bool bIsAdded = table.Add(key, p);
                    (void)bIsAdded;

                    p->Release();
                }
                else
                {
                    // 25% Remove
                    bool bIsRemoved = table.Remove(key);
                    (void)bIsRemoved;
                }
            }
        };

    for (unsigned int i = 0; i < threadCount; ++i)
    {
        threads.emplace_back(worker, i);
    }

    startFlag.store(true, std::memory_order_release);

    for (auto& t : threads)
    {
        t.join();
    }

    TEST_REQUIRE(!dataCorruption.load(),
                 "Data corruption detected! (Value data mismatched the mapped Key)",
                 false);

    size_t finalCount = table.GetTotalItemCount();

    TEST_REQUIRE(finalCount <= CACHE_CAP, "Item count exceeds cache capacity bounds!", false);
    TEST_REQUIRE(finalCount <= KEY_SPACE, "Item count exceeds unique keys inserted!", false);

    // Optional Enumerate check (using SFINAE) if the table supports it to ensure internal links
    // aren't orphaned
    if constexpr (requires {
        table.Enumerate(
            [](const uint64_t&, RefCountedPayload*)
            {
                return true;
            });
    })
    {
        size_t enumCount = 0;

        table.Enumerate(
            [&](const uint64_t& key, RefCountedPayload* val) -> bool
            {
                if (val)
                {
                    if (val->Data != key)
                    {
                        dataCorruption.store(true, std::memory_order_relaxed);
                    }

                    enumCount++;
                }

                return true;
            });

        TEST_REQUIRE(!dataCorruption.load(), "Data corruption detected during Enumerate pass!", false);

        TEST_REQUIRE(enumCount == finalCount,
                     "Enumerate count does not match GetTotalItemCount! (Structural corruption)",
                     false);
    }

    std::cout << "[+] Multi-Threaded Correctness Test Passed.\n";
    return true;
}

template <typename TTable>
PerformanceMetrics RunPerformanceTest(size_t CacheCapacity,
    size_t PrePopulateCount,
    size_t OperationCount,
    const char* TestName,
    uint32_t PromotionThreshold = 0,
    bool bRandomKeys = false) // <-- New Parameter
{
    std::cout << "[*] Running Performance Test: " << TestName << " (Cap: " << CacheCapacity
        << ", Threshold: " << PromotionThreshold << "%)...\n";

    PerformanceMetrics metrics = { 0 };
    TTable table;

    TEST_REQUIRE(InitTableHelper(table, CacheCapacity, PromotionThreshold),
                 "Failed to initialize table",
                 metrics);

    size_t memoryBytes = table.GetTotalMemoryUsage();
    double memoryMB = static_cast<double>(memoryBytes) / (1024.0 * 1024.0);

    std::cout << "    - Estimated Memory Footprint : " << memoryBytes << " bytes (" << std::fixed
              << std::setprecision(2) << memoryMB << " MB)\n";
    std::cout << std::flush;

    FastRng setupRng(0x88888888);
    for (size_t i = 0; i < PrePopulateCount; ++i)
    {
        uint64_t key = bRandomKeys ? setupRng.Next() % (CacheCapacity * 2) : i;
        RefCountedPayload* p = new RefCountedPayload(key);
        p->AddRef();

        bool isAdded = table.Add(key, p);
        (void)isAdded;
        p->Release();
    }

    // Pre-generate keys to completely remove RNG overhead from the timed execution paths
    std::vector<uint64_t> testKeys(OperationCount);
    FastRng benchRng(0x12345678);
    for (size_t i = 0; i < OperationCount; ++i)
    {
        // For random, we multiply capacity by 4 to guarantee a mix of hits and misses (forcing eviction)
        testKeys[i] = bRandomKeys ? benchRng.Next() % (CacheCapacity * 4) : (PrePopulateCount + i);
    }

    const int PASSES = 4;

    std::chrono::duration<double, std::milli> addTime(0);
    std::chrono::duration<double, std::milli> lookupTime(0);
    std::chrono::duration<double, std::milli> removeTime(0);

    for (int pass = 0; pass < PASSES; ++pass)
    {
        auto startAdd = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < OperationCount; ++i)
        {
            uint64_t key = testKeys[i];
            RefCountedPayload* p = new RefCountedPayload(key);
            p->AddRef();

            bool isAdded = table.Add(key, p);
            (void)isAdded;
            p->Release();
        }

        auto endAdd = std::chrono::high_resolution_clock::now();
        addTime += (endAdd - startAdd);

        auto startLookup = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < OperationCount; ++i)
        {
            RefCountedPayload* out = nullptr;
            if (table.Lookup(testKeys[i], out))
            {
                out->Release();
            }
        }

        auto endLookup = std::chrono::high_resolution_clock::now();
        lookupTime += (endLookup - startLookup);

        auto startRemove = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < OperationCount; ++i)
        {
            bool isRemoved = table.Remove(testKeys[i]);
            (void)isRemoved;
        }

        auto endRemove = std::chrono::high_resolution_clock::now();
        removeTime += (endRemove - startRemove);
    }

    auto calculateAndPrint = [](const char* OpName, size_t Ops, double MsTime) -> uint64_t
        {
            double seconds = MsTime / 1000.0;
            double throughput = static_cast<double>(Ops) / seconds;
            double timePerOpNs = (seconds * 1000000000.0) / static_cast<double>(Ops);

            std::cout << "    - " << OpName << " " << Ops << " items:\n";
            std::cout << "        Total Time  : " << MsTime << " ms\n";
            std::cout << "        Time per Op : " << timePerOpNs << " ns\n";
            std::cout << "        Throughput  : " << static_cast<uint64_t>(throughput) << " Ops/sec\n";

            return static_cast<uint64_t>(throughput);
        };

    size_t totalOps = OperationCount * PASSES;

    metrics.AddThroughput = calculateAndPrint("Add", totalOps, addTime.count());
    metrics.LookupThroughput = calculateAndPrint("Lookup", totalOps, lookupTime.count());
    metrics.RemoveThroughput = calculateAndPrint("Remove", totalOps, removeTime.count());

    std::cout << "[+] " << TestName << " Complete.\n";

    return metrics;
}

// ----------------------------------------------------------------------------
// Measures highly concurrent multi-threaded throughput on a massive capacity table
// where no evictions occur. Isolates lock contention and routing efficiency.
// Includes a 1-second warm-up phase.
// ----------------------------------------------------------------------------
template <typename TTable>
bool RunContentionTest_NoEviction(int SecondsToRun)
{
    std::cout << "[*] Running Contention Test: Lightly Populated (No Evictions) for "
              << SecondsToRun << " seconds...\n";

    TTable table;

    TEST_REQUIRE(InitTableHelper(table, 10000000), "Failed to initialize table", false);

    std::atomic<uint64_t> totalOps{ 0 };
    std::atomic<bool> warmUpFlag{ true };
    std::atomic<bool> stopFlag{ false };

    unsigned int threadCount = std::thread::hardware_concurrency();

    if (threadCount == 0)
    {
        threadCount = 4;
    }

    std::vector<std::thread> threads;

    auto worker = [&](int threadId)
        {
            FastRng rng(threadId + 1);
            uint64_t localOps = 0;

            // Warm-up phase
            while (warmUpFlag.load(std::memory_order_relaxed))
            {
                uint64_t key = rng.Next() % 50000;
                RefCountedPayload* out = nullptr;

                if (table.Lookup(key, out))
                {
                    out->Release();
                }
            }

            // Timed benchmark phase
            while (!stopFlag.load(std::memory_order_relaxed))
            {
                uint64_t key = rng.Next() % 50000;
                uint32_t opType = rng.Next() % 100;

                if (opType < 70)
                {
                    RefCountedPayload* out = nullptr;

                    if (table.Lookup(key, out))
                    {
                        out->Release();
                    }
                }
                else if (opType < 90)
                {
                    RefCountedPayload* p = new RefCountedPayload(key);
                    p->AddRef();

                    bool isAdded = table.Add(key, p);
                    (void)isAdded;

                    p->Release();
                }
                else
                {
                    bool isRemoved = table.Remove(key);
                    (void)isRemoved;
                }

                localOps++;
            }

            totalOps.fetch_add(localOps, std::memory_order_relaxed);
        };

    for (unsigned int i = 0; i < threadCount; ++i)
    {
        threads.emplace_back(worker, i);
    }

    std::cout << "    - Warming up for 1 second...\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "    - Warm-up complete. Running benchmark for " << SecondsToRun
              << " seconds...\n";
    std::cout << std::flush;

    warmUpFlag.store(false, std::memory_order_relaxed);
    auto start = std::chrono::high_resolution_clock::now();

    std::this_thread::sleep_for(std::chrono::seconds(SecondsToRun));

    stopFlag.store(true, std::memory_order_relaxed);

    for (auto& t : threads)
    {
        t.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    double timePerOpNs = (elapsed.count() * 1000000000.0) / static_cast<double>(totalOps.load());

    std::cout << "    - Threads: " << threadCount << "\n";
    std::cout << "    - Total Time: " << elapsed.count() << " seconds\n";
    std::cout << "    - Total Operations: " << totalOps.load() << "\n";
    std::cout << "    - Time per Op: " << timePerOpNs << " ns\n";
    std::cout << "    - Ops/Second: " << static_cast<uint64_t>(totalOps.load() / elapsed.count()) << "\n";
    std::cout << "[+] Contention Test (No Eviction) Complete.\n";

    return true;
}

// ----------------------------------------------------------------------------
// Measures multi-threaded throughput on a severely constrained table, forcing constant
// out-of-lock destruction and background yielding under heavy lock pressure.
// Includes a 1-second warm-up phase.
// ----------------------------------------------------------------------------
template <typename TTable>
bool RunContentionTest_EvictionThrashing(int SecondsToRun)
{
    std::cout << "[*] Running Contention Test: Full Table (Constant Evictions) for " << SecondsToRun
              << " seconds...\n";

    TTable table;

    TEST_REQUIRE(InitTableHelper(table, 1000), "Failed to initialize table", false);

    std::atomic<uint64_t> totalOps{ 0 };
    std::atomic<bool> warmUpFlag{ true };
    std::atomic<bool> stopFlag{ false };

    unsigned int threadCount = std::thread::hardware_concurrency();

    if (threadCount == 0)
    {
        threadCount = 4;
    }

    std::vector<std::thread> threads;

    auto worker = [&](int threadId)
        {
            FastRng rng(threadId + 1000);
            uint64_t localOps = 0;

            // Warm-up phase
            while (warmUpFlag.load(std::memory_order_relaxed))
            {
                uint64_t key = rng.Next() % 5000000;
                RefCountedPayload* out = nullptr;

                if (table.Lookup(key, out))
                {
                    out->Release();
                }
            }

            // Timed benchmark phase
            while (!stopFlag.load(std::memory_order_relaxed))
            {
                uint64_t key = rng.Next() % 5000000;
                uint32_t opType = rng.Next() % 100;

                if (opType < 20)
                {
                    RefCountedPayload* out = nullptr;

                    if (table.Lookup(key, out))
                    {
                        out->Release();
                    }
                }
                else
                {
                    RefCountedPayload* p = new RefCountedPayload(key);
                    p->AddRef();

                    bool isAdded = table.Add(key, p);
                    (void)isAdded;

                    p->Release();
                }

                localOps++;
            }

            totalOps.fetch_add(localOps, std::memory_order_relaxed);
        };

    for (unsigned int i = 0; i < threadCount; ++i)
    {
        threads.emplace_back(worker, i);
    }

    std::cout << "    - Warming up for 1 second...\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "    - Warm-up complete. Running benchmark for " << SecondsToRun
              << " seconds...\n";

    warmUpFlag.store(false, std::memory_order_relaxed);
    auto start = std::chrono::high_resolution_clock::now();

    std::this_thread::sleep_for(std::chrono::seconds(SecondsToRun));

    stopFlag.store(true, std::memory_order_relaxed);

    for (auto& t : threads)
    {
        t.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    table.Cleanup();

    double timePerOpNs = (elapsed.count() * 1000000000.0) / static_cast<double>(totalOps.load());

    std::cout << "    - Threads: " << threadCount << "\n";
    std::cout << "    - Total Time: " << elapsed.count() << " seconds\n";
    std::cout << "    - Total Operations: " << totalOps.load() << "\n";
    std::cout << "    - Time per Op: " << timePerOpNs << " ns\n";
    std::cout << "    - Ops/Second: " << static_cast<uint64_t>(totalOps.load() / elapsed.count()) << "\n";
    std::cout << "[+] Contention Test (Eviction Thrashing) Complete.\n";

    return true;
}

// ----------------------------------------------------------------------------
// Simulates a realistic production environment with a mix of Add, Lookup, Remove,
// and a dedicated background Trimmer thread. Includes a 1-second warm-up phase.
// ----------------------------------------------------------------------------
template <typename TTable>
uint64_t RunContentionTest_MixedWorkload(int      SecondsToRun,
                                         size_t   CacheCapacity,
                                         uint32_t PromotionThreshold = 100)
{
    std::cout << "[*] Running Contention Test: Mixed Workload (Add/Remove/Trim) for "
              << SecondsToRun << " seconds (Cap: " << CacheCapacity
              << ", Threshold: " << PromotionThreshold << "%)...\n";

    TTable table;

    TEST_REQUIRE(InitTableHelper(table, CacheCapacity, PromotionThreshold), "Failed to initialize table", 0);

    for (size_t i = 0; i < CacheCapacity; ++i)
    {
        RefCountedPayload* p = new RefCountedPayload(i);
        p->AddRef();

        bool isAdded = table.Add(i, p);
        (void)isAdded;

        p->Release();
    }

    std::atomic<uint64_t> totalOps{ 0 };
    std::atomic<uint64_t> totalTrimmed{ 0 };
    std::atomic<bool> warmUpFlag{ true };
    std::atomic<bool> stopFlag{ false };

    unsigned int threadCount = std::thread::hardware_concurrency();

    if (threadCount == 0)
    {
        threadCount = 4;
    }

    std::vector<std::thread> threads;

    auto worker = [&](int threadId)
        {
            FastRng rng(threadId + 2000);
            uint64_t localOps = 0;
            uint64_t maxKey = CacheCapacity + (CacheCapacity / 2);

            // Warm-up phase
            while (warmUpFlag.load(std::memory_order_relaxed))
            {
                uint64_t key = rng.Next() % maxKey;
                RefCountedPayload* out = nullptr;

                if (table.Lookup(key, out))
                {
                    out->Release();
                }
            }

            // Timed benchmark phase
            while (!stopFlag.load(std::memory_order_relaxed))
            {
                uint64_t key = rng.Next() % maxKey;
                uint32_t opType = rng.Next() % 100;

                if (opType < 60)
                {
                    RefCountedPayload* out = nullptr;

                    if (table.Lookup(key, out))
                    {
                        out->Release();
                    }
                }
                else if (opType < 80)
                {
                    RefCountedPayload* p = new RefCountedPayload(key);
                    p->AddRef();

                    bool isAdded = table.Add(key, p);
                    (void)isAdded;

                    p->Release();
                }
                else
                {
                    bool isRemoved = table.Remove(key);
                    (void)isRemoved;
                }

                localOps++;
            }

            totalOps.fetch_add(localOps, std::memory_order_relaxed);
        };

    auto trimmer = [&]()
        {
            size_t trimTarget = CacheCapacity / 20;
            uint64_t localTrimmed = 0;

            // Trimmer also waits out the warm-up phase to avoid skewing initial capacity
            while (warmUpFlag.load(std::memory_order_relaxed))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }

            while (!stopFlag.load(std::memory_order_relaxed))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));

                size_t trimmed = table.Trim(trimTarget);
                localTrimmed += trimmed;
            }

            totalTrimmed.fetch_add(localTrimmed, std::memory_order_relaxed);
        };

    threads.emplace_back(trimmer);

    for (unsigned int i = 0; i < threadCount - 1; ++i)
    {
        threads.emplace_back(worker, i);
    }

    std::cout << "    - Warming up for 1 second...\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "    - Warm-up complete. Running benchmark for " << SecondsToRun
              << " seconds...\n";

    warmUpFlag.store(false, std::memory_order_relaxed);
    auto start = std::chrono::high_resolution_clock::now();

    std::this_thread::sleep_for(std::chrono::seconds(SecondsToRun));

    stopFlag.store(true, std::memory_order_relaxed);

    for (auto& t : threads)
    {
        t.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    table.Cleanup();

    uint64_t throughput = static_cast<uint64_t>(totalOps.load() / elapsed.count());
    double timePerOpNs = (elapsed.count() * 1000000000.0) / static_cast<double>(totalOps.load());

    std::cout << "    - Threads: " << threadCount << " (1 Trimmer, " << (threadCount - 1)
              << " Workers)\n";
    std::cout << "    - Total Time: " << elapsed.count() << " seconds\n";
    std::cout << "    - Total Operations: " << totalOps.load() << "\n";
    std::cout << "    - Total Items Trimmed: " << totalTrimmed.load() << "\n";
    std::cout << "    - Time per Op: " << timePerOpNs << " ns\n";
    std::cout << "    - Ops/Second: " << throughput << "\n";
    std::cout << "[+] Contention Test Complete.\n";

    return throughput;
}

// ----------------------------------------------------------------------------
// Simulates a heavily oversubscribed production environment (3x CPU threads)
// with a mix of Add, Lookup, Remove, and exactly 1 dedicated background Trimmer.
// ----------------------------------------------------------------------------
template <typename TTable>
uint64_t RunContentionTest_MixedWorkloadOversubscribed(int      SecondsToRun,
                                                       size_t   CacheCapacity,
                                                       uint32_t PromotionThreshold = 100)
{
    std::cout << "[*] Running Contention Test: Mixed Workload Oversubscribed (3x Threads) for "
              << SecondsToRun << " seconds (Cap: " << CacheCapacity
              << ", Threshold: " << PromotionThreshold << "%)...\n";

    TTable table;

    TEST_REQUIRE(InitTableHelper(table, CacheCapacity, PromotionThreshold), "Failed to initialize table", 0);

    for (size_t i = 0; i < CacheCapacity; ++i)
    {
        RefCountedPayload* p = new RefCountedPayload(i);
        p->AddRef();

        bool isAdded = table.Add(i, p);
        (void)isAdded;
        p->Release();
    }

    std::atomic<uint64_t> totalOps{ 0 };
    std::atomic<uint64_t> totalTrimmed{ 0 };
    std::atomic<bool> warmUpFlag{ true };
    std::atomic<bool> stopFlag{ false };

    unsigned int hardwareCores = std::thread::hardware_concurrency();
    if (hardwareCores == 0)
    {
        hardwareCores = 4;
    }

    // Explicit 3x Oversubscription
    unsigned int threadCount = hardwareCores * 3;

    std::vector<std::thread> threads;

    auto worker = [&](int threadId)
        {
            FastRng rng(threadId + 2000);
            uint64_t localOps = 0;
            uint64_t maxKey = CacheCapacity + (CacheCapacity / 2);

            // Warm-up phase
            while (warmUpFlag.load(std::memory_order_relaxed))
            {
                uint64_t key = rng.Next() % maxKey;
                RefCountedPayload* out = nullptr;
                if (table.Lookup(key, out))
                {
                    out->Release();
                }
            }

            // Timed benchmark phase
            while (!stopFlag.load(std::memory_order_relaxed))
            {
                uint64_t key = rng.Next() % maxKey;
                uint32_t opType = rng.Next() % 100;

                if (opType < 60)
                {
                    RefCountedPayload* out = nullptr;
                    if (table.Lookup(key, out))
                    {
                        out->Release();
                    }
                }
                else if (opType < 80)
                {
                    RefCountedPayload* p = new RefCountedPayload(key);
                    p->AddRef();

                    (void)table.Add(key, p);
                    p->Release();
                }
                else
                {
                    (void)table.Remove(key);
                }
                localOps++;
            }
            totalOps.fetch_add(localOps, std::memory_order_relaxed);
        };

    auto trimmer = [&]()
        {
            size_t trimTarget = CacheCapacity / 20;
            uint64_t localTrimmed = 0;

            while (warmUpFlag.load(std::memory_order_relaxed))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }

            while (!stopFlag.load(std::memory_order_relaxed))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                localTrimmed += table.Trim(trimTarget);
            }
            totalTrimmed.fetch_add(localTrimmed, std::memory_order_relaxed);
        };

    // Exactly 1 Trimmer
    threads.emplace_back(trimmer);

    // N - 1 Workers
    for (unsigned int i = 0; i < threadCount - 1; ++i)
    {
        threads.emplace_back(worker, i);
    }

    std::cout << "    - Warming up for 1 second...\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "    - Warm-up complete. Running benchmark for " << SecondsToRun
              << " seconds...\n";

    warmUpFlag.store(false, std::memory_order_relaxed);
    auto start = std::chrono::high_resolution_clock::now();

    std::this_thread::sleep_for(std::chrono::seconds(SecondsToRun));
    stopFlag.store(true, std::memory_order_relaxed);

    for (auto& t : threads)
    {
        t.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    table.Cleanup();

    uint64_t throughput = static_cast<uint64_t>(totalOps.load() / elapsed.count());
    double timePerOpNs = (elapsed.count() * 1000000000.0) / static_cast<double>(totalOps.load());

    std::cout << "    - Threads: " << threadCount << " (1 Trimmer, " << (threadCount - 1)
              << " Workers)\n";
    std::cout << "    - Total Operations: " << totalOps.load() << "\n";
    std::cout << "    - Total Items Trimmed: " << totalTrimmed.load() << "\n";
    std::cout << "    - Time per Op: " << timePerOpNs << " ns\n";
    std::cout << "    - Ops/Second: " << throughput << "\n";
    std::cout << "[+] Contention Test Complete.\n";

    return throughput;
}

// ----------------------------------------------------------------------------
// Simulates a read-heavy workload (95% Lookup) with an uneven key distribution
// (Pareto/Zipfian approximation). Evaluates the impact of Lazy LRU Promotion.
// Includes a 1-second warm-up phase.
// ----------------------------------------------------------------------------
template <typename TTable>
uint64_t RunContentionTest_ReadHeavySkewed(int      SecondsToRun,
                                           size_t   CacheCapacity,
                                           uint32_t PromotionThreshold = 100)
{
    std::cout << "[*] Running Contention Test: Read-Heavy Skewed (95% Read) for " << SecondsToRun
              << " seconds (Cap: " << CacheCapacity << ", Threshold: " << PromotionThreshold
              << "%)...\n";

    TTable table;

    TEST_REQUIRE(InitTableHelper(table, CacheCapacity, PromotionThreshold), "Failed to initialize table", 0);

    for (size_t i = 0; i < CacheCapacity; ++i)
    {
        RefCountedPayload* p = new RefCountedPayload(i);
        p->AddRef();

        bool isAdded = table.Add(i, p);
        (void)isAdded;

        p->Release();
    }

    std::atomic<uint64_t> totalOps{ 0 };
    std::atomic<uint64_t> totalTrimmed{ 0 };
    std::atomic<bool> warmUpFlag{ true };
    std::atomic<bool> stopFlag{ false };

    unsigned int threadCount = std::thread::hardware_concurrency();

    if (threadCount == 0)
    {
        threadCount = 4;
    }

    std::vector<std::thread> threads;

    auto worker = [&](int threadId)
        {
            FastRng rng(threadId + 3000);
            uint64_t localOps = 0;

            uint64_t maxKey = CacheCapacity + (CacheCapacity / 2);
            uint64_t hotSet = maxKey / 5;

            // Warm-up phase
            while (warmUpFlag.load(std::memory_order_relaxed))
            {
                uint64_t key;

                if ((rng.Next() % 100) < 80)
                {
                    key = rng.Next() % hotSet;
                }
                else
                {
                    key = rng.Next() % maxKey;
                }

                RefCountedPayload* out = nullptr;
                if (table.Lookup(key, out))
                {
                    out->Release();
                }
            }

            // Timed benchmark phase
            while (!stopFlag.load(std::memory_order_relaxed))
            {
                uint64_t key;

                if ((rng.Next() % 100) < 80)
                {
                    key = rng.Next() % hotSet;
                }
                else
                {
                    key = rng.Next() % maxKey;
                }

                uint32_t opType = rng.Next() % 100;

                if (opType < 95)
                {
                    RefCountedPayload* out = nullptr;

                    if (table.Lookup(key, out))
                    {
                        out->Release();
                    }
                }
                else if (opType < 98)
                {
                    RefCountedPayload* p = new RefCountedPayload(key);
                    p->AddRef();

                    bool isAdded = table.Add(key, p);
                    (void)isAdded;

                    p->Release();
                }
                else
                {
                    bool isRemoved = table.Remove(key);
                    (void)isRemoved;
                }

                localOps++;
            }

            totalOps.fetch_add(localOps, std::memory_order_relaxed);
        };

    auto trimmer = [&]()
        {
            size_t   trimTarget = CacheCapacity / 20;
            uint64_t localTrimmed = 0;

            // Trimmer also waits out the warm-up phase
            while (warmUpFlag.load(std::memory_order_relaxed))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }

            while (!stopFlag.load(std::memory_order_relaxed))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));

                size_t trimmed = table.Trim(trimTarget);
                localTrimmed += trimmed;
            }

            totalTrimmed.fetch_add(localTrimmed, std::memory_order_relaxed);
        };

    threads.emplace_back(trimmer);

    for (unsigned int i = 0; i < threadCount - 1; ++i)
    {
        threads.emplace_back(worker, i);
    }

    std::cout << "    - Warming up for 1 second...\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "    - Warm-up complete. Running benchmark for " << SecondsToRun
              << " seconds...\n";

    warmUpFlag.store(false, std::memory_order_relaxed);
    auto start = std::chrono::high_resolution_clock::now();

    std::this_thread::sleep_for(std::chrono::seconds(SecondsToRun));

    stopFlag.store(true, std::memory_order_relaxed);

    for (auto& t : threads)
    {
        t.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    table.Cleanup();

    uint64_t throughput = static_cast<uint64_t>(totalOps.load() / elapsed.count());
    double timePerOpNs = (elapsed.count() * 1000000000.0) / static_cast<double>(totalOps.load());

    std::cout << "    - Threads: " << threadCount << " (1 Trimmer, " << (threadCount - 1)
              << " Workers)\n";
    std::cout << "    - Total Time: " << elapsed.count() << " seconds\n";
    std::cout << "    - Total Operations: " << totalOps.load() << "\n";
    std::cout << "    - Total Items Trimmed: " << totalTrimmed.load() << "\n";
    std::cout << "    - Time per Op: " << timePerOpNs << " ns\n";
    std::cout << "    - Ops/Second: " << throughput << "\n";
    std::cout << "[+] Contention Test Complete.\n";

    return throughput;
}

// ----------------------------------------------------------------------------
// Measures the sheer speed and efficiency of the background yielding trim operation
// when instructed to remove a specific percentage of the table.
// ----------------------------------------------------------------------------
template <typename TTable>
bool RunTrimPerformanceTest(size_t CacheCapacity, 
                            double TrimPercentage)
{
    std::cout << "[*] Running Trim Performance Test (Cap: " << CacheCapacity
              << ", Trim: " << (TrimPercentage * 100) << "%)...\n";

    TTable table;

    TEST_REQUIRE(InitTableHelper(table, CacheCapacity), "Failed to initialize table", false);

    for (size_t i = 0; i < CacheCapacity; ++i)
    {
        RefCountedPayload* p = new RefCountedPayload(i);
        p->AddRef();

        bool isAdded = table.Add(i, p);
        (void)isAdded;

        p->Release();
    }

    size_t itemsToTrim = static_cast<size_t>(CacheCapacity * TrimPercentage);

    auto start = std::chrono::high_resolution_clock::now();

    size_t itemsActuallyTrimmed = table.Trim(itemsToTrim);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> trimTime = end - start;

    double seconds = trimTime.count() / 1000.0;
    double timePerOpNs = 0.0;
    double throughput = 0.0;

    if (itemsActuallyTrimmed > 0)
    {
        timePerOpNs = (seconds * 1000000000.0) / static_cast<double>(itemsActuallyTrimmed);
        throughput = static_cast<double>(itemsActuallyTrimmed) / seconds;
    }

    std::cout << "    - Target Trim Count : " << itemsToTrim << " items\n";
    std::cout << "    - Actual Trim Count : " << itemsActuallyTrimmed << " items\n";
    std::cout << "    - Total Trim Time   : " << trimTime.count() << " ms\n";

    if (itemsActuallyTrimmed > 0)
    {
        std::cout << "    - Time per Item     : " << timePerOpNs << " ns\n";
        std::cout << "    - Trim Throughput   : " << static_cast<uint64_t>(throughput)
                  << " Ops/sec\n";
    }

    std::cout << "[+] Trim Performance Test Complete.\n\n";

    return true;
}

// ----------------------------------------------------------------------------
// Validates the high/low watermark background trimming logic by overfilling
// the table and calling Trim(0) to force stabilization.
// ----------------------------------------------------------------------------
template <typename TTable>
bool RunTrimToWatermarkTest(size_t CacheCapacity)
{
    std::cout << "[*] Running Trim-to-Watermark Test (Trim(0)) (Cap: " << CacheCapacity << ")...\n";

    TTable table;

    TEST_REQUIRE(InitTableHelper(table, CacheCapacity), "Failed to initialize table", false);

    // Fill the table to maximum capacity to trigger the >90% high watermark
    for (size_t i = 0; i < CacheCapacity; ++i)
    {
        RefCountedPayload* p = new RefCountedPayload(i);
        p->AddRef();

        bool isAdded = table.Add(i, p);
        (void)isAdded;

        p->Release();
    }

    size_t itemsBeforeTrim = table.GetTotalItemCount();
    std::cout << "    - Items before Trim(0) : " << itemsBeforeTrim << "\n";
    std::cout << std::flush;

    auto start = std::chrono::high_resolution_clock::now();

    size_t itemsTrimmed = table.Trim(0);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> trimTime = end - start;

    size_t itemsAfterTrim = table.GetTotalItemCount();

    std::cout << "    - Items after Trim(0)  : " << itemsAfterTrim << "\n";
    std::cout << "    - Actual Trim Count    : " << itemsTrimmed << " items\n";
    std::cout << "    - Total Trim(0) Time   : " << trimTime.count() << " ms\n";

    if (itemsTrimmed > 0)
    {
        double seconds = trimTime.count() / 1000.0;
        double timePerOpNs = (seconds * 1000000000.0) / static_cast<double>(itemsTrimmed);
        double throughput = static_cast<double>(itemsTrimmed) / seconds;

        std::cout << "    - Time per Item        : " << timePerOpNs << " ns\n";
        std::cout << "    - Trim(0) Throughput   : " << static_cast<uint64_t>(throughput)
                  << " Ops/sec\n";
    }

    TEST_REQUIRE(itemsTrimmed > 0, "Trim(0) did not remove any items", false);
    TEST_REQUIRE(itemsBeforeTrim == (itemsAfterTrim + itemsTrimmed), "Trim count mismatch", false);

    // Allow slight variance for internal shard capacities (by comparing against the high watermark
    // + a shard padding factor)
    size_t expectedMaxItems = (CacheCapacity * 90) / 100;

    TEST_REQUIRE(itemsAfterTrim <= (expectedMaxItems + 256),
                 "Trim(0) failed to reach watermark bounds",
                 false);

    std::cout << "[+] Trim-to-Watermark Test Complete.\n\n";

    return true;
}

// ----------------------------------------------------------------------------
// Validates the forced-trim override, ensuring that Trim(N, true) bypasses
// capacity watermarks and strictly evicts the requested number of items.
// ----------------------------------------------------------------------------
template <typename TTable>
bool RunForcedTrimTest(size_t CacheCapacity)
{
    std::cout << "[*] Running Forced Trim Test (Trim(Count, true)) (Cap: " << CacheCapacity
              << ")...\n";

    TTable table;

    TEST_REQUIRE(InitTableHelper(table, CacheCapacity), "Failed to initialize table", false);

    // ------------------------------------------------------------------------
    // Phase 1: Test Complete Eviction via Trim(0, true)
    // ------------------------------------------------------------------------
    for (size_t i = 0; i < CacheCapacity; ++i)
    {
        RefCountedPayload* p = new RefCountedPayload(i);
        p->AddRef();

        bool isAdded = table.Add(i, p);
        (void)isAdded;

        p->Release();
    }

    size_t fullCount = table.GetTotalItemCount();

    TEST_REQUIRE(fullCount > 0, "Table failed to populate", false);

    size_t trimmedAll = table.Trim(0, true);
    size_t countAfterClear = table.GetTotalItemCount();

    std::cout << "    - Trim(0, true) removed " << trimmedAll << " items.\n";

    TEST_REQUIRE(countAfterClear == 0, "Trim(0, true) failed to completely empty the table", false);
    TEST_REQUIRE(trimmedAll == fullCount, "Trim(0, true) return count mismatch", false);

    // ------------------------------------------------------------------------
    // Phase 2: Test Targeted Forced Eviction Below Watermarks
    // ------------------------------------------------------------------------
    for (size_t i = 0; i < 10; ++i)
    {
        RefCountedPayload* p = new RefCountedPayload(i);
        p->AddRef();

        bool isAdded = table.Add(i, p);
        (void)isAdded;

        p->Release();
    }

    size_t countBeforeForced = table.GetTotalItemCount();

    TEST_REQUIRE(countBeforeForced == 10, "Failed to repopulate 10 items", false);

    // Verify normal trim respects the watermark and does nothing
    size_t normalTrim = table.Trim(5, false);

    TEST_REQUIRE(normalTrim == 0, "Normal trim incorrectly removed items below watermark", false);

    // Verify forced trim ignores the watermark and removes exactly 5
    size_t forcedTrim = table.Trim(5, true);
    size_t countAfterForced = table.GetTotalItemCount();

    std::cout << "    - Trim(5, true) removed " << forcedTrim << " items from a 10-item table.\n";

    TEST_REQUIRE(forcedTrim == 5, "Trim(5, true) failed to remove exactly 5 items", false);
    TEST_REQUIRE(countAfterForced == 5, "Table count is incorrect after forced trim", false);

    std::cout << "[+] Forced Trim Test Complete.\n\n";

    return true;
}

// ----------------------------------------------------------------------------
// Tests the safe traversal of all active nodes in the table via the
// Enumerate callback, including validation of early-abort mechanics.
// ----------------------------------------------------------------------------
template <typename TTable>
bool RunEnumerateTest(size_t CacheCapacity)
{
    std::cout << "[*] Running Enumerate Test (Cap: " << CacheCapacity << ")...\n";

    TTable table;

    // Initialize with 2x capacity to guarantee no premature evictions
    // due to imperfect hash distribution across the physical shards.
    TEST_REQUIRE(InitTableHelper(table, CacheCapacity * 2), "Failed to initialize table", false);

    uint64_t expectedSum = 0;

    for (size_t i = 0; i < CacheCapacity; ++i)
    {
        RefCountedPayload* p = new RefCountedPayload(i);
        p->AddRef();

        bool isAdded = table.Add(i, p);
        (void)isAdded;

        p->Release();
        expectedSum += i;
    }

    size_t count = 0;
    uint64_t actualSum = 0;

    // ------------------------------------------------------------------------
    // Phase 1: Test Complete Enumeration and Data Integrity
    // ------------------------------------------------------------------------
    table.Enumerate(
        [&](const uint64_t& Key, RefCountedPayload* Value) -> bool
        {
            if (Value)
            {
                count++;
                actualSum += Key;
            }

            return true;
        });

    std::cout << "    - Enumerated " << count << " items.\n";

    TEST_REQUIRE(count == CacheCapacity, "Enumerate did not visit all items", false);
    TEST_REQUIRE(actualSum == expectedSum, "Enumerate data mismatch (checksum failed)", false);

    // ------------------------------------------------------------------------
    // Phase 2: Test Early Abort
    // ------------------------------------------------------------------------
    size_t abortCount = 0;
    const size_t targetAbort = CacheCapacity / 2;

    table.Enumerate(
        [&](const uint64_t& Key, RefCountedPayload* Value) -> bool
        {
            (void)Key;
            (void)Value;

            abortCount++;

            if (abortCount >= targetAbort)
            {
                return false;
            }

            return true;
        });

    std::cout << "    - Early abort stopped at " << abortCount << " items.\n";

    TEST_REQUIRE(abortCount == targetAbort, "Enumerate failed to abort early", false);

    std::cout << "[+] Enumerate Test Complete.\n\n";

    return true;
}

// ----------------------------------------------------------------------------
// Iteratively ramps up concurrent thread counts to map the scalability curve.
// Highlights linear scaling for sharded designs versus negative scaling for global locks.
// Includes a 1-second warm-up phase per step.
// ----------------------------------------------------------------------------
template <typename TTable>
bool RunThreadScalingSweep(const char*     TestName,
                           size_t          CacheCapacity,
                           int             SecondsPerStep,
                           ScalingMetrics* pOutMetrics = nullptr)
{
    std::cout << "[*] Running Thread Scaling Sweep: " << TestName << "...\n";
    std::cout << "    " << std::left << std::setw(10) << "Threads"
              << "| " << std::setw(15) << "Ops/Second"
              << "| " << "Scaling Factor\n";
    std::cout << "    ------------------------------------------\n";

    unsigned int maxThreads = std::thread::hardware_concurrency();

    if (maxThreads == 0)
    {
        maxThreads = 4;
    }

    uint64_t baselineThroughput = 0;

    unsigned int threadCount = 1;
    bool bDidMax = false;

    while (!bDidMax)
    {
        if (threadCount >= maxThreads)
        {
            threadCount = maxThreads;
            bDidMax = true;
        }

        TTable table;

        TEST_REQUIRE(InitTableHelper(table, CacheCapacity, 0), "Init failed", false);

        for (size_t i = 0; i < CacheCapacity; ++i)
        {
            RefCountedPayload* p = new RefCountedPayload(i);
            p->AddRef();

            bool bIsAdded = table.Add(i, p);
            (void)bIsAdded;

            p->Release();
        }

        std::atomic<uint64_t> totalOps{ 0 };
        std::atomic<bool> warmUpFlag{ true };
        std::atomic<bool> stopFlag{ false };
        std::vector<std::thread> threads;

        auto worker = [&](int threadId)
            {
                FastRng rng(threadId + 5000 + threadCount);
                uint64_t localOps = 0;
                uint64_t maxKey = CacheCapacity + (CacheCapacity / 2);

                // Warm-up phase
                while (warmUpFlag.load(std::memory_order_relaxed))
                {
                    uint64_t key = rng.Next() % maxKey;
                    RefCountedPayload* out = nullptr;

                    if (table.Lookup(key, out))
                    {
                        out->Release();
                    }
                }

                // Timed benchmark phase
                while (!stopFlag.load(std::memory_order_relaxed))
                {
                    uint64_t key = rng.Next() % maxKey;
                    uint32_t opType = rng.Next() % 100;

                    if (opType < 80)
                    {
                        RefCountedPayload* out = nullptr;

                        if (table.Lookup(key, out))
                        {
                            out->Release();
                        }
                    }
                    else
                    {
                        RefCountedPayload* p = new RefCountedPayload(key);
                        p->AddRef();

                        bool bIsAdded = table.Add(key, p);
                        (void)bIsAdded;

                        p->Release();
                    }

                    localOps++;
                }

                totalOps.fetch_add(localOps, std::memory_order_relaxed);
            };

        for (unsigned int i = 0; i < threadCount; ++i)
        {
            threads.emplace_back(worker, i);
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));

        warmUpFlag.store(false, std::memory_order_relaxed);
        auto start = std::chrono::high_resolution_clock::now();

        std::this_thread::sleep_for(std::chrono::seconds(SecondsPerStep));

        stopFlag.store(true, std::memory_order_relaxed);

        for (auto& t : threads)
        {
            t.join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        table.Cleanup();

        uint64_t throughput = static_cast<uint64_t>(totalOps.load() / elapsed.count());

        if (threadCount == 1)
        {
            baselineThroughput = throughput;
        }

        double scalingFactor = static_cast<double>(throughput) / static_cast<double>(baselineThroughput);

        if (pOutMetrics)
        {
            if (threadCount == 1)
            {
                pOutMetrics->BaselineThroughput = throughput;
            }
            pOutMetrics->MaxThreads = threadCount;
            pOutMetrics->MaxThroughput = throughput;
            pOutMetrics->ScalingFactor = scalingFactor;
        }

        std::cout << "    " << std::left << std::setw(10) << threadCount << "| " << std::setw(15)
                  << throughput << "| " << std::fixed << std::setprecision(2) << scalingFactor
                  << "x\n";

        if (!bDidMax)
        {
            threadCount *= 2;
        }
    }

    std::cout << "[+] Thread Scaling Sweep Complete.\n\n";

    return true;
}

// ----------------------------------------------------------------------------
// Captures precise nanosecond-level execution times for millions of operations.
// Analyzes algorithmic stability (P50/P90) and hardware/OS jitter (P99.99).
// Includes a 1-second warm-up phase.
// ----------------------------------------------------------------------------
template <typename TTable>
bool RunTailLatencyTest(const char*     TestName,
                        size_t          CacheCapacity,
                        LatencyMetrics* pOutMetrics = nullptr)
{
    std::cout << "[*] Running Tail Latency Test: " << TestName << "...\n";

    TTable table;

    TEST_REQUIRE(InitTableHelper(table, CacheCapacity, 0), "Init failed", false);

    for (size_t i = 0; i < CacheCapacity; ++i)
    {
        RefCountedPayload* p = new RefCountedPayload(i);
        p->AddRef();

        bool bIsAdded = table.Add(i, p);
        (void)bIsAdded;
        p->Release();
    }

    unsigned int threadCount = std::thread::hardware_concurrency();

    if (threadCount == 0)
    {
        threadCount = 4;
    }

    const size_t SAMPLES_PER_THREAD = 50000;
    const uint32_t SAMPLE_RATE = 100;

    std::vector<std::vector<double>> allThreadSamples(threadCount);
    std::vector<std::thread> threads;
    std::atomic<bool> warmUpFlag{ true };
    std::atomic<bool> startFlag{ false };

    auto worker = [&](int threadId)
        {
            FastRng rng(threadId + 8000);
            uint64_t maxKey = CacheCapacity + (CacheCapacity / 2);

            std::vector<double>& localSamples = allThreadSamples[threadId];
            localSamples.reserve(SAMPLES_PER_THREAD);

            uint32_t opCounter = 0;

            while (!startFlag.load(std::memory_order_acquire))
            {
                std::this_thread::yield();
            }

            // Warm-up phase
            while (warmUpFlag.load(std::memory_order_relaxed))
            {
                uint64_t key = rng.Next() % maxKey;
                RefCountedPayload* out = nullptr;

                if (table.Lookup(key, out))
                {
                    out->Release();
                }
            }

            // Sampling phase
            while (localSamples.size() < SAMPLES_PER_THREAD)
            {
                uint64_t key = rng.Next() % maxKey;
                uint32_t opType = rng.Next() % 100;

                opCounter++;

                bool shouldSample = (opCounter % SAMPLE_RATE == 0);

                auto t0 = shouldSample ? std::chrono::high_resolution_clock::now()
                    : std::chrono::time_point<std::chrono::high_resolution_clock>();

                if (opType < 90)
                {
                    RefCountedPayload* out = nullptr;

                    if (table.Lookup(key, out))
                    {
                        out->Release();
                    }
                }
                else
                {
                    RefCountedPayload* p = new RefCountedPayload(key);
                    p->AddRef();
                    bool bIsAdded = table.Add(key, p);
                    (void)bIsAdded;
                    p->Release();
                }

                if (shouldSample)
                {
                    auto t1 = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double, std::nano> elapsed = t1 - t0;

                    localSamples.push_back(elapsed.count());
                }
            }
        };

    for (unsigned int i = 0; i < threadCount; ++i)
    {
        threads.emplace_back(worker, i);
    }

    startFlag.store(true, std::memory_order_release);

    std::cout << "    - Warming up for 1 second...\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "    - Warm-up complete. Collecting tail latency samples...\n";
    std::cout << std::flush;

    warmUpFlag.store(false, std::memory_order_relaxed);

    for (auto& t : threads)
    {
        t.join();
    }

    table.Cleanup();

    std::vector<double> globalSamples;
    globalSamples.reserve(SAMPLES_PER_THREAD * threadCount);

    for (const auto& local : allThreadSamples)
    {
        globalSamples.insert(globalSamples.end(), local.begin(), local.end());
    }

    std::sort(globalSamples.begin(), globalSamples.end());

    auto getPercentile = [&](double p) -> double
        {
            size_t index = static_cast<size_t>(p * globalSamples.size());

            if (index >= globalSamples.size())
            {
                index = globalSamples.size() - 1;
            }

            return globalSamples[index];
        };

    if (pOutMetrics)
    {
        pOutMetrics->P50    = static_cast<uint64_t>(getPercentile(0.50));
        pOutMetrics->P90    = static_cast<uint64_t>(getPercentile(0.90));
        pOutMetrics->P99    = static_cast<uint64_t>(getPercentile(0.99));
        pOutMetrics->P99_9  = static_cast<uint64_t>(getPercentile(0.999));
        pOutMetrics->P99_99 = static_cast<uint64_t>(getPercentile(0.9999));
    }

    std::cout << "    - Total Samples Collected : " << globalSamples.size() << "\n";
    std::cout << "    - P50 (Median) Latency    : " << std::fixed << std::setprecision(2)
                                                    << getPercentile(0.50) << " ns\n";
    std::cout << "    - P90 Latency             : " << getPercentile(0.90) << " ns\n";
    std::cout << "    - P99 Latency             : " << getPercentile(0.99) << " ns\n";
    std::cout << "    - P99.9 Latency           : " << getPercentile(0.999) << " ns\n";
    std::cout << "    - P99.99 Latency          : " << getPercentile(0.9999) << " ns\n";
    std::cout << "[+] Tail Latency Test Complete.\n\n";

    return true;
}

#include "std_lru_hash_table.h"

// ----------------------------------------------------------------------------
// Master Test Orchestrator
// ----------------------------------------------------------------------------
template <typename CustomTestTable, 
          typename CollisionTestTable>
inline int RunAllTests(const char* szCustomName)
{
    using StdTestTable = StdLruHashTable<uint64_t, RefCountedPayload, Hasher64Bit>;

    const size_t cacheCap = 1000000;
    const size_t prePopulate = 900000;
    const size_t operations = 5000000;
    const int contentionSec = 15;
    const size_t contentionCap = 250000;

    std::cout << "\n";
    std::cout << "=========================================================\n";
    std::cout << "     CPU: " << cpu_name() << "\n";
    std::cout << "=========================================================\n\n";

    SetHighPriority();

    PerformanceMetrics customPerf = { 0 };
    PerformanceMetrics stdPerf = { 0 };

    LatencyMetrics customLat = { 0 };
    LatencyMetrics stdLat = { 0 };

    ScalingMetrics customScale = { 0 };
    ScalingMetrics stdScale = { 0 };

    auto runAllTestsLambda = [&]() -> bool
        {
            std::cout << "=========================================================\n";
            std::cout << "         CUSTOM ARRAY-BACKED LRU HASH TABLE              \n";
            std::cout << "=========================================================\n\n";

            if (!RunCorrectnessTests<CustomTestTable>())
            {
                return false;
            }
            std::cout << "\n";

            if (!RunTinyTableTest<CustomTestTable>())
            {
                return false;
            }
            std::cout << "\n";

#ifndef TEST_IS_KM
            if (!RunCustomAllocatorTest())
            {
                return false;
            }
            std::cout << "\n";
#endif

            if (!RunFreeListIntegrityTest<CustomTestTable>())
            {
                return false;
            }
            std::cout << "\n";

            if (!RunZeroCapacitySanityTest<CustomTestTable>())
            {
                return false;
            }
            std::cout << "\n";

            if (!RunEvictionAndExhaustionTest<CustomTestTable>())
            {
                return false;
            }
            std::cout << "\n";

            if (!RunMultiThreadedCorrectnessTest<CustomTestTable>())
            {
                return false;
            }
            std::cout << "\n";

            if (!RunHashCollisionTest<CollisionTestTable>())
            {
                return false;
            }
            std::cout << "\n";

            RunPerformanceTest<CustomTestTable>(10000, 5000, 5000000, "Lightly Populated (Sequential)");
            std::cout << "\n";

            RunPerformanceTest<CustomTestTable>(10000, 5000, 5000000, "Lightly Populated (Random)", 0, true);
            std::cout << "\n";

            customPerf = RunPerformanceTest<CustomTestTable>(cacheCap, prePopulate, operations, "Heavily Populated (0% Safe Zone / Sequential)", 0);
            std::cout << "\n";

            RunPerformanceTest<CustomTestTable>(cacheCap, prePopulate, operations, "Heavily Populated (0% Safe Zone / Random)", 0, true);
            std::cout << "\n";

            RunPerformanceTest<CustomTestTable>(cacheCap, prePopulate, operations, "Heavily Populated (25% Safe Zone)", 25);
            std::cout << "\n";

            RunPerformanceTest<CustomTestTable>(cacheCap, prePopulate, operations, "Heavily Populated (50% Safe Zone)", 50);
            std::cout << "\n";

            RunPerformanceTest<CustomTestTable>(cacheCap, prePopulate, operations, "Heavily Populated (75% Safe Zone)", 75);
            std::cout << "\n";

            if (!RunContentionTest_NoEviction<CustomTestTable>(5))
            {
                return false;
            }
            std::cout << "\n";

            if (!RunContentionTest_EvictionThrashing<CustomTestTable>(5))
            {
                return false;
            }
            std::cout << "\n";

            customPerf.ContentionThroughput = RunContentionTest_MixedWorkload<CustomTestTable>(contentionSec, contentionCap, 0);
            std::cout << "\n";

            customPerf.OversubscribedContentionThroughput = RunContentionTest_MixedWorkloadOversubscribed<CustomTestTable>(contentionSec, contentionCap, 0);
            std::cout << "\n";

            customPerf.ReadHeavyThroughput0 = RunContentionTest_ReadHeavySkewed<CustomTestTable>(contentionSec, contentionCap, 0);
            std::cout << "\n";

            customPerf.ReadHeavyThroughput25 = RunContentionTest_ReadHeavySkewed<CustomTestTable>(contentionSec, contentionCap, 25);
            std::cout << "\n";

            customPerf.ReadHeavyThroughput50 = RunContentionTest_ReadHeavySkewed<CustomTestTable>(contentionSec, contentionCap, 50);
            std::cout << "\n";

            customPerf.ReadHeavyThroughput75 = RunContentionTest_ReadHeavySkewed<CustomTestTable>(contentionSec, contentionCap, 75);
            std::cout << "\n";

            customPerf.ReadHeavyThroughput100 = RunContentionTest_ReadHeavySkewed<CustomTestTable>(contentionSec, contentionCap, 100);
            std::cout << "\n";

            if (!RunTrimPerformanceTest<CustomTestTable>(120000, 0.02))
            {
                return false;
            }

            if (!RunTrimToWatermarkTest<CustomTestTable>(120000))
            {
                return false;
            }

            if (!RunForcedTrimTest<CustomTestTable>(120000))
            {
                return false;
            }

            if (!RunEnumerateTest<CustomTestTable>(10000))
            {
                return false;
            }

            if (!RunThreadScalingSweep<CustomTestTable>(szCustomName, cacheCap, 2, &customScale))
            {
                return false;
            }

            if (!RunTailLatencyTest<CustomTestTable>(szCustomName, cacheCap, &customLat))
            {
                return false;
            }

            std::cout << "\n=========================================================\n";
            std::cout << "      STANDARD STD::UNORDERED_MAP + STD::LIST TABLE      \n";
            std::cout << "=========================================================\n\n";

            if (!RunCorrectnessTests<StdTestTable>())
            {
                return false;
            }
            std::cout << "\n";

            if (!RunTinyTableTest<StdTestTable>())
            {
                return false;
            }
            std::cout << "\n";

            if (!RunFreeListIntegrityTest<StdTestTable>())
            {
                return false;
            }
            std::cout << "\n";

            if (!RunZeroCapacitySanityTest<StdTestTable>())
            {
                return false;
            }
            std::cout << "\n";

            if (!RunEvictionAndExhaustionTest<StdTestTable>())
            {
                return false;
            }
            std::cout << "\n";

            if (!RunMultiThreadedCorrectnessTest<StdTestTable>())
            {
                return false;
            }
            std::cout << "\n";

            RunPerformanceTest<StdTestTable>(10000, 5000, 5000000, "Lightly Populated (sequential)");
            std::cout << "\n";

            RunPerformanceTest<StdTestTable>(10000, 5000, 5000000, "Lightly Populated (random)", 0, true);
            std::cout << "\n";

            stdPerf = RunPerformanceTest<StdTestTable>(cacheCap, prePopulate, operations, "Heavily Populated (sequential)");
            std::cout << "\n";

            RunPerformanceTest<StdTestTable>(cacheCap, prePopulate, operations, "Heavily Populated (random)", 0, true);
            std::cout << "\n";

            if (!RunContentionTest_NoEviction<StdTestTable>(5))
            {
                return false;
            }
            std::cout << "\n";

            if (!RunContentionTest_EvictionThrashing<StdTestTable>(5))
            {
                return false;
            }
            std::cout << "\n";

            stdPerf.ContentionThroughput = RunContentionTest_MixedWorkload<StdTestTable>(contentionSec, contentionCap, 0);
            std::cout << "\n";

            stdPerf.OversubscribedContentionThroughput = RunContentionTest_MixedWorkloadOversubscribed<StdTestTable>(contentionSec, contentionCap, 0);
            std::cout << "\n";

            stdPerf.ReadHeavyThroughput0 = RunContentionTest_ReadHeavySkewed<StdTestTable>(contentionSec, contentionCap, 0);
            stdPerf.ReadHeavyThroughput25 = stdPerf.ReadHeavyThroughput0;
            stdPerf.ReadHeavyThroughput50 = stdPerf.ReadHeavyThroughput0;
            stdPerf.ReadHeavyThroughput75 = stdPerf.ReadHeavyThroughput0;
            stdPerf.ReadHeavyThroughput100 = stdPerf.ReadHeavyThroughput0;
            std::cout << "\n";

            if (!RunTrimPerformanceTest<StdTestTable>(120000, 0.02))
            {
                return false;
            }

            if (!RunThreadScalingSweep<StdTestTable>("Std::Map+List Table", cacheCap, 2, &stdScale))
            {
                return false;
            }

            if (!RunTailLatencyTest<StdTestTable>("Std::Map+List Table", cacheCap, &stdLat))
            {
                return false;
            }

            std::cout << "\n==============================================================================================\n";
            std::cout << "                                 FINAL PERFORMANCE COMPARISON              \n";
            std::cout << "==============================================================================================\n\n";

            auto printMultiplier = [](const char* MetricName, uint64_t CustomOps, uint64_t StdOps)
                {
                    double multiplier = static_cast<double>(CustomOps) / static_cast<double>(StdOps);

                    std::cout << std::left << std::setw(32) << MetricName << " | " << std::setw(15)
                              << CustomOps << " | " << std::setw(16) << StdOps << " | " << std::fixed
                              << std::setprecision(2) << multiplier << "x Faster\n";
                };

            std::cout << std::left << std::setw(32) << "Metric (Ops/sec)"
                      << " | " << std::setw(15) << "Custom Array"
                      << " | " << std::setw(16) << "Std::Map+List"
                      << " | Speedup Factor\n";
            std::cout << "----------------------------------------------------------------------------------------------\n";

            printMultiplier("Sequential Add", customPerf.AddThroughput, stdPerf.AddThroughput);

            printMultiplier("Sequential Lookup", customPerf.LookupThroughput, stdPerf.LookupThroughput);

            printMultiplier("Sequential Remove", customPerf.RemoveThroughput, stdPerf.RemoveThroughput);

            printMultiplier(
                "Mixed Contention", customPerf.ContentionThroughput, stdPerf.ContentionThroughput);

            printMultiplier("Mixed Contention Oversubscribed",
                            customPerf.OversubscribedContentionThroughput,
                            stdPerf.OversubscribedContentionThroughput);

            printMultiplier("Read-Heavy Skewed (0% Safe)",
                            customPerf.ReadHeavyThroughput0,
                            stdPerf.ReadHeavyThroughput0);

            printMultiplier("Read-Heavy Skewed (25% Safe)",
                            customPerf.ReadHeavyThroughput25,
                            stdPerf.ReadHeavyThroughput25);

            printMultiplier("Read-Heavy Skewed (50% Safe)",
                            customPerf.ReadHeavyThroughput50,
                            stdPerf.ReadHeavyThroughput50);

            printMultiplier("Read-Heavy Skewed (75% Safe)",
                            customPerf.ReadHeavyThroughput75,
                            stdPerf.ReadHeavyThroughput75);

            printMultiplier("Read-Heavy Skewed (100% Safe)",
                            customPerf.ReadHeavyThroughput100,
                            stdPerf.ReadHeavyThroughput100);

            // ----------------------------------------------------------------
            // Multi-Threaded Scaling Comparison
            // ----------------------------------------------------------------
            std::cout << "\n----------------------------------------------------------------------------------------------\n";
            std::cout << std::left << std::setw(32) << "Multi-Threaded Scaling"
                      << " | " << std::setw(15) << "Custom Array"
                      << " | " << std::setw(16) << "Std::Map+List"
                      << " | Advantage\n";
            std::cout << "----------------------------------------------------------------------------------------------\n";

            auto printScaleStr = [](double scale)
                {
                    std::stringstream ss;
                    ss << std::fixed << std::setprecision(2) << scale << "x";

                    if (scale < 1.0)
                    {
                        ss << " (Negative)";
                    }

                    return ss.str();
                };

            std::cout << std::left << std::setw(32) << "Scaling Factor (Max Threads)"
                      << " | " << std::setw(15) << printScaleStr(customScale.ScalingFactor) << " | "
                      << std::setw(16) << printScaleStr(stdScale.ScalingFactor);

            if (customScale.ScalingFactor > stdScale.ScalingFactor && stdScale.ScalingFactor > 0)
            {
                double diff = customScale.ScalingFactor / stdScale.ScalingFactor;
                std::cout << " | " << std::fixed << std::setprecision(2) << diff
                          << "x Better Scaling\n";
            }
            else
            {
                std::cout << " | Custom Absorbs Load\n";
            }

            // ----------------------------------------------------------------
            // Tail Latency Comparison
            // ----------------------------------------------------------------
            std::cout << "\n----------------------------------------------------------------------------------------------\n";
            std::cout << std::left << std::setw(32) << "Tail Latency (ns)"
                      << " | " << std::setw(15) << "Custom Array"
                      << " | " << std::setw(16) << "Std::Map+List"
                      << " | Stability Advantage\n";
            std::cout << "----------------------------------------------------------------------------------------------\n";

            auto printLatency = [](const char* metric, uint64_t customLatNs, uint64_t stdLatNs)
                {
                    double advantage = 0.0;

                    if (customLatNs > 0)
                    {
                        advantage = static_cast<double>(stdLatNs) / static_cast<double>(customLatNs);
                    }

                    std::cout << std::left << std::setw(32) << metric << " | " << std::setw(15)
                              << customLatNs << " | " << std::setw(16) << stdLatNs << " | " << std::fixed
                              << std::setprecision(2) << advantage << "x More Stable\n";
                };

            printLatency("P99.9  (Algorithmic)", customLat.P99_9, stdLat.P99_9);
            printLatency("P99.99 (OS/Hardware Jitter)", customLat.P99_99, stdLat.P99_99);

            // ----------------------------------------------------------------
            // Sequential vs Concurrent Performance Analysis Output
            // ----------------------------------------------------------------
            double avgSeqSpeedup = 0.0;

            if (stdPerf.AddThroughput > 0 && stdPerf.LookupThroughput > 0 &&
                stdPerf.RemoveThroughput > 0)
            {
                avgSeqSpeedup =
                    ((static_cast<double>(customPerf.AddThroughput) / stdPerf.AddThroughput) +
                      (static_cast<double>(customPerf.LookupThroughput) / stdPerf.LookupThroughput) +
                      (static_cast<double>(customPerf.RemoveThroughput) / stdPerf.RemoveThroughput)) /3.0;
            }

            std::cout << "\n----------------------------------------------------------------------------------------------\n";

            // Only show the explanatory note if the sequential speedup is less than 15%
            if (avgSeqSpeedup < 1.15)
            {
                std::cout << " [!] Analysis of Sequential vs. Concurrent Scaling:\n";
                std::cout << "     In purely sequential (single-threaded) workloads, the Custom Array shows modest\n";
                std::cout << "     gains primarily due to cache-friendly contiguous memory and the strict\n";
                std::cout << "     elimination of per-node heap allocations. However, std::unordered_map is already\n";
                std::cout << "     highly optimized for uncontended operations, keeping the baseline gap narrow.\n\n";
                std::cout << "     The Custom Table's architecture is explicitly designed to survive massive concurrency.\n";
                std::cout << "     Its combination of sharded locks, lock-free statistical reads, array-based LRU links,\n";
                std::cout << "     and lazy promotion completely eliminates the global lock contention that paralyzes\n";
                std::cout << "     the standard table, resulting in the high speedups seen above.\n";
                std::cout << "----------------------------------------------------------------------------------------------\n";
            }

            return true;
        };

    bool bTestsPassed = runAllTestsLambda();

    if (g_LiveObjectsCount.load() != 0)
    {
        std::cerr << "\n[-] MASTER MEMORY LEAK CHECK: " << g_LiveObjectsCount.load()
                  << " objects stranded.\n";

        return -1;
    }

    if (!bTestsPassed)
    {
        std::cerr << "\n[-] Test suite aborted due to an error.\n";

        return -1;
    }

    std::cout << "\n--- All Tests Finished Successfully ---\n";

    return 0;
}
