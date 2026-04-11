/*
* Apache LRU Hash Table Sample
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
// This sample demonstrates usage and performance characteristics of the 
// LruHashTable with string-based keys.
//
// CORRECTNESS & USAGE GUIDE:
// A structured, step-by-step walkthrough demonstrating practical usage scenarios
// of the cache. It covers Small String Optimization (SSO) keys, handling of
// larger heap-allocated strings, and zero-allocation lookups using borrowed
// string views (e.g., network inputs). The guide also illustrates collision
// resolution strategies (KeepIfExists vs. ReplaceIfExists) and demonstrates
// safe key removal with validation.
//
// HIGH-PERFORMANCE SCALING SUITE:
// A multithreaded benchmarking harness (built with std::jthread and std::latch)
// designed to measure aggregate throughput in millions of operations per second.
// It evaluates scaling behavior from a single thread up to oversubscribed core
// counts across multiple workloads—Add, Lookup, Remove, and an 80/20 mixed
// read/write pattern. Benchmarks are executed using both pure SSO datasets and
// mixed-length key distributions to reflect realistic usage conditions.
// ----------------------------------------------------------------------------

#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <random>
#include <algorithm>
#include <format>
#include <iostream>
#include <string>
#include <latch>

#ifndef _WIN32
#include <sys/time.h>
#include <sys/resource.h>
#endif

#include "lru_hash_table.h"
#include "lru_string_key.h"

// ------------------------------------------------------------------------
// SAMPLE PAYLOAD
// ------------------------------------------------------------------------
class alignas(64) SessionContext
{
private:
    std::atomic<int> m_refs{ 1 };

protected:
    virtual ~SessionContext() noexcept = default;

public:
    uint64_t AccountId;
    int      AccessLevel;
    bool     IsActive;

    SessionContext(uint64_t accId,
                   int      accessLvl) noexcept : AccountId(accId),
                                                  AccessLevel(accessLvl),
                                                  IsActive(true)
    {}    

    void AddRef() noexcept
    {
        m_refs.fetch_add(1, std::memory_order_relaxed);
    }

    void Release() noexcept
    {
        if (m_refs.fetch_sub(1, std::memory_order_release) == 1)
        {
            std::atomic_thread_fence(std::memory_order_acquire);
            delete this;
        }
    }
};

// ------------------------------------------------------------------------
//  HASH TABLE TYPE DEFINITION
// ------------------------------------------------------------------------
using HashTable = LruHashTable<CustomStringKey,
                               SessionContext,
                               FastStringHasher,
                               DefaultNumaAllocator,
                               AdaptiveSpinPolicy>;

// ------------------------------------------------------------------------
// CORRECTNESS & TUTORIAL SAMPLE
// ------------------------------------------------------------------------
void correctness_sample()
{
    std::cout << "\n==========================================================================\n";
    std::cout << " LRU HASH TABLE STRING KEY: CORRECTNESS & USAGE GUIDE   \n";
    std::cout << "==========================================================================\n";

    HashTable cache;
    if (!cache.Initialize(1000))
    {
        std::cerr << "[ERROR] Cache initialization failed.\n";
        return;
    }

    std::cout << "[INIT] Cache initialized successfully.\n\n";

    // ------------------------------------------------------------------------
    // SCENARIO 1: Short Strings (Small String Optimization - SSO)
    // ------------------------------------------------------------------------
    std::cout << "--- 1. SSO Keys (<= 22 characters) ---\n";
    std::string shortStr = "user_12345";
    CustomStringKey ssoKey(shortStr);

    SessionContext* session1 = new SessionContext(101, 1);
    if (cache.Add(ssoKey, session1))
    {
        std::cout << "  [+] Added SSO Key: '" << shortStr << "'\n";
    }
    else
    {
        std::cout << "  [x] Failed to add SSO Key.\n";
        session1->Release();
        return;
    }
    session1->Release(); // Hash table now owns the reference

    SessionContext* foundSession = nullptr;
    if (cache.Lookup(ssoKey, foundSession))
    {
        std::cout << "  [v] Lookup Successful! Account ID: " << foundSession->AccountId << "\n";
        foundSession->Release();
    }
    else
    {
        std::cout << "  [x] Lookup Failed.\n";
        return;
    }

    // ------------------------------------------------------------------------
    // SCENARIO 2: Long Strings (Heap Allocated)
    // ------------------------------------------------------------------------
    std::cout << "\n--- 2. Long Keys (Heap Allocated, > 22 characters) ---\n";
    std::string longStr = "session_token_xyz_987654321_extended_auth_string";
    CustomStringKey heapKey(longStr);

    SessionContext* session2 = new SessionContext(202, 2);
    if (cache.Add(heapKey, session2))
    {
        std::cout << "  [+] Added Long Key: '" << longStr << "'\n";
    }
    else
    {
        std::cout << "  [x] Failed to add Long Key.\n";
        session2->Release();
        return;
    }
    session2->Release();

    if (cache.Lookup(heapKey, foundSession))
    {
        std::cout << "  [v] Lookup Successful! Account ID: " << foundSession->AccountId << "\n";
        foundSession->Release();
    }
    else
    {
        std::cout << "  [x] Lookup Failed.\n";
        return;
    }

    // ------------------------------------------------------------------------
    // SCENARIO 3: Zero-Allocation Lookup (Borrowing)
    // ------------------------------------------------------------------------
    std::cout << "\n--- 3. Zero-Allocation Lookup (Borrowed View) ---\n";
    // We simulate receiving a network request string
    std::string networkRequest = "session_token_xyz_987654321_extended_auth_string";

    // We use the Borrow tag to prevent memory allocation during the lookup
    CustomStringKey borrowKey(networkRequest, CustomStringKey::Borrow{});
    if (cache.Lookup(borrowKey, foundSession))
    {
        std::cout << "  [v] Borrowed Lookup Successful! Account ID: " << foundSession->AccountId << "\n";
        foundSession->Release();
    }
    else
    {
        std::cout << "  [x] Borrowed Lookup Failed.\n";
        return;
    }

    // ------------------------------------------------------------------------
    // SCENARIO 4: Collision & Overwrite Policies
    // ------------------------------------------------------------------------
    std::cout << "\n--- 4. Overwrite Policies (Keep vs Replace) ---\n";

    SessionContext* session3 = new SessionContext(999, 5); // Different Account ID

    // Default Behavior: KeepIfExists (Get-Or-Add)
    std::cout << "  Attempting to Add existing key with 'KeepIfExists'...\n";
    SessionContext* existingSession = nullptr;
    if (!cache.Add(ssoKey, session3, &existingSession, AddAction::KeepIfExists))
    {
        std::cout << "  [-] Add returned false (Key exists). Existing value protected.\n";
    }

    if (existingSession != nullptr)
    {
        if (existingSession->AccountId == 101)
        {
            std::cout << "  [v] Verified out parameter captured existing Account ID: " << existingSession->AccountId << " (Expected: 101)\n";
        }
        else
        {
            std::cout << "  [x] Error: Account ID changed unexpectedly to: " << existingSession->AccountId << "\n";
            existingSession->Release();
            session3->Release();
            return;
        }
        existingSession->Release();
    }
    else
    {
        std::cout << "  [x] Error: Out parameter was not populated during KeepIfExists collision.\n";
        session3->Release();
        return;
    }

    // Because the API is _In_, the table did NOT consume our failed insertion payload.
    // We retain ownership and must cleanly release it ourselves.
    session3->Release();

    // Explicit Behavior: ReplaceIfExists (Upsert)
    std::cout << "  Attempting to Add existing key with 'ReplaceIfExists'...\n";
    SessionContext* session4 = new SessionContext(999, 5);
    SessionContext* replacedSession = nullptr;

    if (cache.Add(ssoKey, session4, &replacedSession, AddAction::ReplaceIfExists))
    {
        std::cout << "  [+] Add returned true. Value replaced.\n";
    }
    else
    {
        std::cout << "  [x] Error: ReplaceIfExists failed to return true.\n";
        session4->Release();
        if (replacedSession) replacedSession->Release();
        return;
    }
    session4->Release();

    if (replacedSession != nullptr)
    {
        if (replacedSession->AccountId == 101)
        {
            std::cout << "  [v] Verified out parameter captured the REPLACED Account ID: " << replacedSession->AccountId << " (Expected: 101)\n";
        }
        else
        {
            std::cout << "  [x] Error: Replaced Account ID is incorrect: " << replacedSession->AccountId << "\n";
            replacedSession->Release();
            return;
        }
        replacedSession->Release();
    }
    else
    {
        std::cout << "  [x] Error: Out parameter was not populated during ReplaceIfExists.\n";
        return;
    }

    if (cache.Lookup(ssoKey, foundSession))
    {
        if (foundSession->AccountId == 999)
        {
            std::cout << "  [v] Verified Account ID in cache updated to: " << foundSession->AccountId << " (Expected: 999)\n";
        }
        else
        {
            std::cout << "  [x] Error: Account ID not updated, remains: " << foundSession->AccountId << "\n";
            foundSession->Release();
            return;
        }
        foundSession->Release();
    }
    else
    {
        std::cout << "  [x] Error: Key lost after ReplaceIfExists attempt.\n";
        return;
    }

    // ------------------------------------------------------------------------
    // SCENARIO 5: Removal
    // ------------------------------------------------------------------------
    std::cout << "\n--- 5. Key Removal ---\n";
    if (cache.Remove(ssoKey))
    {
        std::cout << "  [-] Key removed successfully.\n";
    }
    else
    {
        std::cout << "  [x] Error: Failed to remove key.\n";
        return;
    }

    if (!cache.Lookup(ssoKey, foundSession))
    {
        std::cout << "  [v] Verified key is no longer in cache.\n";
    }
    else
    {
        std::cout << "  [x] Error: Key still found after removal!\n";
        foundSession->Release();
        return;
    }

    std::cout << "==========================================================================\n";
    std::cout << " CORRECTNESS TESTS PASSED\n";
    std::cout << "==========================================================================\n\n";
}

// ------------------------------------------------------------------------
// BENCHMARK CONFIGURATION
// ------------------------------------------------------------------------
constexpr size_t UNIQUE_KEYS      = 100'000;
constexpr size_t CACHE_CAPACITY   = 1'000'000;
constexpr size_t TOTAL_OPERATIONS = 100'000'000;

enum class TestMode
{
    AddOnly,
    LookupOnly,
    RemoveOnly,
    Mixed80_20
};

struct BenchmarkResult
{
    double TimeSeconds;
    double OpsPerSecond;
};

inline void SetHighPriority()
{
#ifdef _WIN32
    SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS);
#else
    setpriority(PRIO_PROCESS, 0, -10);
#endif
}

// ------------------------------------------------------------------------
// BENCHMARK WORKER
// ------------------------------------------------------------------------
void BenchmarkWorker(HashTable&                          cache,
                     const std::vector<CustomStringKey>& localKeys,
                     std::latch&                         workersReady,
                     std::latch&                         startSignal,
                     size_t                              opsPerThread,
                     TestMode                            mode,
                     uint32_t                            threadId)
{
    SessionContext* dummySession = new SessionContext(threadId, 1);

    workersReady.count_down();
    startSignal.wait();

    size_t keyCount = localKeys.size();
    size_t keyIndex = 0;

    for (size_t i = 0; i < opsPerThread; ++i)
    {
        const CustomStringKey& tokenKey = localKeys[keyIndex++];

        if (keyIndex >= keyCount) [[unlikely]]
        {
            keyIndex = 0;
        }

        if (mode == TestMode::AddOnly)
        {
            // Pass nullptr for the OutExistingValue parameter
            if (cache.Add(tokenKey, dummySession, nullptr, AddAction::ReplaceIfExists))
            {
                // Discard result
            }
        }
        else if (mode == TestMode::LookupOnly)
        {
            SessionContext* foundSession = nullptr;
            if (cache.Lookup(tokenKey, foundSession))
            {
                foundSession->Release();
            }
        }
        else if (mode == TestMode::RemoveOnly)
        {
            if (cache.Remove(tokenKey))
            {
                // Discard result
            }
        }
        else if (mode == TestMode::Mixed80_20)
        {
            if ((i % 10) < 8)
            {
                SessionContext* foundSession = nullptr;
                if (cache.Lookup(tokenKey, foundSession))
                {
                    foundSession->Release();
                }
            }
            else
            {
                // Pass nullptr for the OutExistingValue parameter
                if (cache.Add(tokenKey, dummySession, nullptr, AddAction::ReplaceIfExists))
                {
                    // Discard result
                }
            }
        }
    }

    dummySession->Release();
}

// ------------------------------------------------------------------------
// BENCHMARK ORCHESTRATOR
// ------------------------------------------------------------------------
BenchmarkResult RunBenchmark(uint32_t                            threadCount,
                             const std::vector<CustomStringKey>& masterKeys,
                             TestMode                            mode)
{
    HashTable cache;
    if (!cache.Initialize(CACHE_CAPACITY, 25))
    {
        std::cerr << "Cache initialization failed.\n";
        return { 0.0, 0.0 };
    }

    size_t opsPerThread = TOTAL_OPERATIONS / threadCount;

    if (mode == TestMode::RemoveOnly)
    {
        opsPerThread = masterKeys.size() / threadCount;
    }

    if (mode == TestMode::LookupOnly || mode == TestMode::RemoveOnly || mode == TestMode::Mixed80_20)
    {
        for (const auto& key : masterKeys)
        {
            SessionContext* prepopSession = new SessionContext(8080, 1);
            if (cache.Add(key, prepopSession))
            {
                // Discard result
            }
            prepopSession->Release();
        }
    }

    std::vector<std::vector<CustomStringKey>> threadLocalKeys(threadCount);

    for (uint32_t i = 0; i < threadCount; ++i)
    {
        threadLocalKeys[i] = masterKeys;

        std::mt19937 rng(1337 + i);
        std::shuffle(threadLocalKeys[i].begin(), threadLocalKeys[i].end(), rng);
    }

    std::vector<std::jthread> threads;
    threads.reserve(threadCount);

    std::latch workersReady(threadCount);
    std::latch startSignal(1);

    for (uint32_t i = 0; i < threadCount; ++i)
    {
        threads.emplace_back(BenchmarkWorker,
                             std::ref(cache),
                             std::ref(threadLocalKeys[i]),
                             std::ref(workersReady),
                             std::ref(startSignal),
                             opsPerThread,
                             mode,
                             i);
    }

    workersReady.wait();

    auto startTime = std::chrono::high_resolution_clock::now();
    startSignal.count_down();

    // std::jthread automatically joins upon destruction, 
    // but we need to block here to calculate the elapsed time correctly.
    for (auto& t : threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;

    size_t totalPerformedOps = opsPerThread * threadCount;
    double opsPerSecond = totalPerformedOps / elapsed.count();

    return { elapsed.count(), opsPerSecond };
}

void RunScalingSuite(uint32_t                            hwCores, 
                     const std::vector<CustomStringKey>& masterKeys, 
                     TestMode                            mode, 
                     const std::string&                  label)
{
    std::cout << "--- SCALING BENCHMARK: " << label << " ---\n";

    std::cout << std::format("{:<12}{:<15}{:<25}\n", "Threads", "Time (s)", "Throughput (M Ops/sec)");
    std::cout << std::string(52, '-') << "\n";

    auto PrintRow = [&](uint32_t threads)
        {
            BenchmarkResult res = RunBenchmark(threads, masterKeys, mode);
            std::cout << std::format("{:<12}{:<15.3f}{:<25.2f}\n",
                         threads,
                         res.TimeSeconds,
                         (res.OpsPerSecond / 1'000'000.0));
        };

    PrintRow(1);

    uint32_t halfCores = hwCores / 2;
    if (halfCores > 1)
    {
        PrintRow(halfCores);
    }

    PrintRow(hwCores);
    PrintRow(hwCores * 2);
    PrintRow(hwCores * 3);

    std::cout << "\n";
}

void string_key_benchmark()
{
    uint32_t hwCores = std::thread::hardware_concurrency();
    if (hwCores == 0)
    {
        hwCores = 4;
    }

    std::cout << "Detected Hardware Threads: " << hwCores << "\n\n";

    // ------------------------------------------------------------------------
    // DATASET 1: 100% SSO Keys
    // ------------------------------------------------------------------------
    std::vector<CustomStringKey> ssoKeys;
    ssoKeys.reserve(UNIQUE_KEYS);

    for (size_t i = 0; i < UNIQUE_KEYS; ++i)
    {
        ssoKeys.emplace_back("tkn_" + std::to_string(i));
    }

    std::cout << "====================================================\n";
    std::cout << " [TEST 1] PURE SSO KEYS (100% < 22 chars)\n";
    std::cout << "====================================================\n";

    RunScalingSuite(hwCores, ssoKeys, TestMode::AddOnly, "ADD (Hot Loop Isolation)");
    RunScalingSuite(hwCores, ssoKeys, TestMode::LookupOnly, "LOOKUP (MRU Promotion)");
    RunScalingSuite(hwCores, ssoKeys, TestMode::Mixed80_20, "MIXED (80% Read / 20% Add)");
    RunScalingSuite(hwCores, ssoKeys, TestMode::RemoveOnly, "REMOVE (Complete Unlink)");

    // ------------------------------------------------------------------------
    // DATASET 2: 50% SSO / 50% Long Keys (Heap)
    // ------------------------------------------------------------------------
    std::vector<CustomStringKey> mixedKeys;
    mixedKeys.reserve(UNIQUE_KEYS);

    for (size_t i = 0; i < UNIQUE_KEYS; ++i)
    {
        if (i % 2 == 0)
        {
            mixedKeys.emplace_back("tkn_" + std::to_string(i));
        }
        else
        {
            mixedKeys.emplace_back("tkn_session_extended_network_string_" + std::to_string(i));
        }
    }

    std::cout << "====================================================\n";
    std::cout << " [TEST 2] MIXED KEYS (50% SSO / 50% HEAP)\n";
    std::cout << "====================================================\n";

    RunScalingSuite(hwCores, mixedKeys, TestMode::AddOnly, "ADD (Hot Loop Isolation)");
    RunScalingSuite(hwCores, mixedKeys, TestMode::LookupOnly, "LOOKUP (MRU Promotion)");
    RunScalingSuite(hwCores, mixedKeys, TestMode::Mixed80_20, "MIXED (80% Read / 20% Add)");
    RunScalingSuite(hwCores, mixedKeys, TestMode::RemoveOnly, "REMOVE (Complete Unlink)");
}

int main()
{
    // 1. Run Correctness & Tutorial Code
    correctness_sample();

    SetHighPriority();

    // 2. Run High-Performance Scaling Suite
    string_key_benchmark();

    return 0;
}