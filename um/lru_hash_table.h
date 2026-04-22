/*
* Apache LRU Hash Table
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

#pragma once

#include <cstdint>
#include <cstring>
#include <new>
#include <thread>
#include <mutex>
#include <vector>
#include <atomic>

#if defined(__linux__)
#include <time.h>
#endif

// ----------------------------------------------------------------------------
// SAL Annotation Fallbacks for Cross-Platform Compilation
// ----------------------------------------------------------------------------
#ifndef _In_
#define _In_
#endif
#ifndef _Out_
#define _Out_
#endif
#ifndef _Inout_
#define _Inout_
#endif
#ifndef _In_opt_
#define _In_opt_
#endif
#ifndef _Out_opt_ 
#define _Out_opt_
#endif

// ----------------------------------------------------------------------------
// Cross-Platform NUMA & Memory Management Helpers
// ----------------------------------------------------------------------------
#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <unistd.h>
    // NOTE: On Linux, you must link with -lnuma (e.g., g++ ... -lnuma)
#include <numa.h>
#include <numaif.h>
#elif defined(__APPLE__) || defined(__unix__)
#include <unistd.h>
#else
#include <stdlib.h>
#endif

// ----------------------------------------------------------------------------
// Intrinsic Headers
// ----------------------------------------------------------------------------
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#if defined(__i386__) || defined(__x86_64__)
#include <x86intrin.h> 
#endif
#endif

// ----------------------------------------------------------------------------
// Cache Line Constants
//
// We explicitly avoid std::hardware_destructive_interference_size due to ABI 
// instability warnings ([-Winterference-size]) across different compiler flags.
// 
// Apply 128-byte alignment strictly for Apple Silicon to prevent false sharing,
// while preserving optimal 64-byte density for x86_64 and standard Linux ARM64.
// ----------------------------------------------------------------------------
#if defined(__APPLE__) && defined(__aarch64__)
    constexpr size_t CACHE_LINE_SIZE = 128;
#else        
    constexpr size_t CACHE_LINE_SIZE = 64;
#endif

constexpr size_t CACHE_LINE_MASK = CACHE_LINE_SIZE - 1;

// ----------------------------------------------------------------------------
// Cross-Platform Hardware Pause Macro
// ----------------------------------------------------------------------------
// Why we pause:
// In a tight spin loop, a thread repeatedly polls a shared variable,
// consuming execution resources and potentially causing contention
// (especially on SMT/Hyper-Threaded cores).
//
// The CPU pause/yield instruction is a hint that the thread is in a
// spin-wait loop. It:
// - Inserts a short delay
// - Reduces power usage and contention
// - Improves performance of sibling threads on the same core
//
// It does NOT yield the thread's timeslice to the OS and does NOT act
// as a memory barrier.
// ----------------------------------------------------------------------------
#if defined(_MSC_VER)
#if defined(_M_ARM64) || defined(_M_ARM)
#define CPU_PAUSE() __yield()
#else
#define CPU_PAUSE() _mm_pause()
#endif
#elif defined(__GNUC__) || defined(__clang__)
#if defined(__i386__) || defined(__x86_64__)
#define CPU_PAUSE() __builtin_ia32_pause()
#elif defined(__aarch64__) || defined(__arm__)
#define CPU_PAUSE() asm volatile("yield" ::: "memory")
#else
#define CPU_PAUSE() ((void)0)
#endif
#else
#define CPU_PAUSE() ((void)0)
#endif

#if defined(_MSC_VER) || defined(_WIN32)
#include <windows.h>
inline void OsYield() noexcept
{
    SwitchToThread();
}
#elif defined(__linux__) || defined(__APPLE__)
#include <sched.h>
inline void OsYield() noexcept
{
    sched_yield();
}
#else
inline void OsYield() noexcept
{
    std::this_thread::yield();
}
#endif

// ============================================================================
// SPIN WAIT POLICIES
// These policies are injected at compile-time via C++ templates. 
// This guarantees zero runtime overhead (no virtual function calls or branch 
// prediction penalties) inside the critical nanosecond spin-loops.
// ============================================================================

// ----------------------------------------------------------------------------
// Adaptive Spin Policy
// ----------------------------------------------------------------------------
// Architecture:
// A hybrid spin policy combining pure user-space spinning with OS-assisted backoff.
//
// The thread initially spins using CPU pause instructions to efficiently handle
// short lock hold times without incurring scheduler overhead.
//
// If the spin limit is exceeded, the policy assumes that continued spinning is
// no longer productive (e.g., due to contention or possible lock-holder preemption),
// and temporarily yields or sleeps to allow other threads to make progress.
//
// On Linux, nanosleep is used instead of sched_yield(), which may not provide
// effective backoff under the CFS scheduler. This also helps reduce excessive
// spin-wait overhead in virtualized environments.
//
// Best For:
// General-purpose server workloads, thread pools, and virtualized environments
// where a balance between low-latency spinning and starvation avoidance is needed.
// ----------------------------------------------------------------------------
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
            // SLOW PATH: Lock is severely contented.
#if defined(__linux__)
            // Linux hypervisor mitigation:
            // sched_yield() is largely ignored by the Linux CFS scheduler. 
            // In a virtualized environment, this causes Pause Loop Exiting (PLE) VM-Exit storms.
            // We force a hard deschedule via nanosleep to guarantee the hypervisor 
            // switches to the preempted vCPU that holds the lock.
            struct timespec ts = { 0, 500 }; // Sleep for 500 nanoseconds
            nanosleep(&ts, nullptr);
#else
            // Windows SwitchToThread() strictly relinquishes the time slice.
            OsYield();            
#endif
            // Reset the phase to resume pure spinning when we wake up.
            SpinPhase = 0;
        }
    }
};

// ----------------------------------------------------------------------------
// Exponential Backoff Spin Policy
// ----------------------------------------------------------------------------
// Architecture:
// An adaptive spin policy using exponential backoff to reduce contention.
// Instead of polling at a constant rate, threads progressively increase
// the number of pause instructions between retries (1, 2, 4, ... up to a cap),
// reducing synchronized contention and cache-coherency traffic.
//
// This improves throughput under high contention by desynchronizing threads,
// at the cost of slightly increased tail latency.
//
// Fallback / Starvation Protection:
// Tracks the total number of pause instructions executed. Once a threshold
// is exceeded, the policy assumes continued spinning is unproductive and
// temporarily deschedules the thread (e.g., nanosleep on Linux). This helps
// avoid wasted CPU time and improves forward progress, particularly in
// oversubscribed or virtualized environments.
//
// Best For:
// High-contention workloads such as servers and concurrent data structures,
// where throughput is preferred over strict fairness.
// ----------------------------------------------------------------------------
struct ExponentialBackoffPolicy
{
    // Cap the maximum consecutive pauses to a much lower threshold. 
    // This prevents the thread from being "blind" to a lock release for too long.    
    static constexpr uint32_t MAX_BACKOFF_PAUSES = 8;

    // Track TOTAL PAUSES rather than total iterations. This explicitly matches
    // the industry-standard 4000 total pause limit of the AdaptiveSpinPolicy,
    // guaranteeing we yield to the OS at the exact same time to prevent starvation.
    static constexpr uint32_t YIELD_THRESHOLD_PAUSES = 4000;

    static inline void SpinWait(_Inout_ uint64_t& SpinState) noexcept
    {
        // Unpack state
        // High 32 bits: Total pauses executed so far.
        // Low 32 bits: Current pause batch size.
        uint32_t totalPauses   = static_cast<uint32_t>(SpinState >> 32);
        uint32_t currentPauses = static_cast<uint32_t>(SpinState & 0xFFFFFFFF);

        if (totalPauses == 0)
        {
            currentPauses = 1;
        }

        if (totalPauses < YIELD_THRESHOLD_PAUSES)
        {
            // HARDWARE BACKOFF: Execute the current batch
            for (uint32_t i = 0; i < currentPauses; ++i)
            {
                CPU_PAUSE();
            }

            // Update our total pause counter
            totalPauses += currentPauses;

            // EXPONENTIAL GROWTH: Double for the next iteration, up to the cap
            uint32_t nextPauses = currentPauses << 1;
            if (nextPauses > MAX_BACKOFF_PAUSES)
            {
                nextPauses = MAX_BACKOFF_PAUSES;
            }

            // REPACK STATE
            SpinState = (static_cast<uint64_t>(totalPauses) << 32) | nextPauses;
        }
        else
        {
            // SLOW PATH: Lock is severely contended. Yield to the OS scheduler.
#if defined(__linux__)
            // Consider switching to user futex
            struct timespec ts = { 0, 500 };
            nanosleep(&ts, nullptr);
#else
            OsYield();
#endif
            // Reset state upon waking up
            SpinState = 0;
        }
    }
};

// ------------------------------------------------------------------------
// SpinLock
// Ultra-low latency, exclusive Test-and-Test-and-Set (TTAS) spinlock.
//
// Architecture:
// Designed specifically for microscopic critical sections. It implements 
// the TTAS pattern to mitigate MESI protocol bus floods ("Cache Line Bouncing") 
// on multi-socket / multi-core systems while the lock is held.
// 
// Instead of repeatedly executing expensive atomic exchange instructions, 
// waiting threads spin locally on a relaxed memory read. This keeps the 
// underlying cache line in a Shared (Read-Only) state within the L1 cache.
// Note: While this drastically reduces continuous interconnect traffic, 
// releasing the lock may still trigger a burst of Read-For-Ownership 
// requests from waiting threads.
// 
// The hardware back-off strategy (e.g., yielding to the OS vs. executing 
// CPU pause instructions) is decoupled via the TSpinPolicy template.
// ------------------------------------------------------------------------
template <typename TSpinPolicy>
class SpinLock
{
private:
    std::atomic<bool> m_IsLocked{ false };

public:
    inline void lock() noexcept
    {
        uint64_t spinPhase = 0;

        while (true)
        {
            // Spin on relaxed read (L1 cache)
            while (m_IsLocked.load(std::memory_order_relaxed))
            {
                TSpinPolicy::SpinWait(spinPhase);
            }

            // exchange() is extremely fast on x86/ARM for bools
            if (!m_IsLocked.exchange(true, std::memory_order_acquire))
            {
                return;
            }

            CPU_PAUSE();
        }
    }

    inline void unlock() noexcept
    {
        m_IsLocked.store(false, std::memory_order_release);
    }
};

// ----------------------------------------------------------------------------
// Default NUMA Allocator Policy
// Serves as the default memory provider for the LruHashTable. 
// Can be replaced by any class implementing this static interface.
// ----------------------------------------------------------------------------
struct DefaultNumaAllocator
{
    // Returns a const reference to avoid all heap allocations and copies
    static const std::vector<uint32_t>& GetValidNodes() noexcept
    {
        // Use static variable to be initialized exactly ONCE in a thread-safe manner
        static const std::vector<uint32_t> validNodes = []()
            {
                try
                {
                    std::vector<uint32_t> nodes;

#if defined(_WIN32)
                    ULONG highestNode = 0;
                    if (GetNumaHighestNodeNumber(&highestNode))
                    {
                        const USHORT maxNode = static_cast<USHORT>(highestNode);
                        for (USHORT i = 0; i <= maxNode; ++i)
                        {
                            ULONGLONG availableMemory = 0;
                            if (GetNumaAvailableMemoryNodeEx(i, &availableMemory))
                            {
                                nodes.push_back(i);
                            }
                        }
                    }
#elif defined(__linux__)
                    if (numa_available() >= 0)
                    {
                        int highestNode = numa_max_node();
                        for (int i = 0; i <= highestNode; ++i)
                        {
                            if (numa_bitmask_isbitset(numa_all_nodes_ptr, i))
                            {
                                nodes.push_back(i);
                            }
                        }
                    }
#endif
                    // Absolute fallback
                    if (nodes.empty())
                    {
                        nodes.push_back(0);
                    }

                    // Shrink to fit to minimize memory footprint of the cached vector
                    nodes.shrink_to_fit();
                    return nodes;
                }
                catch (const std::bad_alloc&)
                {
                    // Absolute fallback if vector heap allocation fails.
                    // We return a vector with just Node 0. 
                    // Note: Returning {0} technically requires a small allocation, 
                    // but if the system cannot fulfill a 4-byte allocation at this 
                    // stage, process termination due to out of memory is imminent anyway.
                    static const std::vector<uint32_t> fallback{ 0 };
                    return fallback;                    
                }
            }();

        return validNodes;
    }

    static inline void* Allocate(_In_ size_t   size,
                                 _In_ uint32_t node) noexcept
    {
#if defined(_WIN32)
        void* ptr = VirtualAllocExNuma(GetCurrentProcess(),
                                       NULL,
                                       size,
                                       MEM_RESERVE | MEM_COMMIT,
                                       PAGE_READWRITE,
                                       node);
        if (!ptr)
        {
            // Fallback to standard VirtualAlloc
            ptr = VirtualAlloc(NULL,
                               size,
                               MEM_RESERVE | MEM_COMMIT,
                               PAGE_READWRITE);
        }

        return ptr;
#elif defined(__linux__)
        if (numa_available() >= 0)
        {
            void* ptr = numa_alloc_onnode(size, node);
            if (!ptr)
            {
                // Fallback to local NUMA node allocation if the target node is OOM
                ptr = numa_alloc_local(size);
            }
            return ptr;
        }

        // If NUMA is completely unavailable, use standard aligned allocation
        return ::operator new[](size, std::align_val_t{ CACHE_LINE_SIZE }, std::nothrow);
#else
        // macOS / Unsupported platforms
        return ::operator new[](size, std::align_val_t{ CACHE_LINE_SIZE }, std::nothrow);
#endif
    }

    static inline void Free(_In_ void* ptr,
                            _In_ size_t size) noexcept
    {
        if (!ptr)
        {
            return;
        }

#if defined(_WIN32)
        // Size must be 0 when using MEM_RELEASE
        VirtualFree(ptr, 0, MEM_RELEASE);
#elif defined(__linux__)
        if (numa_available() >= 0)
        {
            // Since we only used numa_alloc_* in the Allocate block, 
            // this is 100% safe.
            numa_free(ptr, size);
        }
        else
        {
            ::operator delete[](ptr, std::align_val_t{ CACHE_LINE_SIZE });
        }
#else
        ::operator delete[](ptr, std::align_val_t{ CACHE_LINE_SIZE });
#endif
    }
};

// ----------------------------------------------------------------------------
// Cross-Platform Hardware Prefetch Macro (x86/x64 & ARM64)
// Automatically translates to PREFETCHT0 on Intel/AMD and PRFM on ARM.
// ----------------------------------------------------------------------------
#if defined(_MSC_VER)
#if defined(_M_ARM64) || defined(_M_ARM)
#include <intrin.h>
#define CACHE_PREFETCH(ptr) __prefetch(ptr)
#else
#include <xmmintrin.h>
#define CACHE_PREFETCH(ptr) _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0)
#endif
#elif defined(__GNUC__) || defined(__clang__)
    // Universal builtin for Linux (GCC) and macOS (Apple Clang)
#define CACHE_PREFETCH(ptr) __builtin_prefetch(ptr, 0, 3)
#else
#define CACHE_PREFETCH(ptr) ((void)0)
#endif

// ----------------------------------------------------------------------------
// Collision Resolution Policy
// ----------------------------------------------------------------------------
enum class AddAction
{
    KeepIfExists,    // Default: Get-Or-Add (addresses TOCTOU races)
    ReplaceIfExists  // Upsert (Explicit overwrite)
};

// ----------------------------------------------------------------------------
// High-Performance Array-Backed LRU Hash Table
//
// Architecture:
// - Sharded Partitioning: The table is divided into multiple independent shards 
//   based on logical processor count to minimize global lock contention.
// 
// - Flat Arrays (Mega-Block): Nodes and Buckets are pre-allocated in unified, 
//   contiguous arrays. Linked lists (LRU and Hash Collision chains) utilize 
//   32-bit array indices instead of 64-bit pointers. This halves the per-node 
//   memory footprint, drastically reduces heap/pool fragmentation, and 
//   maximizes L1/L2/TLB cache locality.
// 
// - Reference Counting: TValue MUST be a pointer to an object implementing 
//   AddRef() and Release().
// 
// - Out-of-Lock Destruction: To prevent deadlocks, objects are always released 
//   and destructed after the shard lock has been completely dropped.
// 
// - Yielding Trim: Background eviction gracefully drops the lock after every 
//   single object release to maintain foreground I/O responsiveness.
// 
// - Lazy LRU Promotion: Uses a Generation counter to probabilistically age 
//   items, skipping expensive exclusive lock upgrades for "hot" cache hits.
// 
// - Watermark Eviction: Trim uses ActiveCount to bypass healthy shards in O(1).
// 
// - NUMA Awareness: Shard memory is explicitly bound to specific NUMA nodes 
//   to evenly distribute memory controller load and maximize bus bandwidth.
// 
// Requirement:
// - TValue must implement thread-safe AddRef()/Release() semantics (e.g., 
//   atomic reference counting).
// ----------------------------------------------------------------------------
template <typename TKey, 
          typename TValue, 
          typename THasher, 
          typename TAllocator      = DefaultNumaAllocator,
          typename TSpinWaitPolicy = AdaptiveSpinPolicy>
class LruHashTable
{
    // Enforce that TValue destructor is noexcept
    static_assert(std::is_nothrow_copy_constructible<TKey>::value,
                  "TKey must be completely noexcept copy constructible to prevent node leaks during Add()");

    // Enforce that AddRef is strictly noexcept
    static_assert(noexcept(std::declval<TValue*>()->AddRef()),
                  "TValue::AddRef() MUST be declared noexcept to prevent state corruption during cache overwrites");

    // Enforce that Release is strictly noexcept
    static_assert(noexcept(std::declval<TValue*>()->Release()),
                  "TValue::Release() MUST be declared noexcept to prevent permanent capacity leaks during node eviction");

public:
    static constexpr uint32_t INVALID_INDEX = 0xFFFFFFFF;

    // ------------------------------------------------------------------------
    // Intrusive Hash/LRU Node
    // Contains both the collision chain pointers and the doubly-linked 
    // LRU list pointers using 32-bit array indices inside the Mega-Block.
    // ------------------------------------------------------------------------
    struct LruNode
    {
        // --------------------------------------------------------------------
        // 1. HOT PATH: Hash Traversal
        // Grouped exactly at Offset 0. The CPU fetches these together in one 
        // read. If the hash fails, it immediately uses HashNext to jump.
        // --------------------------------------------------------------------
        uint64_t Hash;         // Cached hash to avoid re-computation
        uint32_t HashNext;     // Index-based hash collision chain
        uint32_t LruPrev;      // Pulled up to eliminate the 4-byte padding hole

        // --------------------------------------------------------------------
        // 2. MATCH PATH: Key Verification
        // Kept as high as possible so it remains in the same dynamic cache 
        // line as the hash variables above.
        // --------------------------------------------------------------------
        TKey     Key;          // Hash table key (Starts exactly at 16-byte boundary)

        // --------------------------------------------------------------------
        // 3. COLDER PATH: Payload & Eviction Meta
        // Accessed only upon a definitive hash/key match or during eviction.
        // --------------------------------------------------------------------
        TValue* Value;         // Raw pointer to intrusive ref-counted object. 
                               // Design decision: Raw pointer is used instead of 
                               // std::shared_ptr to minimize footprint and improve 
                               // cache locality.
        uint64_t LastPromoted; // Tracks the "age" relative to Shard::Generation
        uint32_t LruNext;      // Index-based linked list (Less Recently Used)
    };

    // ------------------------------------------------------------------------
    // Cache Shard
    // Aligned to dynamic cache line size to prevent false sharing across CPU 
    // cache lines. If two shards share a CPU cache line, a lock acquisition 
    // on Shard A would inadvertently invalidate the cache line for a CPU 
    // accessing Shard B. Each shard independently manages its own capacity, 
    // locks, and LRU queue.
    // ------------------------------------------------------------------------    
    struct alignas(CACHE_LINE_SIZE) Shard
    {
        // CACHE LINE 0: The Lock. 
        // Spinning threads will blast this line, but it won't interfere 
        // with the lock-holder's data access.
        alignas(CACHE_LINE_SIZE) SpinLock<TSpinWaitPolicy> Lock;     // Lock for synchronizing shard access

        // Cache Line 1: Read-Heavy Hash Data
        // Exclusively owned by the thread that successfully acquires the lock
        alignas(CACHE_LINE_SIZE) LruNode* Nodes;                    // Flat array segment of pre-allocated nodes
        uint32_t* Buckets;                  // Array segment of hash bucket heads

        void* RawMemoryBlock;           // Base pointer for allocator deallocation
        size_t                AllocationSize;           // Exact size allocated on the node

        uint32_t              BucketMask;               // Bitwise mask used to route a hash to a specific bucket index efficiently
        uint32_t              Capacity;                 // Maximum number of active LruNodes this specific shard can hold

        // CACHE LINE 2: Write-Heavy LRU State
        // Mutated on every Add/Overwrite MRU promotion
        alignas(CACHE_LINE_SIZE) uint32_t  LruHead;                  // MRU pointer
        uint32_t              LruTail;                  // LRU (Eviction Target)
        uint32_t              FreeHead;                 // Unused node stack

        uint64_t              Generation;               // Incremented on every MRU push
        uint64_t              ThresholdAge;             // Precomputed promotion age limit

        // CACHE LINE 3: Active Count
        // Mutated independently, lock-free statistical reads
        alignas(CACHE_LINE_SIZE) std::atomic<uint32_t> ActiveCount;  // Tracks live items to enable O(1) capacity checks
    };    

private:
    Shard* m_Shards;              // Dynamically allocated array representing the sharded cache architecture
    uint32_t m_ShardCount;          // Total number of initialized shards. Guaranteed to be a power of 2
    uint32_t m_PromotionThreshold;  // Percentage threshold (0-100) determining how aggressively nodes are promoted to MRU

    // ----------------------------------------------------------------------------
    // GetSystemLogicalCoreCount
    // Truly portable hardware concurrency resolution
    // ----------------------------------------------------------------------------
    inline uint32_t GetSystemLogicalCoreCount() noexcept
    {
#if defined(_WIN32)
        // Windows: Use the >64 core aware API
        uint32_t ulCount = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
        if (ulCount > 0)
        {
            return ulCount;
        }
#elif defined(__linux__) || defined(__APPLE__) || defined(__unix__)
        // POSIX (Linux, macOS, FreeBSD): Query the online processor count
        long lCount = sysconf(_SC_NPROCESSORS_ONLN);
        if (lCount > 0)
        {
            return static_cast<uint32_t>(std::min<long>(lCount, UINT32_MAX));
        }
#endif

        // Absolute Fallback Path
        // If the OS APIs fail or we are on an unknown platform, try the standard 
        // library. If the standard library returns 0, assume 1 core to prevent 
        // divide-by-zero crashes in sizing algorithms.
        uint32_t ulFallback = std::thread::hardware_concurrency();
        return (ulFallback > 0) ? ulFallback : 1;
    }

    // ------------------------------------------------------------------------
    // MixHash (stripped-down version of SplitMix64)
    // Applies a final avalanche step to the user-provided hash. 
    // This forces entropy into the lower bits to prevent clustering when 
    // dealing with weak hash functions (like identity hashes or sequential IDs),
    // ensuring better distribution across power-of-2 shard masks.
    // For the hash table only lower bits are actually matter, thus second part
    // of SplitMix64 is omitted. This saves about 4-5 CPU cycles.
    // ------------------------------------------------------------------------
    static constexpr uint64_t MixHash(_In_ uint64_t hash) noexcept
    {
        uint64_t mixed = hash ^ (hash >> 30);
        mixed *= 0xbf58476d1ce4e5b9ULL;
        mixed ^= (mixed >> 27);

        return mixed;
    }
   
    // ------------------------------------------------------------------------
    // UnlinkLru (Internal)
    // Severs a node from the doubly-linked LRU list using 32-bit indices.
    // Caller must hold the shard lock exclusively.
    // ------------------------------------------------------------------------
    inline void UnlinkLru(_In_ Shard* ShardPtr,
                          _In_ uint32_t Index) noexcept
    {
        uint32_t prev = ShardPtr->Nodes[Index].LruPrev;
        uint32_t next = ShardPtr->Nodes[Index].LruNext;

        if (prev != INVALID_INDEX)
        {
            ShardPtr->Nodes[prev].LruNext = next;
        }
        else
        {
            ShardPtr->LruHead = next;
        }

        if (next != INVALID_INDEX)
        {
            ShardPtr->Nodes[next].LruPrev = prev;
        }
        else
        {
            ShardPtr->LruTail = prev;
        }
    }

    // ------------------------------------------------------------------------
    // PushMru (Internal)
    // Inserts a node at the absolute front (Head) of the LRU list, marking 
    // it as the Most Recently Used item, and updates its Generation stamp.
    // Caller must hold the shard lock exclusively.
    // ------------------------------------------------------------------------
    inline void PushMru(_In_ Shard* ShardPtr,
                        _In_ uint32_t Index) noexcept
    {
        ShardPtr->Generation++;
        ShardPtr->Nodes[Index].LastPromoted = ShardPtr->Generation;

        ShardPtr->Nodes[Index].LruPrev = INVALID_INDEX;
        ShardPtr->Nodes[Index].LruNext = ShardPtr->LruHead;

        if (ShardPtr->LruHead != INVALID_INDEX)
        {
            ShardPtr->Nodes[ShardPtr->LruHead].LruPrev = Index;
        }

        ShardPtr->LruHead = Index;

        // If the list was empty, this node is both the Head and the Tail
        if (ShardPtr->LruTail == INVALID_INDEX)
        {
            ShardPtr->LruTail = Index;
        }
    }

    // ------------------------------------------------------------------------
    // RemoveFromHashChain (Internal)
    // Removes a node from the singly-linked hash collision chain.
    // Caller must hold the shard lock exclusively.
    // ------------------------------------------------------------------------
    void RemoveFromHashChain(_In_ Shard* ShardPtr,
                             _In_ uint32_t Index) noexcept
    {        
        uint32_t bucketIdx = static_cast<uint32_t>(ShardPtr->Nodes[Index].Hash & ShardPtr->BucketMask);

        uint32_t curr = ShardPtr->Buckets[bucketIdx];
        uint32_t prev = INVALID_INDEX;

        while (curr != INVALID_INDEX)
        {
            if (curr == Index) [[likely]]
            {
                if (prev == INVALID_INDEX)
                {
                    ShardPtr->Buckets[bucketIdx] = ShardPtr->Nodes[curr].HashNext;
                }
                else
                {
                    ShardPtr->Nodes[prev].HashNext = ShardPtr->Nodes[curr].HashNext;
                }
                break;
            }

            prev = curr;
            curr = ShardPtr->Nodes[curr].HashNext;
        }
    }

public:
    LruHashTable() noexcept : m_Shards(nullptr),
                              m_ShardCount(0),
                              m_PromotionThreshold(0)
    {}

    ~LruHashTable() noexcept
    {
        Cleanup();
    }

    // ------------------------------------------------------------------------
    // Initialize
    // Calculates power-of-2 shard counts based on logical processors to 
    // prevent lock contention. Allocates the Mega-Block flat arrays
    // via the custom allocator and initializes the free list. Ensures absolute 
    // CACHE_LINE_SIZE alignment for buckets and precomputes thresholds.
    // ------------------------------------------------------------------------
    [[nodiscard]]
    bool Initialize(_In_ const size_t TotalEntries,
                    _In_ uint32_t     PromotionThreshold = 0) noexcept
    {
        Cleanup();

        // Prevent arithmetic overflow on massive capacity requests
        if (TotalEntries > (SIZE_MAX / sizeof(LruNode)))
        {
            return false;
        }

        // Clamp threshold to a valid 0-100 percentage range
        if (PromotionThreshold > 100)
        {
            PromotionThreshold = 100;
        }

        m_PromotionThreshold = PromotionThreshold;

        uint32_t numProcs = GetSystemLogicalCoreCount();
        if (numProcs == 0) [[unlikely]]
        {
            numProcs = 4;
        }

        // Aggressive dispersion for massive core counts
        uint32_t targetShards = numProcs * 32;

        // To prevent premature evictions due to imperfect hash distribution 
        // (hash clustering), a shard needs enough capacity to absorb variance.
        // 64 items is the minimum safe floor to absorb statistical clustering.
        uint32_t minItemsPerShard = 64;

        // Calculate the maximum number of shards we can support
        uint32_t maxShardsForCapacity = static_cast<uint32_t>(TotalEntries / minItemsPerShard);

        // Absolute fallback for tiny tables (e.g., TotalEntries < 64)
        // This forces tiny tables into a single shard, completely preventing 
        // hash-clustering evictions and allowing tests to pass
        if (maxShardsForCapacity == 0)
        {
            maxShardsForCapacity = 1;
        }

        // Clamp our hardware target to our capacity reality
        if (targetShards > maxShardsForCapacity)
        {
            targetShards = maxShardsForCapacity;
        }

        m_ShardCount = 1;

        // We round DOWN to the nearest power of 2 here 
        while ((m_ShardCount << 1) <= targetShards)
        {
            m_ShardCount <<= 1;
        }

        // Enforce the modern hardware ceiling
        if (m_ShardCount > 4096)
        {
            m_ShardCount = 4096;
        }

        // Get the valid NUMA nodes
        const std::vector<uint32_t>& validNumaNodes = TAllocator::GetValidNodes();
        uint32_t primaryNode = validNumaNodes.empty() ? 0 : validNumaNodes[0];

        // Any exception thrown by TAllocator (e.g., std::bad_alloc) will be 
        // caught, memory will be cleanly rolled back, and the function will 
        // return false without crashing the process
        try
        {
            // Allocate the Shards block on a first NUMA node
            m_Shards = static_cast<Shard*>(TAllocator::Allocate(sizeof(Shard) * m_ShardCount, primaryNode));
            if (!m_Shards) [[unlikely]]
            {
                return false;
            }

            // Explicitly construct the Shards via placement new and zero the 
            // RawMemoryBlock pointers immediately to make Cleanup() safe in the event of 
            // a partial allocation failure.
            for (uint32_t i = 0; i < m_ShardCount; ++i)
            {
                new (&m_Shards[i]) Shard();
                m_Shards[i].RawMemoryBlock = nullptr;
            }

            // Use ceiling division to prevent capacity loss from truncation
            uint32_t capacityPerShard = static_cast<uint32_t>((TotalEntries + m_ShardCount - 1) / m_ShardCount);

            // Absolute floor to prevent zero-capacity configuration bugs on microscopic tables
            if (capacityPerShard < 8)
            {
                capacityPerShard = 8;
            }

            uint32_t bucketCountPerShard = 1;

            // Prevent infinite loop on edge-case massive capacities
            while (bucketCountPerShard < capacityPerShard && bucketCountPerShard != 0x80000000)
            {
                bucketCountPerShard <<= 1;
            }

            // Calculate bucket size and pad it to the nearest dynamic cache boundary for relative alignment
            size_t bucketBytesPerShard = sizeof(uint32_t) * bucketCountPerShard;
            bucketBytesPerShard = (bucketBytesPerShard + CACHE_LINE_MASK) & ~(size_t)CACHE_LINE_MASK;

            size_t nodeBytesPerShard = sizeof(LruNode) * capacityPerShard;

            // We allocate Buckets and Nodes in a single block per shard, +CACHE_LINE_SIZE for manual alignment
            size_t allocationSizePerShard = bucketBytesPerShard + nodeBytesPerShard + CACHE_LINE_SIZE;

            // To eliminate "if (threshold == 0)" branches from the Lookup hot-path, 
            // we map the percentage to a absolute generation delta
            uint64_t precomputedThresholdAge;
            if (m_PromotionThreshold == 0)
            {
                // 0% Safe Zone (Strict LRU): Any age >= 0 will trigger promotion
                precomputedThresholdAge = 0;
            }
            else if (m_PromotionThreshold == 100)
            {
                // 100% Safe Zone (FIFO): Age will never be >= UINT64_MAX, so no promotion
                precomputedThresholdAge = UINT64_MAX;
            }
            else
            {
                // Standard Threshold: Calculate relative distance in the LRU queue
                precomputedThresholdAge = (static_cast<uint64_t>(capacityPerShard) * m_PromotionThreshold) / 100;
            }

            uint32_t numValidNodes = static_cast<uint32_t>(validNumaNodes.size());

            for (uint32_t i = 0; i < m_ShardCount; ++i)
            {
                uint32_t targetPhysicalNode = validNumaNodes[i % numValidNodes];

                void* rawBlock = TAllocator::Allocate(allocationSizePerShard, targetPhysicalNode);
                if (!rawBlock) [[unlikely]]
                {
                    Cleanup();
                    return false;
                }

                m_Shards[i].RawMemoryBlock = rawBlock;
                m_Shards[i].AllocationSize = allocationSizePerShard;

                // Shift the raw pointer to the next absolute dynamic cache boundary
                uintptr_t uPtr = reinterpret_cast<uintptr_t>(rawBlock);
                uPtr = (uPtr + CACHE_LINE_MASK) & ~(uintptr_t)CACHE_LINE_MASK;

                m_Shards[i].Buckets = reinterpret_cast<uint32_t*>(uPtr);
                m_Shards[i].Nodes = reinterpret_cast<LruNode*>(uPtr + bucketBytesPerShard);

                // Initialize all buckets to INVALID_INDEX (empty)
                std::memset(m_Shards[i].Buckets, 0xFF, bucketBytesPerShard);

                m_Shards[i].Capacity   = capacityPerShard;
                m_Shards[i].BucketMask = bucketCountPerShard - 1;
                m_Shards[i].ActiveCount.store(0, std::memory_order_relaxed);
                m_Shards[i].LruHead    = INVALID_INDEX;
                m_Shards[i].LruTail    = INVALID_INDEX;
                m_Shards[i].FreeHead   = 0;
                m_Shards[i].Generation = 0;

                // Store the precomputed age limit for branchless promotion logic.
                m_Shards[i].ThresholdAge = precomputedThresholdAge;

                // Chain all newly allocated nodes into the Free List
                for (uint32_t n = 0; n < capacityPerShard - 1; ++n)
                {
                    m_Shards[i].Nodes[n].HashNext = n + 1;
                }

                m_Shards[i].Nodes[capacityPerShard - 1].HashNext = INVALID_INDEX;
            }
        }
        catch (...)
        {
            // Clean up any partially initialized shards to prevent memory leaks.
            Cleanup();
            return false;
        }

        return true;
    }

    // ------------------------------------------------------------------------
    // GetTotalMemoryUsage
    // Calculates the total heap memory currently allocated by the hash 
    // table and its internal structures. This method does not acquire any 
    // locks. Its goal is to provide approximate/statistical information for 
    // telemetry, prioritizing zero-contention over strict synchronization.
    // ------------------------------------------------------------------------
   [[nodiscard]]
    size_t GetTotalMemoryUsage() const noexcept
    {
        if (!m_Shards) [[unlikely]]
        {
            return 0;
        }

        size_t totalBytes = sizeof(Shard) * m_ShardCount;

        for (uint32_t i = 0; i < m_ShardCount; ++i)
        {
            totalBytes += m_Shards[i].AllocationSize;
        }

        return totalBytes;
    }

    // ------------------------------------------------------------------------
    // GetTotalItemCount
    // Provides an approximate, statistical snapshot of the global item count.
    // This method does not acquire any locks; it aggregates relaxed atomic 
    // reads across all shards. Its goal is to provide fast, lock-free 
    // telemetry without interrupting concurrent foreground I/O operations.
    // ------------------------------------------------------------------------
    [[nodiscard]]
    size_t GetTotalItemCount() const noexcept
    {
        if (!m_Shards) [[unlikely]]
        {
            return 0;
        }

        size_t totalItems = 0;

        for (uint32_t i = 0; i < m_ShardCount; ++i)
        {
            // Lock-free statistical read. 32-bit reads are natively atomic on modern processors.
            totalItems += m_Shards[i].ActiveCount.load(std::memory_order_relaxed);
        }

        return totalItems;
    }

    // ------------------------------------------------------------------------
    // Cleanup
    // Traverses all valid nodes, calls explicit destructors for keys, and 
    // safely releases all TValue references before routing memory destruction
    // back through TAllocator::Free.
    // 
    // WARNING: This method performs NO synchronization. Calling Cleanup() or 
    // destructing the table while ANY threads are actively executing Add, 
    // Lookup, Remove, or Trim will result in Undefined Behavior (typically an 
    // Access Violation or Segmentation Fault). The caller is strictly responsible 
    // for quiescing all concurrent access prior to destruction.
    // ------------------------------------------------------------------------
    void Cleanup() noexcept
    {
        if (m_Shards)
        {
            for (uint32_t i = 0; i < m_ShardCount; ++i)
            {
                if (m_Shards[i].RawMemoryBlock)
                {
                    uint32_t curr = m_Shards[i].LruHead;

                    // Traverse the active LRU chain to destroy lingering objects
                    while (curr != INVALID_INDEX)
                    {
                        uint32_t next = m_Shards[i].Nodes[curr].LruNext;

                        // Issue software prefetch for the next node in the random LRU chain
                        if (next != INVALID_INDEX) [[likely]]
                        {
                            CACHE_PREFETCH(&m_Shards[i].Nodes[next]);
                        }

                        // Destructor Order Rule: Destroy Key BEFORE dropping the Value.
                        // If the Key struct relies on the Value for its own cleanup logic,
                        // this guarantees the Value payload is still alive.
                        m_Shards[i].Nodes[curr].Key.~TKey();

                        if (m_Shards[i].Nodes[curr].Value)
                        {
                            // This expensive call hides the prefetch memory latency
                            m_Shards[i].Nodes[curr].Value->Release();
                        }

                        curr = next;
                    }

                    TAllocator::Free(m_Shards[i].RawMemoryBlock, m_Shards[i].AllocationSize);
                }

                // Explicitly destruct the Shard to clean up std::shared_mutex and std::atomic
                m_Shards[i].~Shard();
            }

            TAllocator::Free(m_Shards, sizeof(Shard) * m_ShardCount);
            m_Shards = nullptr;
        }

        m_ShardCount = 0;
    }

    // ------------------------------------------------------------------------
    // Add    
    // Provides O(1) expected insertion time. If the target shard is at capacity, 
    // it enforces limits by evicting the current Least Recently Used (LRU) tail. 
    // To maintain maximum foreground responsiveness and prevent deadlocks, all 
    // expensive operations (Key destruction and Value Release) occur strictly 
    // OUTSIDE the critical section lock.
    //
    // Collision & Overwrite Semantics:
    // - KeepIfExists: Acts as a thread-safe "Get-Or-Add". If the key already 
    // exists, the table is untouched, and the function returns false.
    // - ReplaceIfExists: Acts as an "Upsert". If the key exists, its Value is 
    //   overwritten and the original Value is released outside the lock.
    //
    // The OutExistingValue Parameter:
    // An optional pointer to capture the state of the table upon a collision.
    // - If Action == KeepIfExists: Captures the pre-existing Value that blocked 
    // the insertion.
    // - If Action == ReplaceIfExists: Captures the old Value that was successfully 
    // overwritten.
    // In both scenarios, the returned Value has its AddRef() automatically 
    // incremented under the lock. The caller assumes ownership and MUST call 
    // Release().
    // ------------------------------------------------------------------------    
    [[nodiscard]]  
    bool Add(_In_      const TKey& Key,
             _In_      TValue* InValue,
             _Out_opt_ TValue** OutExistingValue = nullptr,
             _In_      AddAction   Action = AddAction::KeepIfExists)
    {
        if (!m_Shards || InValue == nullptr) [[unlikely]]
        {
            return false;
        }

        uint64_t hash  = THasher::ComputeHash(Key);
        uint64_t mixed = MixHash(hash);

        uint32_t shardIdx = static_cast<uint32_t>(mixed & (m_ShardCount - 1));
        Shard* shard = &m_Shards[shardIdx];

        uint32_t bucketIdx = static_cast<uint32_t>(hash & shard->BucketMask);

        // Track a node cleaned in a previous iteration to prevent livelock.
        uint32_t reservedIdx = INVALID_INDEX;

        while (true)
        {
            std::unique_lock<decltype(shard->Lock)> lockGuard(shard->Lock);

            uint32_t curr = shard->Buckets[bucketIdx];

            // 1. Check for existing Key (Overwrite/Collision path)
            while (curr != INVALID_INDEX)
            {
                if (shard->Nodes[curr].Hash == hash && shard->Nodes[curr].Key == Key) [[unlikely]]
                {
                    TValue* valueToRelease = nullptr;

                    // If Key already exists. Either return the existing value or replace it based on the Action
                    if (Action == AddAction::KeepIfExists)
                    {
                        valueToRelease = nullptr;

                        // Populate the out parameter if requested by the caller
                        if (OutExistingValue != nullptr)
                        {
                            if (shard->Nodes[curr].Value) [[likely]]
                            {
                                shard->Nodes[curr].Value->AddRef();
                            }

                            *OutExistingValue = shard->Nodes[curr].Value;
                        }
                    }
                    else
                    {
                        valueToRelease = shard->Nodes[curr].Value;
                        shard->Nodes[curr].Value = InValue;

                        if (InValue) [[likely]]
                        {
                            InValue->AddRef();
                        }

                        // If the caller wants the old replaced value, 
                        // transfer ownership instead of releasing it internally
                        if (OutExistingValue != nullptr)
                        {
                            *OutExistingValue = valueToRelease;
                            valueToRelease = nullptr;
                        }
                    }

                    // Promote to MRU regardless of ThresholdAge for explicit Adds/Updates
                    if (curr != shard->LruHead)
                    {
                        UnlinkLru(shard, curr);
                        PushMru(shard, curr);
                    }

                    // Return reserved node to FreeList if we didn't need it
                    if (reservedIdx != INVALID_INDEX)
                    {
                        shard->Nodes[reservedIdx].HashNext = shard->FreeHead;
                        shard->FreeHead = reservedIdx;
                    }

                    lockGuard.unlock();

                    if (valueToRelease)
                    {
                        valueToRelease->Release();
                    }

                    return (Action == AddAction::ReplaceIfExists);
                }

                curr = shard->Nodes[curr].HashNext;
            }

            // 2. Procure a Node for Insertion
            uint32_t targetIdx = INVALID_INDEX;

            if (reservedIdx != INVALID_INDEX)
            {
                targetIdx = reservedIdx;
                reservedIdx = INVALID_INDEX;
                shard->ActiveCount.fetch_add(1, std::memory_order_relaxed);
            }
            else if (shard->FreeHead != INVALID_INDEX) [[likely]]
            {
                targetIdx = shard->FreeHead;
                shard->FreeHead = shard->Nodes[targetIdx].HashNext;
                shard->ActiveCount.fetch_add(1, std::memory_order_relaxed);
            }
            else [[unlikely]]
            {
                // Shard is Full: Evict the LRU Tail
                targetIdx = shard->LruTail;

                if (targetIdx == INVALID_INDEX) [[unlikely]]
                {
                    // All nodes are currently "in-flight" being destructed by other 
                    // threads. Yield to let them finish and return to the FreeList
                    lockGuard.unlock();

                    OsYield();
                    continue;
                }

                RemoveFromHashChain(shard, targetIdx);
                UnlinkLru(shard, targetIdx);

                TValue* evictedValue = shard->Nodes[targetIdx].Value;

                // Clear Value first to hide this node from concurrent Lookups
                shard->Nodes[targetIdx].Value = nullptr;
                shard->ActiveCount.fetch_sub(1, std::memory_order_relaxed);

                lockGuard.unlock();

                // Destruct Key and Release Value outside the lock
                shard->Nodes[targetIdx].Key.~TKey();

                if (evictedValue)
                {
                    evictedValue->Release();
                }

                reservedIdx = targetIdx;
                continue;
            }

            // 3. Set metadata FIRST and only link the node LAST to prevent 
            // Lookup() from seeing an inconsistent or half-constructed node           
            shard->Nodes[targetIdx].Hash = hash;
            new (&shard->Nodes[targetIdx].Key) TKey(Key);

            // Value is assigned AFTER the Key is fully constructed
            shard->Nodes[targetIdx].Value = InValue;

            if (InValue) [[likely]]
            {
                InValue->AddRef();
            }

            // Link into the hash chain and LRU MRU list ONLY after the node 
            // is completely ready for a Lookup() thread to find
            shard->Nodes[targetIdx].HashNext = shard->Buckets[bucketIdx];
            shard->Buckets[bucketIdx] = targetIdx;

            PushMru(shard, targetIdx);

            return true;
        }
    }

    // ------------------------------------------------------------------------
    // Lookup
    // Fast-path lookup using an exclusive lock. 
    // 
    // ARCHITECTURAL NOTE ON EXCLUSIVE LOCKING:
    // Counter-intuitively, this method acquires an EXCLUSIVE lock immediately 
    // rather than a SHARED lock, even for highly read-skewed workloads. 
    // Since the critical section is microscopic (a few array index lookups and 
    // integer comparisons), the overhead of a shared-to-exclusive lock upgrade 
    // is a massive net negative.
    // 
    // To promote an LRU node using a shared lock, a thread must:
    // 1. Drop the shared lock.
    // 2. Acquire the exclusive lock.
    // 3. Perform a full ABA hazard mitigation loop (re-traversing the hash chain 
    //    to ensure the node wasn't deleted or recycled during the unlocked gap).
    // 
    // This state-machine and ABA mitigation loop executes significantly more CPU 
    // instructions and generates more NUMA interconnect traffic (cache line 
    // bouncing from atomic reader-count updates) than simply locking the code 
    // exclusively from the start. An exclusive lock guarantees immediate, 
    // branchless, in-place promotion without the hazard overhead.
    // ------------------------------------------------------------------------
    [[nodiscard]]
    bool Lookup(_In_  const TKey& Key,
                _Out_ TValue*&    OutValue) noexcept
    {
        if (!m_Shards) [[unlikely]]
        {
            return false;
        }

        uint64_t hash  = THasher::ComputeHash(Key);
        uint64_t mixed = MixHash(hash);

        uint32_t shardIdx = static_cast<uint32_t>(mixed & (m_ShardCount - 1));
        Shard* shard    = &m_Shards[shardIdx];

        uint32_t bucketIdx = static_cast<uint32_t>(hash & shard->BucketMask);

        // Exclusive lock acquisition
        std::unique_lock<decltype(shard->Lock)> lockGuard(shard->Lock);

        uint32_t curr = shard->Buckets[bucketIdx];

        while (curr != INVALID_INDEX)
        {
            // Prefetch the next link in the collision chain to hide memory latency 
            // while the CPU performs the Key comparison
            uint32_t next = shard->Nodes[curr].HashNext;
            if (next != INVALID_INDEX) [[likely]]
            {
                CACHE_PREFETCH(&shard->Nodes[next]);
            }

            if (shard->Nodes[curr].Hash == hash && shard->Nodes[curr].Key == Key) [[likely]]
            {
                TValue* val = shard->Nodes[curr].Value;

                // Secure the payload while still under the shard lock
                val->AddRef();
                OutValue = val;

                // Optimized Threshold & Promotion Logic
                if (curr != shard->LruHead) [[unlikely]]
                {
                    // Check if node age exceeds precomputed shard threshold.
                    // Subtraction is safe against 64-bit wrap-around.
                    if ((shard->Generation - shard->Nodes[curr].LastPromoted) >= shard->ThresholdAge)
                    {
                        UnlinkLru(shard, curr);
                        PushMru(shard, curr);
                    }
                }

                return true;
            }

            curr = next;
        }

        return false;
    }

    // ------------------------------------------------------------------------
    // Remove
    // Manually purges a key from the cache. The node is completely detached,
    // destructed outside the lock to prevent deadlocks, and then pushed 
    // to the FreeHead stack. Uses RAII to guarantee exception safety.
    // ------------------------------------------------------------------------    
    bool Remove(_In_ const TKey& Key)
    {
        if (!m_Shards) [[unlikely]]
        {
            return false;
        }

        uint64_t hash  = THasher::ComputeHash(Key);
        uint64_t mixed = MixHash(hash);

        uint32_t shardIdx = static_cast<uint32_t>(mixed & (m_ShardCount - 1));
        Shard* shard    = &m_Shards[shardIdx];

        uint32_t bucketIdx = static_cast<uint32_t>(hash & shard->BucketMask);

        // Acquire lock using RAII
        std::unique_lock<decltype(shard->Lock)> lockGuard(shard->Lock);

        uint32_t curr = shard->Buckets[bucketIdx];
        uint32_t prev = INVALID_INDEX;

        while (curr != INVALID_INDEX)
        {
            if (shard->Nodes[curr].Hash == hash && shard->Nodes[curr].Key == Key) [[likely]]
            {
                if (prev == INVALID_INDEX)
                {
                    shard->Buckets[bucketIdx] = shard->Nodes[curr].HashNext;
                }
                else
                {
                    shard->Nodes[prev].HashNext = shard->Nodes[curr].HashNext;
                }

                UnlinkLru(shard, curr);

                TValue* valueToRelease = shard->Nodes[curr].Value;
                shard->Nodes[curr].Value = nullptr;

                shard->ActiveCount.fetch_sub(1, std::memory_order_relaxed);

                // Explicit unlock before destruction
                lockGuard.unlock();

                // Destruct actual key and payload outside lock
                shard->Nodes[curr].Key.~TKey();

                if (valueToRelease) [[likely]]
                {
                    valueToRelease->Release();
                }

                // Relock to safely push the clean node to the FreeList
                lockGuard.lock();

                shard->Nodes[curr].HashNext = shard->FreeHead;
                shard->FreeHead = curr;

                // lockGuard destructor safely releases the lock as we return
                return true;
            }

            prev = curr;
            curr = shard->Nodes[curr].HashNext;
        }

        // lockGuard destructor safely releases the lock as we return
        return false;
    }

    // ------------------------------------------------------------------------
    // Yielding Batch Trim
    // Evicts LRU items to maintain healthy shard capacities. 
    //
    // - Watermark Mode (Count = 0): Uses lock-free atomic reads to bypass 
    //   healthy shards in O(1). Only triggers if a shard exceeds 90% capacity, 
    //   trimming it down to 85% to prevent eviction thrashing.
    // 
    // - Yielding Execution: To prevent starving foreground I/O threads, the 
    //   critical section is scope-locked for *each individual node* evicted. 
    //   All Key destruction and Value Release() calls occur entirely outside 
    //   the lock.
    // ------------------------------------------------------------------------
    size_t Trim(_In_ const size_t Count = 0,
                _In_ const bool   Force = false)
    {
        if (!m_Shards) [[unlikely]]
        {
            return 0;
        }

        // If Count is 0, we enter "Watermark Mode" where the cache autonomously 
        // cleans itself down to a healthy percentage. Otherwise, we trim a specific number.
        bool   TrimToWatermark = (Count == 0);
        size_t totalTrimmed    = 0;

        for (uint32_t i = 0; i < m_ShardCount; ++i)
        {
            Shard* shard = &m_Shards[i];

            // Define proportional watermarks based on the specific shard's capacity.
            // High Watermark (90%): The threshold where background trimming activates.
            // Low Watermark (85%): The target threshold we trim down to.
            // This 5% gap prevents "trim thrashing" (constantly trimming 1 item at 90%).
            size_t highWatermark = (shard->Capacity * 90) / 100;
            size_t lowWatermark  = (shard->Capacity * 85) / 100;

            if (!Force)
            {
                if (shard->ActiveCount.load(std::memory_order_relaxed) < highWatermark)
                {
                    continue;
                }
            }

            while (shard->ActiveCount.load(std::memory_order_relaxed) > 0 &&
                  (Force || shard->ActiveCount.load(std::memory_order_relaxed) > lowWatermark) &&
                  (TrimToWatermark || totalTrimmed < Count))
            {
                // Acquire RAII lock for this specific iteration
                std::unique_lock<decltype(shard->Lock)> lockGuard(shard->Lock);

                // Double-check state under lock
                if (shard->ActiveCount.load(std::memory_order_relaxed) > 0 &&
                    (Force || shard->ActiveCount.load(std::memory_order_relaxed) > lowWatermark))
                {
                    uint32_t targetIdx = shard->LruTail;

                    if (targetIdx != INVALID_INDEX) [[likely]]
                    {
                        uint32_t nextTail = shard->Nodes[targetIdx].LruPrev;

                        // Issue a software prefetch to the CPU. While we are busy 
                        // doing the math to detach the current tail
                        if (nextTail != INVALID_INDEX) [[likely]]
                        {
                            CACHE_PREFETCH(&shard->Nodes[nextTail]);
                        }

                        RemoveFromHashChain(shard, targetIdx);
                        UnlinkLru(shard, targetIdx);

                        TValue* valueToRelease = shard->Nodes[targetIdx].Value;
                        shard->Nodes[targetIdx].Value = nullptr;

                        shard->ActiveCount.fetch_sub(1, std::memory_order_relaxed);

                        // Explicit unlock before destruction
                        lockGuard.unlock();

                        // Deferred destruction logic outside lock
                        shard->Nodes[targetIdx].Key.~TKey();

                        if (valueToRelease) [[likely]]
                        {
                            valueToRelease->Release();
                        }

                        // Relock to safely return node to FreeList
                        lockGuard.lock();

                        shard->Nodes[targetIdx].HashNext = shard->FreeHead;
                        shard->FreeHead = targetIdx;

                        totalTrimmed++;

                        // lockGuard automatically unlocks at the end of the loop scope
                        continue;
                    }
                }

                // If we reach here, the condition failed after locking, or targetIdx was invalid.
                // lockGuard automatically unlocks as we break.
                break;
            }

            if (!TrimToWatermark && totalTrimmed >= Count)
            {
                break;
            }
        }

        return totalTrimmed;
    }

    // ------------------------------------------------------------------------
    // Enumerate
    // Safely iterates over all active items in the table, from Most Recently 
    // Used (MRU) to Least Recently Used (LRU) across all shards.
    //     
    // Issues a CACHE_PREFETCH software hint for the next node in the LRU chain 
    // BEFORE invoking the user callback. This helps mitigate DRAM latency by 
    // requesting the hardware to load the next node asynchronously while the 
    // current callback executes.
    // 
    // WARNING: The callback is invoked while holding an EXCLUSIVE lock on the 
    // current shard. The callback MUST NOT attempt to modify the table 
    // (Add/Remove/Trim) to prevent recursive deadlocks.
    // ------------------------------------------------------------------------    
    template <typename TCallback>
    void Enumerate(_In_ const TCallback& Callback) const
    {
        if (!m_Shards) [[unlikely]]
        {
            return;
        }

        for (uint32_t i = 0; i < m_ShardCount; ++i)
        {
            Shard* shard = &m_Shards[i];

            // RAII Exclusive Lock
            std::unique_lock<decltype(shard->Lock)> lockGuard(shard->Lock);

            uint32_t curr = shard->LruHead;

            while (curr != INVALID_INDEX)
            {
                // Fetch the index of the next chronological node and hint the 
                // CPU to load it while we process the user callback.
                uint32_t next = shard->Nodes[curr].LruNext;

                if (next != INVALID_INDEX) [[likely]]
                {
                    CACHE_PREFETCH(&shard->Nodes[next]);
                }

                // The callback should return false to abort the enumeration early
                if (!Callback(shard->Nodes[curr].Key, shard->Nodes[curr].Value))
                {
                    // lockGuard automatically unlocks as we return
                    return;
                }

                curr = next;
            }

            // lockGuard automatically unlocks at the end of the loop scope
        }
    }
};