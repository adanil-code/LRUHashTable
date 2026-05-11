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

#if defined(_M_AMD64) || defined(_M_IX86)
#include <xmmintrin.h>
#define CACHE_PREFETCH(ptr) _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0)
#elif defined(_M_ARM64)
#define CACHE_PREFETCH(ptr) __prefetch(ptr)
#else
#define CACHE_PREFETCH(ptr) ((void)0)
#endif

#ifdef _KERNEL_MODE
#include <ntddk.h>
#include <intrin.h>

// ----------------------------------------------------------------------------
// Placement New Overloads
// Standard kernel-mode placement new override. 
// Allows us to construct C++ objects directly into our pre-allocated array memory 
// without triggering pool allocations on the hot path.
// ----------------------------------------------------------------------------
#ifndef KM_PLACEMENT_NEW
#define KM_PLACEMENT_NEW
inline void* __cdecl operator new(SIZE_T Size,
                                  void*  Object) noexcept
{
    UNREFERENCED_PARAMETER(Size);
    return Object;
}

inline void __cdecl operator delete(void* Object,
                                    void* Memory)
{
    UNREFERENCED_PARAMETER(Object);
    UNREFERENCED_PARAMETER(Memory);
}
#endif // KM_PLACEMENT_NEW

#else
// Pull in standard placement new for user-mode tests
#include <new>
#endif // _KERNEL_MODE

#include <limits.h>

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
template <typename TKey, typename TValue, typename THasher>
class CLruHashTable
{
public:
    static const ULONG  POOL_TAG      = 'LruR';
    static const UINT32 INVALID_INDEX = 0xFFFFFFFF;

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
        // read. If the hash fails, it immediately uses ulHashNext to jump.
        // --------------------------------------------------------------------
        UINT64  ullHash;        // Offset 0: Stored to avoid re-hashing during eviction checks
        UINT32  ulHashNext;     // Offset 8: Index of the next node in this hash bucket's chain
        UINT32  ulLruPrev;      // Offset 12: Pulled up to eliminate 4-byte padding hole

        // --------------------------------------------------------------------
        // 2. MATCH PATH: Key Verification
        // Kept as high as possible so it remains in the same 64-byte L1 cache 
        // line as the hash variables above.
        // --------------------------------------------------------------------
        TKey    tKey;           // Offset 16: Hash table key

        // --------------------------------------------------------------------
        // 3. COLDER PATH: Payload & Eviction Meta
        // Accessed only upon a definitive hash/key match or during eviction.
        // --------------------------------------------------------------------
        TValue* pValue;           // Raw pointer to intrusive ref-counted object
        UINT64  ullLastPromoted;  // Tracks the "age" relative to Shard::ullGeneration
        UINT32  ulLruNext;        // Placed at the tail.
    };

    // ------------------------------------------------------------------------
    // Cache Shard
    // Aligned to 64 bytes to prevent false sharing across CPU cache lines.
    // If two shards share a CPU cache line, a lock acquisition on Shard A 
    // would inadvertently invalidate the cache line for a CPU accessing Shard B.
    // Each shard independently manages its own capacity, locks, and LRU queue.
    // ------------------------------------------------------------------------    
    struct alignas(64) Shard
    {
        // CACHE LINE 0: The Lock. 
        // Spinning threads will blast this line, but it won't interfere 
        // with the lock-holder's data access
        alignas(64) EX_PUSH_LOCK lockPush;         // Pointer-sized reader/writer lock (<= APC_LEVEL) for synchronizing shard access

        // Cache Line 1: Read-Heavy Hash Data
        // Exclusively owned by the thread that successfully acquires the lock
        alignas(64) LruNode* pNodes;               // Pointer to the segment of the unified flat node block for this shard
        UINT32*              pulBuckets;           // Pointer to the segment of the unified flat bucket block for this shard

        PVOID                pRawMemoryBlock;      // Base pointer for NUMA deallocation
        SIZE_T               szAllocationSize;     // Exact size allocated on the NUMA node

        UINT32               ulBucketMask;         // Bitwise mask used to route a hash to a specific bucket index efficiently
        UINT32               ulCapacity;           // Maximum number of active LruNodes this specific shard can hold        

        // CACHE LINE 2: Write-Heavy LRU State
        // Mutated on every Add/Overwrite MRU promotion
        alignas(64) UINT32   ulLruHead;            // Index of the Most Recently Used (MRU) node. Top of the active LRU chain
        UINT32               ulLruTail;            // Index of the Least Recently Used (LRU) node. Primary target for eviction
        UINT32               ulFreeHead;           // Index of the first unused node in the pre-allocated array chain

        UINT64               ullGeneration;        // Incremented on every MRU push. Used for probabilistic lazy promotion
        UINT64               ullThresholdAge;      // Precomputed promotion age limit

        // CACHE LINE 3: Active Count
        // Mutated independently, lock-free statistical reads
        alignas(64) volatile UINT32 ulActiveCount; // Tracks live items. Volatile prevents register-caching UB on lock-free reads
    };

private:
    Shard* m_pShards;               // Dynamically allocated array representing the sharded cache architecture
    ULONG  m_ulShardCount;          // Total number of initialized shards. Guaranteed to be a power of 2
    UINT32 m_ulPromotionThreshold;  // Percentage threshold (0-100) determining how aggressively nodes are promoted to MRU

    // ------------------------------------------------------------------------
    // MixHash (stripped-down version of SplitMix64)
    // Applies a final avalanche step to the user-provided hash. 
    // This forces entropy into the lower bits to prevent clustering when 
    // dealing with weak hash functions (like identity hashes or sequential IDs),
    // ensuring better distribution across power-of-2 shard masks.
    // For the hash table only lower bits are actually matter, thus second part
    // of SplitMix64 is omitted. This saves about 4-5 CPU cycles.
    // ------------------------------------------------------------------------
    static constexpr UINT64 MixHash(UINT64 hash) noexcept
    {
        UINT64 mixed = hash ^ (hash >> 30);
        mixed *= 0xbf58476d1ce4e5b9ULL;
        mixed ^= (mixed >> 27);

        return mixed;
    }

    // ------------------------------------------------------------------------
    // UnlinkLru (Internal)
    // Severs a node from the doubly-linked LRU list using 32-bit indices.
    // Caller must hold the shard lock exclusively.
    // ------------------------------------------------------------------------
    __forceinline void UnlinkLru(_In_ Shard* pShardPtr,
                                 _In_ UINT32 ulIndex) noexcept
    {
        PAGED_CODE();

        UINT32 ulPrev = pShardPtr->pNodes[ulIndex].ulLruPrev;
        UINT32 ulNext = pShardPtr->pNodes[ulIndex].ulLruNext;

        if (ulPrev != INVALID_INDEX)
        {
            pShardPtr->pNodes[ulPrev].ulLruNext = ulNext;
        }
        else
        {
            pShardPtr->ulLruHead = ulNext;
        }

        if (ulNext != INVALID_INDEX)
        {
            pShardPtr->pNodes[ulNext].ulLruPrev = ulPrev;
        }
        else
        {
            pShardPtr->ulLruTail = ulPrev;
        }
    }

    // ------------------------------------------------------------------------
    // PushMru (Internal)
    // Inserts a node at the absolute front (Head) of the LRU list, marking 
    // it as the Most Recently Used item, and updates its Generation stamp.
    // Caller must hold the shard lock exclusively.
    // ------------------------------------------------------------------------
    __forceinline void PushMru(_In_ Shard* pShardPtr,
                               _In_ UINT32 ulIndex) noexcept
    {
        PAGED_CODE();

        pShardPtr->ullGeneration++;
        pShardPtr->pNodes[ulIndex].ullLastPromoted = pShardPtr->ullGeneration;

        pShardPtr->pNodes[ulIndex].ulLruPrev = INVALID_INDEX;
        pShardPtr->pNodes[ulIndex].ulLruNext = pShardPtr->ulLruHead;

        if (pShardPtr->ulLruHead != INVALID_INDEX)
        {
            pShardPtr->pNodes[pShardPtr->ulLruHead].ulLruPrev = ulIndex;
        }

        pShardPtr->ulLruHead = ulIndex;

        // If the list was empty, this node is both the Head and the Tail
        if (pShardPtr->ulLruTail == INVALID_INDEX)
        {
            pShardPtr->ulLruTail = ulIndex;
        }
    }

    // ------------------------------------------------------------------------
    // RemoveFromHashChain (Internal)
    // Removes a node from the singly-linked hash collision chain.
    // Caller must hold the shard lock exclusively.
    // ------------------------------------------------------------------------
    void RemoveFromHashChain(_In_ Shard* pShardPtr,
                             _In_ UINT32 ulIndex) noexcept
    {
        PAGED_CODE();
        
        UINT32 ulBucketIdx = static_cast<UINT32>(pShardPtr->pNodes[ulIndex].ullHash & pShardPtr->ulBucketMask);

        UINT32 ulCurr = pShardPtr->pulBuckets[ulBucketIdx];
        UINT32 ulPrev = INVALID_INDEX;

        while (ulCurr != INVALID_INDEX)
        {
            if (ulCurr == ulIndex)
            {
                if (ulPrev == INVALID_INDEX)
                {
                    pShardPtr->pulBuckets[ulBucketIdx] = pShardPtr->pNodes[ulCurr].ulHashNext;
                }
                else
                {
                    pShardPtr->pNodes[ulPrev].ulHashNext = pShardPtr->pNodes[ulCurr].ulHashNext;
                }
                break;
            }

            ulPrev = ulCurr;
            ulCurr = pShardPtr->pNodes[ulCurr].ulHashNext;
        }
    }

public:
    CLruHashTable() noexcept : m_pShards(NULL),
                               m_ulShardCount(0),
                               m_ulPromotionThreshold(0)
    {
        PAGED_CODE();
    }

    ~CLruHashTable() noexcept
    {
        PAGED_CODE();
        Cleanup();
    }

    // ------------------------------------------------------------------------
    // Initialize
    // Allocates non-paged pool memory for the hash table array. Dynamically 
    // calculates shard count based on active processor cores to avoid 
    // cache-line ping-pong under heavy reader loads. Ensures absolute 64-byte 
    // alignment for buckets and precomputes age thresholds.
    // 
    // The shard metadata array AND the mega-blocks are exclusively allocated
    // from non-paged pool memory. This guarantees the cache remains memory-resident,
    // avoiding expensive disk I/O page faults while locks are held.
    // ------------------------------------------------------------------------
    NTSTATUS Initialize(_In_ SIZE_T uTotalEntries,
                        _In_ UINT32 ulPromotionThreshold = 0) noexcept
    {
        PAGED_CODE();

        Cleanup();

        // Prevent absolute arithmetic overflow on massive capacity requests
        if (uTotalEntries > (~(SIZE_T)0 / sizeof(LruNode)))
        {
            return STATUS_INVALID_PARAMETER;
        }

        // Clamp threshold to a valid 0-100 percentage range
        if (ulPromotionThreshold > 100)
        {
            ulPromotionThreshold = 100;
        }

        m_ulPromotionThreshold = ulPromotionThreshold;

        ULONG ulNumProcs = KeQueryActiveProcessorCountEx(ALL_PROCESSOR_GROUPS);

        // Aggressive dispersion for massive core counts
        ULONG ulTargetShards = ulNumProcs * 32;

        // To prevent premature evictions due to imperfect hash distribution 
        // (hash clustering), a shard needs enough capacity to absorb variance.
        // 64 items is the minimum safe floor to absorb statistical clustering.
        ULONG ulMinItemsPerShard = 64;

        // Calculate the maximum number of shards we can support without 
        // starving any shard of items
        ULONG ulMaxShardsForCapacity = static_cast<ULONG>(uTotalEntries / ulMinItemsPerShard);

        // Absolute fallback for legitimately tiny tables (e.g., uTotalEntries < 64).
        // This forces tiny tables into a single shard, completely preventing 
        // hash-clustering evictions.
        if (ulMaxShardsForCapacity == 0)
        {
            ulMaxShardsForCapacity = 1;
        }

        // Clamp our hardware target to our capacity reality
        if (ulTargetShards > ulMaxShardsForCapacity)
        {
            ulTargetShards = ulMaxShardsForCapacity;
        }

        m_ulShardCount = 1;

        // We round DOWN to the nearest power of 2 here 
        while ((m_ulShardCount << 1) <= ulTargetShards)
        {
            m_ulShardCount <<= 1;
        }

        // Enforce the modern hardware ceiling
        if (m_ulShardCount > 4096)
        {
            m_ulShardCount = 4096;
        }

        // Prevent unrealistic capacity requests
        if ((uTotalEntries / m_ulShardCount) >= (INVALID_INDEX - 1))
        {
            return STATUS_INVALID_PARAMETER;
        }

        // Non-Paged Pool Allocation with NUMA.
        // The contiguous Shard array is pinned in physical RAM. Because it
        // is a single array, we bind the entire block to the local NUMA node.
        POOL_EXTENDED_PARAMETER shardExtParams = { 0 };

        shardExtParams.Type          = PoolExtendedParameterNumaNode;
        shardExtParams.Optional      = 1;
        shardExtParams.PreferredNode = KeGetCurrentNodeNumber();

        m_pShards = static_cast<Shard*>(ExAllocatePool3(POOL_FLAG_NON_PAGED,
                                                        sizeof(Shard) * m_ulShardCount,
                                                        POOL_TAG,
                                                        &shardExtParams,
                                                        1));
        if (!m_pShards)
        {
            return STATUS_INSUFFICIENT_RESOURCES;
        }
        
        // Use ceiling division to prevent capacity loss from truncation
        UINT32 ulCapacityPerShard = (UINT32)((uTotalEntries + m_ulShardCount - 1) / m_ulShardCount);

        // Absolute floor to prevent zero-capacity configuration bugs on microscopic tables
        if (ulCapacityPerShard < 8)
        {
            ulCapacityPerShard = 8;
        }

        UINT32 ulBucketCountPerShard = 1;

        while (ulBucketCountPerShard < ulCapacityPerShard && ulBucketCountPerShard != 0x80000000)
        {
            ulBucketCountPerShard <<= 1;
        }

        // Calculate bucket size and pad it to the nearest 64 bytes for relative alignment
        SIZE_T uBucketBytesPerShard = sizeof(UINT32) * ulBucketCountPerShard;
        uBucketBytesPerShard = (uBucketBytesPerShard + 63) & ~(SIZE_T)63;

        SIZE_T uNodeBytesPerShard = sizeof(LruNode) * ulCapacityPerShard;

        // Over-allocate by 64 bytes to allow for manual absolute alignment
        SIZE_T uAllocationSizePerShard = uBucketBytesPerShard + uNodeBytesPerShard + 64;

        // To avoid "if (threshold == 0)" branches from the Lookup hot-path, 
        // we map the percentage to a absolute generation delta 
        UINT64 ullPrecomputedThresholdAge;

        if (m_ulPromotionThreshold == 0)
        {
            // 0% Safe Zone (Strict LRU): Age is always >= 0. Triggers immediate promotion
            ullPrecomputedThresholdAge = 0;
        }
        else if (m_ulPromotionThreshold == 100)
        {
            // 100% Safe Zone (FIFO): ullAge will never exceed _UI64_MAX. Completely halts promotion
            ullPrecomputedThresholdAge = _UI64_MAX;
        }
        else
        {
            // Standard Threshold: Calculate relative distance in the LRU queue
            ullPrecomputedThresholdAge = ((UINT64)ulCapacityPerShard * m_ulPromotionThreshold) / 100;
        }

        // Map Active NUMA Topology
        USHORT usHighestNode = KeQueryHighestNodeNumber();
        ULONG ulMaxNodes     = usHighestNode + 1;

        USHORT* pusActiveNodes = static_cast<USHORT*>(ExAllocatePool2(POOL_FLAG_PAGED,
                                                                      ulMaxNodes * sizeof(USHORT),
                                                                      POOL_TAG));
        ULONG ulActiveNodeCount = 0;

        if (pusActiveNodes)
        {
            for (USHORT usNode = 0; usNode <= usHighestNode; ++usNode)
            {
                GROUP_AFFINITY Affinity = { 0 };                
                KeQueryNodeActiveAffinity(usNode, &Affinity, NULL);

                // Verify the node actually has active processors assigned to it
                if (Affinity.Mask != 0)
                {
                    pusActiveNodes[ulActiveNodeCount++] = usNode;
                }
            }
        }

        // If allocation failed or OS returned 0 active nodes
        if (ulActiveNodeCount == 0)
        {
            ulActiveNodeCount = ulMaxNodes;
        }

        // Allocate Shard Mega-Blocks
        for (ULONG ulIdx = 0; ulIdx < m_ulShardCount; ++ulIdx)
        {
            USHORT usTargetNode;

            // Route the shard using our compacted list of truly active nodes
            if (pusActiveNodes && pusActiveNodes[ulIdx % ulActiveNodeCount] <= usHighestNode)
            {
                usTargetNode = pusActiveNodes[ulIdx % ulActiveNodeCount];
            }
            else
            {
                usTargetNode = (USHORT)(ulIdx % ulActiveNodeCount);
            }

            POOL_EXTENDED_PARAMETER extendedParams = { 0 };

            extendedParams.Type          = PoolExtendedParameterNumaNode;
            extendedParams.Optional      = 1;
            extendedParams.PreferredNode = usTargetNode;

            PVOID pRawBlock = ExAllocatePool3(POOL_FLAG_NON_PAGED,
                                              uAllocationSizePerShard,
                                              POOL_TAG,
                                              &extendedParams,
                                              1);
            if (!pRawBlock)
            {
                if (pusActiveNodes)
                {
                    ExFreePoolWithTag(pusActiveNodes, POOL_TAG);
                }

                Cleanup();
                return STATUS_INSUFFICIENT_RESOURCES;
            }

            m_pShards[ulIdx].pRawMemoryBlock  = pRawBlock;
            m_pShards[ulIdx].szAllocationSize = uAllocationSizePerShard;

            // Shift the raw pointer to the next absolute 64-byte boundary.
            UINT_PTR uPtr = reinterpret_cast<UINT_PTR>(pRawBlock);
            uPtr = (uPtr + 63) & ~(UINT_PTR)63;

            m_pShards[ulIdx].pulBuckets = reinterpret_cast<UINT32*>(uPtr);
            m_pShards[ulIdx].pNodes = reinterpret_cast<LruNode*>(uPtr + uBucketBytesPerShard);

            // Initialize all buckets to INVALID_INDEX (empty)
            RtlFillMemory(m_pShards[ulIdx].pulBuckets,
                          uBucketBytesPerShard,
                          0xFF);

            ExInitializePushLock(&m_pShards[ulIdx].lockPush);

            m_pShards[ulIdx].ulCapacity      = ulCapacityPerShard;
            m_pShards[ulIdx].ulBucketMask    = ulBucketCountPerShard - 1;
            m_pShards[ulIdx].ulActiveCount   = 0;
            m_pShards[ulIdx].ulLruHead       = INVALID_INDEX;
            m_pShards[ulIdx].ulLruTail       = INVALID_INDEX;
            m_pShards[ulIdx].ulFreeHead      = 0;
            m_pShards[ulIdx].ullGeneration   = 0;
            m_pShards[ulIdx].ullThresholdAge = ullPrecomputedThresholdAge;

            for (UINT32 ulNodeIdx = 0; ulNodeIdx < ulCapacityPerShard - 1; ++ulNodeIdx)
            {
                m_pShards[ulIdx].pNodes[ulNodeIdx].ulHashNext = ulNodeIdx + 1;
            }

            m_pShards[ulIdx].pNodes[ulCapacityPerShard - 1].ulHashNext = INVALID_INDEX;
        }

        // Clean up the temporary active node map array
        if (pusActiveNodes)
        {
            ExFreePoolWithTag(pusActiveNodes, POOL_TAG);
        }

        return STATUS_SUCCESS;
    }

    // ------------------------------------------------------------------------
    // GetTotalMemoryUsage
    // Calculates the total heap memory currently allocated by the hash 
    // table and its internal structures. This method does not acquire any 
    // locks. Its goal is to provide approximate/statistical information for 
    // telemetry, prioritizing zero-contention over strict synchronization.
    // ------------------------------------------------------------------------
    [[nodiscard]]
    SIZE_T GetTotalMemoryUsage() const noexcept
    {
        PAGED_CODE();

        if (!m_pShards)
        {
            return 0;
        }

        SIZE_T uTotalBytes = sizeof(Shard) * m_ulShardCount;

        for (ULONG ulIdx = 0; ulIdx < m_ulShardCount; ++ulIdx)
        {
            uTotalBytes += m_pShards[ulIdx].szAllocationSize;
        }

        return uTotalBytes;
    }

    // ------------------------------------------------------------------------
    // GetTotalItemCount
    // Provides an approximate, statistical snapshot of the global item count.
    // This method does not acquire any locks; it aggregates relaxed atomic 
    // reads across all shards. Its goal is to provide fast, lock-free 
    // telemetry without interrupting concurrent foreground I/O operations.
    // ------------------------------------------------------------------------
    [[nodiscard]]
    SIZE_T GetTotalItemCount() const noexcept
    {
        PAGED_CODE();

        if (!m_pShards)
        {
            return 0;
        }

        SIZE_T uTotalItems = 0;

        for (ULONG ulIdx = 0; ulIdx < m_ulShardCount; ++ulIdx)
        {
            // Lock-free statistical read. 32-bit reads are natively atomic on modern processors.
            uTotalItems += m_pShards[ulIdx].ulActiveCount;
        }

        return uTotalItems;
    }

    // ------------------------------------------------------------------------
    // Cleanup
    // Traverses all valid nodes, calls explicit destructors for keys, and 
    // safely releases all TValue references before freeing the mega-blocks.
    // 
    // WARNING: This method performs NO synchronization. Calling Cleanup() or 
    // destructing the table while ANY threads are actively executing Add, 
    // Lookup, Remove, or Trim will result in a fatal system Bug Check 
    // (e.g., invalid pool access or use-after-free). The caller is strictly 
    // responsible for quiescing all concurrent access prior to destruction. 
    // ------------------------------------------------------------------------
    void Cleanup() noexcept
    {
        PAGED_CODE();

        if (m_pShards)
        {
            for (ULONG ulIdx = 0; ulIdx < m_ulShardCount; ++ulIdx)
            {
                if (m_pShards[ulIdx].pRawMemoryBlock)
                {
                    UINT32 ulCurr = m_pShards[ulIdx].ulLruHead;

                    // Traverse the active LRU chain to destroy lingering objects.
                    // Locks are not required here because teardown assumes exclusive access.
                    while (ulCurr != INVALID_INDEX)
                    {
                        UINT32 ulNext = m_pShards[ulIdx].pNodes[ulCurr].ulLruNext;

                        if (ulNext != INVALID_INDEX)
                        {
                            CACHE_PREFETCH(&m_pShards[ulIdx].pNodes[ulNext]);
                        }

                        m_pShards[ulIdx].pNodes[ulCurr].tKey.~TKey();

                        if (m_pShards[ulIdx].pNodes[ulCurr].pValue)
                        {
                            m_pShards[ulIdx].pNodes[ulCurr].pValue->Release();
                            m_pShards[ulIdx].pNodes[ulCurr].pValue = NULL;
                        }

                        ulCurr = ulNext;
                    }

                    ExFreePoolWithTag(m_pShards[ulIdx].pRawMemoryBlock,
                                      POOL_TAG);
                }
            }

            ExFreePoolWithTag(m_pShards,
                              POOL_TAG);

            m_pShards = NULL;
        }

        m_ulShardCount = 0;
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
    // - KeepIfExists: Acts as a thread-safe "Get-Or-Add". If the key already exists, 
    //   the table is untouched, and the function returns false.
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
    BOOLEAN Add(_In_      const TKey& tKey,
                _In_      TValue*     pInValue,
                _Out_opt_ TValue**    ppOutExistingValue = NULL,
                _In_      AddAction   Action = AddAction::KeepIfExists) noexcept
    {
        PAGED_CODE();

        if (pInValue == NULL || !m_pShards)
        {
            return FALSE;
        }

        UINT64 ullHash  = THasher::ComputeHash(tKey);
        UINT64 ullMixed = MixHash(ullHash);

        ULONG  ulShardIdx = (ULONG)(ullMixed & (m_ulShardCount - 1));
        Shard* pShard     = &m_pShards[ulShardIdx];

        UINT32 ulBucketIdx = (UINT32)(ullHash & pShard->ulBucketMask);

        // Track the node we cleaned across loop iterations to prevent steal/livelock
        UINT32 ulReservedIdx = INVALID_INDEX;

        while (TRUE)
        {
            KeEnterCriticalRegion();
            ExAcquirePushLockExclusive(&pShard->lockPush);

            UINT32 ulCurr = pShard->pulBuckets[ulBucketIdx];

            // 1. Check if the key already exists (Collision path)
            while (ulCurr != INVALID_INDEX)
            {
                if (pShard->pNodes[ulCurr].ullHash == ullHash && pShard->pNodes[ulCurr].tKey == tKey)
                {
                    TValue* pValueToRelease = NULL;

                    // If Key already exists. 
                    // Either return the existing value or replace it based on the Action.
                    if (Action == AddAction::KeepIfExists)
                    {
                        pValueToRelease = NULL;

                        if (ppOutExistingValue != NULL)
                        {
                            if (pShard->pNodes[ulCurr].pValue)
                            {
                                pShard->pNodes[ulCurr].pValue->AddRef();
                            }

                            *ppOutExistingValue = pShard->pNodes[ulCurr].pValue;
                        }
                    }
                    else
                    {
                        pValueToRelease = pShard->pNodes[ulCurr].pValue;
                        pShard->pNodes[ulCurr].pValue = pInValue;

                        if (pInValue)
                        {
                            pInValue->AddRef();
                        }

                        // If caller wants the evicted value, transfer ownership
                        if (ppOutExistingValue != NULL)
                        {
                            *ppOutExistingValue = pValueToRelease;
                            pValueToRelease = NULL; // Caller is now responsible for calling Release()
                        }
                    }

                    if (ulCurr != pShard->ulLruHead)
                    {
                        UnlinkLru(pShard, ulCurr);
                        PushMru(pShard, ulCurr);
                    }

                    if (ulReservedIdx != INVALID_INDEX)
                    {
                        pShard->pNodes[ulReservedIdx].ulHashNext = pShard->ulFreeHead;
                        pShard->ulFreeHead = ulReservedIdx;
                    }

                    ExReleasePushLockExclusive(&pShard->lockPush);
                    KeLeaveCriticalRegion();

                    if (pValueToRelease)
                    {
                        pValueToRelease->Release();
                    }

                    return (Action == AddAction::ReplaceIfExists) ? TRUE : FALSE;
                }

                ulCurr = pShard->pNodes[ulCurr].ulHashNext;
            }

            // 2. Cache Miss - Procure a new node
            UINT32 ulTargetIdx = INVALID_INDEX;

            if (ulReservedIdx != INVALID_INDEX)
            {
                // Fast Path A: We brought our own clean node from a previous loop iteration!
                ulTargetIdx   = ulReservedIdx;
                ulReservedIdx = INVALID_INDEX;
                
                pShard->ulActiveCount++;
            }
            else if (pShard->ulFreeHead != INVALID_INDEX)
            {
                // Fast Path B: Cache has unused capacity. Pop from the Free List.
                ulTargetIdx        = pShard->ulFreeHead;
                pShard->ulFreeHead = pShard->pNodes[ulTargetIdx].ulHashNext;
                
                pShard->ulActiveCount++;
            }
            else
            {
                // Slow Path: Cache is full. Evict the LRU Tail.
                ulTargetIdx = pShard->ulLruTail;

                if (ulTargetIdx == INVALID_INDEX)
                {
                    // Extreme Contention Edge Case: All nodes are currently "in-flight" 
                    // being destructed by other threads. Drop the lock and yield the 
                    // CPU to let the preempted threads return the nodes to the FreeList.
                    ExReleasePushLockExclusive(&pShard->lockPush);
                    KeLeaveCriticalRegion();

                    // Yield to the OS scheduler to let preempted threads finish Release()
                    LARGE_INTEGER yieldTimeout;
                    yieldTimeout.QuadPart = -10000; // 1 millisecond relative delay
                    KeDelayExecutionThread(KernelMode, FALSE, &yieldTimeout);

                    continue;
                }

                RemoveFromHashChain(pShard, ulTargetIdx);
                UnlinkLru(pShard, ulTargetIdx);

                TValue* pEvictedValue = pShard->pNodes[ulTargetIdx].pValue;

                // Clear Value first to hide this node from concurrent Lookups
                pShard->pNodes[ulTargetIdx].pValue = NULL;                
                pShard->ulActiveCount--;

                ExReleasePushLockExclusive(&pShard->lockPush);
                KeLeaveCriticalRegion();

                // Destruct Key and Value OUTSIDE the lock to prevent deadlocks
                pShard->pNodes[ulTargetIdx].tKey.~TKey();

                if (pEvictedValue)
                {
                    pEvictedValue->Release();
                }

                // Hold onto the node locally and restart the state machine
                ulReservedIdx = ulTargetIdx;

                continue;
            }

            // 3. Set metadata FIRST and only link the node LAST to prevent 
            // Lookup() from seeing an inconsistent or half-constructed node.
            pShard->pNodes[ulTargetIdx].ullHash = ullHash;
            new (&pShard->pNodes[ulTargetIdx].tKey) TKey(tKey);

            // Value is assigned AFTER the Key is fully constructed.
            pShard->pNodes[ulTargetIdx].pValue = pInValue;
            if (pInValue)
            {
                pInValue->AddRef();
            }

            // Link into the hash chain and LRU MRU list ONLY after the node 
            // is completely ready for a Lookup() thread to find.
            pShard->pNodes[ulTargetIdx].ulHashNext = pShard->pulBuckets[ulBucketIdx];
            pShard->pulBuckets[ulBucketIdx] = ulTargetIdx;

            PushMru(pShard, ulTargetIdx);

            ExReleasePushLockExclusive(&pShard->lockPush);
            KeLeaveCriticalRegion();

            return TRUE;
        }
    }

    // ------------------------------------------------------------------------
    // Lookup
    // Fast-path lookup using an exclusive lock. 
    // 
    // ARCHITECTURAL NOTE ON EXCLUSIVE LOCKING:
    // Counter-intuitively, this method acquires an EXCLUSIVE push lock immediately 
    // rather than a SHARED lock, even for highly read-skewed workloads. 
    // Because the critical section is microscopic (a few array index lookups and 
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
    // instructions and generates more NUMA interconnect traffic (cache line bouncing 
    // from atomic reader-count updates) than simply locking the code exclusively 
    // from the start. An exclusive lock guarantees immediate, branchless, in-place 
    // promotion without the hazard overhead.
    // ------------------------------------------------------------------------
    [[nodiscard]]
    BOOLEAN Lookup(_In_  const TKey& tKey,
                   _Out_ TValue*&    pOutValue) noexcept
    {
        PAGED_CODE();

        pOutValue = NULL;

        if (!m_pShards)
        {
            return FALSE;
        }

        UINT64 ullHash  = THasher::ComputeHash(tKey);
        UINT64 ullMixed = MixHash(ullHash);

        ULONG  ulShardIdx = (ULONG)(ullMixed & (m_ulShardCount - 1));
        Shard* pShard     = &m_pShards[ulShardIdx];

        UINT32 ulBucketIdx = (UINT32)(ullHash & pShard->ulBucketMask);
        
        // Exclusive lock acquisition
        KeEnterCriticalRegion();
        ExAcquirePushLockExclusive(&pShard->lockPush);

        UINT32 ulCurr = pShard->pulBuckets[ulBucketIdx];

        while (ulCurr != INVALID_INDEX)
        {
            // Prefetch the next link in the collision chain to hide memory 
            // latency while the CPU performs the Key comparison.
            UINT32 ulNext = pShard->pNodes[ulCurr].ulHashNext;

            if (ulNext != INVALID_INDEX)
            {
                CACHE_PREFETCH(&pShard->pNodes[ulNext]);
            }

            if (pShard->pNodes[ulCurr].ullHash == ullHash && pShard->pNodes[ulCurr].tKey == tKey)
            {
                TValue* pVal = pShard->pNodes[ulCurr].pValue;
                pVal->AddRef();
                pOutValue = pVal;

                // Optimized Threshold & Promotion Logic
                if (ulCurr != pShard->ulLruHead)
                {
                    // Check if node age exceeds precomputed shard threshold.
                    // Subtraction is safe against 64-bit wrap-around.
                    if ((pShard->ullGeneration - pShard->pNodes[ulCurr].ullLastPromoted) >= pShard->ullThresholdAge)
                    {
                        UnlinkLru(pShard, ulCurr);
                        PushMru(pShard, ulCurr);
                    }
                }

                ExReleasePushLockExclusive(&pShard->lockPush);
                KeLeaveCriticalRegion();

                return TRUE;
            }

            ulCurr = ulNext;
        }

        ExReleasePushLockExclusive(&pShard->lockPush);
        KeLeaveCriticalRegion();

        return FALSE;
    }

    // ------------------------------------------------------------------------
    // Remove
    // Manually purges a key from the cache. The node is completely detached,
    // destructed outside the lock, and then pushed to the FreeHead stack.
    // ------------------------------------------------------------------------    
    BOOLEAN Remove(_In_ const TKey& tKey) noexcept
    {
        PAGED_CODE();

        if (!m_pShards)
        {
            return FALSE;
        }

        UINT64 ullHash  = THasher::ComputeHash(tKey);
        UINT64 ullMixed = MixHash(ullHash);

        ULONG  ulShardIdx = (ULONG)(ullMixed & (m_ulShardCount - 1));
        Shard* pShard     = &m_pShards[ulShardIdx];

        UINT32 ulBucketIdx = (UINT32)(ullHash & pShard->ulBucketMask);

        KeEnterCriticalRegion();
        ExAcquirePushLockExclusive(&pShard->lockPush);

        UINT32 ulCurr = pShard->pulBuckets[ulBucketIdx];
        UINT32 ulPrev = INVALID_INDEX;

        while (ulCurr != INVALID_INDEX)
        {
            if (pShard->pNodes[ulCurr].ullHash == ullHash && pShard->pNodes[ulCurr].tKey == tKey)
            {
                if (ulPrev == INVALID_INDEX)
                {
                    pShard->pulBuckets[ulBucketIdx] = pShard->pNodes[ulCurr].ulHashNext;
                }
                else
                {
                    pShard->pNodes[ulPrev].ulHashNext = pShard->pNodes[ulCurr].ulHashNext;
                }

                UnlinkLru(pShard, ulCurr);

                TValue* pValueToRelease = pShard->pNodes[ulCurr].pValue;

                pShard->pNodes[ulCurr].pValue = NULL;
                pShard->ulActiveCount--;

                ExReleasePushLockExclusive(&pShard->lockPush);
                KeLeaveCriticalRegion();

                // Destruct actual key and payload OUTSIDE lock
                pShard->pNodes[ulCurr].tKey.~TKey();

                if (pValueToRelease)
                {
                    pValueToRelease->Release();
                }

                // Relock to push the clean node to the FreeList
                KeEnterCriticalRegion();
                ExAcquirePushLockExclusive(&pShard->lockPush);

                pShard->pNodes[ulCurr].ulHashNext = pShard->ulFreeHead;
                pShard->ulFreeHead = ulCurr;

                ExReleasePushLockExclusive(&pShard->lockPush);
                KeLeaveCriticalRegion();

                return TRUE;
            }

            ulPrev = ulCurr;
            ulCurr = pShard->pNodes[ulCurr].ulHashNext;
        }

        ExReleasePushLockExclusive(&pShard->lockPush);
        KeLeaveCriticalRegion();

        return FALSE;
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
    SIZE_T Trim(_In_ SIZE_T  szCount = 0,
                _In_ BOOLEAN bForce  = FALSE) noexcept
    {
        PAGED_CODE();

        if (!m_pShards)
        {
            return 0;
        }

        // If Count is 0, we enter "Watermark Mode" where the cache autonomously 
        // cleans itself down to a healthy percentage. Otherwise, we trim a specific number.
        BOOLEAN bTrimToWatermark = (szCount == 0);
        SIZE_T  uTotalTrimmed    = 0;

        for (ULONG ulIdx = 0; ulIdx < m_ulShardCount; ++ulIdx)
        {
            Shard* pShard = &m_pShards[ulIdx];

            // Define proportional watermarks based on the specific shard's capacity.
            // High Watermark (90%): The threshold where background trimming activates.
            // Low Watermark (85%): The target threshold we trim down to.
            // This 5% gap prevents "trim thrashing" (constantly trimming 1 item at 90%).
            SIZE_T uHighWatermark = (pShard->ulCapacity * 90) / 100;
            SIZE_T uLowWatermark  = (pShard->ulCapacity * 85) / 100;

            if (!bForce)
            {
                if (pShard->ulActiveCount < uHighWatermark)
                {
                    continue;
                }
            }

            while (pShard->ulActiveCount > 0 &&
                   (bForce || pShard->ulActiveCount > uLowWatermark) &&
                   (bTrimToWatermark || uTotalTrimmed < szCount))
            {
                KeEnterCriticalRegion();
                ExAcquirePushLockExclusive(&pShard->lockPush);

                if (pShard->ulActiveCount > 0 &&
                    (bForce || pShard->ulActiveCount > uLowWatermark))
                {
                    UINT32 ulTargetIdx = pShard->ulLruTail;

                    if (ulTargetIdx != INVALID_INDEX)
                    {
                        // Traverse backwards up the LRU chain to prefetch the 
                        // node that will become the NEW tail on the next loop.
                        UINT32 ulNextTail = pShard->pNodes[ulTargetIdx].ulLruPrev;

                        // Issue a software prefetch to the CPU. While we are busy 
                        // doing the math to detach the current tail
                        if (ulNextTail != INVALID_INDEX)
                        {
                            CACHE_PREFETCH(&pShard->pNodes[ulNextTail]);
                        }

                        RemoveFromHashChain(pShard, ulTargetIdx);
                        UnlinkLru(pShard, ulTargetIdx);

                        TValue* pValueToRelease = pShard->pNodes[ulTargetIdx].pValue;

                        pShard->pNodes[ulTargetIdx].pValue = NULL;                        
                        pShard->ulActiveCount--;

                        ExReleasePushLockExclusive(&pShard->lockPush);
                        KeLeaveCriticalRegion();

                        // Deferred destruction logic
                        pShard->pNodes[ulTargetIdx].tKey.~TKey();

                        if (pValueToRelease)
                        {
                            pValueToRelease->Release();
                        }

                        // Relock to return node to FreeList
                        KeEnterCriticalRegion();
                        ExAcquirePushLockExclusive(&pShard->lockPush);

                        pShard->pNodes[ulTargetIdx].ulHashNext = pShard->ulFreeHead;
                        pShard->ulFreeHead = ulTargetIdx;

                        ExReleasePushLockExclusive(&pShard->lockPush);
                        KeLeaveCriticalRegion();

                        uTotalTrimmed++;

                        continue;
                    }
                }

                ExReleasePushLockExclusive(&pShard->lockPush);
                KeLeaveCriticalRegion();

                break;
            }

            if (!bTrimToWatermark && uTotalTrimmed >= szCount)
            {
                break;
            }
        }

        return uTotalTrimmed;
    }

    // ------------------------------------------------------------------------
    // Enumerate
    // Safely iterates over all active items in the table, from Most Recently 
    // Used (MRU) to Least Recently Used (LRU) across all shards.
    // 
    // ARCHITECTURAL NOTE ON EXCLUSIVE LOCKING:
    // Just like Lookup(), we acquire an EXCLUSIVE push lock here. This avoids 
    // the atomic overhead and NUMA interconnect traffic associated with 
    // managing a shared reader count. 
    // 
    // WARNING: The callback is invoked at <= APC_LEVEL while holding an        
    // EXCLUSIVE lock on the current shard. The callback MUST NOT attempt 
    // to modify the table (Add/Remove/Trim) to prevent recursive deadlocks.    
    // 
    // I/O EXCEPTION: Heavy blocking I/O (e.g., FltWriteFile, ZwWriteFile) is 
    // strictly forbidden during live, concurrent driver operation. However, 
    // it is perfectly safe to perform synchronous disk I/O directly inside 
    // the callback ONLY if the table is being enumerated during a fully 
    // quiesced teardown phase (such as DriverUnload or InstanceTeardown) 
    // where no concurrent threads are accessing the cache.
    // ------------------------------------------------------------------------    
    template <typename TCallback>
    VOID Enumerate(_In_ const TCallback &Callback) noexcept
    {
        PAGED_CODE();

        if (!m_pShards)
        {
            return;
        }

        for (ULONG ulIdx = 0; ulIdx < m_ulShardCount; ++ulIdx)
        {
            Shard* pShard = &m_pShards[ulIdx];

            KeEnterCriticalRegion();
            ExAcquirePushLockExclusive(&pShard->lockPush);

            UINT32 ulCurr = pShard->ulLruHead;

            while (ulCurr != INVALID_INDEX)
            {
                // Fetch the index of the next chronological node and hint the 
                // CPU to load it while we process the user callback
                UINT32 ulNext = pShard->pNodes[ulCurr].ulLruNext;

                if (ulNext != INVALID_INDEX)
                {
                    CACHE_PREFETCH(&pShard->pNodes[ulNext]);
                }

                // The callback should return FALSE to abort the enumeration early
                if (!Callback(pShard->pNodes[ulCurr].tKey, pShard->pNodes[ulCurr].pValue))
                {
                    ExReleasePushLockExclusive(&pShard->lockPush);
                    KeLeaveCriticalRegion();

                    return;
                }

                ulCurr = ulNext;
            }

            ExReleasePushLockExclusive(&pShard->lockPush);
            KeLeaveCriticalRegion();
        }
    }
};
