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

#pragma once

#include <cstdint>
#include <list>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <type_traits>

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

namespace LruDetail
{
    // ------------------------------------------------------------------------
    // Type trait to detect if the TMutex type supports shared locking 
    // (e.g., std::shared_mutex, std::shared_timed_mutex).
    // ------------------------------------------------------------------------
    template <typename T, 
              typename = void>
    struct supports_shared_lock : std::false_type
    {};

    template <typename T>
    struct supports_shared_lock<T, std::void_t<decltype(std::declval<T>().lock_shared())>> : std::true_type
    {};

    template <typename T>
    inline constexpr bool supports_shared_lock_v = supports_shared_lock<T>::value;
}

// ----------------------------------------------------------------------------
// Textbook Thread-Safe, Exception-Safe LRU Hash Table
// ----------------------------------------------------------------------------
template <typename TKey, 
          typename TValue, 
          typename THasher = std::hash<TKey>, 
          typename TMutex  = std::shared_mutex>
class StdLruHashTable
{
    // Enforce that move constructor is noexcept
    static_assert(std::is_nothrow_move_constructible<TKey>::value,
                  "TKey should be noexcept move constructible for strong exception safety");

    // Enforce that AddRef is strictly noexcept
    static_assert(noexcept(std::declval<TValue*>()->AddRef()),
                  "TValue::AddRef() MUST be declared noexcept to prevent state corruption during cache overwrites");

    // Enforce that Release is strictly noexcept
    static_assert(noexcept(std::declval<TValue*>()->Release()),
                  "TValue::Release() MUST be declared noexcept to prevent permanent capacity leaks during node eviction");

private:
    using LruList = std::list<std::pair<TKey, TValue*>>;
    using LruMap  = std::unordered_map<TKey, typename LruList::iterator, THasher>;

    mutable TMutex      m_Lock;
    LruList             m_Items;
    LruMap              m_Map;
    std::atomic<size_t> m_Capacity;

public:
    StdLruHashTable() noexcept : m_Capacity(0)
    {}

    ~StdLruHashTable() noexcept
    {
        // Destructors must not throw. If Cleanup's lock fails, swallow it 
        // to prevent std::terminate during stack unwinding.
        try
        {
            Cleanup();
        }
        catch (...)
        {
        }
    }

    [[nodiscard]]
    bool Initialize(_In_ const size_t TotalEntries)
    {
        Cleanup();

        std::unique_lock<TMutex> lock(m_Lock);

        size_t boundedCapacity = TotalEntries;

        if (boundedCapacity < 8)
        {
            boundedCapacity = 8;
        }

        m_Capacity.store(boundedCapacity, std::memory_order_relaxed);

        try
        {
            m_Map.reserve(boundedCapacity);
        }
        catch (...)
        {
            m_Capacity.store(0, std::memory_order_relaxed);

            return false;
        }

        return true;
    }

    [[nodiscard]]
    size_t GetTotalMemoryUsage() const
    {
        size_t totalBytes = sizeof(*this);

        if constexpr (LruDetail::supports_shared_lock_v<TMutex>)
        {
            std::shared_lock<TMutex> lock(m_Lock);

            totalBytes += m_Map.bucket_count() * sizeof(void*);
            totalBytes += m_Map.size() * (sizeof(void*) * 2 + sizeof(TKey) + sizeof(typename LruList::iterator));
            totalBytes += m_Items.size() * (sizeof(void*) * 2 + sizeof(TKey) + sizeof(TValue*));
        }
        else
        {
            std::lock_guard<TMutex> lock(m_Lock);

            totalBytes += m_Map.bucket_count() * sizeof(void*);
            totalBytes += m_Map.size() * (sizeof(void*) * 2 + sizeof(TKey) + sizeof(typename LruList::iterator));
            totalBytes += m_Items.size() * (sizeof(void*) * 2 + sizeof(TKey) + sizeof(TValue*));
        }

        return totalBytes;
    }

    [[nodiscard]]
    size_t GetTotalItemCount() const
    {
        if constexpr (LruDetail::supports_shared_lock_v<TMutex>)
        {
            std::shared_lock<TMutex> lock(m_Lock);
            return m_Map.size();
        }
        else
        {
            std::lock_guard<TMutex> lock(m_Lock);
            return m_Map.size();
        }
    }

    void Cleanup()
    {
        LruList itemsToRelease;

        // Steal the list contents within the lock, then drop the lock.
        // This ensures Release() cannot cause deadlocks or lock inversion.
        {
            std::unique_lock<TMutex> lock(m_Lock);

            itemsToRelease = std::move(m_Items);
            m_Map.clear();
            m_Capacity.store(0, std::memory_order_relaxed);
        }

        for (auto& item : itemsToRelease)
        {
            if (item.second)
            {
                item.second->Release();
            }
        }
    }

    [[nodiscard]]
    bool Add(_In_     const TKey& Key,
             _In_opt_ TValue* Value)
    {
        // Safe lock-free early out
        if (m_Capacity.load(std::memory_order_relaxed) == 0) [[unlikely]]
        {
            return false;
        }

        TValue* valueToRelease = nullptr;
        TValue* evictedValue   = nullptr;
        bool    success        = true;

        try
        {
            std::unique_lock<TMutex> lock(m_Lock);

            auto it = m_Map.find(Key);

            // 1. Check for Overwrite
            if (it != m_Map.end()) [[unlikely]]
            {
                valueToRelease = it->second->second;

                it->second->second = Value;

                if (Value) [[likely]]
                {
                    Value->AddRef();
                }

                // Splice to MRU
                if (it->second != m_Items.begin())
                {
                    m_Items.splice(m_Items.begin(), m_Items, it->second);
                }
            }
            else
            {
                // 2. Cache Miss: Insert new item safely FIRST
                m_Items.emplace_front(Key, Value);

                try
                {
                    m_Map.emplace(Key, m_Items.begin());
                }
                catch (...)
                {
                    // Rollback list insertion if map allocation fails
                    m_Items.pop_front();
                    throw;
                }

                if (Value) [[likely]]
                {
                    Value->AddRef();
                }

                // 3. Evict if at capacity (Only evaluated after a successful insertion)
                if (m_Map.size() > m_Capacity.load(std::memory_order_relaxed)) [[likely]]
                {
                    if (!m_Items.empty()) [[likely]]
                    {
                        auto last = m_Items.end();
                        --last;

                        evictedValue = last->second;

                        m_Map.erase(last->first);
                        m_Items.pop_back();
                    }
                }
            }
        }
        catch (...)
        {
            success = false;
        }

        // Lock is fully released here before triggering reference decrements
        if (valueToRelease)
        {
            valueToRelease->Release();
        }

        if (evictedValue) [[likely]]
        {
            evictedValue->Release();
        }

        return success;
    }

    [[nodiscard]]
    bool Lookup(_In_  const TKey& Key,
                _Out_ TValue*&    OutValue)
    {
        OutValue = nullptr;

        if (m_Capacity.load(std::memory_order_relaxed) == 0) [[unlikely]]
        {
            return false;
        }

        if constexpr (LruDetail::supports_shared_lock_v<TMutex>)
        {
            // SHARED MUTEX PATH (Lock Upgrade & ABA Mitigation)
            {
                std::shared_lock<TMutex> sharedLock(m_Lock);

                auto it = m_Map.find(Key);

                if (it != m_Map.end()) [[likely]]
                {
                    if (it->second->second) [[likely]]
                    {
                        // Secures memory against ABA recycling during lock upgrade
                        it->second->second->AddRef();
                    }

                    OutValue = it->second->second;

                    // Fast Path: Already at MRU
                    if (it->second == m_Items.begin()) [[unlikely]]
                    {
                        return true;
                    }
                }
                else
                {
                    return false;
                }
            }

            // Slow Path: Needs MRU promotion
            {
                std::unique_lock<TMutex> exclusiveLock(m_Lock);

                auto verifyIt = m_Map.find(Key);

                // Safe pointer comparison due to AddRef() above
                if (verifyIt != m_Map.end() && verifyIt->second->second == OutValue) [[likely]]
                {
                    if (verifyIt->second != m_Items.begin()) [[likely]]
                    {
                        m_Items.splice(m_Items.begin(), m_Items, verifyIt->second);
                    }

                    return true;
                }
            }

            // Phantom Hit: The node was evicted and replaced during the lock upgrade gap
            if (OutValue) [[likely]]
            {
                OutValue->Release();
                OutValue = nullptr;
            }

            return false;
        }
        else
        {
            // EXCLUSIVE MUTEX PATH
            std::lock_guard<TMutex> lock(m_Lock);

            auto it = m_Map.find(Key);

            if (it != m_Map.end()) [[likely]]
            {
                OutValue = it->second->second;

                if (OutValue) [[likely]]
                {
                    OutValue->AddRef();
                }

                // MRU promotion
                if (it->second != m_Items.begin()) [[likely]]
                {
                    m_Items.splice(m_Items.begin(), m_Items, it->second);
                }

                return true;
            }

            return false;
        }
    }

    [[nodiscard]]
    bool Remove(_In_ const TKey& Key)
    {
        if (m_Capacity.load(std::memory_order_relaxed) == 0) [[unlikely]]
        {
            return false;
        }

        TValue* valueToRelease = nullptr;

        {
            std::unique_lock<TMutex> lock(m_Lock);

            auto it = m_Map.find(Key);

            if (it != m_Map.end()) [[likely]]
            {
                valueToRelease = it->second->second;

                m_Items.erase(it->second);
                m_Map.erase(it);
            }
            else
            {
                return false;
            }
        }

        if (valueToRelease) [[likely]]
        {
            valueToRelease->Release();
        }

        return true;
    }

    size_t Trim(_In_ const size_t Count)
    {
        if (m_Capacity.load(std::memory_order_relaxed) == 0 || Count == 0) [[unlikely]]
        {
            return 0;
        }

        size_t totalTrimmed = 0;

        while (totalTrimmed < Count)
        {
            TValue* valueToRelease = nullptr;
            bool    nodeTrimmed = false;

            {
                std::unique_lock<TMutex> lock(m_Lock);

                if (!m_Items.empty()) [[likely]]
                {
                    auto last = m_Items.end();
                    --last;

                    valueToRelease = last->second;

                    m_Map.erase(last->first);
                    m_Items.pop_back();

                    totalTrimmed++;
                    nodeTrimmed = true;
                }
            }

            if (nodeTrimmed) [[likely]]
            {
                if (valueToRelease) [[likely]]
                {
                    valueToRelease->Release();
                }
            }
            else [[unlikely]]
            {
                break;
            }
        }

        return totalTrimmed;
    }
};