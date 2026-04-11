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

// ----------------------------------------------------------------------------
// CLASS PURPOSE
// CustomStringKey is a 24-byte string key optimized for use in LruHashTable, 
// enabling efficient packing within 64-byte nodes for improved cache locality. 
// It implements Small String Optimization (SSO) for strings up to 22 characters, 
// eliminating heap allocations entirely.
// For longer strings, it falls back to either a reference-counted heap 
// allocation or a zero-allocation borrowed pointer, both paired with a 
// cached 32-bit hash to accelerate lookups. The design allows SSO keys to be 
// compared and hashed using 64-bit block operations, while non-SSO keys benefit
// from cached hash-based filtering with fallback to string comparison with 
// minimal memory overhead.
// ----------------------------------------------------------------------------

#pragma once

#include <atomic>
#include <string_view>
#include <cstring>
#include <algorithm>
#include <memory>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4200)
#endif

// ----------------------------------------------------------------------------
// FLEXIBLE HEAP BUFFER
// Used exclusively for strings exceeding 22 characters.
// ----------------------------------------------------------------------------
struct SharedStringBuffer
{
    // Thread-safe reference count to manage the lifetime of the allocated heap block.
    // Initializes to 1 upon creation.
    std::atomic<uint32_t> Refs{ 1 };

    // The exact number of characters in the string, excluding the null terminator.
    uint32_t              Length;

    // Flexible array member containing the actual null-terminated string characters.
    // Memory for this array is allocated dynamically contiguous with the struct.
    char                  Data[];
};

// ----------------------------------------------------------------------------
// THE 24-BYTE BLOCK-OPTIMIZED KEY for perfect 64-byte LRUNode packing.
// Supports Small String Optimization (SSO), Ref-Counted Heap allocations, 
// and Zero-Allocation Unowned memory borrows for extreme performance.
// ----------------------------------------------------------------------------
class CustomStringKey
{
public:
    static constexpr uint16_t HEAP_FLAG    = 0xFFFF;
    static constexpr uint16_t UNOWNED_FLAG = 0xFFFE;

    union
    {
        // View 1: 2 bytes length + 22 chars capacity
        // Used when the string is small enough to fit directly inside the object (SSO).
        struct
        {
            // Indicates the current state of the key. 
            // Values 0-22 represent the length of an SSO string. 
            // Values 0xFFFF and 0xFFFE represent HEAP_FLAG and UNOWNED_FLAG respectively.
            uint16_t m_state;

            // Inline buffer for small strings. Avoids dynamic memory allocation entirely.
            char     m_sso_data[22];
        };

        // View 2: Heap Backed / Unowned Borrowed Pointer
        // Used when the string exceeds 22 characters or is explicitly borrowed.
        struct
        {
            // Shares the exact same memory address as m_state. 
            // Will contain either HEAP_FLAG or UNOWNED_FLAG.
            uint16_t m_magic;

            // Explicit padding to cleanly align the following 32-bit hash cache.
            uint16_t m_padding;

            // Precomputed 32-bit hash value. 
            // Caching this avoids expensive pointer chasing and string hashing during dictionary lookups.
            uint32_t m_hash_cache;

            union
            {
                // Pointer to a reference-counted, heap-allocated buffer. 
                // Active when m_magic == HEAP_FLAG.
                SharedStringBuffer* m_heap_ptr;

                // Raw pointer to externally owned memory (e.g., string literals or long-lived buffers).
                // Active when m_magic == UNOWNED_FLAG.
                const char* m_unowned_ptr;
            };

            // Explicit length tracker used ONLY when m_magic is UNOWNED_FLAG. 
            // Placed here (bytes 16-23) to utilize free space and prevent length truncation on huge strings.
            size_t m_unowned_length;
        };

        // View 3: The Fast-Math Blocks
        // Treats the entire 24-byte object as an array of three 64-bit integers.
        // Used for ultra-fast equality checks (memcmp equivalent) and fast hashing of SSO strings.
        uint64_t m_blocks[3];
    };

    struct Borrow {};

    // ------------------------------------------------------------------------
    // CONSTRUCTORS
    // ------------------------------------------------------------------------
    CustomStringKey() noexcept
    {
        m_blocks[0] = 0;
        m_blocks[1] = 0;
        m_blocks[2] = 0;
    }

    explicit CustomStringKey(std::string_view sv)
    {
        m_blocks[0] = 0;
        m_blocks[1] = 0;
        m_blocks[2] = 0;

        if (sv.length() <= 22)
        {
            m_state = static_cast<uint16_t>(sv.length());
            std::memcpy(m_sso_data, sv.data(), m_state);
        }
        else
        {
            m_magic      = HEAP_FLAG;
            m_padding    = 0;
            m_hash_cache = static_cast<uint32_t>(Compute32BitHash(sv));

            void* raw = ::operator new(sizeof(SharedStringBuffer) + sv.length() + 1);
            
            m_heap_ptr = std::construct_at(static_cast<SharedStringBuffer*>(raw));

            m_heap_ptr->Length = static_cast<uint32_t>(sv.length());
            std::memcpy(m_heap_ptr->Data, sv.data(), sv.length());
            m_heap_ptr->Data[sv.length()] = '\0';
        }
    }

    // Zero-allocation constructor for fast lookups
    CustomStringKey(std::string_view sv, Borrow) noexcept
    {
        m_blocks[0] = 0;
        m_blocks[1] = 0;
        m_blocks[2] = 0;

        if (sv.length() <= 22)
        {
            m_state = static_cast<uint16_t>(sv.length());
            std::memcpy(m_sso_data, sv.data(), m_state);
        }
        else
        {
            m_magic          = UNOWNED_FLAG;
            m_padding        = 0;
            m_hash_cache     = static_cast<uint32_t>(Compute32BitHash(sv));
            m_unowned_ptr    = sv.data();
            m_unowned_length = sv.length();
        }
    }

    // STRICTLY NOEXCEPT COPY CONSTRUCTOR - LRUHashTable requirement
    CustomStringKey(const CustomStringKey& other) noexcept
    {
        m_blocks[0] = other.m_blocks[0];
        m_blocks[1] = other.m_blocks[1];
        m_blocks[2] = other.m_blocks[2];

        if (m_state == HEAP_FLAG && m_heap_ptr)
        {
            m_heap_ptr->Refs.fetch_add(1, std::memory_order_relaxed);
        }
    }

    CustomStringKey& operator=(const CustomStringKey& other) noexcept
    {
        if (this != &other) [[likely]]
        {            
            std::destroy_at(this);
            std::construct_at(this, other);
        }

        return *this;
    }

    ~CustomStringKey() noexcept
    {
        if (m_state == HEAP_FLAG && m_heap_ptr)
        {
            if (m_heap_ptr->Refs.fetch_sub(1, std::memory_order_acq_rel) == 1)
            {                
                std::destroy_at(m_heap_ptr);
                ::operator delete(m_heap_ptr);
            }
        }
    }

    // ------------------------------------------------------------------------
    // METHODS
    // ------------------------------------------------------------------------
    [[nodiscard]] inline std::string_view GetView() const noexcept
    {
        if (m_state == HEAP_FLAG)
        {
            return std::string_view(m_heap_ptr->Data, m_heap_ptr->Length);
        }

        if (m_state == UNOWNED_FLAG)
        {
            return std::string_view(m_unowned_ptr, m_unowned_length);
        }

        return std::string_view(m_sso_data, m_state);
    }

    [[nodiscard]] inline bool operator==(const CustomStringKey& other) const noexcept
    {
        // Fast path for exact memory match (SSO vs SSO, or identical ptrs/hashes)
        if (m_blocks[0] == other.m_blocks[0] &&
            m_blocks[1] == other.m_blocks[1] &&
            m_blocks[2] == other.m_blocks[2])
        {
            return true;
        }

        // Fallback for logical equality across different memory locations
        return GetView() == other.GetView();
    }

    [[nodiscard]] static uint32_t Compute32BitHash(std::string_view sv) noexcept
    {
        const char* data = sv.data();
        size_t len = sv.size();

        // Seed with length to differentiate anagrams/shifted strings
        uint64_t hash = len * 0xcbf29ce484222325ULL;

        // Process 8 bytes at a time
        while (len >= 8)
        {
            uint64_t block;
            std::memcpy(&block, data, 8);

            hash ^= block;
            hash *= 0x100000001b3ULL;

            data += 8;
            len -= 8;
        }

        // Process the remaining 0-7 bytes
        if (len > 0) [[likely]]
        {
            uint64_t tail = 0;
            std::memcpy(&tail, data, len);
            hash ^= tail;
            hash *= 0x100000001b3ULL;
        }

        // Combine upper and lower bits into a highly entropic 32-bit result
        return static_cast<uint32_t>(hash ^ (hash >> 32));
    }
};

// ----------------------------------------------------------------------------
// FAST 64-BIT HASHER
// ----------------------------------------------------------------------------
struct FastStringHasher
{
    [[nodiscard]] static inline uint64_t ComputeHash(const CustomStringKey& key) noexcept
    {
        if (key.m_state == CustomStringKey::HEAP_FLAG || key.m_state == CustomStringKey::UNOWNED_FLAG)
        {
            // Expand the cached 32-bit hash back to 64-bit.            
            uint64_t h = key.m_hash_cache;
            return (h ^ (h << 32));
        }

        // Fast path for SSO completely resident in registers.
        uint64_t h = key.m_blocks[0];

        h ^= key.m_blocks[1] + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= key.m_blocks[2] + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);

        return h;
    }
};

#ifdef _MSC_VER
#pragma warning(pop)
#endif
