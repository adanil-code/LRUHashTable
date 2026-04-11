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

// No TEST_IS_KM defined for User Mode
#include "lru_hash_table.h"
#include "test_lru_hash_common.h"

int main()
{
    std::cout << "\n";
    std::cout << "=========================================================\n";
    std::cout << "         USER-MODE LRU HASH TABLE TEST SUITE             \n";
    std::cout << "=========================================================\n";

#ifdef LRU_USE_EXPONENTIAL_BACKOFF
    using CustomTestTable    = LruHashTable<uint64_t, RefCountedPayload, Hasher64Bit, DefaultNumaAllocator, ExponentialBackoffPolicy>;
    using CollisionTestTable = LruHashTable<uint64_t, RefCountedPayload, DegradedHasher, DefaultNumaAllocator, ExponentialBackoffPolicy>;
#else
    using CustomTestTable    = LruHashTable<uint64_t, RefCountedPayload, Hasher64Bit>;
    using CollisionTestTable = LruHashTable<uint64_t, RefCountedPayload, DegradedHasher>;
#endif

    return RunAllTests<CustomTestTable, CollisionTestTable>("Custom Array Table");
}