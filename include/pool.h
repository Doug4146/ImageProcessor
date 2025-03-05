#ifndef POOL_H
#define POOL_H

#include <stdint.h> // For type size_t


#ifdef _WIN32
    #define MEMORY_ALIGNMENT 32U  // For optimal SIMD handling (on Windows, MinGW)
#else
    #define MEMORY_ALIGNMENT 32  // For POSIX systems
#endif



/**
 * - Structure for representing a simple memory pool.
 * 
 * - Used to optimize memory allocation/deallocation, specifically for managing a preallocated memory block
 *   needed by `Window` structures in the convolution pipeline.
 * 
 * - The structure contains fields that represent the total sizes of the two memory regions, `poolSizeOne` 
 *   and `poolSizeTwo`, along with pointers to the next available memory blocks in each region, `nextFreeOne` 
 *   and `nextFreeTwo`, and pointers to the start of each preallocated memory region, `regionOne` and `regionTwo`.
 */
typedef struct MemoryPool {
        size_t poolSize;   // Size of the total memory region in bytes
        void *nextFree;    // Pointer to the next available memory location in the memory region
        void *memory;      // Pointer to the beginning location of the memory region
} MemoryPool;



// Aligns the input size to that of the maximum alignment requirment of the processor
size_t memory_size_alignment(size_t size);


// Creates and initializes a MemoryPool structure  of `desiredSize` size (before alignment) 
struct MemoryPool *init_memory_pool(size_t desiredSize);


// Allocates requestedSize (bytes) from the memory pool
void *allocate_from_pool(struct MemoryPool *pool, size_t requestedSize);


// Pushes the nextFree MemoryPool field pointer by the size of a previously allocated item
void free_from_pool(struct MemoryPool *pool, void *ptrToStart, size_t size);


// Pushes the nextFree MemoryPool field pointer back to the beginning of the memory pool
void empty_pool(struct MemoryPool *pool);


// Frees the memoryPool structure and its associated memory
void release_entire_memory_pool(struct MemoryPool *pool);



#endif //POOL_H