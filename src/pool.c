#include <stdio.h>
#include <stdlib.h>
#include <stdint.h> // For type size_t
#include <stdalign.h> // For alignof() and max_align_t
#include "pool.h"



size_t memory_size_alignment(size_t size) {

        // Maximum alignment requirement for processor
        size_t alignment = alignof(max_align_t);

        // Adjust size to be a multiple of alignment requirement and return
        return (size + alignment - 1) & ~(alignment - 1);

}


struct MemoryPool *init_memory_pool(size_t desiredSize) {

        // Align the desired memory region sizes to the processor's highest allignment requirment 
        size_t alignedSize = memory_size_alignment(desiredSize);

        // Create a MemoryPool struct
        struct MemoryPool *pool = (struct MemoryPool*)malloc(sizeof(struct MemoryPool));
        if (pool == NULL) {
                fprintf(stderr, "\nFatal error: memory ool could not be created.\n");
		return NULL;
        }

        // Allocate contiguous memory block for both memory regions
        pool->memory = (void*) malloc(alignedSize);
        if (pool->memory == NULL) {
                free(pool);
                fprintf(stderr, "\nFatal error: memory pool could not be created.\n");
		return NULL;
        }

        // Initialize the struct fields
        pool->poolSize = alignedSize;
        pool->nextFree = pool->memory;

        //Return pointer to created memoryPool
        return pool;

}


void *allocate_from_pool(struct MemoryPool *pool, size_t requestedSize) {

        // Align the requested memory size to the processor's alignment requirement
        size_t alignedSize = memory_size_alignment(requestedSize); 

        // Check if there is space availible for the requested memory size in the memory region
        if ((char*)pool->nextFree + alignedSize > (char*)pool->memory + pool->poolSize) {
                fprintf(stderr, "\nFatal error: requested memory exeeds the capacity of the memory pool.\n");
                return NULL;
        }

        // Address of the current availible memory block
        void *ptrToStart = pool->nextFree;

        // Move the nextFree pointer forward by the size of the allocated block
        pool->nextFree = (void*) ((char*)pool->nextFree + alignedSize);

        // Return a pointer to the beginning of the allocated memory block
        return ptrToStart;

}


void free_from_pool(struct MemoryPool *pool, void *ptrToStart, size_t size) {

        // Align the size of the block being freed to the processor's alignment requirement
        size_t alignedSize = memory_size_alignment(size); 

        // Move the nextFreeOne pointer backward to "free" the memory block in region one
        pool->nextFree = (void*) ((char*)pool->nextFree - alignedSize);

}


void empty_pool(struct MemoryPool *pool) {

        // Move the nextFreeOne pointer backward to "free" all currently allocated memory in the pool
        pool->nextFree = pool->memory;

}


void release_entire_memory_pool(struct MemoryPool *pool) {
        free(pool->memory);
        pool->memory = NULL;
        pool->nextFree = NULL;
        free(pool);
} 