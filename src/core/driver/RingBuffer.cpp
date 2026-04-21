#include "RingBuffer.h"

void RingBuffer_Init(PVOID baseAddress, ULONG totalSizeBytes) {
    if (!baseAddress) return;

    RtlZeroMemory(baseAddress, totalSizeBytes);

    SharedMemoryHeader* header = (SharedMemoryHeader*)baseAddress;
    header->capacity = (totalSizeBytes - sizeof(SharedMemoryHeader)) / sizeof(PacketRecordV1);
    header->head = 0;
    header->tail = 0;
}

bool RingBuffer_Push(SharedMemoryHeader* header, PacketRecordV1* buffer_start, const PacketRecordV1* pkt) {
    if (!header || !buffer_start || !pkt) return false;

    // CRITICAL FIX: NEVER read 'capacity' from shared memory for bounds checking!
    // We hardcode the max capacity to prevent user-mode from causing a kernel out-of-bounds write.
    const UINT64 MAX_CAPACITY = ((1024 * 1024 * 16) - sizeof(SharedMemoryHeader)) / sizeof(PacketRecordV1);

    UINT64 safe_head = header->head % MAX_CAPACITY;
    UINT64 safe_tail = header->tail % MAX_CAPACITY;
    
    UINT64 next_head = (safe_head + 1) % MAX_CAPACITY;

    if (next_head == safe_tail) {
        InterlockedIncrement64(reinterpret_cast<volatile LONG64*>(&header->dropped_packets));
        return false;
    }

    // Now it is mathematically impossible to write out of bounds
    buffer_start[safe_head] = *pkt;
    KeMemoryBarrier();
    
    // Update the index safely
    header->head = next_head;

    return true;
}
