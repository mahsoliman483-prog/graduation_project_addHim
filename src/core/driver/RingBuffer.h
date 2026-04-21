#pragma once

#include <ntddk.h>
#include "../PacketRecord.h"

#ifdef __cplusplus
extern "C" {
#endif

// Pushes a single PacketRecordV1 into the lock-free SPSC ring buffer.
// Returns true on success, false if the buffer is full.
bool RingBuffer_Push(SharedMemoryHeader* header, PacketRecordV1* buffer_start, const PacketRecordV1* pkt);

void RingBuffer_Init(PVOID baseAddress, ULONG totalSizeBytes);

#ifdef __cplusplus
}
#endif
