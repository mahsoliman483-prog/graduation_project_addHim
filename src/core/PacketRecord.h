#pragma once
#include <ntddk.h>

/*
 * Memory Layout:
 * [SharedMemoryHeader] immediately followed by [PacketRecordV1 array]
 */

#pragma pack(push, 1)
typedef struct _SharedMemoryHeader {
    UINT32 schema_version;      // offset   0, 4 bytes
    UINT32 _pad0;               // offset   4, 4 bytes
    volatile UINT64 head;       // offset   8, 8 bytes  (alignas(64) removed: ignored inside #pragma pack(1))
    UINT8 pad1[56];             // offset  16, 56 bytes  -- cache-line isolation for head
    volatile UINT64 tail;       // offset  72, 8 bytes
    UINT8 pad2[56];             // offset  80, 56 bytes  -- cache-line isolation for tail
    volatile UINT64 capacity;   // offset 136, 8 bytes
    volatile UINT64 dropped_packets; // offset 144, 8 bytes
    UINT8 pad3[40];             // offset 152, 40 bytes  -- pads struct to exactly 192 bytes
} SharedMemoryHeader;
#pragma pack(pop)

// CRITICAL FIX 5.1 & 6.1: Python reads the packet array starting at byte 192.
// If this assert fires, sizeof() and Python's packet_array_offset=192 are out of sync.
static_assert(sizeof(SharedMemoryHeader) == 192, "SharedMemoryHeader MUST be exactly 192 bytes to match Python offset!");

// Exactly 64 bytes = 1 CPU Cache Line
struct alignas(64) PacketRecordV1 {
    UINT64 mono_ts_ns;
    UINT32 schema_version;
    UINT32 if_index;
    UINT32 captured_len;
    UINT32 wire_len;
    UINT16 src_port;
    UINT16 dst_port;
    UINT8 direction;
    UINT8 ip_version;
    UINT8 proto;
    UINT8 tcp_flags;
    UINT8 src_ip[16];
    UINT8 dst_ip[16];
};

static_assert(sizeof(PacketRecordV1) == 64, "PacketRecordV1 must be exactly 64 bytes");
