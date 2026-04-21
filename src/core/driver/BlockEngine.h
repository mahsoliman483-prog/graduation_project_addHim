#pragma once

#define MAX_BLOCK_RULES 1024
#include <ntddk.h>
#ifndef NDIS61
#define NDIS61 1
#endif
#include <ndis.h>
#include <fwpsk.h>
#include "../PacketRecord.h"

// Explicit padding and byte alignment for clarity
#pragma pack(push, 1)
struct BlockRuleV1 {
    UINT8 ip_version;
    UINT8 proto;
    UINT8 src_ip[16];
    UINT8 dst_ip[16];
    UINT16 src_port;
    UINT16 dst_port;
    UINT8 _pad0[2]; // FIX: Pushes the next 64-bit int to a clean 8-byte boundary
    UINT64 ttl_ms;
    UINT64 timestamp_added;
    volatile LONG is_active;
    UINT8 _pad1[4]; // FIX: Pushes the total struct size to an even 64 bytes
};
#pragma pack(pop)

// Initializes the Block Engine table
void BlockEngine_Init();

// Handles User-Mode IOCTL insertions safely using RCU-style interlocking
NTSTATUS BlockEngine_AddRule(const BlockRuleV1* newRule);

// ClassifyFn evaluation hook
bool ShouldBlockPacket(const PacketRecordV1* pkt);

// Copies all is_active == 1 rules into outBuffer (which must hold at least
// outCapacity entries).  Returns the number of rules written.
ULONG BlockEngine_GetRules(_Out_writes_(outCapacity) BlockRuleV1* outBuffer,
                           _In_ ULONG outCapacity);

// Atomically deactivates the first active rule whose dst_port matches.
// Returns STATUS_SUCCESS if a rule was found and deactivated,
// STATUS_NOT_FOUND otherwise.
NTSTATUS BlockEngine_RemoveRule(_In_ UINT16 dstPort);
