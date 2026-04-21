#include <ntddk.h>
#include "BlockEngine.h"

// MAX_BLOCK_RULES is now defined in BlockEngine.h

BlockRuleV1 g_BlockRules[MAX_BLOCK_RULES];

void BlockEngine_Init() {
    RtlZeroMemory(g_BlockRules, sizeof(g_BlockRules));
}

NTSTATUS BlockEngine_AddRule(const BlockRuleV1* newRule) {
    if (!newRule) return STATUS_INVALID_PARAMETER;

    LARGE_INTEGER currentTime;
    KeQuerySystemTime(&currentTime);

    for (int i = 0; i < MAX_BLOCK_RULES; ++i) {
        BlockRuleV1* rule = &g_BlockRules[i];
        
        LONG active = InterlockedCompareExchange(&rule->is_active, 0, 0);
        bool expired = false;

        if (active == 1) {
            UINT64 expirationTime = rule->timestamp_added + (rule->ttl_ms * 10000ULL); 
            if ((UINT64)currentTime.QuadPart > expirationTime) {
                expired = true;
            }
        }

        if (active == 0 || expired) {
            if (InterlockedCompareExchange(&rule->is_active, 2, active) == active) {
                rule->ip_version = newRule->ip_version;
                rule->proto = newRule->proto;
                rule->src_port = newRule->src_port;
                rule->dst_port = newRule->dst_port;
                rule->ttl_ms = newRule->ttl_ms;
                RtlCopyMemory(rule->src_ip, newRule->src_ip, 16);
                RtlCopyMemory(rule->dst_ip, newRule->dst_ip, 16);
                
                KeQuerySystemTime(&currentTime);
                rule->timestamp_added = (UINT64)currentTime.QuadPart;

                KeMemoryBarrier();
                InterlockedExchange(&rule->is_active, 1);
                return STATUS_SUCCESS;
            }
        }
    }
    return STATUS_INSUFFICIENT_RESOURCES; 
}

bool ShouldBlockPacket(const PacketRecordV1* pkt) {
    if (!pkt) return false;

    LARGE_INTEGER currentTime;
    KeQuerySystemTime(&currentTime);

    for (int i = 0; i < MAX_BLOCK_RULES; ++i) {
        BlockRuleV1* rule = &g_BlockRules[i];

        // Volatile read prevents CPU bus stalling under massive network load
        if (*(volatile LONG*)&rule->is_active != 1) continue; 

        UINT64 expirationTime = rule->timestamp_added + (rule->ttl_ms * 10000ULL);
        if ((UINT64)currentTime.QuadPart > expirationTime) {
            InterlockedCompareExchange(&rule->is_active, 0, 1);
            continue;
        }

        if (rule->ip_version != 0 && rule->ip_version != pkt->ip_version) continue;
        if (rule->proto != 0 && rule->proto != pkt->proto) continue;
        if (rule->src_port != 0 && rule->src_port != pkt->src_port) continue;
        if (rule->dst_port != 0 && rule->dst_port != pkt->dst_port) continue;
        
        bool srcIpAny = true, dstIpAny = true;
        bool srcIpMatch = true, dstIpMatch = true;
        for (int j = 0; j < 16; ++j) {
            if (rule->src_ip[j] != 0) srcIpAny = false;
            if (rule->dst_ip[j] != 0) dstIpAny = false;
            if (rule->src_ip[j] != pkt->src_ip[j]) srcIpMatch = false;
            if (rule->dst_ip[j] != pkt->dst_ip[j]) dstIpMatch = false;
        }

        if (!srcIpAny && !srcIpMatch) continue;
        if (!dstIpAny && !dstIpMatch) continue;

        return true; 
    }
    return false;
}

// ---------------------------------------------------------------------------
// BlockEngine_GetRules
// ---------------------------------------------------------------------------
// Copies every is_active == 1 rule into the caller-supplied buffer.
// The caller MUST supply an output buffer large enough for MAX_BLOCK_RULES
// entries; Driver.cpp ensures this via WdfRequestRetrieveOutputBuffer.
// Returns the number of rules actually written (may be 0).
//
// Safety notes:
//  * We use a volatile read of is_active so the compiler cannot hoist the
//    load out of the loop.
//  * We snapshot fields *after* confirming is_active == 1 and *before*
//    publishing the entry to the caller.  If a concurrent RemoveRule sets
//    is_active to 0 after our check, the caller receives a stale-but-valid
//    copy -- acceptable for a management query.
//  * We operate entirely on kernel-mode stack/pool; no user-mode probing
//    is needed here (Driver.cpp owns the buffer lifetime).
// ---------------------------------------------------------------------------
ULONG BlockEngine_GetRules(
    _Out_writes_(outCapacity) BlockRuleV1* outBuffer,
    _In_ ULONG outCapacity)
{
    if (!outBuffer || outCapacity == 0) return 0;

    ULONG count = 0;

    for (int i = 0; i < MAX_BLOCK_RULES && count < outCapacity; ++i) {
        BlockRuleV1* src = &g_BlockRules[i];

        // Volatile read: prevents the compiler from caching is_active across
        // iterations.  This is the same pattern used in ShouldBlockPacket.
        LONG active = *(volatile LONG*)&src->is_active;
        if (active != 1) continue;

        // Expiry check -- mirror the ShouldBlockPacket logic so the caller
        // only sees rules that are still live.
        LARGE_INTEGER now;
        KeQuerySystemTime(&now);
        UINT64 expirationTime = src->timestamp_added + (src->ttl_ms * 10000ULL);
        if ((UINT64)now.QuadPart > expirationTime) {
            // Opportunistically retire this rule (best-effort, no retry
            // needed -- ShouldBlockPacket will also retire it in-line).
            InterlockedCompareExchange(&src->is_active, 0, 1);
            continue;
        }

        // RtlCopyMemory is safe: src is in our own non-paged kernel array,
        // outBuffer points into a WDF-managed output buffer (non-paged).
        RtlCopyMemory(&outBuffer[count], src, sizeof(BlockRuleV1));
        ++count;
    }

    return count;
}

// ---------------------------------------------------------------------------
// BlockEngine_RemoveRule
// ---------------------------------------------------------------------------
// Finds the first active rule matching dstPort and deactivates it atomically.
//
// Safety notes:
//  * InterlockedCompareExchange(target, 0, 1) only succeeds when the cell
//    is still active (== 1).  If ShouldBlockPacket or AddRule wins the race
//    and changes is_active first, our CAS fails harmlessly and we continue
//    scanning -- no retry spin is needed because each slot is independent.
//  * dstPort == 0 is treated as a wildcard-remove safety guard: we refuse
//    it to avoid accidentally wiping all "any-port" rules.
// ---------------------------------------------------------------------------
NTSTATUS BlockEngine_RemoveRule(_In_ UINT16 dstPort)
{
    if (dstPort == 0) return STATUS_INVALID_PARAMETER;

    for (int i = 0; i < MAX_BLOCK_RULES; ++i) {
        BlockRuleV1* rule = &g_BlockRules[i];

        // Volatile read so the compiler does not skip the check.
        LONG active = *(volatile LONG*)&rule->is_active;
        if (active != 1) continue;

        if (rule->dst_port != dstPort) continue;

        // Atomically transition 1 -> 0.  If we lose the race (e.g.,
        // ShouldBlockPacket expired the rule and already set it to 0),
        // the CAS returns the old value which != 1, so we keep scanning.
        if (InterlockedCompareExchange(&rule->is_active, 0, 1) == 1) {
            // Ensure the cleared state is visible to all CPUs before we
            // return -- especially important for ShouldBlockPacket running
            // concurrently on other cores.
            KeMemoryBarrier();
            return STATUS_SUCCESS;
        }
    }

    return STATUS_NOT_FOUND;
}