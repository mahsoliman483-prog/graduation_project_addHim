#include <ntddk.h>
#include <wdf.h>
#ifndef NDIS61
#define NDIS61 1
#endif
#include <ndis.h>
#include <initguid.h>
#include <fwpsk.h>
#include <fwpmk.h>
#pragma comment(lib, "ndis.lib")
#pragma comment(lib, "fwpkclnt.lib")
#include "../PacketRecord.h"
#include "RingBuffer.h"
#include "BlockEngine.h"

#pragma warning(push)
#pragma warning(disable: 4201) 

extern PVOID g_SharedMemoryKernelBase;
extern PKEVENT g_PacketEvent;

volatile LONG g_BatchCount = 0;
// Must be 8-byte aligned for InterlockedExchange64 -- alignment fault = BSOD
__declspec(align(8)) LARGE_INTEGER g_LastBatchRunTime = {0};

UINT32 calloutIds[4] = {0};
HANDLE g_EngineHandle = NULL; 
KSPIN_LOCK g_RingBufferLock;

EXTERN_C const GUID DECLSPEC_SELECTANY SEC_AI_CALLOUT_IN_V4  = { 0x11111111, 0x1111, 0x1111, { 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11 } };
EXTERN_C const GUID DECLSPEC_SELECTANY SEC_AI_CALLOUT_OUT_V4 = { 0x22222222, 0x2222, 0x2222, { 0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22 } };
EXTERN_C const GUID DECLSPEC_SELECTANY SEC_AI_CALLOUT_IN_V6  = { 0x33333333, 0x3333, 0x3333, { 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33 } };
EXTERN_C const GUID DECLSPEC_SELECTANY SEC_AI_CALLOUT_OUT_V6 = { 0x44444444, 0x4444, 0x4444, { 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44 } };

extern "C" void NTAPI ClassifyFn(
    const FWPS_INCOMING_VALUES0* inFixedValues,
    const FWPS_INCOMING_METADATA_VALUES0* inMetaValues,
    void* layerData,
    const FWPS_FILTER0* filter,
    UINT64 flowContext,
    FWPS_CLASSIFY_OUT0* classifyOut
);

#pragma alloc_text (NONPAGE, ClassifyFn)

UINT32 GetTcpMss(PNET_BUFFER_LIST nbl) {
    NDIS_TCP_LARGE_SEND_OFFLOAD_NET_BUFFER_LIST_INFO lsoInfo;
    lsoInfo.Value = NET_BUFFER_LIST_INFO(nbl, TcpLargeSendNetBufferListInfo);
    if (lsoInfo.Value != 0) {
        if (lsoInfo.LsoV2Transmit.Type == 1) return lsoInfo.LsoV2Transmit.MSS;
        else return lsoInfo.LsoV1Transmit.MSS;
    }
    return 0;
}

NTSTATUS NTAPI NotifyFn(
    FWPS_CALLOUT_NOTIFY_TYPE notifyType,
    const GUID* filterKey,
    FWPS_FILTER0* filter) {
    UNREFERENCED_PARAMETER(notifyType);
    UNREFERENCED_PARAMETER(filterKey);
    UNREFERENCED_PARAMETER(filter);
    return STATUS_SUCCESS;
}

void NTAPI ClassifyFn(
    const FWPS_INCOMING_VALUES0* inFixedValues,
    const FWPS_INCOMING_METADATA_VALUES0* inMetaValues,
    void* layerData,
    const FWPS_FILTER0* filter,
    UINT64 flowContext,
    FWPS_CLASSIFY_OUT0* classifyOut
) {
    UNREFERENCED_PARAMETER(inMetaValues);
    UNREFERENCED_PARAMETER(filter);
    UNREFERENCED_PARAMETER(flowContext);

    // Always set a default action FIRST so any early return is safe.
    // With FWP_ACTION_CALLOUT_INSPECTION this is not strictly required,
    // but it's critical correctness hygiene -- a missing action with a
    // TERMINATING callout causes an immediate bug check.
    bool canWriteAction = false;
    if (classifyOut && (classifyOut->rights & FWPS_RIGHT_ACTION_WRITE)) {
        canWriteAction = true;
        
        // CRITICAL FIX: Because this is now a TERMINATING filter, we MUST default to PERMIT.
        // We intentionally do NOT clear the FWPS_RIGHT_ACTION_WRITE flag here. 
        // This allows lower-priority filters (like Windows Defender Firewall) to still block the packet if they need to.
        classifyOut->actionType = FWP_ACTION_PERMIT; 
    }

    if (!layerData || !g_SharedMemoryKernelBase) return;

    PNET_BUFFER_LIST nbl = (PNET_BUFFER_LIST)layerData;
    PNET_BUFFER nb = NET_BUFFER_LIST_FIRST_NB(nbl);
    if (!nb) return;

    PacketRecordV1 pkt = {0};
    LARGE_INTEGER ts;
    KeQuerySystemTime(&ts);
    pkt.mono_ts_ns = ts.QuadPart * 100; 
    pkt.schema_version = 1;

    UINT32 packetLength = NET_BUFFER_DATA_LENGTH(nb);
    UINT32 mss = GetTcpMss(nbl);
    pkt.wire_len = (mss > 0 && mss < packetLength) ? mss : packetLength;
    pkt.captured_len = pkt.wire_len;

    // CRITICAL FIX 1.1 & 7.1: Extract protocol, IPs, and ports FIRST.
    // The old code checked pkt.proto == 6 for TCP flags BEFORE pkt.proto was assigned,
    // so tcp_flags was always 0. Additionally, src/dst IP and port were never populated at all.
    if (inFixedValues->layerId == FWPS_LAYER_INBOUND_TRANSPORT_V4) {
        pkt.ip_version = 4; pkt.direction = 1;
        pkt.proto   = inFixedValues->incomingValue[FWPS_FIELD_INBOUND_TRANSPORT_V4_IP_PROTOCOL].value.uint8;
        pkt.src_port = inFixedValues->incomingValue[FWPS_FIELD_INBOUND_TRANSPORT_V4_IP_REMOTE_PORT].value.uint16;
        pkt.dst_port = inFixedValues->incomingValue[FWPS_FIELD_INBOUND_TRANSPORT_V4_IP_LOCAL_PORT].value.uint16;
        // WFP delivers IPv4 addresses in host-byte order; swap to network-byte order for Python
        UINT32 srcIp = RtlUlongByteSwap(inFixedValues->incomingValue[FWPS_FIELD_INBOUND_TRANSPORT_V4_IP_REMOTE_ADDRESS].value.uint32);
        UINT32 dstIp = RtlUlongByteSwap(inFixedValues->incomingValue[FWPS_FIELD_INBOUND_TRANSPORT_V4_IP_LOCAL_ADDRESS].value.uint32);
        RtlCopyMemory(pkt.src_ip, &srcIp, 4);
        RtlCopyMemory(pkt.dst_ip, &dstIp, 4);
    } else if (inFixedValues->layerId == FWPS_LAYER_OUTBOUND_TRANSPORT_V4) {
        pkt.ip_version = 4; pkt.direction = 0;
        pkt.proto   = inFixedValues->incomingValue[FWPS_FIELD_OUTBOUND_TRANSPORT_V4_IP_PROTOCOL].value.uint8;
        pkt.src_port = inFixedValues->incomingValue[FWPS_FIELD_OUTBOUND_TRANSPORT_V4_IP_LOCAL_PORT].value.uint16;
        pkt.dst_port = inFixedValues->incomingValue[FWPS_FIELD_OUTBOUND_TRANSPORT_V4_IP_REMOTE_PORT].value.uint16;
        UINT32 srcIp = RtlUlongByteSwap(inFixedValues->incomingValue[FWPS_FIELD_OUTBOUND_TRANSPORT_V4_IP_LOCAL_ADDRESS].value.uint32);
        UINT32 dstIp = RtlUlongByteSwap(inFixedValues->incomingValue[FWPS_FIELD_OUTBOUND_TRANSPORT_V4_IP_REMOTE_ADDRESS].value.uint32);
        RtlCopyMemory(pkt.src_ip, &srcIp, 4);
        RtlCopyMemory(pkt.dst_ip, &dstIp, 4);
    } else if (inFixedValues->layerId == FWPS_LAYER_INBOUND_TRANSPORT_V6) {
        pkt.ip_version = 6; pkt.direction = 1;
        pkt.proto   = inFixedValues->incomingValue[FWPS_FIELD_INBOUND_TRANSPORT_V6_IP_PROTOCOL].value.uint8;
        pkt.src_port = inFixedValues->incomingValue[FWPS_FIELD_INBOUND_TRANSPORT_V6_IP_REMOTE_PORT].value.uint16;
        pkt.dst_port = inFixedValues->incomingValue[FWPS_FIELD_INBOUND_TRANSPORT_V6_IP_LOCAL_PORT].value.uint16;
        // IPv6 addresses from WFP are already in network-byte order
        RtlCopyMemory(pkt.src_ip, inFixedValues->incomingValue[FWPS_FIELD_INBOUND_TRANSPORT_V6_IP_REMOTE_ADDRESS].value.byteArray16->byteArray16, 16);
        RtlCopyMemory(pkt.dst_ip, inFixedValues->incomingValue[FWPS_FIELD_INBOUND_TRANSPORT_V6_IP_LOCAL_ADDRESS].value.byteArray16->byteArray16, 16);
    } else if (inFixedValues->layerId == FWPS_LAYER_OUTBOUND_TRANSPORT_V6) {
        pkt.ip_version = 6; pkt.direction = 0;
        pkt.proto   = inFixedValues->incomingValue[FWPS_FIELD_OUTBOUND_TRANSPORT_V6_IP_PROTOCOL].value.uint8;
        pkt.src_port = inFixedValues->incomingValue[FWPS_FIELD_OUTBOUND_TRANSPORT_V6_IP_LOCAL_PORT].value.uint16;
        pkt.dst_port = inFixedValues->incomingValue[FWPS_FIELD_OUTBOUND_TRANSPORT_V6_IP_REMOTE_PORT].value.uint16;
        RtlCopyMemory(pkt.src_ip, inFixedValues->incomingValue[FWPS_FIELD_OUTBOUND_TRANSPORT_V6_IP_LOCAL_ADDRESS].value.byteArray16->byteArray16, 16);
        RtlCopyMemory(pkt.dst_ip, inFixedValues->incomingValue[FWPS_FIELD_OUTBOUND_TRANSPORT_V6_IP_REMOTE_ADDRESS].value.byteArray16->byteArray16, 16);
    }

    // Now that pkt.proto is correctly populated, TCP flag extraction will work.
    pkt.tcp_flags = 0;
    if (packetLength >= 20 && pkt.proto == 6) { // 6 = IPPROTO_TCP
        UCHAR safeBuffer[20];
        PVOID pData = NdisGetDataBuffer(nb, sizeof(safeBuffer), safeBuffer, 1, 0);
        if (pData) {
            PUCHAR buffer = (PUCHAR)pData;
            pkt.tcp_flags = buffer[13];
        }
    }

    SharedMemoryHeader* header = (SharedMemoryHeader*)g_SharedMemoryKernelBase;
    PacketRecordV1* records = (PacketRecordV1*)(header + 1);

    KIRQL oldIrql;
    KeAcquireSpinLock(&g_RingBufferLock, &oldIrql);
    RingBuffer_Push(header, records, &pkt);
    KeReleaseSpinLock(&g_RingBufferLock, oldIrql);

    if (ShouldBlockPacket(&pkt)) {
        if (canWriteAction) {
            classifyOut->actionType = FWP_ACTION_BLOCK;
            classifyOut->rights &= ~FWPS_RIGHT_ACTION_WRITE;
        }
    }

    LONG currentCount = InterlockedIncrement(&g_BatchCount);
    LARGE_INTEGER currentTime;
    KeQuerySystemTime(&currentTime);

    // ATOMIC READ: InterlockedCompareExchange64 is the standard WDK idiom for an
    // atomic 64-bit read on ALL architectures including x86, where a plain
    // LONGLONG read at DISPATCH_LEVEL on multi-core is NOT guaranteed to be atomic.
    // Comperand=0, Exchange=0: only swaps if the current value is 0 (initial state),
    // which is harmless. In all other cases it is a pure read with no side-effects.
    LONGLONG lastRunTime = InterlockedCompareExchange64(
        (volatile LONGLONG*)&g_LastBatchRunTime.QuadPart, 0LL, 0LL);
    LONGLONG elapsed = currentTime.QuadPart - lastRunTime;

    if (currentCount >= 1024 || elapsed >= 50000) {
        if (g_PacketEvent) KeSetEvent(g_PacketEvent, 0, FALSE);
        InterlockedExchange(&g_BatchCount, 0);
        InterlockedExchange64(&g_LastBatchRunTime.QuadPart, currentTime.QuadPart);
    }
}

NTSTATUS RegisterWfpCallouts(PDEVICE_OBJECT deviceObject) {
    KeInitializeSpinLock(&g_RingBufferLock);

    FWPS_CALLOUT0 callout = {0};
    callout.classifyFn = (FWPS_CALLOUT_CLASSIFY_FN0)ClassifyFn;
    callout.notifyFn = (FWPS_CALLOUT_NOTIFY_FN0)NotifyFn;
    callout.flowDeleteFn = NULL;

    NTSTATUS status;

    // Guard each registration individually. On early return, calloutIds[i] for any
    // failed registration stays 0. DriverUnload already checks != 0 before
    // unregistering, so partial registration is safely cleaned up at unload.
    callout.calloutKey = SEC_AI_CALLOUT_IN_V4;
    status = FwpsCalloutRegister0(deviceObject, &callout, &calloutIds[0]);
    if (!NT_SUCCESS(status)) return status;

    callout.calloutKey = SEC_AI_CALLOUT_OUT_V4;
    status = FwpsCalloutRegister0(deviceObject, &callout, &calloutIds[1]);
    if (!NT_SUCCESS(status)) return status;

    callout.calloutKey = SEC_AI_CALLOUT_IN_V6;
    status = FwpsCalloutRegister0(deviceObject, &callout, &calloutIds[2]);
    if (!NT_SUCCESS(status)) return status;

    callout.calloutKey = SEC_AI_CALLOUT_OUT_V6;
    status = FwpsCalloutRegister0(deviceObject, &callout, &calloutIds[3]);

    return status;
}

NTSTATUS RegisterBfeFilters() {
    FWPM_SESSION0 session = {0};
    session.flags = FWPM_SESSION_FLAG_DYNAMIC; 
    
    NTSTATUS status = FwpmEngineOpen0(NULL, RPC_C_AUTHN_WINNT, NULL, &session, &g_EngineHandle);
    if (!NT_SUCCESS(status)) return status;

    FwpmTransactionBegin0(g_EngineHandle, 0);

    FWPM_CALLOUT0 mCallout = {0};
    FWPM_FILTER0 filter = {0};

    // --- IPv4 Inbound ---
    mCallout.calloutKey = SEC_AI_CALLOUT_IN_V4;
    mCallout.displayData.name = (wchar_t*)L"SecAI_IN_V4";
    mCallout.applicableLayer = FWPM_LAYER_INBOUND_TRANSPORT_V4;
    FwpmCalloutAdd0(g_EngineHandle, &mCallout, NULL, NULL);

    filter.displayData.name = (wchar_t*)L"SecAI_Filter_IN_V4"; // REQUIRED: NULL name can crash BFE
    filter.layerKey = FWPM_LAYER_INBOUND_TRANSPORT_V4;
    filter.action.type = FWP_ACTION_CALLOUT_TERMINATING; // Change from INSPECTION
    filter.action.calloutKey = SEC_AI_CALLOUT_IN_V4;
    filter.weight.type = FWP_EMPTY;
    FwpmFilterAdd0(g_EngineHandle, &filter, NULL, NULL);

    // --- IPv4 Outbound ---
    RtlZeroMemory(&mCallout, sizeof(mCallout));
    RtlZeroMemory(&filter, sizeof(filter));
    mCallout.calloutKey = SEC_AI_CALLOUT_OUT_V4;
    mCallout.displayData.name = (wchar_t*)L"SecAI_OUT_V4";
    mCallout.applicableLayer = FWPM_LAYER_OUTBOUND_TRANSPORT_V4;
    FwpmCalloutAdd0(g_EngineHandle, &mCallout, NULL, NULL);

    filter.displayData.name = (wchar_t*)L"SecAI_Filter_OUT_V4";
    filter.layerKey = FWPM_LAYER_OUTBOUND_TRANSPORT_V4;
    filter.action.type = FWP_ACTION_CALLOUT_TERMINATING; // Change from INSPECTION
    filter.action.calloutKey = SEC_AI_CALLOUT_OUT_V4;
    filter.weight.type = FWP_EMPTY;
    FwpmFilterAdd0(g_EngineHandle, &filter, NULL, NULL);

    // --- IPv6 Inbound ---
    RtlZeroMemory(&mCallout, sizeof(mCallout));
    RtlZeroMemory(&filter, sizeof(filter));
    mCallout.calloutKey = SEC_AI_CALLOUT_IN_V6;
    mCallout.displayData.name = (wchar_t*)L"SecAI_IN_V6";
    mCallout.applicableLayer = FWPM_LAYER_INBOUND_TRANSPORT_V6;
    FwpmCalloutAdd0(g_EngineHandle, &mCallout, NULL, NULL);

    filter.displayData.name = (wchar_t*)L"SecAI_Filter_IN_V6";
    filter.layerKey = FWPM_LAYER_INBOUND_TRANSPORT_V6;
    filter.action.type = FWP_ACTION_CALLOUT_TERMINATING; // Change from INSPECTION
    filter.action.calloutKey = SEC_AI_CALLOUT_IN_V6;
    filter.weight.type = FWP_EMPTY;
    FwpmFilterAdd0(g_EngineHandle, &filter, NULL, NULL);

    // --- IPv6 Outbound ---
    RtlZeroMemory(&mCallout, sizeof(mCallout));
    RtlZeroMemory(&filter, sizeof(filter));
    mCallout.calloutKey = SEC_AI_CALLOUT_OUT_V6;
    mCallout.displayData.name = (wchar_t*)L"SecAI_OUT_V6";
    mCallout.applicableLayer = FWPM_LAYER_OUTBOUND_TRANSPORT_V6;
    FwpmCalloutAdd0(g_EngineHandle, &mCallout, NULL, NULL);

    filter.displayData.name = (wchar_t*)L"SecAI_Filter_OUT_V6";
    filter.layerKey = FWPM_LAYER_OUTBOUND_TRANSPORT_V6;
    filter.action.type = FWP_ACTION_CALLOUT_TERMINATING; // Change from INSPECTION
    filter.action.calloutKey = SEC_AI_CALLOUT_OUT_V6;
    filter.weight.type = FWP_EMPTY;
    FwpmFilterAdd0(g_EngineHandle, &filter, NULL, NULL);

    FwpmTransactionCommit0(g_EngineHandle);
    return STATUS_SUCCESS;
}

void UnregisterBfeFilters() {
    if (g_EngineHandle) {
        FwpmEngineClose0(g_EngineHandle); 
        g_EngineHandle = NULL;
    }
}
#pragma warning(pop)