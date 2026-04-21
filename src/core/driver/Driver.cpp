#include <ntddk.h>
#include <wdf.h>
#ifndef NDIS61
#define NDIS61 1
#endif
#include <ndis.h>
#include <fwpsk.h>
#include "Ioctls.h"
#include "RingBuffer.h"
#include "BlockEngine.h"

PVOID g_SharedMemoryKernelBase = NULL;
PMDL g_SharedMemoryMdl = NULL;

PKEVENT g_PacketEvent = NULL;
HANDLE g_PacketEventHandle = NULL;

PVOID g_SharedMemoryUserBase = NULL;
extern UINT32 calloutIds[4];

const ULONG SHARED_MEMORY_SIZE = 1024 * 1024 * 16; 

extern "C" DRIVER_INITIALIZE DriverEntry;
extern "C" EVT_WDF_DRIVER_UNLOAD DriverUnload;

extern NTSTATUS RegisterWfpCallouts(PDEVICE_OBJECT deviceObject);
extern NTSTATUS RegisterBfeFilters();
extern void UnregisterBfeFilters();

PVOID MapSharedMemoryToUserSpace() {
    if (!g_SharedMemoryMdl) return NULL;
    PVOID userBase = NULL;
    __try {
        userBase = MmMapLockedPagesSpecifyCache(
            g_SharedMemoryMdl, UserMode, MmCached, NULL, FALSE, NormalPagePriority | MdlMappingNoExecute
        );
        g_SharedMemoryUserBase = userBase;
    } __except(EXCEPTION_EXECUTE_HANDLER) {
        userBase = NULL;
    }
    return userBase;
}

NTSTATUS InitializeSharedMemory() {
    PHYSICAL_ADDRESS lowAddress, highAddress, skipBytes;
    lowAddress.QuadPart = 0; highAddress.QuadPart = -1; skipBytes.QuadPart = 0;

    g_SharedMemoryMdl = MmAllocatePagesForMdl(lowAddress, highAddress, skipBytes, SHARED_MEMORY_SIZE);
    if (!g_SharedMemoryMdl) return STATUS_INSUFFICIENT_RESOURCES;

    g_SharedMemoryKernelBase = MmGetSystemAddressForMdlSafe(g_SharedMemoryMdl, NormalPagePriority | MdlMappingNoExecute);
    if (!g_SharedMemoryKernelBase) {
        MmFreePagesFromMdl(g_SharedMemoryMdl);
        IoFreeMdl(g_SharedMemoryMdl);
        g_SharedMemoryMdl = NULL;
        return STATUS_INSUFFICIENT_RESOURCES;
    }

    RingBuffer_Init(g_SharedMemoryKernelBase, SHARED_MEMORY_SIZE);
    return STATUS_SUCCESS;
}

NTSTATUS InitializePacketEvent() {
    UNICODE_STRING eventName;
    RtlInitUnicodeString(&eventName, L"\\BaseNamedObjects\\SecAIPacketEvent");

    g_PacketEvent = IoCreateNotificationEvent(&eventName, &g_PacketEventHandle);
    if (!g_PacketEvent) return STATUS_UNSUCCESSFUL;
    
    KeClearEvent(g_PacketEvent);
    return STATUS_SUCCESS;
}

VOID DriverUnload(_In_ WDFDRIVER Driver) {
    UNREFERENCED_PARAMETER(Driver);

    // STEP 1: Remove BFE filters so no new ClassifyFn calls are dispatched
    UnregisterBfeFilters();

    // STEP 2: Unregister WFP callouts and WAIT for all in-flight ClassifyFn
    //         calls to fully drain. This MUST happen before freeing shared
    //         memory, otherwise a racing ClassifyFn will write to freed
    //         memory -> BSOD (bug check 0xD1 or 0x50).
    for (int i = 0; i < 4; i++) {
        if (calloutIds[i] != 0) {
            FwpsCalloutUnregisterById0(calloutIds[i]);
            calloutIds[i] = 0;
        }
    }

    // STEP 3: Now safe to free shared memory -- no callout can touch it
    if (g_SharedMemoryMdl) {
        MmFreePagesFromMdl(g_SharedMemoryMdl);
        IoFreeMdl(g_SharedMemoryMdl);
        g_SharedMemoryMdl = NULL;
        g_SharedMemoryKernelBase = NULL;
    }

    // STEP 4: Close event handle
    if (g_PacketEventHandle) {
        ZwClose(g_PacketEventHandle);
        g_PacketEventHandle = NULL;
        g_PacketEvent = NULL;
    }
}

VOID EvtIoDeviceControl(_In_ WDFQUEUE Queue, _In_ WDFREQUEST Request, _In_ size_t OutputBufferLength, _In_ size_t InputBufferLength, _In_ ULONG IoControlCode) {
    UNREFERENCED_PARAMETER(Queue);
    // InputBufferLength is used by IOCTL_REMOVE_BLOCK_RULE -- do NOT mark unreferenced
    NTSTATUS status = STATUS_INVALID_DEVICE_REQUEST;
    ULONG_PTR information = 0;

    switch (IoControlCode) {
        case IOCTL_START_CAPTURE: {
            if (OutputBufferLength < sizeof(PVOID)) {
                status = STATUS_BUFFER_TOO_SMALL;
                break;
            }
            PVOID userBase = MapSharedMemoryToUserSpace();
            if (userBase) {
                PVOID outBuffer = NULL;
                status = WdfRequestRetrieveOutputBuffer(Request, sizeof(PVOID), &outBuffer, NULL);
                if (NT_SUCCESS(status)) {
                    *(PVOID*)outBuffer = userBase;
                    information = sizeof(PVOID);
                    status = STATUS_SUCCESS;
                }
            } else {
                status = STATUS_UNSUCCESSFUL;
            }
            break;
        }
        // IOCTL_STOP_CAPTURE moved to 0x806 to make room for IOCTL_GET_BLOCK_RULES at 0x803
        case IOCTL_STOP_CAPTURE: { 
            if (g_SharedMemoryUserBase && g_SharedMemoryMdl) {
                MmUnmapLockedPages(g_SharedMemoryUserBase, g_SharedMemoryMdl);
                g_SharedMemoryUserBase = NULL;
            }
            status = STATUS_SUCCESS;
            information = 0;
            break;
        }
        case IOCTL_ADD_BLOCK_RULE: {
            PVOID inBuffer = NULL;
            // Retrieve the struct sent by Python
            status = WdfRequestRetrieveInputBuffer(Request, sizeof(BlockRuleV1), &inBuffer, NULL);
            if (NT_SUCCESS(status) && inBuffer != NULL) {
                // Send it to the Block Engine logic
                status = BlockEngine_AddRule((const BlockRuleV1*)inBuffer);
            }
            information = 0;
            break;
        }
        // ----------------------------------------------------------------
        // IOCTL_GET_BLOCK_RULES  (0x803)
        // Returns all currently-active BlockRuleV1 entries to user-mode.
        // The output buffer must be large enough for MAX_BLOCK_RULES entries;
        // Python should pass ctypes.sizeof(BlockRuleStruct) * 1024 bytes.
        // ----------------------------------------------------------------
        case IOCTL_GET_BLOCK_RULES: {
            const ULONG requiredOut = sizeof(BlockRuleV1) * MAX_BLOCK_RULES;
            if (OutputBufferLength < requiredOut) {
                status = STATUS_BUFFER_TOO_SMALL;
                break;
            }
            PVOID outBuffer = NULL;
            status = WdfRequestRetrieveOutputBuffer(
                Request, requiredOut, &outBuffer, NULL);
            if (!NT_SUCCESS(status) || !outBuffer) break;

            // Safe: outBuffer is a non-paged WDF-managed I/O buffer;
            // BlockEngine_GetRules performs only kernel-mode reads.
            ULONG ruleCount = BlockEngine_GetRules(
                (BlockRuleV1*)outBuffer, MAX_BLOCK_RULES);

            // Tell I/O manager exactly how many bytes to copy back
            // to user-mode (one entry per active rule, not the whole array).
            information = (ULONG_PTR)(ruleCount * sizeof(BlockRuleV1));
            status = STATUS_SUCCESS;
            break;
        }
        // ----------------------------------------------------------------
        // IOCTL_REMOVE_BLOCK_RULE  (0x804)
        // Receives a UINT16 dst_port from Python and atomically deactivates
        // the first matching active rule.
        // ----------------------------------------------------------------
        case IOCTL_REMOVE_BLOCK_RULE: {
            if (InputBufferLength < sizeof(UINT16)) {
                status = STATUS_BUFFER_TOO_SMALL;
                break;
            }
            PVOID inBuffer = NULL;
            status = WdfRequestRetrieveInputBuffer(
                Request, sizeof(UINT16), &inBuffer, NULL);
            if (!NT_SUCCESS(status) || !inBuffer) break;

            UINT16 dstPort = *(UINT16*)inBuffer;
            status = BlockEngine_RemoveRule(dstPort);
            information = 0;
            break;
        }
        default:
            status = STATUS_INVALID_DEVICE_REQUEST;
            break;
    }
    WdfRequestCompleteWithInformation(Request, status, information);
}

VOID EvtFileCleanup(_In_ WDFFILEOBJECT FileObject) {
    UNREFERENCED_PARAMETER(FileObject);
    
    // CRITICAL FIX: Do NOT call MmUnmapLockedPages here!
    // When CMD closes, the OS destroys the page tables automatically.
    // Manually unmapping a dead process context causes an immediate BSOD.
    g_SharedMemoryUserBase = NULL; 
}

NTSTATUS InitializeControlDevice(WDFDRIVER Driver) {
    PWDFDEVICE_INIT deviceInit = WdfControlDeviceInitAllocate(Driver, &SDDL_DEVOBJ_SYS_ALL_ADM_RWX_WORLD_RW_RES_R);
    if (!deviceInit) return STATUS_INSUFFICIENT_RESOURCES;

    // --- NEW CODE: Register File Cleanup ---
    WDF_FILEOBJECT_CONFIG fileConfig;
    WDF_FILEOBJECT_CONFIG_INIT(&fileConfig, WDF_NO_EVENT_CALLBACK, WDF_NO_EVENT_CALLBACK, EvtFileCleanup);
    WdfDeviceInitSetFileObjectConfig(deviceInit, &fileConfig, WDF_NO_OBJECT_ATTRIBUTES);
    // ---------------------------------------

    WdfDeviceInitSetDeviceType(deviceInit, FILE_DEVICE_NETWORK);
    WdfDeviceInitSetCharacteristics(deviceInit, FILE_DEVICE_SECURE_OPEN, FALSE);

    UNICODE_STRING deviceName, symLinkName;
    RtlInitUnicodeString(&deviceName, L"\\Device\\SecAIDriver");
    WdfDeviceInitAssignName(deviceInit, &deviceName);

    WDFDEVICE device;
    NTSTATUS status = WdfDeviceCreate(&deviceInit, WDF_NO_OBJECT_ATTRIBUTES, &device);
    if (!NT_SUCCESS(status)) {
        WdfDeviceInitFree(deviceInit);
        return status;
    }

    WDF_IO_QUEUE_CONFIG queueConfig;
    WDF_IO_QUEUE_CONFIG_INIT_DEFAULT_QUEUE(&queueConfig, WdfIoQueueDispatchSequential);
    queueConfig.EvtIoDeviceControl = EvtIoDeviceControl;

    WDFQUEUE queue;
    status = WdfIoQueueCreate(device, &queueConfig, WDF_NO_OBJECT_ATTRIBUTES, &queue);
    if (!NT_SUCCESS(status)) return status;

    RtlInitUnicodeString(&symLinkName, L"\\DosDevices\\SecAIDriver");
    WdfDeviceCreateSymbolicLink(device, &symLinkName);
    
    WdfControlFinishInitializing(device);

    PDEVICE_OBJECT wdmDevice = WdfDeviceWdmGetDeviceObject(device);
    status = RegisterWfpCallouts(wdmDevice);
    if (NT_SUCCESS(status)) {
        RegisterBfeFilters();
    }

    return STATUS_SUCCESS;
}

extern "C" NTSTATUS DriverEntry(_In_ PDRIVER_OBJECT DriverObject, _In_ PUNICODE_STRING RegistryPath) {
    NTSTATUS status;
    WDF_DRIVER_CONFIG config;

    BlockEngine_Init(); 

    WDF_DRIVER_CONFIG_INIT(&config, WDF_NO_EVENT_CALLBACK);
    config.EvtDriverUnload = DriverUnload;

    status = WdfDriverCreate(DriverObject, RegistryPath, WDF_NO_OBJECT_ATTRIBUTES, &config, WDF_NO_HANDLE);
    if (!NT_SUCCESS(status)) return status;

    WDFDRIVER driver = WdfGetDriver();

    // 1. Initialize Memory and Events FIRST
    status = InitializeSharedMemory();
    if (!NT_SUCCESS(status)) return status;

    status = InitializePacketEvent();
    if (!NT_SUCCESS(status)) return status;

    // 2. FINALLY, Turn on the network tap
    status = InitializeControlDevice(driver);
    if (!NT_SUCCESS(status)) return status;

    return STATUS_SUCCESS;
}