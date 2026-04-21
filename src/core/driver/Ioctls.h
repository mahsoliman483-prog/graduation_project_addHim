#pragma once

#include <ntddk.h>
#include <wdf.h>

#define SEC_AI_DEVICE_TYPE 40000

// Define the IOCTL codes for communication between User-Mode Python / C++ Bridge and the Kernel Driver
#define IOCTL_START_CAPTURE \
    CTL_CODE(SEC_AI_DEVICE_TYPE, 0x800, METHOD_BUFFERED, FILE_ANY_ACCESS)

#define IOCTL_ADD_BLOCK_RULE \
    CTL_CODE(SEC_AI_DEVICE_TYPE, 0x802, METHOD_BUFFERED, FILE_ANY_ACCESS)

// Returns all active BlockRuleV1 entries to user-mode as a flat array
#define IOCTL_GET_BLOCK_RULES \
    CTL_CODE(SEC_AI_DEVICE_TYPE, 0x803, METHOD_BUFFERED, FILE_ANY_ACCESS)

// Receives a dst_port (UINT16) from user-mode and atomically deactivates the matching rule
#define IOCTL_REMOVE_BLOCK_RULE \
    CTL_CODE(SEC_AI_DEVICE_TYPE, 0x804, METHOD_BUFFERED, FILE_ANY_ACCESS)

#define IOCTL_GET_STATS \
    CTL_CODE(SEC_AI_DEVICE_TYPE, 0x805, METHOD_BUFFERED, FILE_ANY_ACCESS)

// NOTE: 0x803 was previously IOCTL_STOP_CAPTURE; it has been moved to 0x806
//       to avoid colliding with IOCTL_GET_BLOCK_RULES. Update any Python code
//       that used the old 0x803 stop value to use 0x806 instead.
#define IOCTL_STOP_CAPTURE \
    CTL_CODE(SEC_AI_DEVICE_TYPE, 0x806, METHOD_BUFFERED, FILE_ANY_ACCESS)
