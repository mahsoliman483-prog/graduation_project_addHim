import numpy as np

# Python mirror of the C++ memory layout for zero-copy sharing.

# SharedMemoryHeader (includes 56-byte padding to prevent false sharing and aligns to 64 bytes)
# The C++ alignas(64) pads the 152 bytes of defined members up to 192 bytes.
header_dtype = np.dtype({
    'names': [
        'schema_version', '_pad0',
        'head', 'pad1',
        'tail', 'pad2',
        'capacity', 'dropped_packets'
    ],
    'formats': [
        np.uint32, np.uint32,
        np.uint64, '56V',
        np.uint64, '56V',
        np.uint64, np.uint64
    ],
    'offsets': [
        0, 4,
        8, 16,
        72, 80,
        136, 144
    ],
    'itemsize': 192
})

# Exactly 64 bytes = 1 CPU Cache Line
packet_dtype = np.dtype({
    'names': [
        'mono_ts_ns', 'schema_version', 'if_index', 'captured_len',
        'wire_len', 'src_port', 'dst_port', 'direction', 'ip_version',
        'proto', 'tcp_flags', 'src_ip', 'dst_ip'
    ],
    'formats': [
        np.uint64, np.uint32, np.uint32, np.uint32,
        np.uint32, np.uint16, np.uint16, np.uint8, np.uint8,
        np.uint8, np.uint8, (np.uint8, 16), (np.uint8, 16)
    ],
    'offsets': [
        0, 8, 12, 16,
        20, 24, 26, 28, 29,
        30, 31, 32, 48
    ],
    'itemsize': 64
})
