#!/usr/bin/env python3
"""Drop one .qpack container's files from the macOS page cache, without sudo.

usage: evict_page_cache.py <model.qpack>

`sudo purge` empties the whole unified buffer cache. For a cache experiment
only the model's own files matter, and for files you can read, macOS will
drop their cached pages on request: map the file shared and read-only, then
msync(MS_INVALIDATE). The script maps, invalidates and unmaps every file in
the container, then checks with mincore(2) that nothing is left resident.
Measured on an M5: a 432 MiB layer file read at 16 GiB/s warm and 6.3 GiB/s
straight after eviction, i.e. from the SSD.
"""
import ctypes
import ctypes.util
import mmap
import os
import sys

libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
MS_INVALIDATE = 0x0002
MAP_FAILED = ctypes.c_void_p(-1).value
PAGE = os.sysconf("SC_PAGE_SIZE")


def evict(path):
    """Returns (bytes, resident fraction after eviction)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        if size == 0:
            return 0, 0.0
        base = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if base in (None, MAP_FAILED):
            raise OSError(ctypes.get_errno(), f"mmap {path}")
        try:
            if libc.msync(base, size, MS_INVALIDATE) != 0:
                raise OSError(ctypes.get_errno(), f"msync {path}")
            pages = (size + PAGE - 1) // PAGE
            vec = ctypes.create_string_buffer(pages)
            if libc.mincore(base, size, vec) != 0:
                raise OSError(ctypes.get_errno(), f"mincore {path}")
            return size, sum(1 for b in vec.raw if b & 1) / pages
        finally:
            libc.munmap(base, size)
    finally:
        os.close(fd)


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    total = left = 0.0
    for root, _, files in os.walk(argv[1]):
        for name in files:
            size, resident = evict(os.path.join(root, name))
            total += size
            left += size * resident
    print(f"evicted {total / 2**30:.2f} GiB from the page cache; "
          f"{left / 2**20:.1f} MiB still resident ({100 * left / max(total, 1):.2f}%)")
    return 0 if left <= 0.01 * total else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
