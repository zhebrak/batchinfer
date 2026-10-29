"""pool=elastic: the KV pool over CUDA virtual memory, so physical memory is held only under live blocks.

The pool's whole virtual range is reserved once (cuMemAddressReserve) and the tensor the kernels and graphs read is
made over it (a DLPack capsule over the pointer: the address never changes, so the graphs captured over it stay
valid). Physical memory comes in pieces of the driver's granularity (2 MiB), created (cuMemCreate), mapped and made
accessible on first use and unmapped and released when nothing live is left under them (executor, kv.py: the
allocator counts live blocks per group of blocks that share pieces); torch's allocator never sees it. Everything is
ctypes over libcuda, no extension and no package. Fake is the same interface over nothing, for the CPU tests and
the executor off the GPU, recording what it was asked.
"""
import ctypes
from ctypes import POINTER, byref, c_int, c_int64, c_size_t, c_uint8, c_uint16, c_uint64, c_void_p

GRANULARITY_FALLBACK = 2 << 20  # the driver's usual minimum; the real one is asked for

# cuda.h: CUmemAllocationProp, CUmemAccessDesc
CU_MEM_ALLOCATION_TYPE_PINNED, CU_MEM_LOCATION_TYPE_DEVICE, CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 1, 1, 3


class _Location(ctypes.Structure):
    _fields_ = [("type", c_int), ("id", c_int)]


class _AllocFlags(ctypes.Structure):
    _fields_ = [("compressionType", c_uint8), ("gpuDirectRDMACapable", c_uint8), ("usage", c_uint16), ("reserved", c_uint8 * 4)]


class _AllocationProp(ctypes.Structure):
    _fields_ = [("type", c_int), ("requestedHandleTypes", c_int), ("location", _Location), ("win32HandleMetaData", c_void_p),
                ("allocFlags", _AllocFlags)]


class _AccessDesc(ctypes.Structure):
    _fields_ = [("location", _Location), ("flags", c_int)]


# dlpack.h: what torch.from_dlpack reads
class _DLDataType(ctypes.Structure):
    _fields_ = [("code", c_uint8), ("bits", c_uint8), ("lanes", c_uint16)]


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", c_int), ("device_id", c_int)]


class _DLTensor(ctypes.Structure):
    _fields_ = [("data", c_void_p), ("device", _DLDevice), ("ndim", c_int), ("dtype", _DLDataType), ("shape", POINTER(c_int64)),
                ("strides", POINTER(c_int64)), ("byte_offset", c_uint64)]


_DELETER = ctypes.CFUNCTYPE(None, c_void_p)
_KEEP = []  # the DLPack structs of every tensor made here, kept for the process: a tensor may outlive its Vmm


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [("dl_tensor", _DLTensor), ("manager_ctx", c_void_p), ("deleter", _DELETER)]


class Vmm:
    """The pool's virtual range on one device: reserve once, map and unmap pieces by offset, a tensor over it."""

    def __init__(self, device_index):
        self.cu = ctypes.CDLL("libcuda.so.1")
        self.cu.cuInit(0)
        self.device = device_index
        self.prop = _AllocationProp(type=CU_MEM_ALLOCATION_TYPE_PINNED, location=_Location(CU_MEM_LOCATION_TYPE_DEVICE, device_index))
        self.access = _AccessDesc(location=_Location(CU_MEM_LOCATION_TYPE_DEVICE, device_index), flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE)
        gran = c_size_t(0)
        self._check(self.cu.cuMemGetAllocationGranularity(byref(gran), byref(self.prop), 0), "cuMemGetAllocationGranularity")
        self.granularity = gran.value or GRANULARITY_FALLBACK
        self.base, self.size, self.handles, self.aliases = None, 0, {}, {}

    def _check(self, code, what):
        if code != 0:
            raise RuntimeError(f"{what} failed with CUresult {code}")

    def reserve(self, size):
        """One virtual range of size bytes (a multiple of the granularity); returns its address."""
        assert self.base is None and size % self.granularity == 0
        ptr = c_uint64(0)
        self._check(self.cu.cuMemAddressReserve(byref(ptr), c_size_t(size), c_size_t(self.granularity), c_uint64(0), c_uint64(0)),
                    "cuMemAddressReserve")
        self.base, self.size = ptr.value, size
        return self.base

    def map(self, offset, size):
        """Physical memory under [offset, offset + size) of the range, created now; a multiple of the granularity."""
        assert offset % self.granularity == 0 and size % self.granularity == 0 and offset not in self.handles
        handle = c_uint64(0)
        self._check(self.cu.cuMemCreate(byref(handle), c_size_t(size), byref(self.prop), c_uint64(0)), "cuMemCreate")
        addr = c_uint64(self.base + offset)
        self._check(self.cu.cuMemMap(addr, c_size_t(size), c_size_t(0), handle, c_uint64(0)), "cuMemMap")
        self._check(self.cu.cuMemSetAccess(addr, c_size_t(size), byref(self.access), c_size_t(1)), "cuMemSetAccess")
        self.handles[offset] = (handle, size)

    def unmap(self, offset):
        """The physical memory under the piece mapped at offset, released to the card."""
        handle, size = self.handles.pop(offset)
        self._check(self.cu.cuMemUnmap(c_uint64(self.base + offset), c_size_t(size)), "cuMemUnmap")
        self._check(self.cu.cuMemRelease(handle), "cuMemRelease")

    def alias(self, offset, source, at):
        """The physical memory `source` mapped at `at` maps again at this range's offset (pool=contiguous: a unit of
        the home range read through a row's region); the memory stays source's."""
        handle, size = source.handles[at]
        addr = c_uint64(self.base + offset)
        self._check(self.cu.cuMemMap(addr, c_size_t(size), c_size_t(0), handle, c_uint64(0)), "cuMemMap")
        self._check(self.cu.cuMemSetAccess(addr, c_size_t(size), byref(self.access), c_size_t(1)), "cuMemSetAccess")
        self.aliases[offset] = size

    def unalias(self, offset):
        size = self.aliases.pop(offset)
        self._check(self.cu.cuMemUnmap(c_uint64(self.base + offset), c_size_t(size)), "cuMemUnmap")

    def tensor(self, shape, dtype):
        """A torch tensor of this shape and dtype over the whole range, through a DLPack capsule over the pointer (a
        uint8 view the tensor is then viewed as dtype): no copy, no owner, the address the graphs bake."""
        import torch
        numel, itemsize = 1, torch.empty((), dtype=dtype).element_size()
        for s in shape:
            numel *= s
        assert numel * itemsize <= self.size
        n = c_int64 * 1
        shape_arr = n(numel * itemsize)
        managed = _DLManagedTensor()
        managed.dl_tensor = _DLTensor(data=c_void_p(self.base), device=_DLDevice(2, self.device), ndim=1,
                                      dtype=_DLDataType(1, 8, 1), shape=shape_arr, strides=None, byte_offset=0)
        managed.manager_ctx, managed.deleter = None, _DELETER()  # NULL: torch frees nothing, the memory is the range's
        _KEEP.extend([shape_arr, managed])  # outlive every tensor made over them, whatever is dropped first
        PyCapsule_New = ctypes.pythonapi.PyCapsule_New
        PyCapsule_New.restype, PyCapsule_New.argtypes = ctypes.py_object, [c_void_p, ctypes.c_char_p, c_void_p]
        capsule = PyCapsule_New(ctypes.cast(byref(managed), c_void_p), b"dltensor", None)
        return torch.from_dlpack(capsule).view(dtype).view(*shape)

    @property
    def mapped_bytes(self):
        return sum(size for _, size in self.handles.values())


class Fake:
    """Vmm's interface over nothing: the bookkeeping, for the CPU tests and the executor off the GPU."""

    def __init__(self, granularity=GRANULARITY_FALLBACK):
        self.granularity, self.base, self.size, self.handles, self.aliases, self.log = granularity, None, 0, {}, {}, []

    def reserve(self, size):
        assert self.base is None and size % self.granularity == 0
        self.base, self.size = 1 << 40, size
        return self.base

    def map(self, offset, size):
        assert offset % self.granularity == 0 and size % self.granularity == 0 and offset + size <= self.size
        assert offset not in self.handles, f"piece at {offset} mapped twice"
        self.handles[offset] = (None, size)
        self.log.append(("map", offset, size))

    def unmap(self, offset):
        self.handles.pop(offset)  # KeyError: an unmap of a piece not mapped
        self.log.append(("unmap", offset))

    def alias(self, offset, source, at):
        assert at in source.handles, f"no memory at {at} to alias"
        assert offset not in self.aliases and offset % self.granularity == 0 and offset + source.handles[at][1] <= self.size
        self.aliases[offset] = source.handles[at][1]
        self.log.append(("alias", offset, at))

    def unalias(self, offset):
        self.aliases.pop(offset)  # KeyError: not aliased
        self.log.append(("unalias", offset))

    def tensor(self, shape, dtype):
        import torch
        return torch.zeros(shape, dtype=dtype)

    @property
    def mapped_bytes(self):
        return sum(size for _, size in self.handles.values())
