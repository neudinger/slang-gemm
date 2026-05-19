const std = @import("std");
const gemm = @import("gemm");

pub const CudaError = enum(c_int) {
    success = 0,
    _,
};

pub const CublasStatus = enum(c_int) {
    success = 0,
    _,
};

pub const Operation = enum(c_int) {
    n = 0,
    t = 1,
    c = 2,
};

pub const MemcpyKind = enum(c_int) {
    host_to_device = 1,
    device_to_host = 2,
    device_to_device = 3,
};

pub const Handle = *opaque {};
pub const Stream = ?*opaque {};
pub const Event = *opaque {};

extern fn cudaGetDeviceCount(count: *c_int) callconv(.c) CudaError;
extern fn cudaMalloc(ptr: *?*anyopaque, size: usize) callconv(.c) CudaError;
extern fn cudaFree(ptr: ?*anyopaque) callconv(.c) CudaError;
extern fn cudaMemcpy(dst: ?*anyopaque, src: ?*const anyopaque, count: usize, kind: MemcpyKind) callconv(.c) CudaError;
extern fn cudaDeviceSynchronize() callconv(.c) CudaError;
extern fn cudaEventCreate(event: *Event) callconv(.c) CudaError;
extern fn cudaEventDestroy(event: Event) callconv(.c) CudaError;
extern fn cudaEventRecord(event: Event, stream: Stream) callconv(.c) CudaError;
extern fn cudaEventSynchronize(event: Event) callconv(.c) CudaError;
extern fn cudaEventElapsedTime(ms: *f32, start: Event, end: Event) callconv(.c) CudaError;

extern fn cublasCreate_v2(handle: *Handle) callconv(.c) CublasStatus;
extern fn cublasDestroy_v2(handle: Handle) callconv(.c) CublasStatus;
extern fn cublasSgemm_v2(handle: Handle, transa: Operation, transb: Operation, m: c_int, n: c_int, k: c_int, alpha: *const f32, a: [*]const f32, lda: c_int, b: [*]const f32, ldb: c_int, beta: *const f32, c: [*]f32, ldc: c_int) callconv(.c) CublasStatus;
extern fn cublasDgemm_v2(handle: Handle, transa: Operation, transb: Operation, m: c_int, n: c_int, k: c_int, alpha: *const f64, a: [*]const f64, lda: c_int, b: [*]const f64, ldb: c_int, beta: *const f64, c: [*]f64, ldc: c_int) callconv(.c) CublasStatus;
extern fn cublasCgemm_v2(handle: Handle, transa: Operation, transb: Operation, m: c_int, n: c_int, k: c_int, alpha: *const gemm.Complex32, a: [*]const gemm.Complex32, lda: c_int, b: [*]const gemm.Complex32, ldb: c_int, beta: *const gemm.Complex32, c: [*]gemm.Complex32, ldc: c_int) callconv(.c) CublasStatus;
extern fn cublasZgemm_v2(handle: Handle, transa: Operation, transb: Operation, m: c_int, n: c_int, k: c_int, alpha: *const gemm.Complex64, a: [*]const gemm.Complex64, lda: c_int, b: [*]const gemm.Complex64, ldb: c_int, beta: *const gemm.Complex64, c: [*]gemm.Complex64, ldc: c_int) callconv(.c) CublasStatus;
extern fn cublasCgemm3m(handle: Handle, transa: Operation, transb: Operation, m: c_int, n: c_int, k: c_int, alpha: *const gemm.Complex32, a: [*]const gemm.Complex32, lda: c_int, b: [*]const gemm.Complex32, ldb: c_int, beta: *const gemm.Complex32, c: [*]gemm.Complex32, ldc: c_int) callconv(.c) CublasStatus;
extern fn cublasZgemm3m(handle: Handle, transa: Operation, transb: Operation, m: c_int, n: c_int, k: c_int, alpha: *const gemm.Complex64, a: [*]const gemm.Complex64, lda: c_int, b: [*]const gemm.Complex64, ldb: c_int, beta: *const gemm.Complex64, c: [*]gemm.Complex64, ldc: c_int) callconv(.c) CublasStatus;
extern fn cublasSgemmStridedBatched(handle: Handle, transa: Operation, transb: Operation, m: c_int, n: c_int, k: c_int, alpha: *const f32, a: [*]const f32, lda: c_int, stride_a: c_longlong, b: [*]const f32, ldb: c_int, stride_b: c_longlong, beta: *const f32, c: [*]f32, ldc: c_int, stride_c: c_longlong, batch_count: c_int) callconv(.c) CublasStatus;
extern fn cublasDgemmStridedBatched(handle: Handle, transa: Operation, transb: Operation, m: c_int, n: c_int, k: c_int, alpha: *const f64, a: [*]const f64, lda: c_int, stride_a: c_longlong, b: [*]const f64, ldb: c_int, stride_b: c_longlong, beta: *const f64, c: [*]f64, ldc: c_int, stride_c: c_longlong, batch_count: c_int) callconv(.c) CublasStatus;
extern fn cublasCgemmStridedBatched(handle: Handle, transa: Operation, transb: Operation, m: c_int, n: c_int, k: c_int, alpha: *const gemm.Complex32, a: [*]const gemm.Complex32, lda: c_int, stride_a: c_longlong, b: [*]const gemm.Complex32, ldb: c_int, stride_b: c_longlong, beta: *const gemm.Complex32, c: [*]gemm.Complex32, ldc: c_int, stride_c: c_longlong, batch_count: c_int) callconv(.c) CublasStatus;
extern fn cublasZgemmStridedBatched(handle: Handle, transa: Operation, transb: Operation, m: c_int, n: c_int, k: c_int, alpha: *const gemm.Complex64, a: [*]const gemm.Complex64, lda: c_int, stride_a: c_longlong, b: [*]const gemm.Complex64, ldb: c_int, stride_b: c_longlong, beta: *const gemm.Complex64, c: [*]gemm.Complex64, ldc: c_int, stride_c: c_longlong, batch_count: c_int) callconv(.c) CublasStatus;
extern fn cublasGemmEx(handle: Handle, transa: Operation, transb: Operation, m: c_int, n: c_int, k: c_int, alpha: ?*const anyopaque, a: ?*const anyopaque, a_type: gemm.DataType, lda: c_int, b: ?*const anyopaque, b_type: gemm.DataType, ldb: c_int, beta: ?*const anyopaque, c: ?*anyopaque, c_type: gemm.DataType, ldc: c_int, compute_type: gemm.ComputeType, algo: gemm.Algo) callconv(.c) CublasStatus;
extern fn cublasSgemmBatched(handle: Handle, transa: Operation, transb: Operation, m: c_int, n: c_int, k: c_int, alpha: *const f32, a_array: ?*const anyopaque, lda: c_int, b_array: ?*const anyopaque, ldb: c_int, beta: *const f32, c_array: ?*anyopaque, ldc: c_int, batch_count: c_int) callconv(.c) CublasStatus;
extern fn cublasDgemmBatched(handle: Handle, transa: Operation, transb: Operation, m: c_int, n: c_int, k: c_int, alpha: *const f64, a_array: ?*const anyopaque, lda: c_int, b_array: ?*const anyopaque, ldb: c_int, beta: *const f64, c_array: ?*anyopaque, ldc: c_int, batch_count: c_int) callconv(.c) CublasStatus;
extern fn cublasCgemmBatched(handle: Handle, transa: Operation, transb: Operation, m: c_int, n: c_int, k: c_int, alpha: *const gemm.Complex32, a_array: ?*const anyopaque, lda: c_int, b_array: ?*const anyopaque, ldb: c_int, beta: *const gemm.Complex32, c_array: ?*anyopaque, ldc: c_int, batch_count: c_int) callconv(.c) CublasStatus;
extern fn cublasZgemmBatched(handle: Handle, transa: Operation, transb: Operation, m: c_int, n: c_int, k: c_int, alpha: *const gemm.Complex64, a_array: ?*const anyopaque, lda: c_int, b_array: ?*const anyopaque, ldb: c_int, beta: *const gemm.Complex64, c_array: ?*anyopaque, ldc: c_int, batch_count: c_int) callconv(.c) CublasStatus;
extern fn cublasCgemm3mEx(handle: Handle, transa: Operation, transb: Operation, m: c_int, n: c_int, k: c_int, alpha: *const gemm.Complex32, a: ?*const anyopaque, a_type: gemm.DataType, lda: c_int, b: ?*const anyopaque, b_type: gemm.DataType, ldb: c_int, beta: *const gemm.Complex32, c: ?*anyopaque, c_type: gemm.DataType, ldc: c_int) callconv(.c) CublasStatus;
extern fn cublasGemmBatchedEx(handle: Handle, transa: Operation, transb: Operation, m: c_int, n: c_int, k: c_int, alpha: ?*const anyopaque, a_array: ?*const anyopaque, a_type: gemm.DataType, lda: c_int, b_array: ?*const anyopaque, b_type: gemm.DataType, ldb: c_int, beta: ?*const anyopaque, c_array: ?*anyopaque, c_type: gemm.DataType, ldc: c_int, batch_count: c_int, compute_type: gemm.ComputeType, algo: gemm.Algo) callconv(.c) CublasStatus;
extern fn cublasGemmStridedBatchedEx(handle: Handle, transa: Operation, transb: Operation, m: c_int, n: c_int, k: c_int, alpha: ?*const anyopaque, a: ?*const anyopaque, a_type: gemm.DataType, lda: c_int, stride_a: c_longlong, b: ?*const anyopaque, b_type: gemm.DataType, ldb: c_int, stride_b: c_longlong, beta: ?*const anyopaque, c: ?*anyopaque, c_type: gemm.DataType, ldc: c_int, stride_c: c_longlong, batch_count: c_int, compute_type: gemm.ComputeType, algo: gemm.Algo) callconv(.c) CublasStatus;

pub fn hasCudaDevice() bool {
    var count: c_int = 0;
    if (cudaGetDeviceCount(&count) != .success) return false;
    return count > 0;
}

pub fn checkCuda(status: CudaError) !void {
    if (status == .success) return;
    return error.CudaFailure;
}

pub fn checkCublas(status: CublasStatus) !void {
    if (status == .success) return;
    return error.CublasFailure;
}

pub const Context = struct {
    handle: Handle,

    pub fn init() !Context {
        var handle: Handle = undefined;
        try checkCublas(cublasCreate_v2(&handle));
        return .{ .handle = handle };
    }

    pub fn deinit(self: Context) void {
        _ = cublasDestroy_v2(self.handle);
    }
};

pub fn DeviceBuffer(comptime T: type) type {
    return struct {
        ptr: [*]T,
        len: usize,

        pub fn alloc(len: usize) !@This() {
            var raw: ?*anyopaque = null;
            try checkCuda(cudaMalloc(&raw, len * @sizeOf(T)));
            return .{ .ptr = @ptrCast(@alignCast(raw.?)), .len = len };
        }

        pub fn free(self: @This()) void {
            _ = cudaFree(@ptrCast(self.ptr));
        }

        pub fn copyFromHost(self: @This(), src: []const T) !void {
            if (src.len > self.len) return error.BufferTooSmall;
            try checkCuda(cudaMemcpy(self.ptr, src.ptr, src.len * @sizeOf(T), .host_to_device));
        }

        pub fn copyToHost(self: @This(), dst: []T) !void {
            if (dst.len > self.len) return error.BufferTooSmall;
            try checkCuda(cudaMemcpy(dst.ptr, self.ptr, dst.len * @sizeOf(T), .device_to_host));
        }
    };
}

pub const RawDeviceBuffer = struct {
    ptr: ?*anyopaque,
    byte_len: usize,

    pub fn alloc(byte_len: usize) !RawDeviceBuffer {
        var raw: ?*anyopaque = null;
        try checkCuda(cudaMalloc(&raw, byte_len));
        return .{ .ptr = raw, .byte_len = byte_len };
    }

    pub fn free(self: RawDeviceBuffer) void {
        _ = cudaFree(self.ptr);
    }

    pub fn copyFromHost(self: RawDeviceBuffer, src: []const u8) !void {
        if (src.len > self.byte_len) return error.BufferTooSmall;
        try checkCuda(cudaMemcpy(self.ptr, src.ptr, src.len, .host_to_device));
    }

    pub fn copyToHost(self: RawDeviceBuffer, dst: []u8) !void {
        if (dst.len > self.byte_len) return error.BufferTooSmall;
        try checkCuda(cudaMemcpy(dst.ptr, self.ptr, dst.len, .device_to_host));
    }
};

pub fn DevicePointerArray(comptime T: type) type {
    return struct {
        ptr: [*][*]T,
        len: usize,

        pub fn alloc(ptrs: []const [*]T) !@This() {
            var raw: ?*anyopaque = null;
            try checkCuda(cudaMalloc(&raw, ptrs.len * @sizeOf([*]T)));
            const result: @This() = .{ .ptr = @ptrCast(@alignCast(raw.?)), .len = ptrs.len };
            try checkCuda(cudaMemcpy(@ptrCast(result.ptr), @ptrCast(ptrs.ptr), ptrs.len * @sizeOf([*]T), .host_to_device));
            return result;
        }

        pub fn free(self: @This()) void {
            _ = cudaFree(@ptrCast(self.ptr));
        }
    };
}

pub fn opFromGemm(op: gemm.Op) Operation {
    return switch (op) {
        .no_trans => .n,
        .trans => .t,
        .conj_trans => .c,
    };
}

pub fn sgemm(ctx: Context, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, alpha: f32, a: [*]const f32, lda: usize, b: [*]const f32, ldb: usize, beta: f32, c: [*]f32, ldc: usize) !void {
    try checkCublas(cublasSgemm_v2(ctx.handle, opFromGemm(op_a), opFromGemm(op_b), int(m), int(n), int(k), &alpha, a, int(lda), b, int(ldb), &beta, c, int(ldc)));
}

pub fn dgemm(ctx: Context, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, alpha: f64, a: [*]const f64, lda: usize, b: [*]const f64, ldb: usize, beta: f64, c: [*]f64, ldc: usize) !void {
    try checkCublas(cublasDgemm_v2(ctx.handle, opFromGemm(op_a), opFromGemm(op_b), int(m), int(n), int(k), &alpha, a, int(lda), b, int(ldb), &beta, c, int(ldc)));
}

pub fn cgemm(ctx: Context, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, alpha: gemm.Complex32, a: [*]const gemm.Complex32, lda: usize, b: [*]const gemm.Complex32, ldb: usize, beta: gemm.Complex32, c: [*]gemm.Complex32, ldc: usize) !void {
    try checkCublas(cublasCgemm_v2(ctx.handle, opFromGemm(op_a), opFromGemm(op_b), int(m), int(n), int(k), &alpha, a, int(lda), b, int(ldb), &beta, c, int(ldc)));
}

pub fn zgemm(ctx: Context, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, alpha: gemm.Complex64, a: [*]const gemm.Complex64, lda: usize, b: [*]const gemm.Complex64, ldb: usize, beta: gemm.Complex64, c: [*]gemm.Complex64, ldc: usize) !void {
    try checkCublas(cublasZgemm_v2(ctx.handle, opFromGemm(op_a), opFromGemm(op_b), int(m), int(n), int(k), &alpha, a, int(lda), b, int(ldb), &beta, c, int(ldc)));
}

pub fn cgemm3m(ctx: Context, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, alpha: gemm.Complex32, a: [*]const gemm.Complex32, lda: usize, b: [*]const gemm.Complex32, ldb: usize, beta: gemm.Complex32, c: [*]gemm.Complex32, ldc: usize) !void {
    try checkCublas(cublasCgemm3m(ctx.handle, opFromGemm(op_a), opFromGemm(op_b), int(m), int(n), int(k), &alpha, a, int(lda), b, int(ldb), &beta, c, int(ldc)));
}

pub fn zgemm3m(ctx: Context, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, alpha: gemm.Complex64, a: [*]const gemm.Complex64, lda: usize, b: [*]const gemm.Complex64, ldb: usize, beta: gemm.Complex64, c: [*]gemm.Complex64, ldc: usize) !void {
    try checkCublas(cublasZgemm3m(ctx.handle, opFromGemm(op_a), opFromGemm(op_b), int(m), int(n), int(k), &alpha, a, int(lda), b, int(ldb), &beta, c, int(ldc)));
}

pub fn sgemmStridedBatched(ctx: Context, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, alpha: f32, a: [*]const f32, lda: usize, stride_a: usize, b: [*]const f32, ldb: usize, stride_b: usize, beta: f32, c: [*]f32, ldc: usize, stride_c: usize, batch_count: usize) !void {
    try checkCublas(cublasSgemmStridedBatched(ctx.handle, opFromGemm(op_a), opFromGemm(op_b), int(m), int(n), int(k), &alpha, a, int(lda), long(stride_a), b, int(ldb), long(stride_b), &beta, c, int(ldc), long(stride_c), int(batch_count)));
}

pub fn dgemmStridedBatched(ctx: Context, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, alpha: f64, a: [*]const f64, lda: usize, stride_a: usize, b: [*]const f64, ldb: usize, stride_b: usize, beta: f64, c: [*]f64, ldc: usize, stride_c: usize, batch_count: usize) !void {
    try checkCublas(cublasDgemmStridedBatched(ctx.handle, opFromGemm(op_a), opFromGemm(op_b), int(m), int(n), int(k), &alpha, a, int(lda), long(stride_a), b, int(ldb), long(stride_b), &beta, c, int(ldc), long(stride_c), int(batch_count)));
}

pub fn cgemmStridedBatched(ctx: Context, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, alpha: gemm.Complex32, a: [*]const gemm.Complex32, lda: usize, stride_a: usize, b: [*]const gemm.Complex32, ldb: usize, stride_b: usize, beta: gemm.Complex32, c: [*]gemm.Complex32, ldc: usize, stride_c: usize, batch_count: usize) !void {
    try checkCublas(cublasCgemmStridedBatched(ctx.handle, opFromGemm(op_a), opFromGemm(op_b), int(m), int(n), int(k), &alpha, a, int(lda), long(stride_a), b, int(ldb), long(stride_b), &beta, c, int(ldc), long(stride_c), int(batch_count)));
}

pub fn zgemmStridedBatched(ctx: Context, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, alpha: gemm.Complex64, a: [*]const gemm.Complex64, lda: usize, stride_a: usize, b: [*]const gemm.Complex64, ldb: usize, stride_b: usize, beta: gemm.Complex64, c: [*]gemm.Complex64, ldc: usize, stride_c: usize, batch_count: usize) !void {
    try checkCublas(cublasZgemmStridedBatched(ctx.handle, opFromGemm(op_a), opFromGemm(op_b), int(m), int(n), int(k), &alpha, a, int(lda), long(stride_a), b, int(ldb), long(stride_b), &beta, c, int(ldc), long(stride_c), int(batch_count)));
}

pub fn sgemmBatched(ctx: Context, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, alpha: f32, a_array: ?*const anyopaque, lda: usize, b_array: ?*const anyopaque, ldb: usize, beta: f32, c_array: ?*anyopaque, ldc: usize, batch_count: usize) !void {
    try checkCublas(cublasSgemmBatched(ctx.handle, opFromGemm(op_a), opFromGemm(op_b), int(m), int(n), int(k), &alpha, a_array, int(lda), b_array, int(ldb), &beta, c_array, int(ldc), int(batch_count)));
}

pub fn dgemmBatched(ctx: Context, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, alpha: f64, a_array: ?*const anyopaque, lda: usize, b_array: ?*const anyopaque, ldb: usize, beta: f64, c_array: ?*anyopaque, ldc: usize, batch_count: usize) !void {
    try checkCublas(cublasDgemmBatched(ctx.handle, opFromGemm(op_a), opFromGemm(op_b), int(m), int(n), int(k), &alpha, a_array, int(lda), b_array, int(ldb), &beta, c_array, int(ldc), int(batch_count)));
}

pub fn cgemmBatched(ctx: Context, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, alpha: gemm.Complex32, a_array: ?*const anyopaque, lda: usize, b_array: ?*const anyopaque, ldb: usize, beta: gemm.Complex32, c_array: ?*anyopaque, ldc: usize, batch_count: usize) !void {
    try checkCublas(cublasCgemmBatched(ctx.handle, opFromGemm(op_a), opFromGemm(op_b), int(m), int(n), int(k), &alpha, a_array, int(lda), b_array, int(ldb), &beta, c_array, int(ldc), int(batch_count)));
}

pub fn zgemmBatched(ctx: Context, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, alpha: gemm.Complex64, a_array: ?*const anyopaque, lda: usize, b_array: ?*const anyopaque, ldb: usize, beta: gemm.Complex64, c_array: ?*anyopaque, ldc: usize, batch_count: usize) !void {
    try checkCublas(cublasZgemmBatched(ctx.handle, opFromGemm(op_a), opFromGemm(op_b), int(m), int(n), int(k), &alpha, a_array, int(lda), b_array, int(ldb), &beta, c_array, int(ldc), int(batch_count)));
}

pub fn gemmEx(ctx: Context, desc: RawGemmExDesc) !void {
    try checkCublas(cublasGemmEx(ctx.handle, opFromGemm(desc.op_a), opFromGemm(desc.op_b), int(desc.m), int(desc.n), int(desc.k), desc.alpha, desc.a, desc.a_type, int(desc.lda), desc.b, desc.b_type, int(desc.ldb), desc.beta, desc.c, desc.c_type, int(desc.ldc), desc.compute_type, desc.algo));
}

pub fn cgemm3mEx(ctx: Context, desc: RawCgemm3mExDesc) !void {
    try checkCublas(cublasCgemm3mEx(ctx.handle, opFromGemm(desc.op_a), opFromGemm(desc.op_b), int(desc.m), int(desc.n), int(desc.k), &desc.alpha, desc.a, desc.a_type, int(desc.lda), desc.b, desc.b_type, int(desc.ldb), &desc.beta, desc.c, desc.c_type, int(desc.ldc)));
}

pub fn gemmStridedBatchedEx(ctx: Context, desc: RawGemmExDesc) !void {
    try checkCublas(cublasGemmStridedBatchedEx(ctx.handle, opFromGemm(desc.op_a), opFromGemm(desc.op_b), int(desc.m), int(desc.n), int(desc.k), desc.alpha, desc.a, desc.a_type, int(desc.lda), long(desc.stride_a), desc.b, desc.b_type, int(desc.ldb), long(desc.stride_b), desc.beta, desc.c, desc.c_type, int(desc.ldc), long(desc.stride_c), int(desc.batch_count), desc.compute_type, desc.algo));
}

pub fn gemmBatchedEx(ctx: Context, desc: RawGemmBatchedExDesc) !void {
    try checkCublas(cublasGemmBatchedEx(ctx.handle, opFromGemm(desc.op_a), opFromGemm(desc.op_b), int(desc.m), int(desc.n), int(desc.k), desc.alpha, desc.a_array, desc.a_type, int(desc.lda), desc.b_array, desc.b_type, int(desc.ldb), desc.beta, desc.c_array, desc.c_type, int(desc.ldc), int(desc.batch_count), desc.compute_type, desc.algo));
}

pub const RawGemmExDesc = struct {
    op_a: gemm.Op,
    op_b: gemm.Op,
    m: usize,
    n: usize,
    k: usize,
    alpha: ?*const anyopaque,
    a: ?*const anyopaque,
    a_type: gemm.DataType,
    lda: usize,
    stride_a: usize = 0,
    b: ?*const anyopaque,
    b_type: gemm.DataType,
    ldb: usize,
    stride_b: usize = 0,
    beta: ?*const anyopaque,
    c: ?*anyopaque,
    c_type: gemm.DataType,
    ldc: usize,
    stride_c: usize = 0,
    batch_count: usize = 1,
    compute_type: gemm.ComputeType,
    algo: gemm.Algo = .default,
};

pub const RawGemmBatchedExDesc = struct {
    op_a: gemm.Op,
    op_b: gemm.Op,
    m: usize,
    n: usize,
    k: usize,
    alpha: ?*const anyopaque,
    a_array: ?*const anyopaque,
    a_type: gemm.DataType,
    lda: usize,
    b_array: ?*const anyopaque,
    b_type: gemm.DataType,
    ldb: usize,
    beta: ?*const anyopaque,
    c_array: ?*anyopaque,
    c_type: gemm.DataType,
    ldc: usize,
    batch_count: usize,
    compute_type: gemm.ComputeType,
    algo: gemm.Algo = .default,
};

pub const RawCgemm3mExDesc = struct {
    op_a: gemm.Op,
    op_b: gemm.Op,
    m: usize,
    n: usize,
    k: usize,
    alpha: gemm.Complex32,
    a: ?*const anyopaque,
    a_type: gemm.DataType,
    lda: usize,
    b: ?*const anyopaque,
    b_type: gemm.DataType,
    ldb: usize,
    beta: gemm.Complex32,
    c: ?*anyopaque,
    c_type: gemm.DataType,
    ldc: usize,
};

pub const GpuTimer = struct {
    start: Event,
    stop: Event,

    pub fn init() !GpuTimer {
        var start: Event = undefined;
        var stop: Event = undefined;
        try checkCuda(cudaEventCreate(&start));
        errdefer _ = cudaEventDestroy(start);
        try checkCuda(cudaEventCreate(&stop));
        return .{ .start = start, .stop = stop };
    }

    pub fn deinit(self: GpuTimer) void {
        _ = cudaEventDestroy(self.start);
        _ = cudaEventDestroy(self.stop);
    }

    pub fn begin(self: GpuTimer) !void {
        try checkCuda(cudaEventRecord(self.start, null));
    }

    pub fn end(self: GpuTimer) !f32 {
        try checkCuda(cudaEventRecord(self.stop, null));
        try checkCuda(cudaEventSynchronize(self.stop));
        var ms: f32 = 0;
        try checkCuda(cudaEventElapsedTime(&ms, self.start, self.stop));
        return ms;
    }
};

pub fn synchronize() !void {
    try checkCuda(cudaDeviceSynchronize());
}

fn int(value: usize) c_int {
    return @intCast(value);
}

fn long(value: usize) c_longlong {
    return @intCast(value);
}
