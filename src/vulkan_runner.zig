const std = @import("std");
const gemm = @import("gemm");
const manifest = @import("manifest");
const vk = @import("vulkan");
const complex_f32_spv = @import("gemm_family_complex_f32_spv");
const complex_f64_spv = @import("gemm_family_complex_f64_spv");
const f16_scalar_spv = @import("gemm_family_f16_scalar_spv");
const f32_spv = @import("gemm_family_f32_tf32_spv");
const f64_spv = @import("gemm_family_f64_spv");
const int8_i32_spv = @import("gemm_family_int8_int32_spv");
const lowp_f32_spv = @import("gemm_family_low_precision_packed_spv");
const nvcoop2_bf16_spv = @import("gemm_nvcoop2_bf16_spv");
const nvcoop2_f16_spv = @import("gemm_nvcoop2_f16_spv");
const nvcoop2_f16_small_spv = @import("gemm_nvcoop2_f16_small_spv");
const nvcoop2_i8_spv = @import("gemm_nvcoop2_i8_spv");

pub const Result = struct {
    ms: f64,
    tflops: f64,
};

pub fn shaderWordsForFamily(family: manifest.ShaderFamily) ?[]const u32 {
    return switch (family) {
        .none => null,
        .f16_scalar => f16_scalar_spv.words[0..],
        .f16_bf16_tensor => nvcoop2_f16_spv.words[0..],
        .f32_tf32 => f32_spv.words[0..],
        .f64 => f64_spv.words[0..],
        .complex_f32 => complex_f32_spv.words[0..],
        .complex_f64 => complex_f64_spv.words[0..],
        .int8_int32 => int8_i32_spv.words[0..],
        .low_precision_f32 => lowp_f32_spv.words[0..],
    };
}

const GemmParams = extern struct {
    op_a: u32 = 0,
    op_b: u32 = 0,
    m: u32,
    n: u32,
    k: u32,
    batch_count: u32 = 1,

    a_offset: u32 = 0,
    b_offset: u32 = 0,
    c_offset: u32 = 0,
    a_row_stride: u32,
    a_col_stride: u32,
    b_row_stride: u32,
    b_col_stride: u32,
    c_row_stride: u32,
    c_col_stride: u32,
    stride_a: u32,
    stride_b: u32,
    stride_c: u32,

    alpha_re: f32 = 1,
    alpha_im: f32 = 0,
    beta_re: f32 = 0,
    beta_im: f32 = 0,
};

const GemmParamsF64 = extern struct {
    op_a: u32 = 0,
    op_b: u32 = 0,
    m: u32,
    n: u32,
    k: u32,
    batch_count: u32 = 1,

    a_offset: u32 = 0,
    b_offset: u32 = 0,
    c_offset: u32 = 0,
    a_row_stride: u32,
    a_col_stride: u32,
    b_row_stride: u32,
    b_col_stride: u32,
    c_row_stride: u32,
    c_col_stride: u32,
    stride_a: u32,
    stride_b: u32,
    stride_c: u32,

    alpha: f64 = 1,
    beta: f64 = 0,
};

const GemmParamsComplexF64 = extern struct {
    op_a: u32 = 0,
    op_b: u32 = 0,
    m: u32,
    n: u32,
    k: u32,
    batch_count: u32 = 1,

    a_offset: u32 = 0,
    b_offset: u32 = 0,
    c_offset: u32 = 0,
    a_row_stride: u32,
    a_col_stride: u32,
    b_row_stride: u32,
    b_col_stride: u32,
    c_row_stride: u32,
    c_col_stride: u32,
    stride_a: u32,
    stride_b: u32,
    stride_c: u32,

    alpha_re: f64 = 1,
    alpha_im: f64 = 0,
    beta_re: f64 = 0,
    beta_im: f64 = 0,
};

const GemmParamsLowp = extern struct {
    base: GemmParams,
    a_type: u32,
    b_type: u32,
};

const GemmParamsI32 = extern struct {
    base: GemmParams,
    a_type: u32,
    b_type: u32,
    c_type: u32,
    alpha: i32 = 1,
    beta: i32 = 0,
};

const Coop2PushConstants = extern struct {
    m: u32,
    n: u32,
    k: u32,
    a_stride: u32,
    b_stride: u32,
    c_stride: u32,
    batch_stride_a: u32,
    batch_stride_b: u32,
    batch_stride_c: u32,
    op_a: u32 = 0,
    op_b: u32 = 0,
};

const Buffer = struct {
    handle: vk.Buffer,
    memory: vk.DeviceMemory,
    mapped: ?[*]u8,
    byte_len: usize,

    fn slice(self: Buffer, comptime T: type) []T {
        return @as([*]T, @ptrCast(@alignCast(self.mapped.?)))[0 .. self.byte_len / @sizeOf(T)];
    }
};

const Vulkan = struct {
    lib: std.DynLib,
    get_instance_proc_addr: vk.PfnGetInstanceProcAddr,
    base: vk.BaseWrapper,
    instance_fns: vk.InstanceWrapper,

    fn open() !Vulkan {
        var lib = try std.DynLib.open("libvulkan.so.1");
        errdefer lib.close();

        const gip = lib.lookup(vk.PfnGetInstanceProcAddr, "vkGetInstanceProcAddr") orelse return error.SymbolNotFound;
        var base = vk.BaseWrapper{ .dispatch = .{} };
        base.dispatch.vkGetInstanceProcAddr = gip;
        base.dispatch.vkCreateInstance = @ptrCast(gip(.null_handle, "vkCreateInstance") orelse return error.SymbolNotFound);

        return .{
            .lib = lib,
            .get_instance_proc_addr = gip,
            .base = base,
            .instance_fns = undefined,
        };
    }

    fn close(self: *Vulkan) void {
        self.lib.close();
    }

    fn createInstance(self: *Vulkan) !vk.Instance {
        var app: vk.ApplicationInfo = .{
            .application_version = vk.makeApiVersion(0, 0, 1, 0).toU32(),
            .engine_version = vk.makeApiVersion(0, 0, 1, 0).toU32(),
            .api_version = vk.API_VERSION_1_4.toU32(),
        };
        app.p_application_name = "slang-gemm";
        app.p_engine_name = "none";

        var info: vk.InstanceCreateInfo = .{};
        info.p_application_info = &app;
        const instance = try self.base.createInstance(&info, null);
        self.instance_fns = vk.InstanceWrapper.load(instance, self.get_instance_proc_addr);
        return instance;
    }
};

const Device = struct {
    handle: vk.Device,
    fns: vk.DeviceWrapper,
    queue: vk.Queue,
    physical_device: vk.PhysicalDevice,
    queue_family: u32,
    supports_float64: bool,

    fn deinit(self: *Device) void {
        _ = self.fns.dispatch.vkDeviceWaitIdle.?(self.handle);
        self.fns.dispatch.vkDestroyDevice.?(self.handle, null);
    }
};

pub fn benchSgemmF32(allocator: std.mem.Allocator, size: usize, iters: usize, warmup: usize, device_substr: ?[]const u8) !?Result {
    return benchSgemmF32Case(allocator, .no_trans, .no_trans, size, size, size, iters, warmup, device_substr);
}

pub fn benchSgemmF32Case(allocator: std.mem.Allocator, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, iters: usize, warmup: usize, device_substr: ?[]const u8) !?Result {
    return benchSgemmF32BatchedCase(allocator, op_a, op_b, m, n, k, 1, iters, warmup, device_substr);
}

pub fn benchSgemmF32BatchedCase(allocator: std.mem.Allocator, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, batch_count: usize, iters: usize, warmup: usize, device_substr: ?[]const u8) !?Result {
    if (m == 0 or n == 0 or k == 0 or iters == 0) return error.InvalidArgument;
    if (batch_count == 0) return error.InvalidArgument;
    if (m > std.math.maxInt(u32) or n > std.math.maxInt(u32) or k > std.math.maxInt(u32) or batch_count > std.math.maxInt(u32)) return error.InvalidArgument;

    var loader = Vulkan.open() catch |err| switch (err) {
        error.FileNotFound, error.SymbolNotFound => return null,
        else => return err,
    };
    defer loader.close();

    const instance = loader.createInstance() catch |err| switch (err) {
        error.IncompatibleDriver => return null,
        else => return err,
    };
    defer loader.instance_fns.dispatch.vkDestroyInstance.?(instance, null);

    const selected = try selectPhysicalDevice(&loader, allocator, instance, device_substr);
    var device = try createDevice(&loader, selected.physical_device, selected.queue_family);
    defer device.deinit();

    const a_rows = storedRows(op_a, m, k);
    const a_cols = storedCols(op_a, m, k);
    const b_rows = storedRows(op_b, k, n);
    const b_cols = storedCols(op_b, k, n);
    const a_len = a_rows * a_cols;
    const b_len = b_rows * b_cols;
    const c_len = m * n;
    var params = try createBuffer(&loader, &device, @sizeOf(GemmParams), .{ .storage_buffer_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, params);
    var a = try createBuffer(&loader, &device, a_len * batch_count * @sizeOf(f32), .{ .storage_buffer_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, a);
    var b = try createBuffer(&loader, &device, b_len * batch_count * @sizeOf(f32), .{ .storage_buffer_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, b);
    var c = try createBuffer(&loader, &device, c_len * batch_count * @sizeOf(f32), .{ .storage_buffer_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, c);

    params.slice(GemmParams)[0] = .{
        .op_a = opCode(op_a),
        .op_b = opCode(op_b),
        .m = @intCast(m),
        .n = @intCast(n),
        .k = @intCast(k),
        .batch_count = @intCast(batch_count),
        .a_row_stride = 1,
        .a_col_stride = @intCast(a_rows),
        .b_row_stride = 1,
        .b_col_stride = @intCast(b_rows),
        .c_row_stride = 1,
        .c_col_stride = @intCast(m),
        .stride_a = @intCast(a_len),
        .stride_b = @intCast(b_len),
        .stride_c = @intCast(c_len),
    };
    fill(a.slice(f32), 0.1);
    fill(b.slice(f32), -0.2);
    @memset(c.slice(f32), 0);

    const shader_module = try createShaderModule(&device, f32_spv.words[0..]);
    defer device.fns.dispatch.vkDestroyShaderModule.?(device.handle, shader_module, null);
    const descriptor_layout = try createDescriptorSetLayout(&device);
    defer device.fns.dispatch.vkDestroyDescriptorSetLayout.?(device.handle, descriptor_layout, null);
    const pipeline_layout = try createPipelineLayout(&device, descriptor_layout);
    defer device.fns.dispatch.vkDestroyPipelineLayout.?(device.handle, pipeline_layout, null);
    const pipeline = try createPipeline(&device, pipeline_layout, shader_module);
    defer device.fns.dispatch.vkDestroyPipeline.?(device.handle, pipeline, null);
    const descriptor_pool = try createDescriptorPool(&device);
    defer device.fns.dispatch.vkDestroyDescriptorPool.?(device.handle, descriptor_pool, null);
    const descriptor_set = try allocateDescriptorSet(&device, descriptor_pool, descriptor_layout);
    updateDescriptorSet(&device, descriptor_set, params, a, b, c);

    const command_pool = try createCommandPool(&device);
    defer device.fns.dispatch.vkDestroyCommandPool.?(device.handle, command_pool, null);
    const warmup_cmd = try allocateCommandBuffer(&device, command_pool);
    const timed_cmd = try allocateCommandBuffer(&device, command_pool);
    const query_pool = try createTimestampQueryPool(&device);
    defer device.fns.dispatch.vkDestroyQueryPool.?(device.handle, query_pool, null);
    const dispatch_x = roundUpDiv(@as(u32, @intCast(n)), 64);
    const dispatch_y = roundUpDiv(@as(u32, @intCast(m)), 64);
    try recordCommands(&device, warmup_cmd, pipeline, pipeline_layout, descriptor_set, dispatch_x, dispatch_y, @intCast(batch_count), 1, .null_handle);
    try recordCommands(&device, timed_cmd, pipeline, pipeline_layout, descriptor_set, dispatch_x, dispatch_y, @intCast(batch_count), iters, query_pool);
    const fence = try createFence(&device);
    defer device.fns.dispatch.vkDestroyFence.?(device.handle, fence, null);

    try requireTimestampQueue(&loader, &device);
    for (0..warmup) |_| try submitCommand(&device, warmup_cmd, fence);
    try submitCommand(&device, timed_cmd, fence);

    var timestamps = [_]u64{ 0, 0 };
    try vkCheck(device.fns.dispatch.vkGetQueryPoolResults.?(
        device.handle,
        query_pool,
        0,
        2,
        @sizeOf(@TypeOf(timestamps)),
        &timestamps,
        @sizeOf(u64),
        vk.QueryResultFlags{ .@"64_bit" = true, .wait_bit = true },
    ));

    const elapsed_ticks = timestamps[1] - timestamps[0];
    const elapsed_ns = @as(f64, @floatFromInt(elapsed_ticks)) * @as(f64, timestampPeriodNs(&loader, device.physical_device));
    const ms = elapsed_ns / @as(f64, @floatFromInt(iters)) / 1.0e6;
    return .{ .ms = ms, .tflops = tflops(m, n, k, ms) * @as(f64, @floatFromInt(batch_count)) };
}

pub fn benchDgemmF64Case(allocator: std.mem.Allocator, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, iters: usize, warmup: usize, device_substr: ?[]const u8) !?Result {
    return benchDgemmF64BatchedCase(allocator, op_a, op_b, m, n, k, 1, iters, warmup, device_substr);
}

pub fn benchDgemmF64BatchedCase(allocator: std.mem.Allocator, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, batch_count: usize, iters: usize, warmup: usize, device_substr: ?[]const u8) !?Result {
    return benchShaderCase(f64, GemmParamsF64, allocator, f64_spv.words[0..], true, 32, op_a, op_b, m, n, k, batch_count, iters, warmup, device_substr);
}

pub fn benchCgemmF32Case(allocator: std.mem.Allocator, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, iters: usize, warmup: usize, device_substr: ?[]const u8) !?Result {
    return benchCgemmF32BatchedCase(allocator, op_a, op_b, m, n, k, 1, iters, warmup, device_substr);
}

pub fn benchCgemmF32BatchedCase(allocator: std.mem.Allocator, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, batch_count: usize, iters: usize, warmup: usize, device_substr: ?[]const u8) !?Result {
    return benchShaderCase(gemm.Complex32, GemmParams, allocator, complex_f32_spv.words[0..], false, 32, op_a, op_b, m, n, k, batch_count, iters, warmup, device_substr);
}

pub fn benchZgemmF64Case(allocator: std.mem.Allocator, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, iters: usize, warmup: usize, device_substr: ?[]const u8) !?Result {
    return benchZgemmF64BatchedCase(allocator, op_a, op_b, m, n, k, 1, iters, warmup, device_substr);
}

pub fn benchZgemmF64BatchedCase(allocator: std.mem.Allocator, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, batch_count: usize, iters: usize, warmup: usize, device_substr: ?[]const u8) !?Result {
    return benchShaderCase(gemm.Complex64, GemmParamsComplexF64, allocator, complex_f64_spv.words[0..], true, 16, op_a, op_b, m, n, k, batch_count, iters, warmup, device_substr);
}

pub fn benchLowpF32Case(allocator: std.mem.Allocator, op_a: gemm.Op, op_b: gemm.Op, a_type: gemm.DataType, b_type: gemm.DataType, m: usize, n: usize, k: usize, iters: usize, warmup: usize, device_substr: ?[]const u8) !?Result {
    return benchLowpF32BatchedCase(allocator, op_a, op_b, a_type, b_type, m, n, k, 1, iters, warmup, device_substr);
}

pub fn benchLowpF32BatchedCase(allocator: std.mem.Allocator, op_a: gemm.Op, op_b: gemm.Op, a_type: gemm.DataType, b_type: gemm.DataType, m: usize, n: usize, k: usize, batch_count: usize, iters: usize, warmup: usize, device_substr: ?[]const u8) !?Result {
    if (m == 0 or n == 0 or k == 0 or iters == 0) return error.InvalidArgument;
    if (batch_count == 0) return error.InvalidArgument;
    if (m > std.math.maxInt(u32) or n > std.math.maxInt(u32) or k > std.math.maxInt(u32) or batch_count > std.math.maxInt(u32)) return error.InvalidArgument;

    var loader = Vulkan.open() catch |err| switch (err) {
        error.FileNotFound, error.SymbolNotFound => return null,
        else => return err,
    };
    defer loader.close();
    const instance = loader.createInstance() catch |err| switch (err) {
        error.IncompatibleDriver => return null,
        else => return err,
    };
    defer loader.instance_fns.dispatch.vkDestroyInstance.?(instance, null);
    const selected = try selectPhysicalDevice(&loader, allocator, instance, device_substr);
    var device = try createDevice(&loader, selected.physical_device, selected.queue_family);
    defer device.deinit();

    const a_rows = storedRows(op_a, m, k);
    const a_cols = storedCols(op_a, m, k);
    const b_rows = storedRows(op_b, k, n);
    const b_cols = storedCols(op_b, k, n);
    const a_len = a_rows * a_cols;
    const b_len = b_rows * b_cols;
    const c_len = m * n;
    var params = try createBuffer(&loader, &device, @sizeOf(GemmParamsLowp), .{ .storage_buffer_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, params);
    var a = try createBuffer(&loader, &device, a_len * batch_count * @sizeOf(u32), .{ .storage_buffer_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, a);
    var b = try createBuffer(&loader, &device, b_len * batch_count * @sizeOf(u32), .{ .storage_buffer_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, b);
    var c = try createBuffer(&loader, &device, c_len * batch_count * @sizeOf(f32), .{ .storage_buffer_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, c);

    params.slice(GemmParamsLowp)[0] = .{ .base = makeParams(GemmParams, op_a, op_b, m, n, k, batch_count, a_rows, b_rows, a_len, b_len, c_len), .a_type = @intCast(@intFromEnum(a_type)), .b_type = @intCast(@intFromEnum(b_type)) };
    fillPackedLowp(a.slice(u32), a_type, 0.1);
    fillPackedLowp(b.slice(u32), b_type, -0.2);
    @memset(c.slice(f32), 0);
    return benchPrepared(&loader, &device, lowp_f32_spv.words[0..], params, a, b, c, m, n, k, batch_count, 32, iters, warmup);
}

pub fn benchInt8I32Case(allocator: std.mem.Allocator, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, iters: usize, warmup: usize, device_substr: ?[]const u8) !?Result {
    return benchInt8I32BatchedCase(allocator, op_a, op_b, m, n, k, 1, iters, warmup, device_substr);
}

pub fn benchInt8I32BatchedCase(allocator: std.mem.Allocator, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, batch_count: usize, iters: usize, warmup: usize, device_substr: ?[]const u8) !?Result {
    if (op_a == .no_trans and op_b == .no_trans) {
        if (try benchNvcoop2I8StridedBatchedCase(allocator, m, n, k, batch_count, iters, warmup, device_substr)) |fast| return fast;
    }
    if (m == 0 or n == 0 or k == 0 or iters == 0) return error.InvalidArgument;
    if (batch_count == 0) return error.InvalidArgument;
    if (m > std.math.maxInt(u32) or n > std.math.maxInt(u32) or k > std.math.maxInt(u32) or batch_count > std.math.maxInt(u32)) return error.InvalidArgument;

    var loader = Vulkan.open() catch |err| switch (err) {
        error.FileNotFound, error.SymbolNotFound => return null,
        else => return err,
    };
    defer loader.close();
    const instance = loader.createInstance() catch |err| switch (err) {
        error.IncompatibleDriver => return null,
        else => return err,
    };
    defer loader.instance_fns.dispatch.vkDestroyInstance.?(instance, null);
    const selected = try selectPhysicalDevice(&loader, allocator, instance, device_substr);
    var device = try createDevice(&loader, selected.physical_device, selected.queue_family);
    defer device.deinit();

    const a_rows = storedRows(op_a, m, k);
    const a_cols = storedCols(op_a, m, k);
    const b_rows = storedRows(op_b, k, n);
    const b_cols = storedCols(op_b, k, n);
    const a_len = a_rows * a_cols;
    const b_len = b_rows * b_cols;
    const c_len = m * n;
    var params = try createBuffer(&loader, &device, @sizeOf(GemmParamsI32), .{ .storage_buffer_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, params);
    var a = try createBuffer(&loader, &device, a_len * batch_count * @sizeOf(u32), .{ .storage_buffer_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, a);
    var b = try createBuffer(&loader, &device, b_len * batch_count * @sizeOf(u32), .{ .storage_buffer_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, b);
    var c = try createBuffer(&loader, &device, c_len * batch_count * @sizeOf(u32), .{ .storage_buffer_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, c);

    params.slice(GemmParamsI32)[0] = .{ .base = makeParams(GemmParams, op_a, op_b, m, n, k, batch_count, a_rows, b_rows, a_len, b_len, c_len), .a_type = @intCast(@intFromEnum(gemm.DataType.r_8i)), .b_type = @intCast(@intFromEnum(gemm.DataType.r_8i)), .c_type = @intCast(@intFromEnum(gemm.DataType.r_32i)) };
    fillPackedInt8(a.slice(u32), 1);
    fillPackedInt8(b.slice(u32), -2);
    @memset(c.slice(u32), 0);
    return benchPrepared(&loader, &device, int8_i32_spv.words[0..], params, a, b, c, m, n, k, batch_count, 32, iters, warmup);
}

pub fn benchNvcoop2I8Case(allocator: std.mem.Allocator, m: usize, n: usize, k: usize, iters: usize, warmup: usize, device_substr: ?[]const u8) !?Result {
    return benchNvcoop2I8StridedBatchedCase(allocator, m, n, k, 1, iters, warmup, device_substr);
}

pub fn benchNvcoop2I8StridedBatchedCase(allocator: std.mem.Allocator, m: usize, n: usize, k: usize, batch_count: usize, iters: usize, warmup: usize, device_substr: ?[]const u8) !?Result {
    if (m == 0 or n == 0 or k == 0 or batch_count == 0 or iters == 0) return error.InvalidArgument;
    if (m > std.math.maxInt(u32) or n > std.math.maxInt(u32) or k > std.math.maxInt(u32) or batch_count > std.math.maxInt(u32)) return error.InvalidArgument;
    if (m % 128 != 0 or n % 256 != 0 or k % 32 != 0) return null;

    var loader = Vulkan.open() catch |err| switch (err) {
        error.FileNotFound, error.SymbolNotFound => return null,
        else => return err,
    };
    defer loader.close();
    const instance = loader.createInstance() catch |err| switch (err) {
        error.IncompatibleDriver => return null,
        else => return err,
    };
    defer loader.instance_fns.dispatch.vkDestroyInstance.?(instance, null);
    const selected = try selectPhysicalDevice(&loader, allocator, instance, device_substr);
    requireNvcoop2I8(&loader, allocator, selected.physical_device) catch |err| switch (err) {
        error.RequiredDeviceExtensionMissing, error.RequiredDeviceFeatureMissing, error.RequiredCoopMatrixPropertyMissing => return null,
        else => return err,
    };
    var device = createNvcoop2I8Device(&loader, selected.physical_device, selected.queue_family) catch |err| switch (err) {
        error.RequiredDeviceExtensionMissing, error.RequiredDeviceFeatureMissing => return null,
        else => return err,
    };
    defer device.deinit();

    const a_stride = m * k;
    const b_stride = k * n;
    const c_stride = m * n;
    var a_stage = try createBuffer(&loader, &device, a_stride * batch_count, .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, a_stage);
    var b_stage = try createBuffer(&loader, &device, b_stride * batch_count, .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, b_stage);
    fillI8Bytes(a_stage.slice(u8), 1);
    fillI8Bytes(b_stage.slice(u8), -2);

    const a_dev = try createBuffer(&loader, &device, a_stage.byte_len, .{ .storage_buffer_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true }, false);
    defer destroyBuffer(&device, a_dev);
    const b_dev = try createBuffer(&loader, &device, b_stage.byte_len, .{ .storage_buffer_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true }, false);
    defer destroyBuffer(&device, b_dev);
    const c_dev = try createBuffer(&loader, &device, c_stride * batch_count * @sizeOf(i32), .{ .storage_buffer_bit = true, .transfer_src_bit = true }, .{ .device_local_bit = true }, false);
    defer destroyBuffer(&device, c_dev);

    const shader_module = try createShaderModule(&device, nvcoop2_i8_spv.words[0..]);
    defer device.fns.dispatch.vkDestroyShaderModule.?(device.handle, shader_module, null);
    const descriptor_layout = try createDescriptorSetLayout3(&device);
    defer device.fns.dispatch.vkDestroyDescriptorSetLayout.?(device.handle, descriptor_layout, null);
    const pipeline_layout = try createPipelineLayoutWithCoop2Push(&device, descriptor_layout);
    defer device.fns.dispatch.vkDestroyPipelineLayout.?(device.handle, pipeline_layout, null);
    const pipeline = try createPipeline(&device, pipeline_layout, shader_module);
    defer device.fns.dispatch.vkDestroyPipeline.?(device.handle, pipeline, null);
    const descriptor_pool = try createDescriptorPool3(&device);
    defer device.fns.dispatch.vkDestroyDescriptorPool.?(device.handle, descriptor_pool, null);
    const descriptor_set = try allocateDescriptorSet(&device, descriptor_pool, descriptor_layout);
    updateDescriptorSet3(&device, descriptor_set, a_dev, b_dev, c_dev);

    const command_pool = try createCommandPool(&device);
    defer device.fns.dispatch.vkDestroyCommandPool.?(device.handle, command_pool, null);
    const upload_cmd = try allocateCommandBuffer(&device, command_pool);
    const warmup_cmd = try allocateCommandBuffer(&device, command_pool);
    const timed_cmd = try allocateCommandBuffer(&device, command_pool);
    const query_pool = try createTimestampQueryPool(&device);
    defer device.fns.dispatch.vkDestroyQueryPool.?(device.handle, query_pool, null);

    try recordUploadCommands(&device, upload_cmd, a_stage, a_dev, b_stage, b_dev);
    try recordNvcoop2Commands(&device, warmup_cmd, pipeline, pipeline_layout, descriptor_set, m, n, k, 128, 256, batch_count, a_stride, b_stride, c_stride, 1, .null_handle);
    try recordNvcoop2Commands(&device, timed_cmd, pipeline, pipeline_layout, descriptor_set, m, n, k, 128, 256, batch_count, a_stride, b_stride, c_stride, iters, query_pool);

    const fence = try createFence(&device);
    defer device.fns.dispatch.vkDestroyFence.?(device.handle, fence, null);
    try requireTimestampQueue(&loader, &device);
    try submitCommand(&device, upload_cmd, fence);
    for (0..warmup) |_| try submitCommand(&device, warmup_cmd, fence);
    try submitCommand(&device, timed_cmd, fence);

    var timestamps = [_]u64{ 0, 0 };
    try vkCheck(device.fns.dispatch.vkGetQueryPoolResults.?(device.handle, query_pool, 0, 2, @sizeOf(@TypeOf(timestamps)), &timestamps, @sizeOf(u64), vk.QueryResultFlags{ .@"64_bit" = true, .wait_bit = true }));
    const elapsed_ticks = timestamps[1] - timestamps[0];
    const elapsed_ns = @as(f64, @floatFromInt(elapsed_ticks)) * @as(f64, timestampPeriodNs(&loader, device.physical_device));
    const ms = elapsed_ns / @as(f64, @floatFromInt(iters)) / 1.0e6;
    return .{ .ms = ms, .tflops = tflops(m, n, k, ms) * @as(f64, @floatFromInt(batch_count)) };
}

pub fn benchNvcoop2F16Case(allocator: std.mem.Allocator, m: usize, n: usize, k: usize, iters: usize, warmup: usize, device_substr: ?[]const u8) !?Result {
    if (m == 0 or n == 0 or k == 0 or iters == 0) return error.InvalidArgument;
    if (m > std.math.maxInt(u32) or n > std.math.maxInt(u32) or k > std.math.maxInt(u32)) return error.InvalidArgument;
    if (m % 128 != 0 or n % 256 != 0 or k % 16 != 0) return null;

    var loader = Vulkan.open() catch |err| switch (err) {
        error.FileNotFound, error.SymbolNotFound => return null,
        else => return err,
    };
    defer loader.close();
    const instance = loader.createInstance() catch |err| switch (err) {
        error.IncompatibleDriver => return null,
        else => return err,
    };
    defer loader.instance_fns.dispatch.vkDestroyInstance.?(instance, null);
    const selected = try selectPhysicalDevice(&loader, allocator, instance, device_substr);
    requireNvcoop2F16(&loader, allocator, selected.physical_device) catch |err| switch (err) {
        error.RequiredDeviceExtensionMissing, error.RequiredDeviceFeatureMissing, error.RequiredCoopMatrixPropertyMissing => return null,
        else => return err,
    };
    var device = createNvcoop2F16Device(&loader, selected.physical_device, selected.queue_family) catch |err| switch (err) {
        error.RequiredDeviceExtensionMissing, error.RequiredDeviceFeatureMissing => return null,
        else => return err,
    };
    defer device.deinit();

    const a_len = m * k;
    const b_len = k * n;
    const c_len = m * n;
    var a_stage = try createBuffer(&loader, &device, a_len * @sizeOf(u16), .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, a_stage);
    var b_stage = try createBuffer(&loader, &device, b_len * @sizeOf(u16), .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, b_stage);
    fillF16Bits(a_stage.slice(u16), 0.1);
    fillF16Bits(b_stage.slice(u16), -0.2);

    const a_dev = try createBuffer(&loader, &device, a_stage.byte_len, .{ .storage_buffer_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true }, false);
    defer destroyBuffer(&device, a_dev);
    const b_dev = try createBuffer(&loader, &device, b_stage.byte_len, .{ .storage_buffer_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true }, false);
    defer destroyBuffer(&device, b_dev);
    const c_dev = try createBuffer(&loader, &device, c_len * @sizeOf(f32), .{ .storage_buffer_bit = true, .transfer_src_bit = true }, .{ .device_local_bit = true }, false);
    defer destroyBuffer(&device, c_dev);

    const shader_module = try createShaderModule(&device, nvcoop2_f16_spv.words[0..]);
    defer device.fns.dispatch.vkDestroyShaderModule.?(device.handle, shader_module, null);
    const descriptor_layout = try createDescriptorSetLayout3(&device);
    defer device.fns.dispatch.vkDestroyDescriptorSetLayout.?(device.handle, descriptor_layout, null);
    const pipeline_layout = try createPipelineLayoutWithCoop2Push(&device, descriptor_layout);
    defer device.fns.dispatch.vkDestroyPipelineLayout.?(device.handle, pipeline_layout, null);
    const pipeline = try createPipeline(&device, pipeline_layout, shader_module);
    defer device.fns.dispatch.vkDestroyPipeline.?(device.handle, pipeline, null);
    const descriptor_pool = try createDescriptorPool3(&device);
    defer device.fns.dispatch.vkDestroyDescriptorPool.?(device.handle, descriptor_pool, null);
    const descriptor_set = try allocateDescriptorSet(&device, descriptor_pool, descriptor_layout);
    updateDescriptorSet3(&device, descriptor_set, a_dev, b_dev, c_dev);

    const command_pool = try createCommandPool(&device);
    defer device.fns.dispatch.vkDestroyCommandPool.?(device.handle, command_pool, null);
    const upload_cmd = try allocateCommandBuffer(&device, command_pool);
    const warmup_cmd = try allocateCommandBuffer(&device, command_pool);
    const timed_cmd = try allocateCommandBuffer(&device, command_pool);
    const query_pool = try createTimestampQueryPool(&device);
    defer device.fns.dispatch.vkDestroyQueryPool.?(device.handle, query_pool, null);

    try recordUploadCommands(&device, upload_cmd, a_stage, a_dev, b_stage, b_dev);
    try recordNvcoop2Commands(&device, warmup_cmd, pipeline, pipeline_layout, descriptor_set, m, n, k, 128, 256, 1, a_len, b_len, c_len, 1, .null_handle);
    try recordNvcoop2Commands(&device, timed_cmd, pipeline, pipeline_layout, descriptor_set, m, n, k, 128, 256, 1, a_len, b_len, c_len, iters, query_pool);

    const fence = try createFence(&device);
    defer device.fns.dispatch.vkDestroyFence.?(device.handle, fence, null);
    try requireTimestampQueue(&loader, &device);
    try submitCommand(&device, upload_cmd, fence);
    for (0..warmup) |_| try submitCommand(&device, warmup_cmd, fence);
    try submitCommand(&device, timed_cmd, fence);

    var timestamps = [_]u64{ 0, 0 };
    try vkCheck(device.fns.dispatch.vkGetQueryPoolResults.?(
        device.handle,
        query_pool,
        0,
        2,
        @sizeOf(@TypeOf(timestamps)),
        &timestamps,
        @sizeOf(u64),
        vk.QueryResultFlags{ .@"64_bit" = true, .wait_bit = true },
    ));

    const elapsed_ticks = timestamps[1] - timestamps[0];
    const elapsed_ns = @as(f64, @floatFromInt(elapsed_ticks)) * @as(f64, timestampPeriodNs(&loader, device.physical_device));
    const ms = elapsed_ns / @as(f64, @floatFromInt(iters)) / 1.0e6;
    return .{ .ms = ms, .tflops = tflops(m, n, k, ms) };
}

pub fn benchNvcoop2F16SmallCase(allocator: std.mem.Allocator, m: usize, n: usize, k: usize, iters: usize, warmup: usize, device_substr: ?[]const u8) !?Result {
    if (m == 0 or n == 0 or k == 0 or iters == 0) return error.InvalidArgument;
    if (m > std.math.maxInt(u32) or n > std.math.maxInt(u32) or k > std.math.maxInt(u32)) return error.InvalidArgument;
    if (m % 64 != 0 or n % 128 != 0 or k % 16 != 0) return null;

    var loader = Vulkan.open() catch |err| switch (err) {
        error.FileNotFound, error.SymbolNotFound => return null,
        else => return err,
    };
    defer loader.close();
    const instance = loader.createInstance() catch |err| switch (err) {
        error.IncompatibleDriver => return null,
        else => return err,
    };
    defer loader.instance_fns.dispatch.vkDestroyInstance.?(instance, null);
    const selected = try selectPhysicalDevice(&loader, allocator, instance, device_substr);
    requireNvcoop2F16(&loader, allocator, selected.physical_device) catch |err| switch (err) {
        error.RequiredDeviceExtensionMissing, error.RequiredDeviceFeatureMissing, error.RequiredCoopMatrixPropertyMissing => return null,
        else => return err,
    };
    var device = createNvcoop2F16Device(&loader, selected.physical_device, selected.queue_family) catch |err| switch (err) {
        error.RequiredDeviceExtensionMissing, error.RequiredDeviceFeatureMissing => return null,
        else => return err,
    };
    defer device.deinit();

    const a_len = m * k;
    const b_len = k * n;
    const c_len = m * n;
    var a_stage = try createBuffer(&loader, &device, a_len * @sizeOf(u16), .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, a_stage);
    var b_stage = try createBuffer(&loader, &device, b_len * @sizeOf(u16), .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, b_stage);
    fillF16Bits(a_stage.slice(u16), 0.1);
    fillF16Bits(b_stage.slice(u16), -0.2);

    const a_dev = try createBuffer(&loader, &device, a_stage.byte_len, .{ .storage_buffer_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true }, false);
    defer destroyBuffer(&device, a_dev);
    const b_dev = try createBuffer(&loader, &device, b_stage.byte_len, .{ .storage_buffer_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true }, false);
    defer destroyBuffer(&device, b_dev);
    const c_dev = try createBuffer(&loader, &device, c_len * @sizeOf(f32), .{ .storage_buffer_bit = true, .transfer_src_bit = true }, .{ .device_local_bit = true }, false);
    defer destroyBuffer(&device, c_dev);

    const shader_module = try createShaderModule(&device, nvcoop2_f16_small_spv.words[0..]);
    defer device.fns.dispatch.vkDestroyShaderModule.?(device.handle, shader_module, null);
    const descriptor_layout = try createDescriptorSetLayout3(&device);
    defer device.fns.dispatch.vkDestroyDescriptorSetLayout.?(device.handle, descriptor_layout, null);
    const pipeline_layout = try createPipelineLayoutWithCoop2Push(&device, descriptor_layout);
    defer device.fns.dispatch.vkDestroyPipelineLayout.?(device.handle, pipeline_layout, null);
    const pipeline = try createPipeline(&device, pipeline_layout, shader_module);
    defer device.fns.dispatch.vkDestroyPipeline.?(device.handle, pipeline, null);
    const descriptor_pool = try createDescriptorPool3(&device);
    defer device.fns.dispatch.vkDestroyDescriptorPool.?(device.handle, descriptor_pool, null);
    const descriptor_set = try allocateDescriptorSet(&device, descriptor_pool, descriptor_layout);
    updateDescriptorSet3(&device, descriptor_set, a_dev, b_dev, c_dev);

    const command_pool = try createCommandPool(&device);
    defer device.fns.dispatch.vkDestroyCommandPool.?(device.handle, command_pool, null);
    const upload_cmd = try allocateCommandBuffer(&device, command_pool);
    const warmup_cmd = try allocateCommandBuffer(&device, command_pool);
    const timed_cmd = try allocateCommandBuffer(&device, command_pool);
    const query_pool = try createTimestampQueryPool(&device);
    defer device.fns.dispatch.vkDestroyQueryPool.?(device.handle, query_pool, null);

    try recordUploadCommands(&device, upload_cmd, a_stage, a_dev, b_stage, b_dev);
    try recordNvcoop2Commands(&device, warmup_cmd, pipeline, pipeline_layout, descriptor_set, m, n, k, 64, 128, 1, a_len, b_len, c_len, 1, .null_handle);
    try recordNvcoop2Commands(&device, timed_cmd, pipeline, pipeline_layout, descriptor_set, m, n, k, 64, 128, 1, a_len, b_len, c_len, iters, query_pool);

    const fence = try createFence(&device);
    defer device.fns.dispatch.vkDestroyFence.?(device.handle, fence, null);
    try requireTimestampQueue(&loader, &device);
    try submitCommand(&device, upload_cmd, fence);
    for (0..warmup) |_| try submitCommand(&device, warmup_cmd, fence);
    try submitCommand(&device, timed_cmd, fence);

    var timestamps = [_]u64{ 0, 0 };
    try vkCheck(device.fns.dispatch.vkGetQueryPoolResults.?(device.handle, query_pool, 0, 2, @sizeOf(@TypeOf(timestamps)), &timestamps, @sizeOf(u64), vk.QueryResultFlags{ .@"64_bit" = true, .wait_bit = true }));
    const elapsed_ticks = timestamps[1] - timestamps[0];
    const elapsed_ns = @as(f64, @floatFromInt(elapsed_ticks)) * @as(f64, timestampPeriodNs(&loader, device.physical_device));
    const ms = elapsed_ns / @as(f64, @floatFromInt(iters)) / 1.0e6;
    return .{ .ms = ms, .tflops = tflops(m, n, k, ms) };
}

pub fn benchNvcoop2F16StridedBatchedCase(allocator: std.mem.Allocator, m: usize, n: usize, k: usize, batch_count: usize, iters: usize, warmup: usize, device_substr: ?[]const u8) !?Result {
    return benchNvcoop2F16StridedBatchedOpCase(allocator, .no_trans, .no_trans, m, n, k, batch_count, iters, warmup, device_substr);
}

pub fn benchNvcoop2F16StridedBatchedOpCase(allocator: std.mem.Allocator, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, batch_count: usize, iters: usize, warmup: usize, device_substr: ?[]const u8) !?Result {
    if (m == 0 or n == 0 or k == 0 or batch_count == 0 or iters == 0) return error.InvalidArgument;
    if (m > std.math.maxInt(u32) or n > std.math.maxInt(u32) or k > std.math.maxInt(u32) or batch_count > std.math.maxInt(u32)) return error.InvalidArgument;
    if (m % 128 != 0 or n % 256 != 0 or k % 16 != 0) return null;

    var loader = Vulkan.open() catch |err| switch (err) {
        error.FileNotFound, error.SymbolNotFound => return null,
        else => return err,
    };
    defer loader.close();
    const instance = loader.createInstance() catch |err| switch (err) {
        error.IncompatibleDriver => return null,
        else => return err,
    };
    defer loader.instance_fns.dispatch.vkDestroyInstance.?(instance, null);
    const selected = try selectPhysicalDevice(&loader, allocator, instance, device_substr);
    requireNvcoop2F16(&loader, allocator, selected.physical_device) catch |err| switch (err) {
        error.RequiredDeviceExtensionMissing, error.RequiredDeviceFeatureMissing, error.RequiredCoopMatrixPropertyMissing => return null,
        else => return err,
    };
    var device = createNvcoop2F16Device(&loader, selected.physical_device, selected.queue_family) catch |err| switch (err) {
        error.RequiredDeviceExtensionMissing, error.RequiredDeviceFeatureMissing => return null,
        else => return err,
    };
    defer device.deinit();

    const a_rows = storedRows(op_a, m, k);
    const a_cols = storedCols(op_a, m, k);
    const b_rows = storedRows(op_b, k, n);
    const b_cols = storedCols(op_b, k, n);
    const a_stride = a_rows * a_cols;
    const b_stride = b_rows * b_cols;
    const c_stride = m * n;
    var a_stage = try createBuffer(&loader, &device, a_stride * batch_count * @sizeOf(u16), .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, a_stage);
    var b_stage = try createBuffer(&loader, &device, b_stride * batch_count * @sizeOf(u16), .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, b_stage);
    fillF16Bits(a_stage.slice(u16), 0.1);
    fillF16Bits(b_stage.slice(u16), -0.2);

    const a_dev = try createBuffer(&loader, &device, a_stage.byte_len, .{ .storage_buffer_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true }, false);
    defer destroyBuffer(&device, a_dev);
    const b_dev = try createBuffer(&loader, &device, b_stage.byte_len, .{ .storage_buffer_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true }, false);
    defer destroyBuffer(&device, b_dev);
    const c_dev = try createBuffer(&loader, &device, c_stride * batch_count * @sizeOf(f32), .{ .storage_buffer_bit = true, .transfer_src_bit = true }, .{ .device_local_bit = true }, false);
    defer destroyBuffer(&device, c_dev);

    const shader_module = try createShaderModule(&device, nvcoop2_f16_spv.words[0..]);
    defer device.fns.dispatch.vkDestroyShaderModule.?(device.handle, shader_module, null);
    const descriptor_layout = try createDescriptorSetLayout3(&device);
    defer device.fns.dispatch.vkDestroyDescriptorSetLayout.?(device.handle, descriptor_layout, null);
    const pipeline_layout = try createPipelineLayoutWithCoop2Push(&device, descriptor_layout);
    defer device.fns.dispatch.vkDestroyPipelineLayout.?(device.handle, pipeline_layout, null);
    const pipeline = try createPipeline(&device, pipeline_layout, shader_module);
    defer device.fns.dispatch.vkDestroyPipeline.?(device.handle, pipeline, null);
    const descriptor_pool = try createDescriptorPool3(&device);
    defer device.fns.dispatch.vkDestroyDescriptorPool.?(device.handle, descriptor_pool, null);
    const descriptor_set = try allocateDescriptorSet(&device, descriptor_pool, descriptor_layout);
    updateDescriptorSet3(&device, descriptor_set, a_dev, b_dev, c_dev);

    const command_pool = try createCommandPool(&device);
    defer device.fns.dispatch.vkDestroyCommandPool.?(device.handle, command_pool, null);
    const upload_cmd = try allocateCommandBuffer(&device, command_pool);
    const warmup_cmd = try allocateCommandBuffer(&device, command_pool);
    const timed_cmd = try allocateCommandBuffer(&device, command_pool);
    const query_pool = try createTimestampQueryPool(&device);
    defer device.fns.dispatch.vkDestroyQueryPool.?(device.handle, query_pool, null);

    try recordUploadCommands(&device, upload_cmd, a_stage, a_dev, b_stage, b_dev);
    try recordNvcoop2CommandsOp(&device, warmup_cmd, pipeline, pipeline_layout, descriptor_set, op_a, op_b, m, n, k, 128, 256, a_cols, b_cols, batch_count, a_stride, b_stride, c_stride, 1, .null_handle);
    try recordNvcoop2CommandsOp(&device, timed_cmd, pipeline, pipeline_layout, descriptor_set, op_a, op_b, m, n, k, 128, 256, a_cols, b_cols, batch_count, a_stride, b_stride, c_stride, iters, query_pool);

    const fence = try createFence(&device);
    defer device.fns.dispatch.vkDestroyFence.?(device.handle, fence, null);
    try requireTimestampQueue(&loader, &device);
    try submitCommand(&device, upload_cmd, fence);
    for (0..warmup) |_| try submitCommand(&device, warmup_cmd, fence);
    try submitCommand(&device, timed_cmd, fence);

    var timestamps = [_]u64{ 0, 0 };
    try vkCheck(device.fns.dispatch.vkGetQueryPoolResults.?(
        device.handle,
        query_pool,
        0,
        2,
        @sizeOf(@TypeOf(timestamps)),
        &timestamps,
        @sizeOf(u64),
        vk.QueryResultFlags{ .@"64_bit" = true, .wait_bit = true },
    ));

    const elapsed_ticks = timestamps[1] - timestamps[0];
    const elapsed_ns = @as(f64, @floatFromInt(elapsed_ticks)) * @as(f64, timestampPeriodNs(&loader, device.physical_device));
    const ms = elapsed_ns / @as(f64, @floatFromInt(iters)) / 1.0e6;
    return .{ .ms = ms, .tflops = tflops(m, n, k, ms) * @as(f64, @floatFromInt(batch_count)) };
}

pub fn benchNvcoop2Bf16Case(allocator: std.mem.Allocator, m: usize, n: usize, k: usize, iters: usize, warmup: usize, device_substr: ?[]const u8) !?Result {
    if (m == 0 or n == 0 or k == 0 or iters == 0) return error.InvalidArgument;
    if (m > std.math.maxInt(u32) or n > std.math.maxInt(u32) or k > std.math.maxInt(u32)) return error.InvalidArgument;
    if (m % 64 != 0 or n % 128 != 0 or k % 16 != 0) return null;

    var loader = Vulkan.open() catch |err| switch (err) {
        error.FileNotFound, error.SymbolNotFound => return null,
        else => return err,
    };
    defer loader.close();
    const instance = loader.createInstance() catch |err| switch (err) {
        error.IncompatibleDriver => return null,
        else => return err,
    };
    defer loader.instance_fns.dispatch.vkDestroyInstance.?(instance, null);
    const selected = try selectPhysicalDevice(&loader, allocator, instance, device_substr);
    requireNvcoop2Bf16(&loader, allocator, selected.physical_device) catch |err| switch (err) {
        error.RequiredDeviceExtensionMissing, error.RequiredDeviceFeatureMissing, error.RequiredCoopMatrixPropertyMissing => return null,
        else => return err,
    };
    var device = createNvcoop2Bf16Device(&loader, selected.physical_device, selected.queue_family) catch |err| switch (err) {
        error.RequiredDeviceExtensionMissing, error.RequiredDeviceFeatureMissing => return null,
        else => return err,
    };
    defer device.deinit();

    const a_len = m * k;
    const b_len = k * n;
    const c_len = m * n;
    var a_stage = try createBuffer(&loader, &device, a_len * @sizeOf(u16), .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, a_stage);
    var b_stage = try createBuffer(&loader, &device, b_len * @sizeOf(u16), .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, b_stage);
    fillBf16Bits(a_stage.slice(u16), 0.1);
    fillBf16Bits(b_stage.slice(u16), -0.2);

    const a_dev = try createBuffer(&loader, &device, a_stage.byte_len, .{ .storage_buffer_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true }, false);
    defer destroyBuffer(&device, a_dev);
    const b_dev = try createBuffer(&loader, &device, b_stage.byte_len, .{ .storage_buffer_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true }, false);
    defer destroyBuffer(&device, b_dev);
    const c_dev = try createBuffer(&loader, &device, c_len * @sizeOf(f32), .{ .storage_buffer_bit = true, .transfer_src_bit = true }, .{ .device_local_bit = true }, false);
    defer destroyBuffer(&device, c_dev);

    const shader_module = try createShaderModule(&device, nvcoop2_bf16_spv.words[0..]);
    defer device.fns.dispatch.vkDestroyShaderModule.?(device.handle, shader_module, null);
    const descriptor_layout = try createDescriptorSetLayout3(&device);
    defer device.fns.dispatch.vkDestroyDescriptorSetLayout.?(device.handle, descriptor_layout, null);
    const pipeline_layout = try createPipelineLayoutWithCoop2Push(&device, descriptor_layout);
    defer device.fns.dispatch.vkDestroyPipelineLayout.?(device.handle, pipeline_layout, null);
    const pipeline = try createPipeline(&device, pipeline_layout, shader_module);
    defer device.fns.dispatch.vkDestroyPipeline.?(device.handle, pipeline, null);
    const descriptor_pool = try createDescriptorPool3(&device);
    defer device.fns.dispatch.vkDestroyDescriptorPool.?(device.handle, descriptor_pool, null);
    const descriptor_set = try allocateDescriptorSet(&device, descriptor_pool, descriptor_layout);
    updateDescriptorSet3(&device, descriptor_set, a_dev, b_dev, c_dev);

    const command_pool = try createCommandPool(&device);
    defer device.fns.dispatch.vkDestroyCommandPool.?(device.handle, command_pool, null);
    const upload_cmd = try allocateCommandBuffer(&device, command_pool);
    const warmup_cmd = try allocateCommandBuffer(&device, command_pool);
    const timed_cmd = try allocateCommandBuffer(&device, command_pool);
    const query_pool = try createTimestampQueryPool(&device);
    defer device.fns.dispatch.vkDestroyQueryPool.?(device.handle, query_pool, null);

    try recordUploadCommands(&device, upload_cmd, a_stage, a_dev, b_stage, b_dev);
    try recordNvcoop2Commands(&device, warmup_cmd, pipeline, pipeline_layout, descriptor_set, m, n, k, 64, 128, 1, a_len, b_len, c_len, 1, .null_handle);
    try recordNvcoop2Commands(&device, timed_cmd, pipeline, pipeline_layout, descriptor_set, m, n, k, 64, 128, 1, a_len, b_len, c_len, iters, query_pool);

    const fence = try createFence(&device);
    defer device.fns.dispatch.vkDestroyFence.?(device.handle, fence, null);
    try requireTimestampQueue(&loader, &device);
    try submitCommand(&device, upload_cmd, fence);
    for (0..warmup) |_| try submitCommand(&device, warmup_cmd, fence);
    try submitCommand(&device, timed_cmd, fence);

    var timestamps = [_]u64{ 0, 0 };
    try vkCheck(device.fns.dispatch.vkGetQueryPoolResults.?(
        device.handle,
        query_pool,
        0,
        2,
        @sizeOf(@TypeOf(timestamps)),
        &timestamps,
        @sizeOf(u64),
        vk.QueryResultFlags{ .@"64_bit" = true, .wait_bit = true },
    ));

    const elapsed_ticks = timestamps[1] - timestamps[0];
    const elapsed_ns = @as(f64, @floatFromInt(elapsed_ticks)) * @as(f64, timestampPeriodNs(&loader, device.physical_device));
    const ms = elapsed_ns / @as(f64, @floatFromInt(iters)) / 1.0e6;
    return .{ .ms = ms, .tflops = tflops(m, n, k, ms) };
}

pub fn benchNvcoop2Bf16StridedBatchedCase(allocator: std.mem.Allocator, m: usize, n: usize, k: usize, batch_count: usize, iters: usize, warmup: usize, device_substr: ?[]const u8) !?Result {
    return benchNvcoop2Bf16StridedBatchedOpCase(allocator, .no_trans, .no_trans, m, n, k, batch_count, iters, warmup, device_substr);
}

pub fn benchNvcoop2Bf16StridedBatchedOpCase(allocator: std.mem.Allocator, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, batch_count: usize, iters: usize, warmup: usize, device_substr: ?[]const u8) !?Result {
    if (m == 0 or n == 0 or k == 0 or batch_count == 0 or iters == 0) return error.InvalidArgument;
    if (m > std.math.maxInt(u32) or n > std.math.maxInt(u32) or k > std.math.maxInt(u32) or batch_count > std.math.maxInt(u32)) return error.InvalidArgument;
    if (m % 64 != 0 or n % 128 != 0 or k % 16 != 0) return null;

    var loader = Vulkan.open() catch |err| switch (err) {
        error.FileNotFound, error.SymbolNotFound => return null,
        else => return err,
    };
    defer loader.close();
    const instance = loader.createInstance() catch |err| switch (err) {
        error.IncompatibleDriver => return null,
        else => return err,
    };
    defer loader.instance_fns.dispatch.vkDestroyInstance.?(instance, null);
    const selected = try selectPhysicalDevice(&loader, allocator, instance, device_substr);
    requireNvcoop2Bf16(&loader, allocator, selected.physical_device) catch |err| switch (err) {
        error.RequiredDeviceExtensionMissing, error.RequiredDeviceFeatureMissing, error.RequiredCoopMatrixPropertyMissing => return null,
        else => return err,
    };
    var device = createNvcoop2Bf16Device(&loader, selected.physical_device, selected.queue_family) catch |err| switch (err) {
        error.RequiredDeviceExtensionMissing, error.RequiredDeviceFeatureMissing => return null,
        else => return err,
    };
    defer device.deinit();

    const a_rows = storedRows(op_a, m, k);
    const a_cols = storedCols(op_a, m, k);
    const b_rows = storedRows(op_b, k, n);
    const b_cols = storedCols(op_b, k, n);
    const a_stride = a_rows * a_cols;
    const b_stride = b_rows * b_cols;
    const c_stride = m * n;
    var a_stage = try createBuffer(&loader, &device, a_stride * batch_count * @sizeOf(u16), .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, a_stage);
    var b_stage = try createBuffer(&loader, &device, b_stride * batch_count * @sizeOf(u16), .{ .transfer_src_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, b_stage);
    fillBf16Bits(a_stage.slice(u16), 0.1);
    fillBf16Bits(b_stage.slice(u16), -0.2);

    const a_dev = try createBuffer(&loader, &device, a_stage.byte_len, .{ .storage_buffer_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true }, false);
    defer destroyBuffer(&device, a_dev);
    const b_dev = try createBuffer(&loader, &device, b_stage.byte_len, .{ .storage_buffer_bit = true, .transfer_dst_bit = true }, .{ .device_local_bit = true }, false);
    defer destroyBuffer(&device, b_dev);
    const c_dev = try createBuffer(&loader, &device, c_stride * batch_count * @sizeOf(f32), .{ .storage_buffer_bit = true, .transfer_src_bit = true }, .{ .device_local_bit = true }, false);
    defer destroyBuffer(&device, c_dev);

    const shader_module = try createShaderModule(&device, nvcoop2_bf16_spv.words[0..]);
    defer device.fns.dispatch.vkDestroyShaderModule.?(device.handle, shader_module, null);
    const descriptor_layout = try createDescriptorSetLayout3(&device);
    defer device.fns.dispatch.vkDestroyDescriptorSetLayout.?(device.handle, descriptor_layout, null);
    const pipeline_layout = try createPipelineLayoutWithCoop2Push(&device, descriptor_layout);
    defer device.fns.dispatch.vkDestroyPipelineLayout.?(device.handle, pipeline_layout, null);
    const pipeline = try createPipeline(&device, pipeline_layout, shader_module);
    defer device.fns.dispatch.vkDestroyPipeline.?(device.handle, pipeline, null);
    const descriptor_pool = try createDescriptorPool3(&device);
    defer device.fns.dispatch.vkDestroyDescriptorPool.?(device.handle, descriptor_pool, null);
    const descriptor_set = try allocateDescriptorSet(&device, descriptor_pool, descriptor_layout);
    updateDescriptorSet3(&device, descriptor_set, a_dev, b_dev, c_dev);

    const command_pool = try createCommandPool(&device);
    defer device.fns.dispatch.vkDestroyCommandPool.?(device.handle, command_pool, null);
    const upload_cmd = try allocateCommandBuffer(&device, command_pool);
    const warmup_cmd = try allocateCommandBuffer(&device, command_pool);
    const timed_cmd = try allocateCommandBuffer(&device, command_pool);
    const query_pool = try createTimestampQueryPool(&device);
    defer device.fns.dispatch.vkDestroyQueryPool.?(device.handle, query_pool, null);

    try recordUploadCommands(&device, upload_cmd, a_stage, a_dev, b_stage, b_dev);
    try recordNvcoop2CommandsOp(&device, warmup_cmd, pipeline, pipeline_layout, descriptor_set, op_a, op_b, m, n, k, 64, 128, a_cols, b_cols, batch_count, a_stride, b_stride, c_stride, 1, .null_handle);
    try recordNvcoop2CommandsOp(&device, timed_cmd, pipeline, pipeline_layout, descriptor_set, op_a, op_b, m, n, k, 64, 128, a_cols, b_cols, batch_count, a_stride, b_stride, c_stride, iters, query_pool);

    const fence = try createFence(&device);
    defer device.fns.dispatch.vkDestroyFence.?(device.handle, fence, null);
    try requireTimestampQueue(&loader, &device);
    try submitCommand(&device, upload_cmd, fence);
    for (0..warmup) |_| try submitCommand(&device, warmup_cmd, fence);
    try submitCommand(&device, timed_cmd, fence);

    var timestamps = [_]u64{ 0, 0 };
    try vkCheck(device.fns.dispatch.vkGetQueryPoolResults.?(device.handle, query_pool, 0, 2, @sizeOf(@TypeOf(timestamps)), &timestamps, @sizeOf(u64), vk.QueryResultFlags{ .@"64_bit" = true, .wait_bit = true }));
    const elapsed_ticks = timestamps[1] - timestamps[0];
    const elapsed_ns = @as(f64, @floatFromInt(elapsed_ticks)) * @as(f64, timestampPeriodNs(&loader, device.physical_device));
    const ms = elapsed_ns / @as(f64, @floatFromInt(iters)) / 1.0e6;
    return .{ .ms = ms, .tflops = tflops(m, n, k, ms) * @as(f64, @floatFromInt(batch_count)) };
}

pub fn benchF16ScalarCase(allocator: std.mem.Allocator, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, batch_count: usize, iters: usize, warmup: usize, device_substr: ?[]const u8) !?Result {
    if (m == 0 or n == 0 or k == 0 or batch_count == 0 or iters == 0) return error.InvalidArgument;
    if (m > std.math.maxInt(u32) or n > std.math.maxInt(u32) or k > std.math.maxInt(u32) or batch_count > std.math.maxInt(u32)) return error.InvalidArgument;

    var loader = Vulkan.open() catch |err| switch (err) {
        error.FileNotFound, error.SymbolNotFound => return null,
        else => return err,
    };
    defer loader.close();
    const instance = loader.createInstance() catch |err| switch (err) {
        error.IncompatibleDriver => return null,
        else => return err,
    };
    defer loader.instance_fns.dispatch.vkDestroyInstance.?(instance, null);
    const selected = try selectPhysicalDevice(&loader, allocator, instance, device_substr);
    var device = try createDevice(&loader, selected.physical_device, selected.queue_family);
    defer device.deinit();

    const a_rows = storedRows(op_a, m, k);
    const a_cols = storedCols(op_a, m, k);
    const b_rows = storedRows(op_b, k, n);
    const b_cols = storedCols(op_b, k, n);
    const a_len = a_rows * a_cols;
    const b_len = b_rows * b_cols;
    const c_len = m * n;
    var params = try createBuffer(&loader, &device, @sizeOf(GemmParams), .{ .storage_buffer_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, params);
    var a = try createBuffer(&loader, &device, a_len * batch_count * @sizeOf(u16), .{ .storage_buffer_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, a);
    var b = try createBuffer(&loader, &device, b_len * batch_count * @sizeOf(u16), .{ .storage_buffer_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, b);
    var c = try createBuffer(&loader, &device, c_len * batch_count * @sizeOf(u16), .{ .storage_buffer_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, c);

    params.slice(GemmParams)[0] = makeParams(GemmParams, op_a, op_b, m, n, k, batch_count, a_rows, b_rows, a_len, b_len, c_len);
    fillF16Bits(a.slice(u16), 0.1);
    fillF16Bits(b.slice(u16), -0.2);
    @memset(c.slice(u16), 0);
    return benchPrepared(&loader, &device, f16_scalar_spv.words[0..], params, a, b, c, m, n, k, batch_count, 16, iters, warmup);
}

fn benchShaderCase(comptime T: type, comptime Params: type, allocator: std.mem.Allocator, shader_words: []const u32, needs_float64: bool, output_tile: u32, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, batch_count: usize, iters: usize, warmup: usize, device_substr: ?[]const u8) !?Result {
    if (m == 0 or n == 0 or k == 0 or iters == 0) return error.InvalidArgument;
    if (batch_count == 0) return error.InvalidArgument;
    if (m > std.math.maxInt(u32) or n > std.math.maxInt(u32) or k > std.math.maxInt(u32) or batch_count > std.math.maxInt(u32)) return error.InvalidArgument;

    var loader = Vulkan.open() catch |err| switch (err) {
        error.FileNotFound, error.SymbolNotFound => return null,
        else => return err,
    };
    defer loader.close();

    const instance = loader.createInstance() catch |err| switch (err) {
        error.IncompatibleDriver => return null,
        else => return err,
    };
    defer loader.instance_fns.dispatch.vkDestroyInstance.?(instance, null);

    const selected = try selectPhysicalDevice(&loader, allocator, instance, device_substr);
    var device = try createDevice(&loader, selected.physical_device, selected.queue_family);
    defer device.deinit();
    if (needs_float64 and !device.supports_float64) return null;

    const a_rows = storedRows(op_a, m, k);
    const a_cols = storedCols(op_a, m, k);
    const b_rows = storedRows(op_b, k, n);
    const b_cols = storedCols(op_b, k, n);
    const a_len = a_rows * a_cols;
    const b_len = b_rows * b_cols;
    const c_len = m * n;
    var params = try createBuffer(&loader, &device, @sizeOf(Params), .{ .storage_buffer_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, params);
    var a = try createBuffer(&loader, &device, a_len * batch_count * @sizeOf(T), .{ .storage_buffer_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, a);
    var b = try createBuffer(&loader, &device, b_len * batch_count * @sizeOf(T), .{ .storage_buffer_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, b);
    var c = try createBuffer(&loader, &device, c_len * batch_count * @sizeOf(T), .{ .storage_buffer_bit = true }, .{ .host_visible_bit = true, .host_coherent_bit = true }, true);
    defer destroyBuffer(&device, c);

    params.slice(Params)[0] = makeParams(Params, op_a, op_b, m, n, k, batch_count, a_rows, b_rows, a_len, b_len, c_len);
    fillTyped(T, a.slice(T), 0.1);
    fillTyped(T, b.slice(T), -0.2);
    zeroTyped(T, c.slice(T));

    const shader_module = try createShaderModule(&device, shader_words);
    defer device.fns.dispatch.vkDestroyShaderModule.?(device.handle, shader_module, null);
    const descriptor_layout = try createDescriptorSetLayout(&device);
    defer device.fns.dispatch.vkDestroyDescriptorSetLayout.?(device.handle, descriptor_layout, null);
    const pipeline_layout = try createPipelineLayout(&device, descriptor_layout);
    defer device.fns.dispatch.vkDestroyPipelineLayout.?(device.handle, pipeline_layout, null);
    const pipeline = try createPipeline(&device, pipeline_layout, shader_module);
    defer device.fns.dispatch.vkDestroyPipeline.?(device.handle, pipeline, null);
    const descriptor_pool = try createDescriptorPool(&device);
    defer device.fns.dispatch.vkDestroyDescriptorPool.?(device.handle, descriptor_pool, null);
    const descriptor_set = try allocateDescriptorSet(&device, descriptor_pool, descriptor_layout);
    updateDescriptorSet(&device, descriptor_set, params, a, b, c);

    const command_pool = try createCommandPool(&device);
    defer device.fns.dispatch.vkDestroyCommandPool.?(device.handle, command_pool, null);
    const warmup_cmd = try allocateCommandBuffer(&device, command_pool);
    const timed_cmd = try allocateCommandBuffer(&device, command_pool);
    const query_pool = try createTimestampQueryPool(&device);
    defer device.fns.dispatch.vkDestroyQueryPool.?(device.handle, query_pool, null);
    const dispatch_x = roundUpDiv(@as(u32, @intCast(n)), output_tile);
    const dispatch_y = roundUpDiv(@as(u32, @intCast(m)), output_tile);
    try recordCommands(&device, warmup_cmd, pipeline, pipeline_layout, descriptor_set, dispatch_x, dispatch_y, @intCast(batch_count), 1, .null_handle);
    try recordCommands(&device, timed_cmd, pipeline, pipeline_layout, descriptor_set, dispatch_x, dispatch_y, @intCast(batch_count), iters, query_pool);
    const fence = try createFence(&device);
    defer device.fns.dispatch.vkDestroyFence.?(device.handle, fence, null);

    try requireTimestampQueue(&loader, &device);
    for (0..warmup) |_| try submitCommand(&device, warmup_cmd, fence);
    try submitCommand(&device, timed_cmd, fence);

    var timestamps = [_]u64{ 0, 0 };
    try vkCheck(device.fns.dispatch.vkGetQueryPoolResults.?(
        device.handle,
        query_pool,
        0,
        2,
        @sizeOf(@TypeOf(timestamps)),
        &timestamps,
        @sizeOf(u64),
        vk.QueryResultFlags{ .@"64_bit" = true, .wait_bit = true },
    ));

    const elapsed_ticks = timestamps[1] - timestamps[0];
    const elapsed_ns = @as(f64, @floatFromInt(elapsed_ticks)) * @as(f64, timestampPeriodNs(&loader, device.physical_device));
    const ms = elapsed_ns / @as(f64, @floatFromInt(iters)) / 1.0e6;
    return .{ .ms = ms, .tflops = tflops(m, n, k, ms) * @as(f64, @floatFromInt(batch_count)) };
}

fn benchPrepared(loader: *Vulkan, device: *Device, shader_words: []const u32, params: Buffer, a: Buffer, b: Buffer, c: Buffer, m: usize, n: usize, k: usize, batch_count: usize, output_tile: u32, iters: usize, warmup: usize) !?Result {
    const shader_module = try createShaderModule(device, shader_words);
    defer device.fns.dispatch.vkDestroyShaderModule.?(device.handle, shader_module, null);
    const descriptor_layout = try createDescriptorSetLayout(device);
    defer device.fns.dispatch.vkDestroyDescriptorSetLayout.?(device.handle, descriptor_layout, null);
    const pipeline_layout = try createPipelineLayout(device, descriptor_layout);
    defer device.fns.dispatch.vkDestroyPipelineLayout.?(device.handle, pipeline_layout, null);
    const pipeline = try createPipeline(device, pipeline_layout, shader_module);
    defer device.fns.dispatch.vkDestroyPipeline.?(device.handle, pipeline, null);
    const descriptor_pool = try createDescriptorPool(device);
    defer device.fns.dispatch.vkDestroyDescriptorPool.?(device.handle, descriptor_pool, null);
    const descriptor_set = try allocateDescriptorSet(device, descriptor_pool, descriptor_layout);
    updateDescriptorSet(device, descriptor_set, params, a, b, c);

    const command_pool = try createCommandPool(device);
    defer device.fns.dispatch.vkDestroyCommandPool.?(device.handle, command_pool, null);
    const warmup_cmd = try allocateCommandBuffer(device, command_pool);
    const timed_cmd = try allocateCommandBuffer(device, command_pool);
    const query_pool = try createTimestampQueryPool(device);
    defer device.fns.dispatch.vkDestroyQueryPool.?(device.handle, query_pool, null);
    const dispatch_x = roundUpDiv(@as(u32, @intCast(n)), output_tile);
    const dispatch_y = roundUpDiv(@as(u32, @intCast(m)), output_tile);
    try recordCommands(device, warmup_cmd, pipeline, pipeline_layout, descriptor_set, dispatch_x, dispatch_y, @intCast(batch_count), 1, .null_handle);
    try recordCommands(device, timed_cmd, pipeline, pipeline_layout, descriptor_set, dispatch_x, dispatch_y, @intCast(batch_count), iters, query_pool);
    const fence = try createFence(device);
    defer device.fns.dispatch.vkDestroyFence.?(device.handle, fence, null);

    try requireTimestampQueue(loader, device);
    for (0..warmup) |_| try submitCommand(device, warmup_cmd, fence);
    try submitCommand(device, timed_cmd, fence);

    var timestamps = [_]u64{ 0, 0 };
    try vkCheck(device.fns.dispatch.vkGetQueryPoolResults.?(
        device.handle,
        query_pool,
        0,
        2,
        @sizeOf(@TypeOf(timestamps)),
        &timestamps,
        @sizeOf(u64),
        vk.QueryResultFlags{ .@"64_bit" = true, .wait_bit = true },
    ));

    const elapsed_ticks = timestamps[1] - timestamps[0];
    const elapsed_ns = @as(f64, @floatFromInt(elapsed_ticks)) * @as(f64, timestampPeriodNs(loader, device.physical_device));
    const ms = elapsed_ns / @as(f64, @floatFromInt(iters)) / 1.0e6;
    return .{ .ms = ms, .tflops = tflops(m, n, k, ms) * @as(f64, @floatFromInt(batch_count)) };
}

const SelectedDevice = struct {
    physical_device: vk.PhysicalDevice,
    queue_family: u32,
};

fn selectPhysicalDevice(loader: *Vulkan, allocator: std.mem.Allocator, instance: vk.Instance, device_substr: ?[]const u8) !SelectedDevice {
    var count: u32 = 0;
    try vkCheck(loader.instance_fns.dispatch.vkEnumeratePhysicalDevices.?(instance, &count, null));
    if (count == 0) return error.NoVulkanDevice;
    const devices = try allocator.alloc(vk.PhysicalDevice, count);
    defer allocator.free(devices);
    try vkCheck(loader.instance_fns.dispatch.vkEnumeratePhysicalDevices.?(instance, &count, devices.ptr));

    var fallback: ?SelectedDevice = null;
    for (devices[0..count]) |physical_device| {
        var props: vk.PhysicalDeviceProperties = undefined;
        loader.instance_fns.dispatch.vkGetPhysicalDeviceProperties.?(physical_device, &props);
        const name = std.mem.sliceTo(&props.device_name, 0);
        if (device_substr) |needle| {
            if (std.mem.indexOf(u8, name, needle) == null) continue;
        }
        if (try findComputeQueueFamily(loader, allocator, physical_device)) |queue_family| {
            if (props.device_type == vk.PhysicalDeviceType.cpu and device_substr == null) {
                if (fallback == null) fallback = .{ .physical_device = physical_device, .queue_family = queue_family };
                continue;
            }
            return .{ .physical_device = physical_device, .queue_family = queue_family };
        }
    }
    if (fallback) |selected| return selected;
    return error.NoMatchingDevice;
}

fn findComputeQueueFamily(loader: *Vulkan, allocator: std.mem.Allocator, physical_device: vk.PhysicalDevice) !?u32 {
    var count: u32 = 0;
    loader.instance_fns.dispatch.vkGetPhysicalDeviceQueueFamilyProperties.?(physical_device, &count, null);
    if (count == 0) return null;
    const families = try allocator.alloc(vk.QueueFamilyProperties, count);
    defer allocator.free(families);
    loader.instance_fns.dispatch.vkGetPhysicalDeviceQueueFamilyProperties.?(physical_device, &count, families.ptr);
    for (families[0..count], 0..) |family, i| {
        if (family.queue_flags.compute_bit) return @intCast(i);
    }
    return null;
}

fn createDevice(loader: *Vulkan, physical_device: vk.PhysicalDevice, queue_family: u32) !Device {
    var available_features: vk.PhysicalDeviceFeatures = undefined;
    loader.instance_fns.dispatch.vkGetPhysicalDeviceFeatures.?(physical_device, &available_features);
    var enabled_features: vk.PhysicalDeviceFeatures = .{};
    if (available_features.shader_float_64 == .true) {
        enabled_features.shader_float_64 = .true;
    }

    var priorities = [_]f32{1.0};
    var queue_info: vk.DeviceQueueCreateInfo = std.mem.zeroes(vk.DeviceQueueCreateInfo);
    queue_info.s_type = .device_queue_create_info;
    queue_info.queue_family_index = queue_family;
    queue_info.queue_count = 1;
    queue_info.p_queue_priorities = priorities[0..].ptr;

    var device_info: vk.DeviceCreateInfo = std.mem.zeroes(vk.DeviceCreateInfo);
    device_info.s_type = .device_create_info;
    device_info.queue_create_info_count = 1;
    var queue_infos = [_]vk.DeviceQueueCreateInfo{queue_info};
    device_info.p_queue_create_infos = queue_infos[0..].ptr;
    device_info.p_enabled_features = &enabled_features;

    var handle: vk.Device = .null_handle;
    try vkCheck(loader.instance_fns.dispatch.vkCreateDevice.?(physical_device, &device_info, null, &handle));
    const fns = vk.DeviceWrapper.load(handle, loader.instance_fns.dispatch.vkGetDeviceProcAddr.?);
    var queue: vk.Queue = .null_handle;
    fns.dispatch.vkGetDeviceQueue.?(handle, queue_family, 0, &queue);
    return .{ .handle = handle, .fns = fns, .queue = queue, .physical_device = physical_device, .queue_family = queue_family, .supports_float64 = enabled_features.shader_float_64 == .true };
}

fn createNvcoop2F16Device(loader: *Vulkan, physical_device: vk.PhysicalDevice, queue_family: u32) !Device {
    var priorities = [_]f32{1.0};
    var queue_info: vk.DeviceQueueCreateInfo = std.mem.zeroes(vk.DeviceQueueCreateInfo);
    queue_info.s_type = .device_queue_create_info;
    queue_info.queue_family_index = queue_family;
    queue_info.queue_count = 1;
    queue_info.p_queue_priorities = priorities[0..].ptr;

    var device_info: vk.DeviceCreateInfo = std.mem.zeroes(vk.DeviceCreateInfo);
    device_info.s_type = .device_create_info;
    device_info.queue_create_info_count = 1;
    var queue_infos = [_]vk.DeviceQueueCreateInfo{queue_info};
    device_info.p_queue_create_infos = queue_infos[0..].ptr;

    var extensions = [_][*:0]const u8{
        vk.extensions.khr_cooperative_matrix.name,
        vk.extensions.nv_cooperative_matrix_2.name,
    };
    device_info.enabled_extension_count = extensions.len;
    device_info.pp_enabled_extension_names = extensions[0..].ptr;

    var coop_features: vk.PhysicalDeviceCooperativeMatrixFeaturesKHR = std.mem.zeroes(vk.PhysicalDeviceCooperativeMatrixFeaturesKHR);
    coop_features.s_type = .physical_device_cooperative_matrix_features_khr;
    coop_features.cooperative_matrix = .true;

    var nvcoop2_features: vk.PhysicalDeviceCooperativeMatrix2FeaturesNV = std.mem.zeroes(vk.PhysicalDeviceCooperativeMatrix2FeaturesNV);
    nvcoop2_features.s_type = .physical_device_cooperative_matrix_2_features_nv;
    nvcoop2_features.cooperative_matrix_workgroup_scope = .true;
    nvcoop2_features.cooperative_matrix_flexible_dimensions = .true;
    nvcoop2_features.cooperative_matrix_tensor_addressing = .true;
    nvcoop2_features.cooperative_matrix_block_loads = .true;
    coop_features.p_next = &nvcoop2_features;

    var storage16_features: vk.PhysicalDevice16BitStorageFeatures = std.mem.zeroes(vk.PhysicalDevice16BitStorageFeatures);
    storage16_features.s_type = .physical_device_16bit_storage_features;
    storage16_features.storage_buffer_16_bit_access = .true;
    storage16_features.p_next = &coop_features;

    var f16_features: vk.PhysicalDeviceShaderFloat16Int8Features = std.mem.zeroes(vk.PhysicalDeviceShaderFloat16Int8Features);
    f16_features.s_type = .physical_device_shader_float16_int8_features;
    f16_features.shader_float_16 = .true;
    f16_features.p_next = &storage16_features;
    device_info.p_next = &f16_features;

    var handle: vk.Device = .null_handle;
    const create_result = loader.instance_fns.dispatch.vkCreateDevice.?(physical_device, &device_info, null, &handle);
    if (create_result == .error_extension_not_present) return error.RequiredDeviceExtensionMissing;
    if (create_result == .error_feature_not_present) return error.RequiredDeviceFeatureMissing;
    try vkCheck(create_result);
    const fns = vk.DeviceWrapper.load(handle, loader.instance_fns.dispatch.vkGetDeviceProcAddr.?);
    var queue: vk.Queue = .null_handle;
    fns.dispatch.vkGetDeviceQueue.?(handle, queue_family, 0, &queue);
    return .{ .handle = handle, .fns = fns, .queue = queue, .physical_device = physical_device, .queue_family = queue_family, .supports_float64 = false };
}

fn createNvcoop2Bf16Device(loader: *Vulkan, physical_device: vk.PhysicalDevice, queue_family: u32) !Device {
    var priorities = [_]f32{1.0};
    var queue_info: vk.DeviceQueueCreateInfo = std.mem.zeroes(vk.DeviceQueueCreateInfo);
    queue_info.s_type = .device_queue_create_info;
    queue_info.queue_family_index = queue_family;
    queue_info.queue_count = 1;
    queue_info.p_queue_priorities = priorities[0..].ptr;

    var device_info: vk.DeviceCreateInfo = std.mem.zeroes(vk.DeviceCreateInfo);
    device_info.s_type = .device_create_info;
    device_info.queue_create_info_count = 1;
    var queue_infos = [_]vk.DeviceQueueCreateInfo{queue_info};
    device_info.p_queue_create_infos = queue_infos[0..].ptr;

    var extensions = [_][*:0]const u8{
        vk.extensions.khr_cooperative_matrix.name,
        vk.extensions.khr_shader_bfloat_16.name,
        vk.extensions.nv_cooperative_matrix_2.name,
    };
    device_info.enabled_extension_count = extensions.len;
    device_info.pp_enabled_extension_names = extensions[0..].ptr;

    var coop_features: vk.PhysicalDeviceCooperativeMatrixFeaturesKHR = std.mem.zeroes(vk.PhysicalDeviceCooperativeMatrixFeaturesKHR);
    coop_features.s_type = .physical_device_cooperative_matrix_features_khr;
    coop_features.cooperative_matrix = .true;

    var nvcoop2_features: vk.PhysicalDeviceCooperativeMatrix2FeaturesNV = std.mem.zeroes(vk.PhysicalDeviceCooperativeMatrix2FeaturesNV);
    nvcoop2_features.s_type = .physical_device_cooperative_matrix_2_features_nv;
    nvcoop2_features.cooperative_matrix_workgroup_scope = .true;
    nvcoop2_features.cooperative_matrix_flexible_dimensions = .true;
    nvcoop2_features.cooperative_matrix_tensor_addressing = .true;
    nvcoop2_features.cooperative_matrix_block_loads = .true;
    coop_features.p_next = &nvcoop2_features;

    var storage16_features: vk.PhysicalDevice16BitStorageFeatures = std.mem.zeroes(vk.PhysicalDevice16BitStorageFeatures);
    storage16_features.s_type = .physical_device_16bit_storage_features;
    storage16_features.storage_buffer_16_bit_access = .true;
    storage16_features.p_next = &coop_features;

    var bf16_features: vk.PhysicalDeviceShaderBfloat16FeaturesKHR = std.mem.zeroes(vk.PhysicalDeviceShaderBfloat16FeaturesKHR);
    bf16_features.s_type = .physical_device_shader_bfloat16_features_khr;
    bf16_features.shader_b_float_16_type = .true;
    bf16_features.shader_b_float_16_cooperative_matrix = .true;
    bf16_features.p_next = &storage16_features;
    device_info.p_next = &bf16_features;

    var handle: vk.Device = .null_handle;
    const create_result = loader.instance_fns.dispatch.vkCreateDevice.?(physical_device, &device_info, null, &handle);
    if (create_result == .error_extension_not_present) return error.RequiredDeviceExtensionMissing;
    if (create_result == .error_feature_not_present) return error.RequiredDeviceFeatureMissing;
    try vkCheck(create_result);
    const fns = vk.DeviceWrapper.load(handle, loader.instance_fns.dispatch.vkGetDeviceProcAddr.?);
    var queue: vk.Queue = .null_handle;
    fns.dispatch.vkGetDeviceQueue.?(handle, queue_family, 0, &queue);
    return .{ .handle = handle, .fns = fns, .queue = queue, .physical_device = physical_device, .queue_family = queue_family, .supports_float64 = false };
}

fn createNvcoop2I8Device(loader: *Vulkan, physical_device: vk.PhysicalDevice, queue_family: u32) !Device {
    var priorities = [_]f32{1.0};
    var queue_info: vk.DeviceQueueCreateInfo = std.mem.zeroes(vk.DeviceQueueCreateInfo);
    queue_info.s_type = .device_queue_create_info;
    queue_info.queue_family_index = queue_family;
    queue_info.queue_count = 1;
    queue_info.p_queue_priorities = priorities[0..].ptr;

    var device_info: vk.DeviceCreateInfo = std.mem.zeroes(vk.DeviceCreateInfo);
    device_info.s_type = .device_create_info;
    device_info.queue_create_info_count = 1;
    var queue_infos = [_]vk.DeviceQueueCreateInfo{queue_info};
    device_info.p_queue_create_infos = queue_infos[0..].ptr;

    var extensions = [_][*:0]const u8{
        vk.extensions.khr_cooperative_matrix.name,
        vk.extensions.nv_cooperative_matrix_2.name,
    };
    device_info.enabled_extension_count = extensions.len;
    device_info.pp_enabled_extension_names = extensions[0..].ptr;

    var coop_features: vk.PhysicalDeviceCooperativeMatrixFeaturesKHR = std.mem.zeroes(vk.PhysicalDeviceCooperativeMatrixFeaturesKHR);
    coop_features.s_type = .physical_device_cooperative_matrix_features_khr;
    coop_features.cooperative_matrix = .true;

    var nvcoop2_features: vk.PhysicalDeviceCooperativeMatrix2FeaturesNV = std.mem.zeroes(vk.PhysicalDeviceCooperativeMatrix2FeaturesNV);
    nvcoop2_features.s_type = .physical_device_cooperative_matrix_2_features_nv;
    nvcoop2_features.cooperative_matrix_workgroup_scope = .true;
    nvcoop2_features.cooperative_matrix_flexible_dimensions = .true;
    nvcoop2_features.cooperative_matrix_tensor_addressing = .true;
    nvcoop2_features.cooperative_matrix_block_loads = .true;
    coop_features.p_next = &nvcoop2_features;

    var i8_features: vk.PhysicalDeviceShaderFloat16Int8Features = std.mem.zeroes(vk.PhysicalDeviceShaderFloat16Int8Features);
    i8_features.s_type = .physical_device_shader_float16_int8_features;
    i8_features.shader_int_8 = .true;
    i8_features.p_next = &coop_features;
    device_info.p_next = &i8_features;

    var handle: vk.Device = .null_handle;
    const create_result = loader.instance_fns.dispatch.vkCreateDevice.?(physical_device, &device_info, null, &handle);
    if (create_result == .error_extension_not_present) return error.RequiredDeviceExtensionMissing;
    if (create_result == .error_feature_not_present) return error.RequiredDeviceFeatureMissing;
    try vkCheck(create_result);
    const fns = vk.DeviceWrapper.load(handle, loader.instance_fns.dispatch.vkGetDeviceProcAddr.?);
    var queue: vk.Queue = .null_handle;
    fns.dispatch.vkGetDeviceQueue.?(handle, queue_family, 0, &queue);
    return .{ .handle = handle, .fns = fns, .queue = queue, .physical_device = physical_device, .queue_family = queue_family, .supports_float64 = false };
}

fn requireNvcoop2F16(loader: *Vulkan, allocator: std.mem.Allocator, physical_device: vk.PhysicalDevice) !void {
    var base_props: vk.PhysicalDeviceProperties = undefined;
    loader.instance_fns.dispatch.vkGetPhysicalDeviceProperties.?(physical_device, &base_props);
    if (base_props.vendor_id != 0x10de) return error.RequiredDeviceFeatureMissing;

    var extension_count: u32 = 0;
    try vkCheck(loader.instance_fns.dispatch.vkEnumerateDeviceExtensionProperties.?(physical_device, null, &extension_count, null));
    const extensions = try allocator.alloc(vk.ExtensionProperties, extension_count);
    defer allocator.free(extensions);
    try vkCheck(loader.instance_fns.dispatch.vkEnumerateDeviceExtensionProperties.?(physical_device, null, &extension_count, extensions.ptr));
    if (!hasDeviceExtension(extensions, vk.extensions.khr_cooperative_matrix.name) or !hasDeviceExtension(extensions, vk.extensions.nv_cooperative_matrix_2.name)) {
        return error.RequiredDeviceExtensionMissing;
    }

    var nv2_features: vk.PhysicalDeviceCooperativeMatrix2FeaturesNV = std.mem.zeroes(vk.PhysicalDeviceCooperativeMatrix2FeaturesNV);
    nv2_features.s_type = .physical_device_cooperative_matrix_2_features_nv;
    var khr_features: vk.PhysicalDeviceCooperativeMatrixFeaturesKHR = std.mem.zeroes(vk.PhysicalDeviceCooperativeMatrixFeaturesKHR);
    khr_features.s_type = .physical_device_cooperative_matrix_features_khr;
    khr_features.p_next = &nv2_features;
    var features2: vk.PhysicalDeviceFeatures2 = std.mem.zeroes(vk.PhysicalDeviceFeatures2);
    features2.s_type = .physical_device_features_2;
    features2.p_next = &khr_features;
    loader.instance_fns.dispatch.vkGetPhysicalDeviceFeatures2.?(physical_device, &features2);
    if (khr_features.cooperative_matrix != .true or
        nv2_features.cooperative_matrix_workgroup_scope != .true or
        nv2_features.cooperative_matrix_flexible_dimensions != .true or
        nv2_features.cooperative_matrix_tensor_addressing != .true or
        nv2_features.cooperative_matrix_block_loads != .true)
    {
        return error.RequiredDeviceFeatureMissing;
    }

    var nv2_props: vk.PhysicalDeviceCooperativeMatrix2PropertiesNV = std.mem.zeroes(vk.PhysicalDeviceCooperativeMatrix2PropertiesNV);
    nv2_props.s_type = .physical_device_cooperative_matrix_2_properties_nv;
    var properties2: vk.PhysicalDeviceProperties2 = std.mem.zeroes(vk.PhysicalDeviceProperties2);
    properties2.s_type = .physical_device_properties_2;
    properties2.p_next = &nv2_props;
    loader.instance_fns.dispatch.vkGetPhysicalDeviceProperties2.?(physical_device, &properties2);
    if (nv2_props.cooperative_matrix_workgroup_scope_max_workgroup_size < 256 or nv2_props.cooperative_matrix_flexible_dimensions_max_dimension < 256) {
        return error.RequiredCoopMatrixPropertyMissing;
    }

    const get_props = loader.instance_fns.dispatch.vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV orelse return error.RequiredDeviceExtensionMissing;
    var count: u32 = 0;
    try vkCheck(get_props(physical_device, &count, null));
    if (count == 0) return error.RequiredCoopMatrixPropertyMissing;
    const props = try allocator.alloc(vk.CooperativeMatrixFlexibleDimensionsPropertiesNV, count);
    defer allocator.free(props);
    for (props) |*prop| {
        prop.* = std.mem.zeroes(vk.CooperativeMatrixFlexibleDimensionsPropertiesNV);
        prop.s_type = .cooperative_matrix_flexible_dimensions_properties_nv;
    }
    try vkCheck(get_props(physical_device, &count, props.ptr));
    for (props[0..count]) |prop| {
        if (128 % prop.m_granularity == 0 and 256 % prop.n_granularity == 0 and 32 % prop.k_granularity == 0 and
            prop.a_type == .float16_khr and prop.b_type == .float16_khr and
            prop.c_type == .float32_khr and prop.result_type == .float32_khr and
            prop.saturating_accumulation == .false and prop.scope == .workgroup_khr and
            prop.workgroup_invocations == 256)
        {
            return;
        }
    }
    return error.RequiredCoopMatrixPropertyMissing;
}

fn requireNvcoop2Bf16(loader: *Vulkan, allocator: std.mem.Allocator, physical_device: vk.PhysicalDevice) !void {
    var base_props: vk.PhysicalDeviceProperties = undefined;
    loader.instance_fns.dispatch.vkGetPhysicalDeviceProperties.?(physical_device, &base_props);
    if (base_props.vendor_id != 0x10de) return error.RequiredDeviceFeatureMissing;

    var extension_count: u32 = 0;
    try vkCheck(loader.instance_fns.dispatch.vkEnumerateDeviceExtensionProperties.?(physical_device, null, &extension_count, null));
    const extensions = try allocator.alloc(vk.ExtensionProperties, extension_count);
    defer allocator.free(extensions);
    try vkCheck(loader.instance_fns.dispatch.vkEnumerateDeviceExtensionProperties.?(physical_device, null, &extension_count, extensions.ptr));
    if (!hasDeviceExtension(extensions, vk.extensions.khr_cooperative_matrix.name) or
        !hasDeviceExtension(extensions, vk.extensions.khr_shader_bfloat_16.name) or
        !hasDeviceExtension(extensions, vk.extensions.nv_cooperative_matrix_2.name))
    {
        return error.RequiredDeviceExtensionMissing;
    }

    var nv2_features: vk.PhysicalDeviceCooperativeMatrix2FeaturesNV = std.mem.zeroes(vk.PhysicalDeviceCooperativeMatrix2FeaturesNV);
    nv2_features.s_type = .physical_device_cooperative_matrix_2_features_nv;
    var khr_features: vk.PhysicalDeviceCooperativeMatrixFeaturesKHR = std.mem.zeroes(vk.PhysicalDeviceCooperativeMatrixFeaturesKHR);
    khr_features.s_type = .physical_device_cooperative_matrix_features_khr;
    khr_features.p_next = &nv2_features;
    var bf16_features: vk.PhysicalDeviceShaderBfloat16FeaturesKHR = std.mem.zeroes(vk.PhysicalDeviceShaderBfloat16FeaturesKHR);
    bf16_features.s_type = .physical_device_shader_bfloat16_features_khr;
    bf16_features.p_next = &khr_features;
    var features2: vk.PhysicalDeviceFeatures2 = std.mem.zeroes(vk.PhysicalDeviceFeatures2);
    features2.s_type = .physical_device_features_2;
    features2.p_next = &bf16_features;
    loader.instance_fns.dispatch.vkGetPhysicalDeviceFeatures2.?(physical_device, &features2);
    if (bf16_features.shader_b_float_16_type != .true or
        bf16_features.shader_b_float_16_cooperative_matrix != .true or
        khr_features.cooperative_matrix != .true or
        nv2_features.cooperative_matrix_workgroup_scope != .true or
        nv2_features.cooperative_matrix_flexible_dimensions != .true or
        nv2_features.cooperative_matrix_tensor_addressing != .true or
        nv2_features.cooperative_matrix_block_loads != .true)
    {
        return error.RequiredDeviceFeatureMissing;
    }

    var nv2_props: vk.PhysicalDeviceCooperativeMatrix2PropertiesNV = std.mem.zeroes(vk.PhysicalDeviceCooperativeMatrix2PropertiesNV);
    nv2_props.s_type = .physical_device_cooperative_matrix_2_properties_nv;
    var properties2: vk.PhysicalDeviceProperties2 = std.mem.zeroes(vk.PhysicalDeviceProperties2);
    properties2.s_type = .physical_device_properties_2;
    properties2.p_next = &nv2_props;
    loader.instance_fns.dispatch.vkGetPhysicalDeviceProperties2.?(physical_device, &properties2);
    if (nv2_props.cooperative_matrix_workgroup_scope_max_workgroup_size < 256 or nv2_props.cooperative_matrix_flexible_dimensions_max_dimension < 128) {
        return error.RequiredCoopMatrixPropertyMissing;
    }

    const get_props = loader.instance_fns.dispatch.vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV orelse return error.RequiredDeviceExtensionMissing;
    var count: u32 = 0;
    try vkCheck(get_props(physical_device, &count, null));
    if (count == 0) return error.RequiredCoopMatrixPropertyMissing;
    const props = try allocator.alloc(vk.CooperativeMatrixFlexibleDimensionsPropertiesNV, count);
    defer allocator.free(props);
    for (props) |*prop| {
        prop.* = std.mem.zeroes(vk.CooperativeMatrixFlexibleDimensionsPropertiesNV);
        prop.s_type = .cooperative_matrix_flexible_dimensions_properties_nv;
    }
    try vkCheck(get_props(physical_device, &count, props.ptr));
    for (props[0..count]) |prop| {
        if (64 % prop.m_granularity == 0 and 128 % prop.n_granularity == 0 and 16 % prop.k_granularity == 0 and
            prop.a_type == .bfloat16_khr and prop.b_type == .bfloat16_khr and
            prop.c_type == .float32_khr and prop.result_type == .float32_khr and
            prop.saturating_accumulation == .false and prop.scope == .workgroup_khr and
            prop.workgroup_invocations == 256)
        {
            return;
        }
    }
    return error.RequiredCoopMatrixPropertyMissing;
}

fn requireNvcoop2I8(loader: *Vulkan, allocator: std.mem.Allocator, physical_device: vk.PhysicalDevice) !void {
    var base_props: vk.PhysicalDeviceProperties = undefined;
    loader.instance_fns.dispatch.vkGetPhysicalDeviceProperties.?(physical_device, &base_props);
    if (base_props.vendor_id != 0x10de) return error.RequiredDeviceFeatureMissing;

    var extension_count: u32 = 0;
    try vkCheck(loader.instance_fns.dispatch.vkEnumerateDeviceExtensionProperties.?(physical_device, null, &extension_count, null));
    const extensions = try allocator.alloc(vk.ExtensionProperties, extension_count);
    defer allocator.free(extensions);
    try vkCheck(loader.instance_fns.dispatch.vkEnumerateDeviceExtensionProperties.?(physical_device, null, &extension_count, extensions.ptr));
    if (!hasDeviceExtension(extensions, vk.extensions.khr_cooperative_matrix.name) or !hasDeviceExtension(extensions, vk.extensions.nv_cooperative_matrix_2.name)) {
        return error.RequiredDeviceExtensionMissing;
    }

    var nv2_features: vk.PhysicalDeviceCooperativeMatrix2FeaturesNV = std.mem.zeroes(vk.PhysicalDeviceCooperativeMatrix2FeaturesNV);
    nv2_features.s_type = .physical_device_cooperative_matrix_2_features_nv;
    var khr_features: vk.PhysicalDeviceCooperativeMatrixFeaturesKHR = std.mem.zeroes(vk.PhysicalDeviceCooperativeMatrixFeaturesKHR);
    khr_features.s_type = .physical_device_cooperative_matrix_features_khr;
    khr_features.p_next = &nv2_features;
    var features2: vk.PhysicalDeviceFeatures2 = std.mem.zeroes(vk.PhysicalDeviceFeatures2);
    features2.s_type = .physical_device_features_2;
    features2.p_next = &khr_features;
    loader.instance_fns.dispatch.vkGetPhysicalDeviceFeatures2.?(physical_device, &features2);
    if (khr_features.cooperative_matrix != .true or
        nv2_features.cooperative_matrix_workgroup_scope != .true or
        nv2_features.cooperative_matrix_flexible_dimensions != .true or
        nv2_features.cooperative_matrix_tensor_addressing != .true or
        nv2_features.cooperative_matrix_block_loads != .true)
    {
        return error.RequiredDeviceFeatureMissing;
    }

    var nv2_props: vk.PhysicalDeviceCooperativeMatrix2PropertiesNV = std.mem.zeroes(vk.PhysicalDeviceCooperativeMatrix2PropertiesNV);
    nv2_props.s_type = .physical_device_cooperative_matrix_2_properties_nv;
    var properties2: vk.PhysicalDeviceProperties2 = std.mem.zeroes(vk.PhysicalDeviceProperties2);
    properties2.s_type = .physical_device_properties_2;
    properties2.p_next = &nv2_props;
    loader.instance_fns.dispatch.vkGetPhysicalDeviceProperties2.?(physical_device, &properties2);
    if (nv2_props.cooperative_matrix_workgroup_scope_max_workgroup_size < 256 or nv2_props.cooperative_matrix_flexible_dimensions_max_dimension < 256) {
        return error.RequiredCoopMatrixPropertyMissing;
    }

    const get_props = loader.instance_fns.dispatch.vkGetPhysicalDeviceCooperativeMatrixFlexibleDimensionsPropertiesNV orelse return error.RequiredDeviceExtensionMissing;
    var count: u32 = 0;
    try vkCheck(get_props(physical_device, &count, null));
    if (count == 0) return error.RequiredCoopMatrixPropertyMissing;
    const props = try allocator.alloc(vk.CooperativeMatrixFlexibleDimensionsPropertiesNV, count);
    defer allocator.free(props);
    for (props) |*prop| {
        prop.* = std.mem.zeroes(vk.CooperativeMatrixFlexibleDimensionsPropertiesNV);
        prop.s_type = .cooperative_matrix_flexible_dimensions_properties_nv;
    }
    try vkCheck(get_props(physical_device, &count, props.ptr));
    for (props[0..count]) |prop| {
        if (128 % prop.m_granularity == 0 and 256 % prop.n_granularity == 0 and 16 % prop.k_granularity == 0 and
            prop.a_type == .sint8_khr and prop.b_type == .sint8_khr and
            prop.c_type == .sint32_khr and prop.result_type == .sint32_khr and
            prop.saturating_accumulation == .false and prop.scope == .workgroup_khr and
            prop.workgroup_invocations == 256)
        {
            return;
        }
    }
    return error.RequiredCoopMatrixPropertyMissing;
}

fn hasDeviceExtension(extensions: []const vk.ExtensionProperties, needle: []const u8) bool {
    for (extensions) |extension| {
        const name = std.mem.sliceTo(&extension.extension_name, 0);
        if (std.mem.eql(u8, name, needle)) return true;
    }
    return false;
}

fn createBuffer(loader: *Vulkan, device: *Device, byte_len: usize, usage: vk.BufferUsageFlags, required: vk.MemoryPropertyFlags, map: bool) !Buffer {
    var info: vk.BufferCreateInfo = std.mem.zeroes(vk.BufferCreateInfo);
    info.s_type = .buffer_create_info;
    info.size = byte_len;
    info.usage = usage;
    info.sharing_mode = .exclusive;

    var handle: vk.Buffer = .null_handle;
    try vkCheck(device.fns.dispatch.vkCreateBuffer.?(device.handle, &info, null, &handle));
    errdefer device.fns.dispatch.vkDestroyBuffer.?(device.handle, handle, null);

    var reqs: vk.MemoryRequirements = undefined;
    device.fns.dispatch.vkGetBufferMemoryRequirements.?(device.handle, handle, &reqs);
    const memory_type = try findMemoryType(loader, device.physical_device, reqs.memory_type_bits, required);

    var alloc: vk.MemoryAllocateInfo = std.mem.zeroes(vk.MemoryAllocateInfo);
    alloc.s_type = .memory_allocate_info;
    alloc.allocation_size = reqs.size;
    alloc.memory_type_index = memory_type;

    var memory: vk.DeviceMemory = .null_handle;
    try vkCheck(device.fns.dispatch.vkAllocateMemory.?(device.handle, &alloc, null, &memory));
    errdefer device.fns.dispatch.vkFreeMemory.?(device.handle, memory, null);
    try vkCheck(device.fns.dispatch.vkBindBufferMemory.?(device.handle, handle, memory, 0));

    var mapped: ?[*]u8 = null;
    if (map) {
        var raw: ?*anyopaque = null;
        try vkCheck(device.fns.dispatch.vkMapMemory.?(device.handle, memory, 0, alloc.allocation_size, .{}, &raw));
        mapped = @ptrCast(raw.?);
    }
    return .{ .handle = handle, .memory = memory, .mapped = mapped, .byte_len = byte_len };
}

fn destroyBuffer(device: *Device, buffer: Buffer) void {
    if (buffer.mapped != null) device.fns.dispatch.vkUnmapMemory.?(device.handle, buffer.memory);
    device.fns.dispatch.vkDestroyBuffer.?(device.handle, buffer.handle, null);
    device.fns.dispatch.vkFreeMemory.?(device.handle, buffer.memory, null);
}

fn findMemoryType(loader: *Vulkan, physical_device: vk.PhysicalDevice, type_bits: u32, required: vk.MemoryPropertyFlags) !u32 {
    var props: vk.PhysicalDeviceMemoryProperties = undefined;
    loader.instance_fns.dispatch.vkGetPhysicalDeviceMemoryProperties.?(physical_device, &props);
    for (props.memory_types[0..props.memory_type_count], 0..) |mem_type, i| {
        const bit: u32 = @as(u32, 1) << @intCast(i);
        if ((type_bits & bit) != 0 and mem_type.property_flags.contains(required)) return @intCast(i);
    }
    return error.MemoryTypeNotFound;
}

fn createShaderModule(device: *Device, words: []const u32) !vk.ShaderModule {
    var info: vk.ShaderModuleCreateInfo = std.mem.zeroes(vk.ShaderModuleCreateInfo);
    info.s_type = .shader_module_create_info;
    info.code_size = words.len * @sizeOf(u32);
    info.p_code = words.ptr;
    var module: vk.ShaderModule = .null_handle;
    try vkCheck(device.fns.dispatch.vkCreateShaderModule.?(device.handle, &info, null, &module));
    return module;
}

fn createDescriptorSetLayout(device: *Device) !vk.DescriptorSetLayout {
    var bindings: [4]vk.DescriptorSetLayoutBinding = undefined;
    for (&bindings, 0..) |*binding, i| {
        binding.* = std.mem.zeroes(vk.DescriptorSetLayoutBinding);
        binding.binding = @intCast(i);
        binding.descriptor_type = .storage_buffer;
        binding.descriptor_count = 1;
        binding.stage_flags = .{ .compute_bit = true };
    }
    var info: vk.DescriptorSetLayoutCreateInfo = std.mem.zeroes(vk.DescriptorSetLayoutCreateInfo);
    info.s_type = .descriptor_set_layout_create_info;
    info.binding_count = bindings.len;
    info.p_bindings = bindings[0..].ptr;
    var layout: vk.DescriptorSetLayout = .null_handle;
    try vkCheck(device.fns.dispatch.vkCreateDescriptorSetLayout.?(device.handle, &info, null, &layout));
    return layout;
}

fn createDescriptorSetLayout3(device: *Device) !vk.DescriptorSetLayout {
    var bindings: [3]vk.DescriptorSetLayoutBinding = undefined;
    for (&bindings, 0..) |*binding, i| {
        binding.* = std.mem.zeroes(vk.DescriptorSetLayoutBinding);
        binding.binding = @intCast(i);
        binding.descriptor_type = .storage_buffer;
        binding.descriptor_count = 1;
        binding.stage_flags = .{ .compute_bit = true };
    }
    var info: vk.DescriptorSetLayoutCreateInfo = std.mem.zeroes(vk.DescriptorSetLayoutCreateInfo);
    info.s_type = .descriptor_set_layout_create_info;
    info.binding_count = bindings.len;
    info.p_bindings = bindings[0..].ptr;
    var layout: vk.DescriptorSetLayout = .null_handle;
    try vkCheck(device.fns.dispatch.vkCreateDescriptorSetLayout.?(device.handle, &info, null, &layout));
    return layout;
}

fn createPipelineLayout(device: *Device, descriptor_layout: vk.DescriptorSetLayout) !vk.PipelineLayout {
    var info: vk.PipelineLayoutCreateInfo = std.mem.zeroes(vk.PipelineLayoutCreateInfo);
    info.s_type = .pipeline_layout_create_info;
    var layouts = [_]vk.DescriptorSetLayout{descriptor_layout};
    info.set_layout_count = 1;
    info.p_set_layouts = layouts[0..].ptr;
    var layout: vk.PipelineLayout = .null_handle;
    try vkCheck(device.fns.dispatch.vkCreatePipelineLayout.?(device.handle, &info, null, &layout));
    return layout;
}

fn createPipelineLayoutWithCoop2Push(device: *Device, descriptor_layout: vk.DescriptorSetLayout) !vk.PipelineLayout {
    var push: vk.PushConstantRange = std.mem.zeroes(vk.PushConstantRange);
    push.stage_flags = .{ .compute_bit = true };
    push.offset = 0;
    push.size = @sizeOf(Coop2PushConstants);

    var info: vk.PipelineLayoutCreateInfo = std.mem.zeroes(vk.PipelineLayoutCreateInfo);
    info.s_type = .pipeline_layout_create_info;
    var layouts = [_]vk.DescriptorSetLayout{descriptor_layout};
    var ranges = [_]vk.PushConstantRange{push};
    info.set_layout_count = 1;
    info.p_set_layouts = layouts[0..].ptr;
    info.push_constant_range_count = 1;
    info.p_push_constant_ranges = ranges[0..].ptr;
    var layout: vk.PipelineLayout = .null_handle;
    try vkCheck(device.fns.dispatch.vkCreatePipelineLayout.?(device.handle, &info, null, &layout));
    return layout;
}

fn createPipeline(device: *Device, pipeline_layout: vk.PipelineLayout, shader_module: vk.ShaderModule) !vk.Pipeline {
    var stage: vk.PipelineShaderStageCreateInfo = std.mem.zeroes(vk.PipelineShaderStageCreateInfo);
    stage.s_type = .pipeline_shader_stage_create_info;
    stage.stage = .{ .compute_bit = true };
    stage.module = shader_module;
    stage.p_name = "main";
    var info: vk.ComputePipelineCreateInfo = std.mem.zeroes(vk.ComputePipelineCreateInfo);
    info.s_type = .compute_pipeline_create_info;
    info.stage = stage;
    info.layout = pipeline_layout;
    var infos = [_]vk.ComputePipelineCreateInfo{info};
    var pipelines = [_]vk.Pipeline{.null_handle};
    try vkCheck(device.fns.dispatch.vkCreateComputePipelines.?(device.handle, .null_handle, 1, infos[0..].ptr, null, pipelines[0..].ptr));
    return pipelines[0];
}

fn createDescriptorPool(device: *Device) !vk.DescriptorPool {
    var size: vk.DescriptorPoolSize = std.mem.zeroes(vk.DescriptorPoolSize);
    size.type = .storage_buffer;
    size.descriptor_count = 4;
    var info: vk.DescriptorPoolCreateInfo = std.mem.zeroes(vk.DescriptorPoolCreateInfo);
    info.s_type = .descriptor_pool_create_info;
    info.max_sets = 1;
    info.pool_size_count = 1;
    var sizes = [_]vk.DescriptorPoolSize{size};
    info.p_pool_sizes = sizes[0..].ptr;
    var pool: vk.DescriptorPool = .null_handle;
    try vkCheck(device.fns.dispatch.vkCreateDescriptorPool.?(device.handle, &info, null, &pool));
    return pool;
}

fn createDescriptorPool3(device: *Device) !vk.DescriptorPool {
    var size: vk.DescriptorPoolSize = std.mem.zeroes(vk.DescriptorPoolSize);
    size.type = .storage_buffer;
    size.descriptor_count = 3;
    var info: vk.DescriptorPoolCreateInfo = std.mem.zeroes(vk.DescriptorPoolCreateInfo);
    info.s_type = .descriptor_pool_create_info;
    info.max_sets = 1;
    info.pool_size_count = 1;
    var sizes = [_]vk.DescriptorPoolSize{size};
    info.p_pool_sizes = sizes[0..].ptr;
    var pool: vk.DescriptorPool = .null_handle;
    try vkCheck(device.fns.dispatch.vkCreateDescriptorPool.?(device.handle, &info, null, &pool));
    return pool;
}

fn allocateDescriptorSet(device: *Device, pool: vk.DescriptorPool, layout: vk.DescriptorSetLayout) !vk.DescriptorSet {
    var info: vk.DescriptorSetAllocateInfo = std.mem.zeroes(vk.DescriptorSetAllocateInfo);
    info.s_type = .descriptor_set_allocate_info;
    info.descriptor_pool = pool;
    info.descriptor_set_count = 1;
    var layouts = [_]vk.DescriptorSetLayout{layout};
    info.p_set_layouts = layouts[0..].ptr;
    var sets = [_]vk.DescriptorSet{.null_handle};
    try vkCheck(device.fns.dispatch.vkAllocateDescriptorSets.?(device.handle, &info, sets[0..].ptr));
    return sets[0];
}

fn updateDescriptorSet(device: *Device, set: vk.DescriptorSet, params: Buffer, a: Buffer, b: Buffer, c: Buffer) void {
    var infos = [_]vk.DescriptorBufferInfo{
        .{ .buffer = params.handle, .offset = 0, .range = params.byte_len },
        .{ .buffer = a.handle, .offset = 0, .range = a.byte_len },
        .{ .buffer = b.handle, .offset = 0, .range = b.byte_len },
        .{ .buffer = c.handle, .offset = 0, .range = c.byte_len },
    };
    var writes: [4]vk.WriteDescriptorSet = undefined;
    for (&writes, 0..) |*write, i| {
        write.* = std.mem.zeroes(vk.WriteDescriptorSet);
        write.s_type = .write_descriptor_set;
        write.dst_set = set;
        write.dst_binding = @intCast(i);
        write.descriptor_count = 1;
        write.descriptor_type = .storage_buffer;
        write.p_buffer_info = @ptrCast(&infos[i]);
    }
    device.fns.dispatch.vkUpdateDescriptorSets.?(device.handle, writes.len, &writes, 0, null);
}

fn updateDescriptorSet3(device: *Device, set: vk.DescriptorSet, a: Buffer, b: Buffer, c: Buffer) void {
    var infos = [_]vk.DescriptorBufferInfo{
        .{ .buffer = a.handle, .offset = 0, .range = a.byte_len },
        .{ .buffer = b.handle, .offset = 0, .range = b.byte_len },
        .{ .buffer = c.handle, .offset = 0, .range = c.byte_len },
    };
    var writes: [3]vk.WriteDescriptorSet = undefined;
    for (&writes, 0..) |*write, i| {
        write.* = std.mem.zeroes(vk.WriteDescriptorSet);
        write.s_type = .write_descriptor_set;
        write.dst_set = set;
        write.dst_binding = @intCast(i);
        write.descriptor_count = 1;
        write.descriptor_type = .storage_buffer;
        write.p_buffer_info = @ptrCast(&infos[i]);
    }
    device.fns.dispatch.vkUpdateDescriptorSets.?(device.handle, writes.len, &writes, 0, null);
}

fn createCommandPool(device: *Device) !vk.CommandPool {
    var info: vk.CommandPoolCreateInfo = std.mem.zeroes(vk.CommandPoolCreateInfo);
    info.s_type = .command_pool_create_info;
    info.queue_family_index = device.queue_family;
    var pool: vk.CommandPool = .null_handle;
    try vkCheck(device.fns.dispatch.vkCreateCommandPool.?(device.handle, &info, null, &pool));
    return pool;
}

fn allocateCommandBuffer(device: *Device, pool: vk.CommandPool) !vk.CommandBuffer {
    var info: vk.CommandBufferAllocateInfo = std.mem.zeroes(vk.CommandBufferAllocateInfo);
    info.s_type = .command_buffer_allocate_info;
    info.command_pool = pool;
    info.level = .primary;
    info.command_buffer_count = 1;
    var buffers = [_]vk.CommandBuffer{.null_handle};
    try vkCheck(device.fns.dispatch.vkAllocateCommandBuffers.?(device.handle, &info, buffers[0..].ptr));
    return buffers[0];
}

fn recordCommands(device: *Device, command_buffer: vk.CommandBuffer, pipeline: vk.Pipeline, pipeline_layout: vk.PipelineLayout, descriptor_set: vk.DescriptorSet, dispatch_x: u32, dispatch_y: u32, dispatch_z: u32, repeat_count: usize, query_pool: vk.QueryPool) !void {
    var begin: vk.CommandBufferBeginInfo = std.mem.zeroes(vk.CommandBufferBeginInfo);
    begin.s_type = .command_buffer_begin_info;
    try vkCheck(device.fns.dispatch.vkBeginCommandBuffer.?(command_buffer, &begin));
    if (query_pool != .null_handle) {
        device.fns.dispatch.vkCmdResetQueryPool.?(command_buffer, query_pool, 0, 2);
        device.fns.dispatch.vkCmdWriteTimestamp.?(command_buffer, .{ .top_of_pipe_bit = true }, query_pool, 0);
    }
    device.fns.dispatch.vkCmdBindPipeline.?(command_buffer, .compute, pipeline);
    var sets = [_]vk.DescriptorSet{descriptor_set};
    device.fns.dispatch.vkCmdBindDescriptorSets.?(command_buffer, .compute, pipeline_layout, 0, 1, sets[0..].ptr, 0, null);
    for (0..repeat_count) |_| {
        device.fns.dispatch.vkCmdDispatch.?(command_buffer, dispatch_x, dispatch_y, dispatch_z);
    }
    if (query_pool != .null_handle) {
        device.fns.dispatch.vkCmdWriteTimestamp.?(command_buffer, .{ .bottom_of_pipe_bit = true }, query_pool, 1);
    }
    try vkCheck(device.fns.dispatch.vkEndCommandBuffer.?(command_buffer));
}

fn recordNvcoop2Commands(device: *Device, command_buffer: vk.CommandBuffer, pipeline: vk.Pipeline, pipeline_layout: vk.PipelineLayout, descriptor_set: vk.DescriptorSet, m: usize, n: usize, k: usize, tile_m: usize, tile_n: usize, batch_count: usize, batch_stride_a: usize, batch_stride_b: usize, batch_stride_c: usize, repeat_count: usize, query_pool: vk.QueryPool) !void {
    try recordNvcoop2CommandsOp(device, command_buffer, pipeline, pipeline_layout, descriptor_set, .no_trans, .no_trans, m, n, k, tile_m, tile_n, k, n, batch_count, batch_stride_a, batch_stride_b, batch_stride_c, repeat_count, query_pool);
}

fn recordNvcoop2CommandsOp(device: *Device, command_buffer: vk.CommandBuffer, pipeline: vk.Pipeline, pipeline_layout: vk.PipelineLayout, descriptor_set: vk.DescriptorSet, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, tile_m: usize, tile_n: usize, a_row_stride: usize, b_row_stride: usize, batch_count: usize, batch_stride_a: usize, batch_stride_b: usize, batch_stride_c: usize, repeat_count: usize, query_pool: vk.QueryPool) !void {
    var begin: vk.CommandBufferBeginInfo = std.mem.zeroes(vk.CommandBufferBeginInfo);
    begin.s_type = .command_buffer_begin_info;
    try vkCheck(device.fns.dispatch.vkBeginCommandBuffer.?(command_buffer, &begin));
    if (query_pool != .null_handle) {
        device.fns.dispatch.vkCmdResetQueryPool.?(command_buffer, query_pool, 0, 2);
        device.fns.dispatch.vkCmdWriteTimestamp.?(command_buffer, .{ .top_of_pipe_bit = true }, query_pool, 0);
    }
    const pc: Coop2PushConstants = .{
        .m = @intCast(m),
        .n = @intCast(n),
        .k = @intCast(k),
        .a_stride = @intCast(a_row_stride),
        .b_stride = @intCast(b_row_stride),
        .c_stride = @intCast(n),
        .batch_stride_a = @intCast(batch_stride_a),
        .batch_stride_b = @intCast(batch_stride_b),
        .batch_stride_c = @intCast(batch_stride_c),
        .op_a = opCode(op_a),
        .op_b = opCode(op_b),
    };
    device.fns.dispatch.vkCmdBindPipeline.?(command_buffer, .compute, pipeline);
    var sets = [_]vk.DescriptorSet{descriptor_set};
    device.fns.dispatch.vkCmdBindDescriptorSets.?(command_buffer, .compute, pipeline_layout, 0, 1, sets[0..].ptr, 0, null);
    device.fns.dispatch.vkCmdPushConstants.?(command_buffer, pipeline_layout, .{ .compute_bit = true }, 0, @sizeOf(Coop2PushConstants), &pc);
    const dispatch_x: u32 = @intCast(n / tile_n);
    const dispatch_y: u32 = @intCast(m / tile_m);
    const dispatch_z: u32 = @intCast(batch_count);
    for (0..repeat_count) |_| {
        device.fns.dispatch.vkCmdDispatch.?(command_buffer, dispatch_x, dispatch_y, dispatch_z);
    }
    if (query_pool != .null_handle) {
        device.fns.dispatch.vkCmdWriteTimestamp.?(command_buffer, .{ .bottom_of_pipe_bit = true }, query_pool, 1);
    }
    try vkCheck(device.fns.dispatch.vkEndCommandBuffer.?(command_buffer));
}

fn recordUploadCommands(device: *Device, command_buffer: vk.CommandBuffer, a_stage: Buffer, a_dev: Buffer, b_stage: Buffer, b_dev: Buffer) !void {
    var begin: vk.CommandBufferBeginInfo = std.mem.zeroes(vk.CommandBufferBeginInfo);
    begin.s_type = .command_buffer_begin_info;
    try vkCheck(device.fns.dispatch.vkBeginCommandBuffer.?(command_buffer, &begin));
    copyBuffer(device, command_buffer, a_stage, a_dev);
    copyBuffer(device, command_buffer, b_stage, b_dev);
    var barriers = [_]vk.BufferMemoryBarrier{
        bufferBarrier(a_dev, .{ .transfer_write_bit = true }, .{ .shader_read_bit = true }),
        bufferBarrier(b_dev, .{ .transfer_write_bit = true }, .{ .shader_read_bit = true }),
    };
    device.fns.dispatch.vkCmdPipelineBarrier.?(command_buffer, .{ .transfer_bit = true }, .{ .compute_shader_bit = true }, .{}, 0, null, barriers.len, barriers[0..].ptr, 0, null);
    try vkCheck(device.fns.dispatch.vkEndCommandBuffer.?(command_buffer));
}

fn copyBuffer(device: *Device, command_buffer: vk.CommandBuffer, src: Buffer, dst: Buffer) void {
    var regions = [_]vk.BufferCopy{
        .{ .src_offset = 0, .dst_offset = 0, .size = src.byte_len },
    };
    device.fns.dispatch.vkCmdCopyBuffer.?(command_buffer, src.handle, dst.handle, 1, regions[0..].ptr);
}

fn bufferBarrier(buffer: Buffer, src_access: vk.AccessFlags, dst_access: vk.AccessFlags) vk.BufferMemoryBarrier {
    return .{
        .s_type = .buffer_memory_barrier,
        .p_next = null,
        .src_access_mask = src_access,
        .dst_access_mask = dst_access,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .buffer = buffer.handle,
        .offset = 0,
        .size = buffer.byte_len,
    };
}

fn createFence(device: *Device) !vk.Fence {
    var info: vk.FenceCreateInfo = std.mem.zeroes(vk.FenceCreateInfo);
    info.s_type = .fence_create_info;
    var fence: vk.Fence = .null_handle;
    try vkCheck(device.fns.dispatch.vkCreateFence.?(device.handle, &info, null, &fence));
    return fence;
}

fn createTimestampQueryPool(device: *Device) !vk.QueryPool {
    var info: vk.QueryPoolCreateInfo = std.mem.zeroes(vk.QueryPoolCreateInfo);
    info.s_type = .query_pool_create_info;
    info.query_type = .timestamp;
    info.query_count = 2;
    var pool: vk.QueryPool = .null_handle;
    try vkCheck(device.fns.dispatch.vkCreateQueryPool.?(device.handle, &info, null, &pool));
    return pool;
}

fn submitCommand(device: *Device, command_buffer: vk.CommandBuffer, fence: vk.Fence) !void {
    var submit: vk.SubmitInfo = std.mem.zeroes(vk.SubmitInfo);
    submit.s_type = .submit_info;
    submit.command_buffer_count = 1;
    var command_buffers = [_]vk.CommandBuffer{command_buffer};
    submit.p_command_buffers = command_buffers[0..].ptr;
    var submits = [_]vk.SubmitInfo{submit};
    try vkCheck(device.fns.dispatch.vkQueueSubmit.?(device.queue, 1, submits[0..].ptr, fence));
    var fences = [_]vk.Fence{fence};
    try vkCheck(device.fns.dispatch.vkWaitForFences.?(device.handle, 1, fences[0..].ptr, .true, std.math.maxInt(u64)));
    try vkCheck(device.fns.dispatch.vkResetFences.?(device.handle, 1, fences[0..].ptr));
}

fn requireTimestampQueue(loader: *Vulkan, device: *Device) !void {
    var count: u32 = 0;
    loader.instance_fns.dispatch.vkGetPhysicalDeviceQueueFamilyProperties.?(device.physical_device, &count, null);
    if (device.queue_family >= count or count > 64) return error.TimestampUnsupported;
    var families: [64]vk.QueueFamilyProperties = undefined;
    loader.instance_fns.dispatch.vkGetPhysicalDeviceQueueFamilyProperties.?(device.physical_device, &count, families[0..].ptr);
    if (families[device.queue_family].timestamp_valid_bits == 0) return error.TimestampUnsupported;
}

fn timestampPeriodNs(loader: *Vulkan, physical_device: vk.PhysicalDevice) f32 {
    var props: vk.PhysicalDeviceProperties = undefined;
    loader.instance_fns.dispatch.vkGetPhysicalDeviceProperties.?(physical_device, &props);
    return props.limits.timestamp_period;
}

fn vkCheck(result: vk.Result) !void {
    if (result == .success) return;
    return error.VulkanFailure;
}

fn fill(dst: []f32, bias: f32) void {
    for (dst, 0..) |*value, i| {
        const raw: f32 = @floatFromInt((i * 19 + 7) % 23);
        value.* = (raw - 11) / 7 + bias;
    }
}

fn fillTyped(comptime T: type, dst: []T, bias: f32) void {
    for (dst, 0..) |*value, i| {
        const raw: f32 = @floatFromInt((i * 19 + 7) % 23);
        const re = (raw - 11) / 7 + bias;
        if (T == f32) {
            value.* = re;
        } else if (T == f64) {
            value.* = @floatCast(re);
        } else if (T == gemm.Complex32) {
            const im_raw: f32 = @floatFromInt((i * 13 + 5) % 17);
            value.* = .{ .re = re, .im = (im_raw - 8) / 11 };
        } else if (T == gemm.Complex64) {
            const im_raw: f32 = @floatFromInt((i * 13 + 5) % 17);
            value.* = .{ .re = @floatCast(re), .im = @floatCast((im_raw - 8) / 11) };
        } else {
            @compileError("unsupported shader benchmark type");
        }
    }
}

fn fillPackedLowp(dst: []u32, data_type: gemm.DataType, bias: f32) void {
    for (dst, 0..) |*value, i| {
        const raw: f32 = @floatFromInt((i * 19 + 7) % 23);
        const v = (raw - 11) / 7 + bias;
        value.* = switch (data_type) {
            .r_16f => f32ToF16Bits(v),
            .r_16bf => f32ToBf16Bits(v),
            else => @bitCast(v),
        };
    }
}

fn fillPackedInt8(dst: []u32, bias: i32) void {
    for (dst, 0..) |*value, i| {
        const raw: i32 = @intCast((i * 7 + 3) % 17);
        value.* = @intCast(@as(u8, @bitCast(@as(i8, @intCast(raw - 8 + bias)))));
    }
}

fn fillI8Bytes(dst: []u8, bias: i32) void {
    for (dst, 0..) |*value, i| {
        const raw: i32 = @intCast((i * 7 + 3) % 17);
        value.* = @bitCast(@as(i8, @intCast(raw - 8 + bias)));
    }
}

fn fillF16Bits(dst: []u16, bias: f32) void {
    for (dst, 0..) |*value, i| {
        const raw: f32 = @floatFromInt((i * 19 + 7) % 23);
        const v = (raw - 11) / 7 + bias;
        value.* = @intCast(f32ToF16Bits(v));
    }
}

fn fillBf16Bits(dst: []u16, bias: f32) void {
    for (dst, 0..) |*value, i| {
        const raw: f32 = @floatFromInt((i * 19 + 7) % 23);
        const v = (raw - 11) / 7 + bias;
        value.* = @truncate(f32ToBf16Bits(v));
    }
}

fn zeroTyped(comptime T: type, dst: []T) void {
    if (T == f32 or T == f64) {
        @memset(dst, 0);
    } else if (T == gemm.Complex32) {
        @memset(dst, .{ .re = 0, .im = 0 });
    } else if (T == gemm.Complex64) {
        @memset(dst, .{ .re = 0, .im = 0 });
    } else {
        @compileError("unsupported shader benchmark type");
    }
}

fn f32ToBf16Bits(value: f32) u32 {
    const bits: u32 = @bitCast(value);
    const rounded = bits + 0x7fff + ((bits >> 16) & 1);
    return rounded >> 16;
}

fn f32ToF16Bits(value: f32) u32 {
    const bits: u32 = @bitCast(value);
    const sign = (bits >> 16) & 0x8000;
    const exponent = @as(i32, @intCast((bits >> 23) & 0xff)) - 127 + 15;
    const mantissa = bits & 0x7fffff;
    if (exponent <= 0) {
        if (exponent < -10) return sign;
        const shifted = (mantissa | 0x800000) >> @intCast(1 - exponent);
        return sign | ((shifted + 0x1000) >> 13);
    }
    if (exponent >= 31) return sign | 0x7c00;
    return sign | (@as(u32, @intCast(exponent)) << 10) | ((mantissa + 0x1000) >> 13);
}

fn makeParams(comptime Params: type, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, batch_count: usize, a_rows: usize, b_rows: usize, a_len: usize, b_len: usize, c_len: usize) Params {
    if (Params == GemmParams) {
        return .{
            .op_a = opCode(op_a),
            .op_b = opCode(op_b),
            .m = @intCast(m),
            .n = @intCast(n),
            .k = @intCast(k),
            .batch_count = @intCast(batch_count),
            .a_row_stride = 1,
            .a_col_stride = @intCast(a_rows),
            .b_row_stride = 1,
            .b_col_stride = @intCast(b_rows),
            .c_row_stride = 1,
            .c_col_stride = @intCast(m),
            .stride_a = @intCast(a_len),
            .stride_b = @intCast(b_len),
            .stride_c = @intCast(c_len),
        };
    } else if (Params == GemmParamsF64) {
        return .{
            .op_a = opCode(op_a),
            .op_b = opCode(op_b),
            .m = @intCast(m),
            .n = @intCast(n),
            .k = @intCast(k),
            .batch_count = @intCast(batch_count),
            .a_row_stride = 1,
            .a_col_stride = @intCast(a_rows),
            .b_row_stride = 1,
            .b_col_stride = @intCast(b_rows),
            .c_row_stride = 1,
            .c_col_stride = @intCast(m),
            .stride_a = @intCast(a_len),
            .stride_b = @intCast(b_len),
            .stride_c = @intCast(c_len),
        };
    } else if (Params == GemmParamsComplexF64) {
        return .{
            .op_a = opCode(op_a),
            .op_b = opCode(op_b),
            .m = @intCast(m),
            .n = @intCast(n),
            .k = @intCast(k),
            .batch_count = @intCast(batch_count),
            .a_row_stride = 1,
            .a_col_stride = @intCast(a_rows),
            .b_row_stride = 1,
            .b_col_stride = @intCast(b_rows),
            .c_row_stride = 1,
            .c_col_stride = @intCast(m),
            .stride_a = @intCast(a_len),
            .stride_b = @intCast(b_len),
            .stride_c = @intCast(c_len),
        };
    } else {
        @compileError("unsupported params type");
    }
}

fn roundUpDiv(value: u32, divisor: u32) u32 {
    return (value + divisor - 1) / divisor;
}

fn opCode(op: gemm.Op) u32 {
    return switch (op) {
        .no_trans => 0,
        .trans => 1,
        .conj_trans => 2,
    };
}

fn storedRows(op: gemm.Op, rows: usize, cols: usize) usize {
    return switch (op) {
        .no_trans => rows,
        .trans, .conj_trans => cols,
    };
}

fn storedCols(op: gemm.Op, rows: usize, cols: usize) usize {
    return switch (op) {
        .no_trans => cols,
        .trans, .conj_trans => rows,
    };
}

fn tflops(m: usize, n: usize, k: usize, ms: f64) f64 {
    const ops = 2.0 * @as(f64, @floatFromInt(m)) * @as(f64, @floatFromInt(n)) * @as(f64, @floatFromInt(k));
    return ops / (ms * 1.0e9);
}
