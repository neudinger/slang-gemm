const std = @import("std");
const gemm = @import("gemm");
const cublas = @import("cublas");
const manifest = @import("manifest");
const vulkan_runner = @import("vulkan_runner");

const CLOCK_MONOTONIC: c_int = 1;
const Timespec = extern struct {
    tv_sec: isize,
    tv_nsec: isize,
};
extern fn clock_gettime(clk_id: c_int, tp: *Timespec) callconv(.c) c_int;

const CFile = opaque {};
extern fn fopen(path: [*:0]const u8, mode: [*:0]const u8) callconv(.c) ?*CFile;
extern fn fwrite(ptr: ?*const anyopaque, size: usize, nmemb: usize, stream: *CFile) callconv(.c) usize;
extern fn fclose(stream: *CFile) callconv(.c) c_int;

const Options = struct {
    sizes: []const usize,
    rect_shapes: []const manifest.Shape,
    iters: usize = 20,
    warmup: usize = 5,
    device_substr: ?[]const u8 = null,
    manifest_all: bool = false,
    fail_on_regression: bool = false,
    fail_every_row: bool = false,
    only_cublas_supported: bool = false,
    tile_compatible_only: bool = false,
    output: ?[]const u8 = null,
};

const TimedResult = struct {
    cublas_ms: f64,
    cublas_tflops: f64,
    scalar_ms: f64,
    scalar_tflops: f64,
    shader: ?vulkan_runner.Result,
};

const Summary = struct {
    accelerated_count: usize = 0,
    accelerated_log_ratio_sum: f64 = 0,
    regression_count: usize = 0,
    strict_count: usize = 0,
    strict_failure_count: usize = 0,
    strict_worst_ratio: f64 = 0,
};

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.c_allocator;
    const args = try init.minimal.args.toSlice(allocator);
    defer allocator.free(args);
    const opts = try parseArgs(allocator, args);
    defer allocator.free(opts.sizes);
    defer allocator.free(opts.rect_shapes);

    var out = try Output.init(allocator, opts.output);
    defer out.deinit(allocator);

    if (!cublas.hasCudaDevice()) {
        try out.write(allocator, "{\"status\":\"invalid\",\"error\":\"no CUDA device available\"}\n");
        return;
    }

    var summary: Summary = .{};
    if (opts.manifest_all) {
        try runManifestAll(allocator, &out, opts, &summary);
    } else {
        for (opts.sizes) |size| {
            const shape: manifest.Shape = .{ .m = size, .n = size, .k = size };
            try runCase(allocator, &out, opts, .{ .api = .regular, .op_a = .no_trans, .op_b = .no_trans, .a_type = .r_32f, .b_type = .r_32f, .c_type = .r_32f, .compute_type = .f32, .shape = shape }, &summary);
        }
    }

    const geo_ratio = if (summary.accelerated_count == 0) null else std.math.exp(summary.accelerated_log_ratio_sum / @as(f64, @floatFromInt(summary.accelerated_count)));
    if (geo_ratio) |ratio| {
        const line = try std.fmt.allocPrint(allocator, "{{\"summary\":\"hardware_accelerated\",\"cases\":{d},\"geomean_shader_cublas_ratio\":{d:.6},\"regressions\":{d},\"strict_cases\":{d},\"strict_failures\":{d},\"strict_worst_ratio\":{d:.6}}}\n", .{ summary.accelerated_count, ratio, summary.regression_count, summary.strict_count, summary.strict_failure_count, summary.strict_worst_ratio });
        defer allocator.free(line);
        try out.write(allocator, line);
    } else {
        const line = try std.fmt.allocPrint(allocator, "{{\"summary\":\"hardware_accelerated\",\"cases\":0,\"geomean_shader_cublas_ratio\":null,\"regressions\":0,\"strict_cases\":{d},\"strict_failures\":{d},\"strict_worst_ratio\":{d:.6}}}\n", .{ summary.strict_count, summary.strict_failure_count, summary.strict_worst_ratio });
        defer allocator.free(line);
        try out.write(allocator, line);
    }

    if (opts.fail_on_regression and geo_ratio != null and geo_ratio.? > 1.0) return error.PerfRegression;
    if (opts.fail_every_row and summary.strict_failure_count != 0) return error.PerfRegression;
}

fn runManifestAll(allocator: std.mem.Allocator, out: *Output, opts: Options, summary: *Summary) !void {
    for (opts.sizes) |size| {
        const shape: manifest.Shape = .{ .m = size, .n = size, .k = size };
        try runCompactCasesForShape(allocator, out, opts, shape, summary);
    }
    for (opts.rect_shapes) |shape| {
        try runCompactCasesForShape(allocator, out, opts, shape, summary);
    }

    // Practical CUDA enum support matrix: enumerate every API/op/compute enum
    // across every CUDA data type as same-type rows, plus explicit mixed
    // tensor/int rows. This covers every enum ID without generating the
    // unusable full A/B/C Cartesian product.
    const manifest_size = if (opts.tile_compatible_only and opts.sizes.len != 0) opts.sizes[0] else 128;
    const shape: manifest.Shape = .{ .m = manifest_size, .n = manifest_size, .k = manifest_size };
    for (manifest.apis) |api| {
        for (manifest.ops) |op_a| {
            for (manifest.ops) |op_b| {
                for (manifest.data_types) |data_type| {
                    for (manifest.compute_types) |compute_type| {
                        const case: manifest.Case = .{
                            .api = api,
                            .op_a = op_a,
                            .op_b = op_b,
                            .a_type = data_type,
                            .b_type = data_type,
                            .c_type = data_type,
                            .compute_type = compute_type,
                            .shape = shape,
                        };
                        if (opts.tile_compatible_only and !case.tileCompatible()) continue;
                        if (opts.fail_every_row and case.cublasSupported() and !case.hardwareAccelerated()) {
                            summary.strict_count += 1;
                            summary.strict_failure_count += 1;
                            try emitStatusOnly(allocator, out, case, .shader_unavailable);
                            continue;
                        }
                        if (case.cublasSupported() and case.shaderAvailable()) {
                            try runCase(allocator, out, opts, case, summary);
                            continue;
                        }
                        if (opts.only_cublas_supported and !case.cublasSupported()) continue;
                        try emitStatusOnly(allocator, out, case, case.baseStatus());
                    }
                }
                const mixed_cases = [_]manifest.Case{
                    .{ .api = api, .op_a = op_a, .op_b = op_b, .a_type = .r_16f, .b_type = .r_16f, .c_type = .r_32f, .compute_type = .f32, .shape = shape },
                    .{ .api = api, .op_a = op_a, .op_b = op_b, .a_type = .r_16bf, .b_type = .r_16bf, .c_type = .r_32f, .compute_type = .f32, .shape = shape },
                    .{ .api = api, .op_a = op_a, .op_b = op_b, .a_type = .r_32f, .b_type = .r_32f, .c_type = .r_32f, .compute_type = .f32_fast_tf32, .shape = shape },
                    .{ .api = api, .op_a = op_a, .op_b = op_b, .a_type = .r_8i, .b_type = .r_8i, .c_type = .r_32i, .compute_type = .i32, .shape = shape },
                };
                for (mixed_cases) |case| {
                    if (opts.tile_compatible_only and !case.tileCompatible()) continue;
                    if (opts.fail_every_row and case.cublasSupported() and !case.hardwareAccelerated()) {
                        summary.strict_count += 1;
                        summary.strict_failure_count += 1;
                        try emitStatusOnly(allocator, out, case, .shader_unavailable);
                        continue;
                    }
                    if (case.cublasSupported() and case.shaderAvailable()) {
                        try runCase(allocator, out, opts, case, summary);
                        continue;
                    }
                    if (opts.only_cublas_supported and !case.cublasSupported()) continue;
                    try emitStatusOnly(allocator, out, case, case.baseStatus());
                }
            }
        }
    }
}

fn runCompactCasesForShape(allocator: std.mem.Allocator, out: *Output, opts: Options, shape: manifest.Shape, summary: *Summary) !void {
    for (manifest.ops) |op_a| {
        for (manifest.ops) |op_b| {
            try runCase(allocator, out, opts, .{
                .api = .regular,
                .op_a = op_a,
                .op_b = op_b,
                .a_type = .r_32f,
                .b_type = .r_32f,
                .c_type = .r_32f,
                .compute_type = .f32,
                .shape = shape,
            }, summary);
        }
    }

    const cases = [_]manifest.Case{
        .{ .api = .regular, .op_a = .trans, .op_b = .no_trans, .a_type = .r_64f, .b_type = .r_64f, .c_type = .r_64f, .compute_type = .f64, .shape = shape },
        .{ .api = .regular, .op_a = .conj_trans, .op_b = .no_trans, .a_type = .c_32f, .b_type = .c_32f, .c_type = .c_32f, .compute_type = .f32, .shape = shape },
        .{ .api = .regular, .op_a = .no_trans, .op_b = .conj_trans, .a_type = .c_64f, .b_type = .c_64f, .c_type = .c_64f, .compute_type = .f64, .shape = shape },
        .{ .api = .complex_3m, .op_a = .no_trans, .op_b = .no_trans, .a_type = .c_32f, .b_type = .c_32f, .c_type = .c_32f, .compute_type = .f32, .shape = shape },
        .{ .api = .gemm_ex, .op_a = .no_trans, .op_b = .no_trans, .a_type = .r_16f, .b_type = .r_16f, .c_type = .r_32f, .compute_type = .f32, .shape = shape },
        .{ .api = .gemm_ex, .op_a = .no_trans, .op_b = .no_trans, .a_type = .r_16bf, .b_type = .r_16bf, .c_type = .r_32f, .compute_type = .f32, .shape = shape },
        .{ .api = .gemm_ex, .op_a = .no_trans, .op_b = .no_trans, .a_type = .r_8i, .b_type = .r_8i, .c_type = .r_32i, .compute_type = .i32, .shape = shape },
        .{ .api = .gemm_strided_batched_ex, .op_a = .no_trans, .op_b = .no_trans, .a_type = .r_16f, .b_type = .r_16f, .c_type = .r_32f, .compute_type = .f32, .shape = shape, .batched = true },
        .{ .api = .gemm_batched_ex, .op_a = .no_trans, .op_b = .no_trans, .a_type = .r_16f, .b_type = .r_16f, .c_type = .r_32f, .compute_type = .f32, .shape = shape, .batched = true },
    };
    for (cases) |case| try runCase(allocator, out, opts, case, summary);
}

fn runCase(allocator: std.mem.Allocator, out: *Output, opts: Options, case: manifest.Case, summary: *Summary) !void {
    if (opts.only_cublas_supported and !case.cublasSupported()) return;
    if (opts.tile_compatible_only and !case.tileCompatible()) return;
    if (opts.fail_every_row and case.cublasSupported() and !case.hardwareAccelerated()) {
        summary.strict_count += 1;
        summary.strict_failure_count += 1;
        return emitStatusOnly(allocator, out, case, .shader_unavailable);
    }
    const status = case.baseStatus();
    if (status != .ok) return emitStatusOnly(allocator, out, case, status);

    const result = try benchRunnableCase(allocator, case, opts);
    const shader_status: manifest.Status = if (result.shader == null) .validation_failed else .ok;
    if (result.shader) |shader| {
        const ratio = shader.ms / result.cublas_ms;
        var final_status = shader_status;
        if (case.hardwareAccelerated()) {
            summary.accelerated_count += 1;
            summary.accelerated_log_ratio_sum += std.math.log(f64, std.math.e, ratio);
            if (ratio > 1.0) {
                summary.regression_count += 1;
                final_status = .perf_regression;
            }
        }
        if (opts.fail_every_row) {
            summary.strict_count += 1;
            if (ratio > summary.strict_worst_ratio) summary.strict_worst_ratio = ratio;
            if (ratio >= 1.0) {
                summary.strict_failure_count += 1;
                final_status = .perf_regression;
            }
        }
        const line = try std.fmt.allocPrint(
            allocator,
            "{{\"status\":\"{s}\",\"api\":\"{s}\",\"family\":\"{s}\",\"shader_family\":\"{s}\",\"shader_implemented\":{},\"dispatch_available\":{},\"op_a\":\"{s}\",\"op_b\":\"{s}\",\"a_type\":\"{s}\",\"b_type\":\"{s}\",\"c_type\":\"{s}\",\"compute_type\":\"{s}\",\"m\":{d},\"n\":{d},\"k\":{d},\"cublas_ms\":{d:.6},\"cublas_tflops\":{d:.4},\"scalar_ms\":{d:.6},\"scalar_tflops\":{d:.4},\"shader_ms\":{d:.6},\"shader_tflops\":{d:.4},\"shader_cublas_ratio\":{d:.6}}}\n",
            .{ final_status.label(), case.api.label(), case.family(), case.shaderFamily().label(), case.shaderImplemented(), case.dispatchAvailable(), manifest.opLabel(case.op_a), manifest.opLabel(case.op_b), manifest.dataTypeLabel(case.a_type), manifest.dataTypeLabel(case.b_type), manifest.dataTypeLabel(case.c_type), manifest.computeTypeLabel(case.compute_type), case.shape.m, case.shape.n, case.shape.k, result.cublas_ms, result.cublas_tflops, result.scalar_ms, result.scalar_tflops, shader.ms, shader.tflops, ratio },
        );
        defer allocator.free(line);
        try out.write(allocator, line);
    } else {
        try emitStatusOnly(allocator, out, case, shader_status);
    }
}

fn emitStatusOnly(allocator: std.mem.Allocator, out: *Output, case: manifest.Case, status: manifest.Status) !void {
    const line = try std.fmt.allocPrint(
        allocator,
        "{{\"status\":\"{s}\",\"api\":\"{s}\",\"family\":\"{s}\",\"shader_family\":\"{s}\",\"shader_implemented\":{},\"dispatch_available\":{},\"op_a\":\"{s}\",\"op_b\":\"{s}\",\"a_type\":\"{s}\",\"b_type\":\"{s}\",\"c_type\":\"{s}\",\"compute_type\":\"{s}\",\"m\":{d},\"n\":{d},\"k\":{d},\"cublas_ms\":null,\"cublas_tflops\":null,\"scalar_ms\":null,\"scalar_tflops\":null,\"shader_ms\":null,\"shader_tflops\":null,\"shader_cublas_ratio\":null}}\n",
        .{ status.label(), case.api.label(), case.family(), case.shaderFamily().label(), case.shaderImplemented(), case.dispatchAvailable(), manifest.opLabel(case.op_a), manifest.opLabel(case.op_b), manifest.dataTypeLabel(case.a_type), manifest.dataTypeLabel(case.b_type), manifest.dataTypeLabel(case.c_type), manifest.computeTypeLabel(case.compute_type), case.shape.m, case.shape.n, case.shape.k },
    );
    defer allocator.free(line);
    try out.write(allocator, line);
}

fn benchRunnableCase(allocator: std.mem.Allocator, case: manifest.Case, opts: Options) !TimedResult {
    if ((case.api == .regular or case.api == .regular_64 or case.api == .batched or case.api == .strided_batched) and case.a_type == .r_64f and case.b_type == .r_64f and case.c_type == .r_64f and case.compute_type == .f64) {
        return benchRegularF64(allocator, case, opts);
    }
    if ((case.api == .regular or case.api == .regular_64 or case.api == .batched or case.api == .strided_batched or case.api == .complex_3m) and case.a_type == .c_32f and case.b_type == .c_32f and case.c_type == .c_32f and case.compute_type == .f32) {
        return benchRegularC32(allocator, case, opts);
    }
    if ((case.api == .regular or case.api == .regular_64 or case.api == .batched or case.api == .strided_batched or case.api == .complex_3m) and case.a_type == .c_64f and case.b_type == .c_64f and case.c_type == .c_64f and case.compute_type == .f64) {
        return benchRegularC64(allocator, case, opts);
    }
    if ((case.api == .gemm_strided_batched_ex or case.api == .gemm_batched_ex) and (case.a_type == .r_16f or case.a_type == .r_16bf) and case.a_type == case.b_type and case.c_type == .r_32f) {
        return benchGemmStridedBatchedExLowpF32(allocator, case, opts);
    }
    if ((case.api == .gemm_ex) and (case.a_type == .r_16f or case.a_type == .r_16bf) and case.a_type == case.b_type and case.c_type == .r_32f) {
        return benchGemmExLowpF32(allocator, case, opts);
    }
    if ((case.api == .gemm_ex or case.api == .gemm_batched_ex or case.api == .gemm_strided_batched_ex) and case.a_type == .r_16f and case.b_type == .r_16f and case.c_type == .r_16f) {
        return benchGemmExF16Scalar(allocator, case, opts);
    }
    if ((case.api == .gemm_ex or case.api == .gemm_batched_ex or case.api == .gemm_strided_batched_ex) and case.a_type == .r_32f and case.b_type == .r_32f and case.c_type == .r_32f) {
        return benchGemmExF32(allocator, case, opts);
    }
    if ((case.api == .gemm_ex or case.api == .gemm_batched_ex or case.api == .gemm_strided_batched_ex) and case.a_type == .r_8i and case.b_type == .r_8i and case.c_type == .r_32i and case.compute_type == .i32) {
        return benchGemmExInt8I32(allocator, case, opts);
    }
    if ((case.api != .regular and case.api != .regular_64 and case.api != .batched and case.api != .strided_batched) or case.a_type != .r_32f or case.b_type != .r_32f or case.c_type != .r_32f or case.compute_type != .f32) return error.InvalidArgument;

    const m = case.shape.m;
    const n = case.shape.n;
    const k = case.shape.k;
    const batch_count = classicBatchCount(case.api);
    const a_rows = storedRows(case.op_a, m, k);
    const a_cols = storedCols(case.op_a, m, k);
    const b_rows = storedRows(case.op_b, k, n);
    const b_cols = storedCols(case.op_b, k, n);
    const a_len = a_rows * a_cols;
    const b_len = b_rows * b_cols;
    const c_len = m * n;
    const a = try allocator.alloc(f32, a_len * batch_count);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, b_len * batch_count);
    defer allocator.free(b);
    const c = try allocator.alloc(f32, c_len * batch_count);
    defer allocator.free(c);
    fill(a, 0.1);
    fill(b, -0.2);
    fill(c, 0.3);

    const ctx = try cublas.Context.init();
    defer ctx.deinit();
    const da = try cublas.DeviceBuffer(f32).alloc(a.len);
    defer da.free();
    const db = try cublas.DeviceBuffer(f32).alloc(b.len);
    defer db.free();
    const dc = try cublas.DeviceBuffer(f32).alloc(c.len);
    defer dc.free();
    try da.copyFromHost(a);
    try db.copyFromHost(b);
    try dc.copyFromHost(c);

    const alpha: f32 = 1;
    const beta: f32 = 0;
    for (0..opts.warmup) |_| try runClassicSgemmLike(ctx, case.api, case.op_a, case.op_b, m, n, k, alpha, da.ptr, a_rows, a_len, db.ptr, b_rows, b_len, beta, dc.ptr, m, c_len, batch_count);
    try cublas.synchronize();

    const timer = try cublas.GpuTimer.init();
    defer timer.deinit();
    try timer.begin();
    for (0..opts.iters) |_| try runClassicSgemmLike(ctx, case.api, case.op_a, case.op_b, m, n, k, alpha, da.ptr, a_rows, a_len, db.ptr, b_rows, b_len, beta, dc.ptr, m, c_len, batch_count);
    const cublas_ms = @as(f64, @floatCast(try timer.end())) / @as(f64, @floatFromInt(opts.iters));

    const scalar_m: usize = @min(m, 256);
    const scalar_n: usize = @min(n, 256);
    const scalar_k: usize = @min(k, 256);
    const scalar_a_rows = storedRows(case.op_a, scalar_m, scalar_k);
    const scalar_a_cols = storedCols(case.op_a, scalar_m, scalar_k);
    const scalar_b_rows = storedRows(case.op_b, scalar_k, scalar_n);
    const scalar_b_cols = storedCols(case.op_b, scalar_k, scalar_n);
    const sa = try allocator.alloc(f32, scalar_a_rows * scalar_a_cols);
    defer allocator.free(sa);
    const sb = try allocator.alloc(f32, scalar_b_rows * scalar_b_cols);
    defer allocator.free(sb);
    const sc = try allocator.alloc(f32, scalar_m * scalar_n);
    defer allocator.free(sc);
    fill(sa, 0.1);
    fill(sb, -0.2);
    fill(sc, 0.3);
    const scalar_start = try nanoTimestamp();
    try gemm.sgemm(.col_major, case.op_a, case.op_b, scalar_m, scalar_n, scalar_k, 1, sa, scalar_a_rows, sb, scalar_b_rows, 0, sc, scalar_m);
    const scalar_end = try nanoTimestamp();
    const scalar_ms = @as(f64, @floatFromInt(scalar_end - scalar_start)) / 1.0e6;
    const shader = vulkan_runner.benchSgemmF32BatchedCase(allocator, case.op_a, case.op_b, m, n, k, batch_count, opts.iters, opts.warmup, opts.device_substr) catch |err| switch (err) {
        error.NoVulkanDevice, error.NoMatchingDevice, error.TimestampUnsupported => null,
        else => return err,
    };

    return .{
        .cublas_ms = cublas_ms,
        .cublas_tflops = tflops(m, n, k, cublas_ms) * @as(f64, @floatFromInt(batch_count)),
        .scalar_ms = scalar_ms,
        .scalar_tflops = tflops(scalar_m, scalar_n, scalar_k, scalar_ms),
        .shader = shader,
    };
}

fn benchRegularF64(allocator: std.mem.Allocator, case: manifest.Case, opts: Options) !TimedResult {
    const m = case.shape.m;
    const n = case.shape.n;
    const k = case.shape.k;
    const batch_count = classicBatchCount(case.api);
    const a_rows = storedRows(case.op_a, m, k);
    const a_cols = storedCols(case.op_a, m, k);
    const b_rows = storedRows(case.op_b, k, n);
    const b_cols = storedCols(case.op_b, k, n);
    const a_len = a_rows * a_cols;
    const b_len = b_rows * b_cols;
    const c_len = m * n;
    const a = try allocator.alloc(f64, a_len * batch_count);
    defer allocator.free(a);
    const b = try allocator.alloc(f64, b_len * batch_count);
    defer allocator.free(b);
    const c = try allocator.alloc(f64, c_len * batch_count);
    defer allocator.free(c);
    fillTyped(f64, a, 0.1);
    fillTyped(f64, b, -0.2);
    fillTyped(f64, c, 0.3);

    const ctx = try cublas.Context.init();
    defer ctx.deinit();
    const da = try cublas.DeviceBuffer(f64).alloc(a.len);
    defer da.free();
    const db = try cublas.DeviceBuffer(f64).alloc(b.len);
    defer db.free();
    const dc = try cublas.DeviceBuffer(f64).alloc(c.len);
    defer dc.free();
    try da.copyFromHost(a);
    try db.copyFromHost(b);
    try dc.copyFromHost(c);

    const alpha: f64 = 1;
    const beta: f64 = 0;
    for (0..opts.warmup) |_| try runClassicDgemmLike(ctx, case.api, case.op_a, case.op_b, m, n, k, alpha, da.ptr, a_rows, a_len, db.ptr, b_rows, b_len, beta, dc.ptr, m, c_len, batch_count);
    try cublas.synchronize();
    const timer = try cublas.GpuTimer.init();
    defer timer.deinit();
    try timer.begin();
    for (0..opts.iters) |_| try runClassicDgemmLike(ctx, case.api, case.op_a, case.op_b, m, n, k, alpha, da.ptr, a_rows, a_len, db.ptr, b_rows, b_len, beta, dc.ptr, m, c_len, batch_count);
    const cublas_ms = @as(f64, @floatCast(try timer.end())) / @as(f64, @floatFromInt(opts.iters));

    const scalar_m: usize = @min(m, 128);
    const scalar_n: usize = @min(n, 128);
    const scalar_k: usize = @min(k, 128);
    const scalar_a_rows = storedRows(case.op_a, scalar_m, scalar_k);
    const scalar_a_cols = storedCols(case.op_a, scalar_m, scalar_k);
    const scalar_b_rows = storedRows(case.op_b, scalar_k, scalar_n);
    const scalar_b_cols = storedCols(case.op_b, scalar_k, scalar_n);
    const sa = try allocator.alloc(f64, scalar_a_rows * scalar_a_cols);
    defer allocator.free(sa);
    const sb = try allocator.alloc(f64, scalar_b_rows * scalar_b_cols);
    defer allocator.free(sb);
    const sc = try allocator.alloc(f64, scalar_m * scalar_n);
    defer allocator.free(sc);
    fillTyped(f64, sa, 0.1);
    fillTyped(f64, sb, -0.2);
    fillTyped(f64, sc, 0.3);
    const scalar_start = try nanoTimestamp();
    try gemm.dgemm(.col_major, case.op_a, case.op_b, scalar_m, scalar_n, scalar_k, 1, sa, scalar_a_rows, sb, scalar_b_rows, 0, sc, scalar_m);
    const scalar_end = try nanoTimestamp();
    const scalar_ms = @as(f64, @floatFromInt(scalar_end - scalar_start)) / 1.0e6;
    const shader = vulkan_runner.benchDgemmF64BatchedCase(allocator, case.op_a, case.op_b, m, n, k, batch_count, opts.iters, opts.warmup, opts.device_substr) catch |err| switch (err) {
        error.NoVulkanDevice, error.NoMatchingDevice, error.TimestampUnsupported, error.VulkanFailure => null,
        else => return err,
    };

    return .{ .cublas_ms = cublas_ms, .cublas_tflops = tflops(m, n, k, cublas_ms) * @as(f64, @floatFromInt(batch_count)), .scalar_ms = scalar_ms, .scalar_tflops = tflops(scalar_m, scalar_n, scalar_k, scalar_ms), .shader = shader };
}

fn benchGemmExF32(allocator: std.mem.Allocator, case: manifest.Case, opts: Options) !TimedResult {
    const m = case.shape.m;
    const n = case.shape.n;
    const k = case.shape.k;
    const batch_count = exBatchCount(case.api);
    const a_rows = storedRows(case.op_a, m, k);
    const a_cols = storedCols(case.op_a, m, k);
    const b_rows = storedRows(case.op_b, k, n);
    const b_cols = storedCols(case.op_b, k, n);
    const stride_a = a_rows * a_cols;
    const stride_b = b_rows * b_cols;
    const stride_c = m * n;
    const a = try allocator.alloc(f32, stride_a * batch_count);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, stride_b * batch_count);
    defer allocator.free(b);
    const c = try allocator.alloc(f32, stride_c * batch_count);
    defer allocator.free(c);
    fill(a, 0.1);
    fill(b, -0.2);
    fill(c, 0.3);

    const ctx = try cublas.Context.init();
    defer ctx.deinit();
    const da = try cublas.RawDeviceBuffer.alloc(a.len * @sizeOf(f32));
    defer da.free();
    const db = try cublas.RawDeviceBuffer.alloc(b.len * @sizeOf(f32));
    defer db.free();
    const dc = try cublas.RawDeviceBuffer.alloc(c.len * @sizeOf(f32));
    defer dc.free();
    try da.copyFromHost(std.mem.sliceAsBytes(a));
    try db.copyFromHost(std.mem.sliceAsBytes(b));
    try dc.copyFromHost(std.mem.sliceAsBytes(c));

    var alpha: f32 = 1;
    var beta: f32 = 0;
    const desc: cublas.RawGemmExDesc = .{ .op_a = case.op_a, .op_b = case.op_b, .m = m, .n = n, .k = k, .alpha = &alpha, .a = da.ptr, .a_type = .r_32f, .lda = a_rows, .stride_a = stride_a, .b = db.ptr, .b_type = .r_32f, .ldb = b_rows, .stride_b = stride_b, .beta = &beta, .c = dc.ptr, .c_type = .r_32f, .ldc = m, .stride_c = stride_c, .batch_count = batch_count, .compute_type = case.compute_type };
    for (0..opts.warmup) |_| try runGemmExLike(ctx, case.api, desc);
    try cublas.synchronize();
    const timer = try cublas.GpuTimer.init();
    defer timer.deinit();
    try timer.begin();
    for (0..opts.iters) |_| try runGemmExLike(ctx, case.api, desc);
    const cublas_ms = @as(f64, @floatCast(try timer.end())) / @as(f64, @floatFromInt(opts.iters));

    const scalar_m: usize = @min(m, 256);
    const scalar_n: usize = @min(n, 256);
    const scalar_k: usize = @min(k, 256);
    const scalar_a_rows = storedRows(case.op_a, scalar_m, scalar_k);
    const scalar_b_rows = storedRows(case.op_b, scalar_k, scalar_n);
    const scalar_ms = try scalarSgemmMs(allocator, case.op_a, case.op_b, scalar_m, scalar_n, scalar_k, scalar_a_rows, scalar_b_rows);
    const shader = vulkan_runner.benchSgemmF32BatchedCase(allocator, case.op_a, case.op_b, m, n, k, batch_count, opts.iters, opts.warmup, opts.device_substr) catch |err| switch (err) {
        error.NoVulkanDevice, error.NoMatchingDevice, error.TimestampUnsupported, error.VulkanFailure => null,
        else => return err,
    };
    return .{ .cublas_ms = cublas_ms, .cublas_tflops = tflops(m, n, k, cublas_ms) * @as(f64, @floatFromInt(batch_count)), .scalar_ms = scalar_ms, .scalar_tflops = tflops(scalar_m, scalar_n, scalar_k, scalar_ms), .shader = shader };
}

fn benchGemmExF16Scalar(allocator: std.mem.Allocator, case: manifest.Case, opts: Options) !TimedResult {
    const m = case.shape.m;
    const n = case.shape.n;
    const k = case.shape.k;
    const batch_count = exBatchCount(case.api);
    const a_rows = storedRows(case.op_a, m, k);
    const a_cols = storedCols(case.op_a, m, k);
    const b_rows = storedRows(case.op_b, k, n);
    const b_cols = storedCols(case.op_b, k, n);
    const stride_a = a_rows * a_cols;
    const stride_b = b_rows * b_cols;
    const stride_c = m * n;
    const a = try allocator.alloc(u16, stride_a * batch_count);
    defer allocator.free(a);
    const b = try allocator.alloc(u16, stride_b * batch_count);
    defer allocator.free(b);
    const c = try allocator.alloc(u16, stride_c * batch_count);
    defer allocator.free(c);
    fillLowp16(a, .r_16f, 0.1);
    fillLowp16(b, .r_16f, -0.2);
    @memset(c, 0);

    const ctx = try cublas.Context.init();
    defer ctx.deinit();
    const da = try cublas.RawDeviceBuffer.alloc(a.len * @sizeOf(u16));
    defer da.free();
    const db = try cublas.RawDeviceBuffer.alloc(b.len * @sizeOf(u16));
    defer db.free();
    const dc = try cublas.RawDeviceBuffer.alloc(c.len * @sizeOf(u16));
    defer dc.free();
    try da.copyFromHost(std.mem.sliceAsBytes(a));
    try db.copyFromHost(std.mem.sliceAsBytes(b));
    try dc.copyFromHost(std.mem.sliceAsBytes(c));

    var alpha_f32: f32 = 1;
    var beta_f32: f32 = 0;
    var alpha_f16: u16 = @intCast(f32ToF16Bits(1));
    var beta_f16: u16 = @intCast(f32ToF16Bits(0));
    const alpha_ptr: ?*const anyopaque = if (case.compute_type == .f16) &alpha_f16 else &alpha_f32;
    const beta_ptr: ?*const anyopaque = if (case.compute_type == .f16) &beta_f16 else &beta_f32;
    const desc: cublas.RawGemmExDesc = .{
        .op_a = case.op_a,
        .op_b = case.op_b,
        .m = m,
        .n = n,
        .k = k,
        .alpha = alpha_ptr,
        .a = da.ptr,
        .a_type = .r_16f,
        .lda = a_rows,
        .stride_a = stride_a,
        .b = db.ptr,
        .b_type = .r_16f,
        .ldb = b_rows,
        .stride_b = stride_b,
        .beta = beta_ptr,
        .c = dc.ptr,
        .c_type = .r_16f,
        .ldc = m,
        .stride_c = stride_c,
        .batch_count = batch_count,
        .compute_type = case.compute_type,
    };
    for (0..opts.warmup) |_| try runGemmExLike(ctx, case.api, desc);
    try cublas.synchronize();
    const timer = try cublas.GpuTimer.init();
    defer timer.deinit();
    try timer.begin();
    for (0..opts.iters) |_| try runGemmExLike(ctx, case.api, desc);
    const cublas_ms = @as(f64, @floatCast(try timer.end())) / @as(f64, @floatFromInt(opts.iters));
    const shader = vulkan_runner.benchF16ScalarCase(allocator, case.op_a, case.op_b, m, n, k, batch_count, opts.iters, opts.warmup, opts.device_substr) catch |err| switch (err) {
        error.NoVulkanDevice, error.NoMatchingDevice, error.TimestampUnsupported, error.VulkanFailure => null,
        else => return err,
    };
    return .{ .cublas_ms = cublas_ms, .cublas_tflops = tflops(m, n, k, cublas_ms) * @as(f64, @floatFromInt(batch_count)), .scalar_ms = 0, .scalar_tflops = 0, .shader = shader };
}

fn benchGemmExLowpF32(allocator: std.mem.Allocator, case: manifest.Case, opts: Options) !TimedResult {
    const m = case.shape.m;
    const n = case.shape.n;
    const k = case.shape.k;
    const batch_count: usize = 1;
    const a_rows = storedRows(case.op_a, m, k);
    const a_cols = storedCols(case.op_a, m, k);
    const b_rows = storedRows(case.op_b, k, n);
    const b_cols = storedCols(case.op_b, k, n);
    const a = try allocator.alloc(u16, a_rows * a_cols);
    defer allocator.free(a);
    const b = try allocator.alloc(u16, b_rows * b_cols);
    defer allocator.free(b);
    const c = try allocator.alloc(f32, m * n);
    defer allocator.free(c);
    fillLowp16(a, case.a_type, 0.1);
    fillLowp16(b, case.b_type, -0.2);
    fill(c, 0.3);

    const ctx = try cublas.Context.init();
    defer ctx.deinit();
    const da = try cublas.RawDeviceBuffer.alloc(a.len * @sizeOf(u16));
    defer da.free();
    const db = try cublas.RawDeviceBuffer.alloc(b.len * @sizeOf(u16));
    defer db.free();
    const dc = try cublas.RawDeviceBuffer.alloc(c.len * @sizeOf(f32));
    defer dc.free();
    try da.copyFromHost(std.mem.sliceAsBytes(a));
    try db.copyFromHost(std.mem.sliceAsBytes(b));
    try dc.copyFromHost(std.mem.sliceAsBytes(c));

    var alpha: f32 = 1;
    var beta: f32 = 0;
    const desc: cublas.RawGemmExDesc = .{
        .op_a = case.op_a,
        .op_b = case.op_b,
        .m = m,
        .n = n,
        .k = k,
        .alpha = &alpha,
        .a = da.ptr,
        .a_type = case.a_type,
        .lda = a_rows,
        .b = db.ptr,
        .b_type = case.b_type,
        .ldb = b_rows,
        .beta = &beta,
        .c = dc.ptr,
        .c_type = case.c_type,
        .ldc = m,
        .compute_type = case.compute_type,
    };
    for (0..opts.warmup) |_| try cublas.gemmEx(ctx, desc);
    try cublas.synchronize();
    const timer = try cublas.GpuTimer.init();
    defer timer.deinit();
    try timer.begin();
    for (0..opts.iters) |_| try cublas.gemmEx(ctx, desc);
    const cublas_ms = @as(f64, @floatCast(try timer.end())) / @as(f64, @floatFromInt(opts.iters));

    const scalar_m: usize = @min(m, 128);
    const scalar_n: usize = @min(n, 128);
    const scalar_k: usize = @min(k, 128);
    const scalar_a_rows = storedRows(case.op_a, scalar_m, scalar_k);
    const scalar_a_cols = storedCols(case.op_a, scalar_m, scalar_k);
    const scalar_b_rows = storedRows(case.op_b, scalar_k, scalar_n);
    const scalar_b_cols = storedCols(case.op_b, scalar_k, scalar_n);
    const sa = try allocator.alloc(u16, scalar_a_rows * scalar_a_cols);
    defer allocator.free(sa);
    const sb = try allocator.alloc(u16, scalar_b_rows * scalar_b_cols);
    defer allocator.free(sb);
    const sc = try allocator.alloc(f32, scalar_m * scalar_n);
    defer allocator.free(sc);
    fillLowp16(sa, case.a_type, 0.1);
    fillLowp16(sb, case.b_type, -0.2);
    fill(sc, 0.3);
    const scalar_start = try nanoTimestamp();
    try gemm.gemmEx(.{ .op_a = case.op_a, .op_b = case.op_b, .m = scalar_m, .n = scalar_n, .k = scalar_k, .alpha = gemm.Scalar.real(1), .a = std.mem.sliceAsBytes(sa), .a_type = case.a_type, .lda = scalar_a_rows, .b = std.mem.sliceAsBytes(sb), .b_type = case.b_type, .ldb = scalar_b_rows, .beta = gemm.Scalar.real(0), .c = std.mem.sliceAsBytes(sc), .c_type = .r_32f, .ldc = scalar_m, .compute_type = case.compute_type });
    const scalar_end = try nanoTimestamp();
    const scalar_ms = @as(f64, @floatFromInt(scalar_end - scalar_start)) / 1.0e6;
    const coop2_shader = if (case.a_type == .r_16f and case.b_type == .r_16f and case.op_a == .no_trans and case.op_b == .no_trans and batch_count == 1 and m <= 256 and n <= 256 and (case.compute_type == .f32 or case.compute_type == .f32_fast_16f))
        (vulkan_runner.benchNvcoop2F16SmallCase(allocator, m, n, k, opts.iters, opts.warmup, opts.device_substr) catch |err| switch (err) {
            error.NoVulkanDevice, error.NoMatchingDevice, error.TimestampUnsupported, error.VulkanFailure => null,
            else => return err,
        })
    else if (case.a_type == .r_16f and case.b_type == .r_16f and (case.compute_type == .f32 or case.compute_type == .f32_fast_16f))
        (vulkan_runner.benchNvcoop2F16StridedBatchedOpCase(allocator, case.op_a, case.op_b, m, n, k, batch_count, opts.iters, opts.warmup, opts.device_substr) catch |err| switch (err) {
            error.NoVulkanDevice, error.NoMatchingDevice, error.TimestampUnsupported, error.VulkanFailure => null,
            else => return err,
        })
    else if (case.a_type == .r_16bf and case.b_type == .r_16bf and case.op_a == .no_trans and case.op_b == .no_trans and (case.compute_type == .f32 or case.compute_type == .f32_fast_16bf))
        (vulkan_runner.benchNvcoop2Bf16StridedBatchedOpCase(allocator, case.op_a, case.op_b, m, n, k, batch_count, opts.iters, opts.warmup, opts.device_substr) catch |err| switch (err) {
            error.NoVulkanDevice, error.NoMatchingDevice, error.TimestampUnsupported, error.VulkanFailure => null,
            else => return err,
        })
    else
        null;
    const shader = if (coop2_shader) |fast| fast else (vulkan_runner.benchLowpF32BatchedCase(allocator, case.op_a, case.op_b, case.a_type, case.b_type, m, n, k, batch_count, opts.iters, opts.warmup, opts.device_substr) catch |err| switch (err) {
        error.NoVulkanDevice, error.NoMatchingDevice, error.TimestampUnsupported, error.VulkanFailure => null,
        else => return err,
    });
    return .{ .cublas_ms = cublas_ms, .cublas_tflops = tflops(m, n, k, cublas_ms), .scalar_ms = scalar_ms, .scalar_tflops = tflops(scalar_m, scalar_n, scalar_k, scalar_ms), .shader = shader };
}

fn benchGemmStridedBatchedExLowpF32(allocator: std.mem.Allocator, case: manifest.Case, opts: Options) !TimedResult {
    const m = case.shape.m;
    const n = case.shape.n;
    const k = case.shape.k;
    const batch_count: usize = 4;
    const a_rows = storedRows(case.op_a, m, k);
    const a_cols = storedCols(case.op_a, m, k);
    const b_rows = storedRows(case.op_b, k, n);
    const b_cols = storedCols(case.op_b, k, n);
    const stride_a = a_rows * a_cols;
    const stride_b = b_rows * b_cols;
    const stride_c = m * n;
    const a = try allocator.alloc(u16, stride_a * batch_count);
    defer allocator.free(a);
    const b = try allocator.alloc(u16, stride_b * batch_count);
    defer allocator.free(b);
    const c = try allocator.alloc(f32, stride_c * batch_count);
    defer allocator.free(c);
    fillLowp16(a, case.a_type, 0.1);
    fillLowp16(b, case.b_type, -0.2);
    fill(c, 0.3);

    const ctx = try cublas.Context.init();
    defer ctx.deinit();
    const da = try cublas.RawDeviceBuffer.alloc(a.len * @sizeOf(u16));
    defer da.free();
    const db = try cublas.RawDeviceBuffer.alloc(b.len * @sizeOf(u16));
    defer db.free();
    const dc = try cublas.RawDeviceBuffer.alloc(c.len * @sizeOf(f32));
    defer dc.free();
    try da.copyFromHost(std.mem.sliceAsBytes(a));
    try db.copyFromHost(std.mem.sliceAsBytes(b));
    try dc.copyFromHost(std.mem.sliceAsBytes(c));

    var alpha: f32 = 1;
    var beta: f32 = 0;
    const desc: cublas.RawGemmExDesc = .{
        .op_a = case.op_a,
        .op_b = case.op_b,
        .m = m,
        .n = n,
        .k = k,
        .alpha = &alpha,
        .a = da.ptr,
        .a_type = case.a_type,
        .lda = a_rows,
        .stride_a = stride_a,
        .b = db.ptr,
        .b_type = case.b_type,
        .ldb = b_rows,
        .stride_b = stride_b,
        .beta = &beta,
        .c = dc.ptr,
        .c_type = case.c_type,
        .ldc = m,
        .stride_c = stride_c,
        .batch_count = batch_count,
        .compute_type = case.compute_type,
    };
    for (0..opts.warmup) |_| try cublas.gemmStridedBatchedEx(ctx, desc);
    try cublas.synchronize();
    const timer = try cublas.GpuTimer.init();
    defer timer.deinit();
    try timer.begin();
    for (0..opts.iters) |_| try cublas.gemmStridedBatchedEx(ctx, desc);
    const cublas_ms = @as(f64, @floatCast(try timer.end())) / @as(f64, @floatFromInt(opts.iters));

    const scalar_m: usize = @min(m, 128);
    const scalar_n: usize = @min(n, 128);
    const scalar_k: usize = @min(k, 128);
    const scalar_batch_count: usize = 2;
    const scalar_a_rows = storedRows(case.op_a, scalar_m, scalar_k);
    const scalar_a_cols = storedCols(case.op_a, scalar_m, scalar_k);
    const scalar_b_rows = storedRows(case.op_b, scalar_k, scalar_n);
    const scalar_b_cols = storedCols(case.op_b, scalar_k, scalar_n);
    const scalar_stride_a = scalar_a_rows * scalar_a_cols;
    const scalar_stride_b = scalar_b_rows * scalar_b_cols;
    const scalar_stride_c = scalar_m * scalar_n;
    const sa = try allocator.alloc(u16, scalar_stride_a * scalar_batch_count);
    defer allocator.free(sa);
    const sb = try allocator.alloc(u16, scalar_stride_b * scalar_batch_count);
    defer allocator.free(sb);
    const sc = try allocator.alloc(f32, scalar_stride_c * scalar_batch_count);
    defer allocator.free(sc);
    fillLowp16(sa, case.a_type, 0.1);
    fillLowp16(sb, case.b_type, -0.2);
    fill(sc, 0.3);
    const scalar_start = try nanoTimestamp();
    try gemm.gemmEx(.{
        .op_a = case.op_a,
        .op_b = case.op_b,
        .m = scalar_m,
        .n = scalar_n,
        .k = scalar_k,
        .alpha = gemm.Scalar.real(1),
        .a = std.mem.sliceAsBytes(sa),
        .a_type = case.a_type,
        .lda = scalar_a_rows,
        .stride_a = scalar_stride_a,
        .b = std.mem.sliceAsBytes(sb),
        .b_type = case.b_type,
        .ldb = scalar_b_rows,
        .stride_b = scalar_stride_b,
        .beta = gemm.Scalar.real(0),
        .c = std.mem.sliceAsBytes(sc),
        .c_type = .r_32f,
        .ldc = scalar_m,
        .stride_c = scalar_stride_c,
        .batch_count = scalar_batch_count,
        .compute_type = case.compute_type,
    });
    const scalar_end = try nanoTimestamp();
    const scalar_ms = @as(f64, @floatFromInt(scalar_end - scalar_start)) / 1.0e6;

    const coop2_shader = if (case.a_type == .r_16f and case.b_type == .r_16f and (case.compute_type == .f32 or case.compute_type == .f32_fast_16f))
        (vulkan_runner.benchNvcoop2F16StridedBatchedOpCase(allocator, case.op_a, case.op_b, m, n, k, batch_count, opts.iters, opts.warmup, opts.device_substr) catch |err| switch (err) {
            error.NoVulkanDevice, error.NoMatchingDevice, error.TimestampUnsupported, error.VulkanFailure => null,
            else => return err,
        })
    else if (case.a_type == .r_16bf and case.b_type == .r_16bf and case.op_a == .no_trans and case.op_b == .no_trans and (case.compute_type == .f32 or case.compute_type == .f32_fast_16bf))
        (vulkan_runner.benchNvcoop2Bf16StridedBatchedOpCase(allocator, case.op_a, case.op_b, m, n, k, batch_count, opts.iters, opts.warmup, opts.device_substr) catch |err| switch (err) {
            error.NoVulkanDevice, error.NoMatchingDevice, error.TimestampUnsupported, error.VulkanFailure => null,
            else => return err,
        })
    else
        null;
    const shader = if (coop2_shader) |fast| fast else (vulkan_runner.benchLowpF32Case(allocator, case.op_a, case.op_b, case.a_type, case.b_type, m, n, k, opts.iters, opts.warmup, opts.device_substr) catch |err| switch (err) {
        error.NoVulkanDevice, error.NoMatchingDevice, error.TimestampUnsupported, error.VulkanFailure => null,
        else => return err,
    });
    return .{
        .cublas_ms = cublas_ms,
        .cublas_tflops = tflops(m, n, k, cublas_ms) * @as(f64, @floatFromInt(batch_count)),
        .scalar_ms = scalar_ms,
        .scalar_tflops = tflops(scalar_m, scalar_n, scalar_k, scalar_ms) * @as(f64, @floatFromInt(scalar_batch_count)),
        .shader = shader,
    };
}

fn benchGemmExInt8I32(allocator: std.mem.Allocator, case: manifest.Case, opts: Options) !TimedResult {
    const m = case.shape.m;
    const n = case.shape.n;
    const k = case.shape.k;
    const batch_count = exBatchCount(case.api);
    const a_rows = storedRows(case.op_a, m, k);
    const a_cols = storedCols(case.op_a, m, k);
    const b_rows = storedRows(case.op_b, k, n);
    const b_cols = storedCols(case.op_b, k, n);
    const stride_a = a_rows * a_cols;
    const stride_b = b_rows * b_cols;
    const stride_c = m * n;
    const a = try allocator.alloc(i8, stride_a * batch_count);
    defer allocator.free(a);
    const b = try allocator.alloc(i8, stride_b * batch_count);
    defer allocator.free(b);
    const c = try allocator.alloc(i32, stride_c * batch_count);
    defer allocator.free(c);
    fillI8(a, 1);
    fillI8(b, -2);
    @memset(c, 0);

    const ctx = try cublas.Context.init();
    defer ctx.deinit();
    const da = try cublas.RawDeviceBuffer.alloc(a.len);
    defer da.free();
    const db = try cublas.RawDeviceBuffer.alloc(b.len);
    defer db.free();
    const dc = try cublas.RawDeviceBuffer.alloc(c.len * @sizeOf(i32));
    defer dc.free();
    try da.copyFromHost(std.mem.sliceAsBytes(a));
    try db.copyFromHost(std.mem.sliceAsBytes(b));
    try dc.copyFromHost(std.mem.sliceAsBytes(c));

    var alpha: i32 = 1;
    var beta: i32 = 0;
    const desc: cublas.RawGemmExDesc = .{ .op_a = case.op_a, .op_b = case.op_b, .m = m, .n = n, .k = k, .alpha = &alpha, .a = da.ptr, .a_type = .r_8i, .lda = a_rows, .stride_a = stride_a, .b = db.ptr, .b_type = .r_8i, .ldb = b_rows, .stride_b = stride_b, .beta = &beta, .c = dc.ptr, .c_type = .r_32i, .ldc = m, .stride_c = stride_c, .batch_count = batch_count, .compute_type = .i32 };
    for (0..opts.warmup) |_| try runGemmExLike(ctx, case.api, desc);
    try cublas.synchronize();
    const timer = try cublas.GpuTimer.init();
    defer timer.deinit();
    try timer.begin();
    for (0..opts.iters) |_| try runGemmExLike(ctx, case.api, desc);
    const cublas_ms = @as(f64, @floatCast(try timer.end())) / @as(f64, @floatFromInt(opts.iters));
    const shader = vulkan_runner.benchInt8I32BatchedCase(allocator, case.op_a, case.op_b, m, n, k, batch_count, opts.iters, opts.warmup, opts.device_substr) catch |err| switch (err) {
        error.NoVulkanDevice, error.NoMatchingDevice, error.TimestampUnsupported, error.VulkanFailure => null,
        else => return err,
    };
    return .{ .cublas_ms = cublas_ms, .cublas_tflops = tflops(m, n, k, cublas_ms) * @as(f64, @floatFromInt(batch_count)), .scalar_ms = 0, .scalar_tflops = 0, .shader = shader };
}

fn benchRegularC32(allocator: std.mem.Allocator, case: manifest.Case, opts: Options) !TimedResult {
    const m = case.shape.m;
    const n = case.shape.n;
    const k = case.shape.k;
    const batch_count = classicBatchCount(case.api);
    const a_rows = storedRows(case.op_a, m, k);
    const a_cols = storedCols(case.op_a, m, k);
    const b_rows = storedRows(case.op_b, k, n);
    const b_cols = storedCols(case.op_b, k, n);
    const a_len = a_rows * a_cols;
    const b_len = b_rows * b_cols;
    const c_len = m * n;
    const a = try allocator.alloc(gemm.Complex32, a_len * batch_count);
    defer allocator.free(a);
    const b = try allocator.alloc(gemm.Complex32, b_len * batch_count);
    defer allocator.free(b);
    const c = try allocator.alloc(gemm.Complex32, c_len * batch_count);
    defer allocator.free(c);
    fillTyped(gemm.Complex32, a, 0.1);
    fillTyped(gemm.Complex32, b, -0.2);
    fillTyped(gemm.Complex32, c, 0.3);

    const ctx = try cublas.Context.init();
    defer ctx.deinit();
    const da = try cublas.DeviceBuffer(gemm.Complex32).alloc(a.len);
    defer da.free();
    const db = try cublas.DeviceBuffer(gemm.Complex32).alloc(b.len);
    defer db.free();
    const dc = try cublas.DeviceBuffer(gemm.Complex32).alloc(c.len);
    defer dc.free();
    try da.copyFromHost(a);
    try db.copyFromHost(b);
    try dc.copyFromHost(c);

    const alpha = gemm.Complex32.init(1, 0);
    const beta = gemm.Complex32.init(0, 0);
    for (0..opts.warmup) |_| {
        try runClassicCgemmLike(ctx, case.api, case.op_a, case.op_b, m, n, k, alpha, da.ptr, a_rows, a_len, db.ptr, b_rows, b_len, beta, dc.ptr, m, c_len, batch_count);
    }
    try cublas.synchronize();
    const timer = try cublas.GpuTimer.init();
    defer timer.deinit();
    try timer.begin();
    for (0..opts.iters) |_| {
        try runClassicCgemmLike(ctx, case.api, case.op_a, case.op_b, m, n, k, alpha, da.ptr, a_rows, a_len, db.ptr, b_rows, b_len, beta, dc.ptr, m, c_len, batch_count);
    }
    const cublas_ms = @as(f64, @floatCast(try timer.end())) / @as(f64, @floatFromInt(opts.iters));

    const scalar_m: usize = @min(m, 96);
    const scalar_n: usize = @min(n, 96);
    const scalar_k: usize = @min(k, 96);
    const scalar_a_rows = storedRows(case.op_a, scalar_m, scalar_k);
    const scalar_a_cols = storedCols(case.op_a, scalar_m, scalar_k);
    const scalar_b_rows = storedRows(case.op_b, scalar_k, scalar_n);
    const scalar_b_cols = storedCols(case.op_b, scalar_k, scalar_n);
    const sa = try allocator.alloc(gemm.Complex32, scalar_a_rows * scalar_a_cols);
    defer allocator.free(sa);
    const sb = try allocator.alloc(gemm.Complex32, scalar_b_rows * scalar_b_cols);
    defer allocator.free(sb);
    const sc = try allocator.alloc(gemm.Complex32, scalar_m * scalar_n);
    defer allocator.free(sc);
    fillTyped(gemm.Complex32, sa, 0.1);
    fillTyped(gemm.Complex32, sb, -0.2);
    fillTyped(gemm.Complex32, sc, 0.3);
    const scalar_start = try nanoTimestamp();
    try gemm.cgemm(.col_major, case.op_a, case.op_b, scalar_m, scalar_n, scalar_k, alpha, sa, scalar_a_rows, sb, scalar_b_rows, beta, sc, scalar_m);
    const scalar_end = try nanoTimestamp();
    const scalar_ms = @as(f64, @floatFromInt(scalar_end - scalar_start)) / 1.0e6;
    const shader = vulkan_runner.benchCgemmF32BatchedCase(allocator, case.op_a, case.op_b, m, n, k, batch_count, opts.iters, opts.warmup, opts.device_substr) catch |err| switch (err) {
        error.NoVulkanDevice, error.NoMatchingDevice, error.TimestampUnsupported, error.VulkanFailure => null,
        else => return err,
    };

    return .{ .cublas_ms = cublas_ms, .cublas_tflops = tflops(m, n, k, cublas_ms) * @as(f64, @floatFromInt(batch_count)), .scalar_ms = scalar_ms, .scalar_tflops = tflops(scalar_m, scalar_n, scalar_k, scalar_ms), .shader = shader };
}

fn benchRegularC64(allocator: std.mem.Allocator, case: manifest.Case, opts: Options) !TimedResult {
    const m = case.shape.m;
    const n = case.shape.n;
    const k = case.shape.k;
    const batch_count = classicBatchCount(case.api);
    const a_rows = storedRows(case.op_a, m, k);
    const a_cols = storedCols(case.op_a, m, k);
    const b_rows = storedRows(case.op_b, k, n);
    const b_cols = storedCols(case.op_b, k, n);
    const a_len = a_rows * a_cols;
    const b_len = b_rows * b_cols;
    const c_len = m * n;
    const a = try allocator.alloc(gemm.Complex64, a_len * batch_count);
    defer allocator.free(a);
    const b = try allocator.alloc(gemm.Complex64, b_len * batch_count);
    defer allocator.free(b);
    const c = try allocator.alloc(gemm.Complex64, c_len * batch_count);
    defer allocator.free(c);
    fillTyped(gemm.Complex64, a, 0.1);
    fillTyped(gemm.Complex64, b, -0.2);
    fillTyped(gemm.Complex64, c, 0.3);

    const ctx = try cublas.Context.init();
    defer ctx.deinit();
    const da = try cublas.DeviceBuffer(gemm.Complex64).alloc(a.len);
    defer da.free();
    const db = try cublas.DeviceBuffer(gemm.Complex64).alloc(b.len);
    defer db.free();
    const dc = try cublas.DeviceBuffer(gemm.Complex64).alloc(c.len);
    defer dc.free();
    try da.copyFromHost(a);
    try db.copyFromHost(b);
    try dc.copyFromHost(c);

    const alpha = gemm.Complex64.init(1, 0);
    const beta = gemm.Complex64.init(0, 0);
    for (0..opts.warmup) |_| {
        try runClassicZgemmLike(ctx, case.api, case.op_a, case.op_b, m, n, k, alpha, da.ptr, a_rows, a_len, db.ptr, b_rows, b_len, beta, dc.ptr, m, c_len, batch_count);
    }
    try cublas.synchronize();
    const timer = try cublas.GpuTimer.init();
    defer timer.deinit();
    try timer.begin();
    for (0..opts.iters) |_| {
        try runClassicZgemmLike(ctx, case.api, case.op_a, case.op_b, m, n, k, alpha, da.ptr, a_rows, a_len, db.ptr, b_rows, b_len, beta, dc.ptr, m, c_len, batch_count);
    }
    const cublas_ms = @as(f64, @floatCast(try timer.end())) / @as(f64, @floatFromInt(opts.iters));

    const scalar_m: usize = @min(m, 64);
    const scalar_n: usize = @min(n, 64);
    const scalar_k: usize = @min(k, 64);
    const scalar_a_rows = storedRows(case.op_a, scalar_m, scalar_k);
    const scalar_a_cols = storedCols(case.op_a, scalar_m, scalar_k);
    const scalar_b_rows = storedRows(case.op_b, scalar_k, scalar_n);
    const scalar_b_cols = storedCols(case.op_b, scalar_k, scalar_n);
    const sa = try allocator.alloc(gemm.Complex64, scalar_a_rows * scalar_a_cols);
    defer allocator.free(sa);
    const sb = try allocator.alloc(gemm.Complex64, scalar_b_rows * scalar_b_cols);
    defer allocator.free(sb);
    const sc = try allocator.alloc(gemm.Complex64, scalar_m * scalar_n);
    defer allocator.free(sc);
    fillTyped(gemm.Complex64, sa, 0.1);
    fillTyped(gemm.Complex64, sb, -0.2);
    fillTyped(gemm.Complex64, sc, 0.3);
    const scalar_start = try nanoTimestamp();
    try gemm.zgemm(.col_major, case.op_a, case.op_b, scalar_m, scalar_n, scalar_k, alpha, sa, scalar_a_rows, sb, scalar_b_rows, beta, sc, scalar_m);
    const scalar_end = try nanoTimestamp();
    const scalar_ms = @as(f64, @floatFromInt(scalar_end - scalar_start)) / 1.0e6;
    const shader = vulkan_runner.benchZgemmF64BatchedCase(allocator, case.op_a, case.op_b, m, n, k, batch_count, opts.iters, opts.warmup, opts.device_substr) catch |err| switch (err) {
        error.NoVulkanDevice, error.NoMatchingDevice, error.TimestampUnsupported, error.VulkanFailure => null,
        else => return err,
    };

    return .{ .cublas_ms = cublas_ms, .cublas_tflops = tflops(m, n, k, cublas_ms) * @as(f64, @floatFromInt(batch_count)), .scalar_ms = scalar_ms, .scalar_tflops = tflops(scalar_m, scalar_n, scalar_k, scalar_ms), .shader = shader };
}

fn classicBatchCount(api: manifest.Api) usize {
    return switch (api) {
        .batched, .strided_batched => 4,
        else => 1,
    };
}

fn exBatchCount(api: manifest.Api) usize {
    return switch (api) {
        .gemm_batched_ex, .gemm_strided_batched_ex => 4,
        else => 1,
    };
}

fn runGemmExLike(ctx: cublas.Context, api: manifest.Api, desc: cublas.RawGemmExDesc) !void {
    switch (api) {
        .gemm_ex => try cublas.gemmEx(ctx, desc),
        .gemm_batched_ex, .gemm_strided_batched_ex => try cublas.gemmStridedBatchedEx(ctx, desc),
        else => return error.InvalidArgument,
    }
}

fn runClassicSgemmLike(ctx: cublas.Context, api: manifest.Api, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, alpha: f32, a: [*]const f32, lda: usize, stride_a: usize, b: [*]const f32, ldb: usize, stride_b: usize, beta: f32, c: [*]f32, ldc: usize, stride_c: usize, batch_count: usize) !void {
    switch (api) {
        .regular, .regular_64 => try cublas.sgemm(ctx, op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        .batched, .strided_batched => try cublas.sgemmStridedBatched(ctx, op_a, op_b, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_count),
        else => return error.InvalidArgument,
    }
}

fn runClassicDgemmLike(ctx: cublas.Context, api: manifest.Api, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, alpha: f64, a: [*]const f64, lda: usize, stride_a: usize, b: [*]const f64, ldb: usize, stride_b: usize, beta: f64, c: [*]f64, ldc: usize, stride_c: usize, batch_count: usize) !void {
    switch (api) {
        .regular, .regular_64 => try cublas.dgemm(ctx, op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        .batched, .strided_batched => try cublas.dgemmStridedBatched(ctx, op_a, op_b, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_count),
        else => return error.InvalidArgument,
    }
}

fn runClassicCgemmLike(ctx: cublas.Context, api: manifest.Api, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, alpha: gemm.Complex32, a: [*]const gemm.Complex32, lda: usize, stride_a: usize, b: [*]const gemm.Complex32, ldb: usize, stride_b: usize, beta: gemm.Complex32, c: [*]gemm.Complex32, ldc: usize, stride_c: usize, batch_count: usize) !void {
    switch (api) {
        .regular, .regular_64 => try cublas.cgemm(ctx, op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        .complex_3m => try cublas.cgemm3m(ctx, op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        .batched, .strided_batched => try cublas.cgemmStridedBatched(ctx, op_a, op_b, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_count),
        else => return error.InvalidArgument,
    }
}

fn runClassicZgemmLike(ctx: cublas.Context, api: manifest.Api, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, alpha: gemm.Complex64, a: [*]const gemm.Complex64, lda: usize, stride_a: usize, b: [*]const gemm.Complex64, ldb: usize, stride_b: usize, beta: gemm.Complex64, c: [*]gemm.Complex64, ldc: usize, stride_c: usize, batch_count: usize) !void {
    switch (api) {
        .regular, .regular_64 => try cublas.zgemm(ctx, op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        .complex_3m => try cublas.zgemm3m(ctx, op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc),
        .batched, .strided_batched => try cublas.zgemmStridedBatched(ctx, op_a, op_b, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c, batch_count),
        else => return error.InvalidArgument,
    }
}

fn parseArgs(allocator: std.mem.Allocator, args: []const []const u8) !Options {
    var sizes_text: []const u8 = "512,1024,2048";
    var rect_text: []const u8 = "";
    var iters: usize = 20;
    var warmup: usize = 5;
    var opts_device_substr: ?[]const u8 = null;
    var manifest_all = false;
    var fail_on_regression = false;
    var fail_every_row = false;
    var only_cublas_supported = false;
    var tile_compatible_only = false;
    var output: ?[]const u8 = null;

    for (args[1..]) |arg| {
        if (std.mem.startsWith(u8, arg, "--sizes=")) {
            sizes_text = arg["--sizes=".len..];
        } else if (std.mem.startsWith(u8, arg, "--rect-sizes=")) {
            rect_text = arg["--rect-sizes=".len..];
        } else if (std.mem.startsWith(u8, arg, "--iters=")) {
            iters = try std.fmt.parseInt(usize, arg["--iters=".len..], 10);
        } else if (std.mem.startsWith(u8, arg, "--warmup=")) {
            warmup = try std.fmt.parseInt(usize, arg["--warmup=".len..], 10);
        } else if (std.mem.eql(u8, arg, "--fail-on-regression")) {
            fail_on_regression = true;
        } else if (std.mem.eql(u8, arg, "--fail-every-row")) {
            fail_every_row = true;
        } else if (std.mem.eql(u8, arg, "--only-cublas-supported")) {
            only_cublas_supported = true;
        } else if (std.mem.eql(u8, arg, "--tile-compatible-only")) {
            tile_compatible_only = true;
        } else if (std.mem.startsWith(u8, arg, "--manifest=")) {
            manifest_all = std.mem.eql(u8, arg["--manifest=".len..], "all");
        } else if (std.mem.startsWith(u8, arg, "--output=")) {
            output = arg["--output=".len..];
        } else if (std.mem.startsWith(u8, arg, "--suite=") or std.mem.startsWith(u8, arg, "--ops=") or std.mem.startsWith(u8, arg, "--types=")) {
            // Accepted for manifest compatibility; current manifest mode
            // enumerates all suites/ops/types when --manifest=all is present.
        } else if (std.mem.startsWith(u8, arg, "--device-substr=")) {
            const value = arg["--device-substr=".len..];
            if (value.len != 0) opts_device_substr = value;
        } else {
            return error.InvalidArgument;
        }
    }

    return .{
        .sizes = try parseSizes(allocator, sizes_text),
        .rect_shapes = try parseRectShapes(allocator, rect_text),
        .iters = iters,
        .warmup = warmup,
        .device_substr = opts_device_substr,
        .manifest_all = manifest_all,
        .fail_on_regression = fail_on_regression,
        .fail_every_row = fail_every_row,
        .only_cublas_supported = only_cublas_supported,
        .tile_compatible_only = tile_compatible_only,
        .output = output,
    };
}

fn parseSizes(allocator: std.mem.Allocator, text: []const u8) ![]usize {
    if (text.len == 0) return allocator.alloc(usize, 0);
    var count: usize = 1;
    for (text) |ch| {
        if (ch == ',') count += 1;
    }
    var sizes = try allocator.alloc(usize, count);
    var split = std.mem.splitScalar(u8, text, ',');
    var i: usize = 0;
    while (split.next()) |part| : (i += 1) {
        sizes[i] = try std.fmt.parseInt(usize, part, 10);
        if (sizes[i] == 0) return error.InvalidArgument;
    }
    return sizes;
}

fn parseRectShapes(allocator: std.mem.Allocator, text: []const u8) ![]manifest.Shape {
    if (text.len == 0) return allocator.alloc(manifest.Shape, 0);
    var count: usize = 1;
    for (text) |ch| {
        if (ch == ',') count += 1;
    }
    var shapes = try allocator.alloc(manifest.Shape, count);
    var split = std.mem.splitScalar(u8, text, ',');
    var i: usize = 0;
    while (split.next()) |part| : (i += 1) {
        var dims = std.mem.splitScalar(u8, part, 'x');
        const m = try std.fmt.parseInt(usize, dims.next() orelse return error.InvalidArgument, 10);
        const n = try std.fmt.parseInt(usize, dims.next() orelse return error.InvalidArgument, 10);
        const k = try std.fmt.parseInt(usize, dims.next() orelse return error.InvalidArgument, 10);
        if (dims.next() != null or m == 0 or n == 0 or k == 0) return error.InvalidArgument;
        shapes[i] = .{ .m = m, .n = n, .k = k };
    }
    return shapes;
}

const Output = struct {
    file: ?*CFile = null,

    fn init(allocator: std.mem.Allocator, path: ?[]const u8) !Output {
        if (path) |p| {
            const z_path = try allocator.dupeZ(u8, p);
            defer allocator.free(z_path);
            const file = fopen(z_path.ptr, "wb") orelse return error.OutputOpenFailed;
            return .{ .file = file };
        }
        return .{};
    }

    fn deinit(self: *Output, allocator: std.mem.Allocator) void {
        _ = allocator;
        if (self.file) |file| _ = fclose(file);
    }

    fn write(self: *Output, allocator: std.mem.Allocator, text: []const u8) !void {
        _ = allocator;
        if (self.file) |file| {
            if (fwrite(text.ptr, 1, text.len, file) != text.len) return error.OutputWriteFailed;
        } else {
            std.debug.print("{s}", .{text});
        }
    }
};

fn fill(dst: []f32, bias: f32) void {
    for (dst, 0..) |*value, i| {
        const raw: f32 = @floatFromInt((i * 19 + 7) % 23);
        value.* = (raw - 11) / 7 + bias;
    }
}

fn scalarSgemmMs(allocator: std.mem.Allocator, op_a: gemm.Op, op_b: gemm.Op, m: usize, n: usize, k: usize, a_rows: usize, b_rows: usize) !f64 {
    const a_cols = storedCols(op_a, m, k);
    const b_cols = storedCols(op_b, k, n);
    const a = try allocator.alloc(f32, a_rows * a_cols);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, b_rows * b_cols);
    defer allocator.free(b);
    const c = try allocator.alloc(f32, m * n);
    defer allocator.free(c);
    fill(a, 0.1);
    fill(b, -0.2);
    fill(c, 0.3);
    const start = try nanoTimestamp();
    try gemm.sgemm(.col_major, op_a, op_b, m, n, k, 1, a, a_rows, b, b_rows, 0, c, m);
    const end = try nanoTimestamp();
    return @as(f64, @floatFromInt(end - start)) / 1.0e6;
}

fn fillTyped(comptime T: type, dst: []T, bias: f32) void {
    for (dst, 0..) |*value, i| {
        const raw: f32 = @floatFromInt((i * 19 + 7) % 23);
        const re = (raw - 11) / 7 + bias;
        if (T == f64) {
            value.* = @floatCast(re);
        } else if (T == gemm.Complex32) {
            const im_raw: f32 = @floatFromInt((i * 13 + 5) % 17);
            value.* = .{ .re = re, .im = (im_raw - 8) / 11 };
        } else if (T == gemm.Complex64) {
            const im_raw: f32 = @floatFromInt((i * 13 + 5) % 17);
            value.* = .{ .re = @floatCast(re), .im = @floatCast((im_raw - 8) / 11) };
        } else {
            @compileError("unsupported benchmark type");
        }
    }
}

fn fillLowp16(dst: []u16, data_type: gemm.DataType, bias: f32) void {
    for (dst, 0..) |*value, i| {
        const raw: f32 = @floatFromInt((i * 19 + 7) % 23);
        const v = (raw - 11) / 7 + bias;
        value.* = switch (data_type) {
            .r_16f => @intCast(f32ToF16Bits(v)),
            .r_16bf => @intCast(f32ToBf16Bits(v)),
            else => 0,
        };
    }
}

fn fillI8(dst: []i8, bias: i32) void {
    for (dst, 0..) |*value, i| {
        const raw: i32 = @intCast((i * 7 + 3) % 17);
        value.* = @intCast(raw - 8 + bias);
    }
}

fn f32ToBf16Bits(value: f32) u32 {
    const bits: u32 = @bitCast(value);
    return bits >> 16;
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

fn nanoTimestamp() !i128 {
    var ts: Timespec = undefined;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return error.ClockFailure;
    return @as(i128, ts.tv_sec) * 1_000_000_000 + ts.tv_nsec;
}
