const std = @import("std");

pub const Layout = enum {
    row_major,
    col_major,
};

pub const Op = enum {
    no_trans,
    trans,
    conj_trans,
};

pub const DataType = enum(c_int) {
    r_16f = 2,
    c_16f = 6,
    r_16bf = 14,
    c_16bf = 15,
    r_32f = 0,
    c_32f = 4,
    r_64f = 1,
    c_64f = 5,
    r_4i = 16,
    c_4i = 17,
    r_4u = 18,
    c_4u = 19,
    r_8i = 3,
    c_8i = 7,
    r_8u = 8,
    c_8u = 9,
    r_16i = 20,
    c_16i = 21,
    r_16u = 22,
    c_16u = 23,
    r_32i = 10,
    c_32i = 11,
    r_32u = 12,
    c_32u = 13,
    r_64i = 24,
    c_64i = 25,
    r_64u = 26,
    c_64u = 27,
    r_8f_e4m3 = 28,
    r_8f_e5m2 = 29,
    r_8f_ue8m0 = 30,
    r_6f_e2m3 = 31,
    r_6f_e3m2 = 32,
    r_4f_e2m1 = 33,
};

pub const cuda_r_8f_ue4m3 = DataType.r_8f_e4m3;

pub const ComputeType = enum(c_int) {
    f16 = 64,
    f16_pedantic = 65,
    f32 = 68,
    f32_pedantic = 69,
    f32_fast_16f = 74,
    f32_fast_16bf = 75,
    f32_fast_tf32 = 77,
    f32_emulated_16bfx9 = 78,
    f64 = 70,
    f64_pedantic = 71,
    f64_emulated_fixedpoint = 79,
    i32 = 72,
    i32_pedantic = 73,
};

pub const Algo = enum(c_int) {
    default = -1,
    algo0 = 0,
    algo1 = 1,
    algo2 = 2,
    algo3 = 3,
    algo4 = 4,
    algo5 = 5,
    algo6 = 6,
    algo7 = 7,
    algo8 = 8,
    algo9 = 9,
    algo10 = 10,
    algo11 = 11,
    algo12 = 12,
    algo13 = 13,
    algo14 = 14,
    algo15 = 15,
    algo16 = 16,
    algo17 = 17,
    algo18 = 18,
    algo19 = 19,
    algo20 = 20,
    algo21 = 21,
    algo22 = 22,
    algo23 = 23,
    default_tensor_op = 99,
    algo0_tensor_op = 100,
    algo1_tensor_op = 101,
    algo2_tensor_op = 102,
    algo3_tensor_op = 103,
    algo4_tensor_op = 104,
    algo5_tensor_op = 105,
    algo6_tensor_op = 106,
    algo7_tensor_op = 107,
    algo8_tensor_op = 108,
    algo9_tensor_op = 109,
    algo10_tensor_op = 110,
    algo11_tensor_op = 111,
    algo12_tensor_op = 112,
    algo13_tensor_op = 113,
    algo14_tensor_op = 114,
    algo15_tensor_op = 115,
    autotune = 999,
};

pub const GemmType = enum {
    f32,
    f64,
    c64,
    c128,
    f16_f32,
    bf16_f32,
};

pub const Complex32 = extern struct {
    re: f32,
    im: f32,

    pub fn init(re: f32, im: f32) Complex32 {
        return .{ .re = re, .im = im };
    }
};

pub const Complex64 = extern struct {
    re: f64,
    im: f64,

    pub fn init(re: f64, im: f64) Complex64 {
        return .{ .re = re, .im = im };
    }
};

pub const Scalar = extern struct {
    re: f64 = 0,
    im: f64 = 0,

    pub fn real(value: f64) Scalar {
        return .{ .re = value, .im = 0 };
    }

    pub fn complex(re: f64, im: f64) Scalar {
        return .{ .re = re, .im = im };
    }
};

pub const MatrixLayout = extern struct {
    offset: usize = 0,
    row_stride: usize,
    col_stride: usize,

    pub fn index(self: MatrixLayout, row: usize, col: usize) usize {
        return self.offset + row * self.row_stride + col * self.col_stride;
    }
};

pub fn matrixLayout(layout: Layout, ld: usize) MatrixLayout {
    return switch (layout) {
        .row_major => .{ .row_stride = ld, .col_stride = 1 },
        .col_major => .{ .row_stride = 1, .col_stride = ld },
    };
}

pub fn storedRows(op: Op, logical_rows: usize, logical_cols: usize) usize {
    return switch (op) {
        .no_trans => logical_rows,
        .trans, .conj_trans => logical_cols,
    };
}

pub fn storedCols(op: Op, logical_rows: usize, logical_cols: usize) usize {
    return switch (op) {
        .no_trans => logical_cols,
        .trans, .conj_trans => logical_rows,
    };
}

pub fn minimumLd(layout: Layout, rows: usize, cols: usize) usize {
    return switch (layout) {
        .row_major => @max(1, cols),
        .col_major => @max(1, rows),
    };
}

pub fn ScalarDesc(comptime T: type) type {
    return struct {
        op_a: Op,
        op_b: Op,
        m: usize,
        n: usize,
        k: usize,
        a: []const T,
        b: []const T,
        c: []T,
        a_layout: MatrixLayout,
        b_layout: MatrixLayout,
        c_layout: MatrixLayout,
        alpha: T,
        beta: T,
        batch_count: usize = 1,
        stride_a: usize = 0,
        stride_b: usize = 0,
        stride_c: usize = 0,
    };
}

pub const GemmExDesc = struct {
    op_a: Op,
    op_b: Op,
    m: usize,
    n: usize,
    k: usize,
    a: []const u16,
    b: []const u16,
    c: []f32,
    a_layout: MatrixLayout,
    b_layout: MatrixLayout,
    c_layout: MatrixLayout,
    alpha: f32 = 1,
    beta: f32 = 0,
    input: GemmType,
    batch_count: usize = 1,
    stride_a: usize = 0,
    stride_b: usize = 0,
    stride_c: usize = 0,
};

pub const ExDesc = struct {
    op_a: Op,
    op_b: Op,
    m: usize,
    n: usize,
    k: usize,
    alpha: Scalar,
    a: []const u8,
    a_type: DataType,
    lda: usize,
    b: []const u8,
    b_type: DataType,
    ldb: usize,
    beta: Scalar,
    c: []u8,
    c_type: DataType,
    ldc: usize,
    compute_type: ComputeType,
    layout: Layout = .col_major,
    batch_count: usize = 1,
    stride_a: usize = 0,
    stride_b: usize = 0,
    stride_c: usize = 0,
    algo: Algo = .default,
    three_m: bool = false,
};

pub const PointerBatchedExDesc = struct {
    op_a: Op,
    op_b: Op,
    m: usize,
    n: usize,
    k: usize,
    alpha: Scalar,
    a: []const []const u8,
    a_type: DataType,
    lda: usize,
    b: []const []const u8,
    b_type: DataType,
    ldb: usize,
    beta: Scalar,
    c: []const []u8,
    c_type: DataType,
    ldc: usize,
    compute_type: ComputeType,
    layout: Layout = .col_major,
    algo: Algo = .default,
    three_m: bool = false,
};

pub fn sgemm(
    layout: Layout,
    op_a: Op,
    op_b: Op,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: []const f32,
    lda: usize,
    b: []const f32,
    ldb: usize,
    beta: f32,
    c: []f32,
    ldc: usize,
) !void {
    try gemm(f32, .{
        .op_a = op_a,
        .op_b = op_b,
        .m = m,
        .n = n,
        .k = k,
        .alpha = alpha,
        .beta = beta,
        .a = a,
        .b = b,
        .c = c,
        .a_layout = matrixLayout(layout, lda),
        .b_layout = matrixLayout(layout, ldb),
        .c_layout = matrixLayout(layout, ldc),
    });
}

pub fn dgemm(
    layout: Layout,
    op_a: Op,
    op_b: Op,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: []const f64,
    lda: usize,
    b: []const f64,
    ldb: usize,
    beta: f64,
    c: []f64,
    ldc: usize,
) !void {
    try gemm(f64, .{
        .op_a = op_a,
        .op_b = op_b,
        .m = m,
        .n = n,
        .k = k,
        .alpha = alpha,
        .beta = beta,
        .a = a,
        .b = b,
        .c = c,
        .a_layout = matrixLayout(layout, lda),
        .b_layout = matrixLayout(layout, ldb),
        .c_layout = matrixLayout(layout, ldc),
    });
}

pub fn cgemm(
    layout: Layout,
    op_a: Op,
    op_b: Op,
    m: usize,
    n: usize,
    k: usize,
    alpha: Complex32,
    a: []const Complex32,
    lda: usize,
    b: []const Complex32,
    ldb: usize,
    beta: Complex32,
    c: []Complex32,
    ldc: usize,
) !void {
    try gemm(Complex32, .{
        .op_a = op_a,
        .op_b = op_b,
        .m = m,
        .n = n,
        .k = k,
        .alpha = alpha,
        .beta = beta,
        .a = a,
        .b = b,
        .c = c,
        .a_layout = matrixLayout(layout, lda),
        .b_layout = matrixLayout(layout, ldb),
        .c_layout = matrixLayout(layout, ldc),
    });
}

pub fn zgemm(
    layout: Layout,
    op_a: Op,
    op_b: Op,
    m: usize,
    n: usize,
    k: usize,
    alpha: Complex64,
    a: []const Complex64,
    lda: usize,
    b: []const Complex64,
    ldb: usize,
    beta: Complex64,
    c: []Complex64,
    ldc: usize,
) !void {
    try gemm(Complex64, .{
        .op_a = op_a,
        .op_b = op_b,
        .m = m,
        .n = n,
        .k = k,
        .alpha = alpha,
        .beta = beta,
        .a = a,
        .b = b,
        .c = c,
        .a_layout = matrixLayout(layout, lda),
        .b_layout = matrixLayout(layout, ldb),
        .c_layout = matrixLayout(layout, ldc),
    });
}

pub const GemmDesc = ExDesc;
pub const StridedBatchedGemmDesc = ExDesc;
pub const BatchedGemmDesc = PointerBatchedExDesc;

pub fn sgemm64(layout: Layout, op_a: Op, op_b: Op, m: u64, n: u64, k: u64, alpha: f32, a: []const f32, lda: u64, b: []const f32, ldb: u64, beta: f32, c: []f32, ldc: u64) !void {
    try sgemm(layout, op_a, op_b, checkedUsize(m), checkedUsize(n), checkedUsize(k), alpha, a, checkedUsize(lda), b, checkedUsize(ldb), beta, c, checkedUsize(ldc));
}

pub fn dgemm64(layout: Layout, op_a: Op, op_b: Op, m: u64, n: u64, k: u64, alpha: f64, a: []const f64, lda: u64, b: []const f64, ldb: u64, beta: f64, c: []f64, ldc: u64) !void {
    try dgemm(layout, op_a, op_b, checkedUsize(m), checkedUsize(n), checkedUsize(k), alpha, a, checkedUsize(lda), b, checkedUsize(ldb), beta, c, checkedUsize(ldc));
}

pub fn cgemm64(layout: Layout, op_a: Op, op_b: Op, m: u64, n: u64, k: u64, alpha: Complex32, a: []const Complex32, lda: u64, b: []const Complex32, ldb: u64, beta: Complex32, c: []Complex32, ldc: u64) !void {
    try cgemm(layout, op_a, op_b, checkedUsize(m), checkedUsize(n), checkedUsize(k), alpha, a, checkedUsize(lda), b, checkedUsize(ldb), beta, c, checkedUsize(ldc));
}

pub fn zgemm64(layout: Layout, op_a: Op, op_b: Op, m: u64, n: u64, k: u64, alpha: Complex64, a: []const Complex64, lda: u64, b: []const Complex64, ldb: u64, beta: Complex64, c: []Complex64, ldc: u64) !void {
    try zgemm(layout, op_a, op_b, checkedUsize(m), checkedUsize(n), checkedUsize(k), alpha, a, checkedUsize(lda), b, checkedUsize(ldb), beta, c, checkedUsize(ldc));
}

pub fn hgemm(
    layout: Layout,
    op_a: Op,
    op_b: Op,
    m: usize,
    n: usize,
    k: usize,
    alpha: u16,
    a: []const u16,
    lda: usize,
    b: []const u16,
    ldb: usize,
    beta: u16,
    c: []u16,
    ldc: usize,
) !void {
    const a_layout = matrixLayout(layout, lda);
    const b_layout = matrixLayout(layout, ldb);
    const c_layout = matrixLayout(layout, ldc);
    const alpha_f32 = f16BitsToF32(alpha);
    const beta_f32 = f16BitsToF32(beta);
    for (0..m) |row| {
        for (0..n) |col| {
            var sum: f32 = 0;
            for (0..k) |kk| {
                const av = switch (op_a) {
                    .no_trans => f16BitsToF32(a[a_layout.index(row, kk)]),
                    .trans, .conj_trans => f16BitsToF32(a[a_layout.index(kk, row)]),
                };
                const bv = switch (op_b) {
                    .no_trans => f16BitsToF32(b[b_layout.index(kk, col)]),
                    .trans, .conj_trans => f16BitsToF32(b[b_layout.index(col, kk)]),
                };
                sum += av * bv;
            }
            const c_index = c_layout.index(row, col);
            c[c_index] = f32ToF16Bits(alpha_f32 * sum + beta_f32 * f16BitsToF32(c[c_index]));
        }
    }
}

pub fn sgemmStridedBatched(layout: Layout, op_a: Op, op_b: Op, m: usize, n: usize, k: usize, alpha: f32, a: []const f32, lda: usize, stride_a: usize, b: []const f32, ldb: usize, stride_b: usize, beta: f32, c: []f32, ldc: usize, stride_c: usize, batch_count: usize) !void {
    try gemm(f32, .{ .op_a = op_a, .op_b = op_b, .m = m, .n = n, .k = k, .alpha = alpha, .beta = beta, .a = a, .b = b, .c = c, .a_layout = matrixLayout(layout, lda), .b_layout = matrixLayout(layout, ldb), .c_layout = matrixLayout(layout, ldc), .batch_count = batch_count, .stride_a = stride_a, .stride_b = stride_b, .stride_c = stride_c });
}

pub fn dgemmStridedBatched(layout: Layout, op_a: Op, op_b: Op, m: usize, n: usize, k: usize, alpha: f64, a: []const f64, lda: usize, stride_a: usize, b: []const f64, ldb: usize, stride_b: usize, beta: f64, c: []f64, ldc: usize, stride_c: usize, batch_count: usize) !void {
    try gemm(f64, .{ .op_a = op_a, .op_b = op_b, .m = m, .n = n, .k = k, .alpha = alpha, .beta = beta, .a = a, .b = b, .c = c, .a_layout = matrixLayout(layout, lda), .b_layout = matrixLayout(layout, ldb), .c_layout = matrixLayout(layout, ldc), .batch_count = batch_count, .stride_a = stride_a, .stride_b = stride_b, .stride_c = stride_c });
}

pub fn cgemmStridedBatched(layout: Layout, op_a: Op, op_b: Op, m: usize, n: usize, k: usize, alpha: Complex32, a: []const Complex32, lda: usize, stride_a: usize, b: []const Complex32, ldb: usize, stride_b: usize, beta: Complex32, c: []Complex32, ldc: usize, stride_c: usize, batch_count: usize) !void {
    try gemm(Complex32, .{ .op_a = op_a, .op_b = op_b, .m = m, .n = n, .k = k, .alpha = alpha, .beta = beta, .a = a, .b = b, .c = c, .a_layout = matrixLayout(layout, lda), .b_layout = matrixLayout(layout, ldb), .c_layout = matrixLayout(layout, ldc), .batch_count = batch_count, .stride_a = stride_a, .stride_b = stride_b, .stride_c = stride_c });
}

pub fn zgemmStridedBatched(layout: Layout, op_a: Op, op_b: Op, m: usize, n: usize, k: usize, alpha: Complex64, a: []const Complex64, lda: usize, stride_a: usize, b: []const Complex64, ldb: usize, stride_b: usize, beta: Complex64, c: []Complex64, ldc: usize, stride_c: usize, batch_count: usize) !void {
    try gemm(Complex64, .{ .op_a = op_a, .op_b = op_b, .m = m, .n = n, .k = k, .alpha = alpha, .beta = beta, .a = a, .b = b, .c = c, .a_layout = matrixLayout(layout, lda), .b_layout = matrixLayout(layout, ldb), .c_layout = matrixLayout(layout, ldc), .batch_count = batch_count, .stride_a = stride_a, .stride_b = stride_b, .stride_c = stride_c });
}

pub fn sgemmBatched(layout: Layout, op_a: Op, op_b: Op, m: usize, n: usize, k: usize, alpha: f32, a: []const []const f32, lda: usize, b: []const []const f32, ldb: usize, beta: f32, c: []const []f32, ldc: usize) !void {
    if (a.len != b.len or a.len != c.len) return error.InvalidBatchCount;
    for (0..a.len) |batch| try sgemm(layout, op_a, op_b, m, n, k, alpha, a[batch], lda, b[batch], ldb, beta, c[batch], ldc);
}

pub fn dgemmBatched(layout: Layout, op_a: Op, op_b: Op, m: usize, n: usize, k: usize, alpha: f64, a: []const []const f64, lda: usize, b: []const []const f64, ldb: usize, beta: f64, c: []const []f64, ldc: usize) !void {
    if (a.len != b.len or a.len != c.len) return error.InvalidBatchCount;
    for (0..a.len) |batch| try dgemm(layout, op_a, op_b, m, n, k, alpha, a[batch], lda, b[batch], ldb, beta, c[batch], ldc);
}

pub fn cgemmBatched(layout: Layout, op_a: Op, op_b: Op, m: usize, n: usize, k: usize, alpha: Complex32, a: []const []const Complex32, lda: usize, b: []const []const Complex32, ldb: usize, beta: Complex32, c: []const []Complex32, ldc: usize) !void {
    if (a.len != b.len or a.len != c.len) return error.InvalidBatchCount;
    for (0..a.len) |batch| try cgemm(layout, op_a, op_b, m, n, k, alpha, a[batch], lda, b[batch], ldb, beta, c[batch], ldc);
}

pub fn zgemmBatched(layout: Layout, op_a: Op, op_b: Op, m: usize, n: usize, k: usize, alpha: Complex64, a: []const []const Complex64, lda: usize, b: []const []const Complex64, ldb: usize, beta: Complex64, c: []const []Complex64, ldc: usize) !void {
    if (a.len != b.len or a.len != c.len) return error.InvalidBatchCount;
    for (0..a.len) |batch| try zgemm(layout, op_a, op_b, m, n, k, alpha, a[batch], lda, b[batch], ldb, beta, c[batch], ldc);
}

pub fn cgemm3m(layout: Layout, op_a: Op, op_b: Op, m: usize, n: usize, k: usize, alpha: Complex32, a: []const Complex32, lda: usize, b: []const Complex32, ldb: usize, beta: Complex32, c: []Complex32, ldc: usize) !void {
    try cgemm(layout, op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

pub fn zgemm3m(layout: Layout, op_a: Op, op_b: Op, m: usize, n: usize, k: usize, alpha: Complex64, a: []const Complex64, lda: usize, b: []const Complex64, ldb: usize, beta: Complex64, c: []Complex64, ldc: usize) !void {
    try zgemm(layout, op_a, op_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

pub fn gemm(comptime T: type, desc: ScalarDesc(T)) !void {
    try validateScalarDesc(T, desc);
    const stride_a = if (desc.stride_a == 0) matrixExtent(desc.a_layout, storedRows(desc.op_a, desc.m, desc.k), storedCols(desc.op_a, desc.m, desc.k)) else desc.stride_a;
    const stride_b = if (desc.stride_b == 0) matrixExtent(desc.b_layout, storedRows(desc.op_b, desc.k, desc.n), storedCols(desc.op_b, desc.k, desc.n)) else desc.stride_b;
    const stride_c = if (desc.stride_c == 0) matrixExtent(desc.c_layout, desc.m, desc.n) else desc.stride_c;

    for (0..desc.batch_count) |batch| {
        const a_base = batch * stride_a;
        const b_base = batch * stride_b;
        const c_base = batch * stride_c;
        for (0..desc.m) |row| {
            for (0..desc.n) |col| {
                var sum = zero(T);
                for (0..desc.k) |kk| {
                    sum = add(T, sum, mul(T, loadOp(T, desc.a, desc.a_layout, desc.op_a, a_base, row, kk), loadOp(T, desc.b, desc.b_layout, desc.op_b, b_base, kk, col)));
                }
                const c_index = c_base + desc.c_layout.index(row, col);
                desc.c[c_index] = add(T, mul(T, desc.alpha, sum), mul(T, desc.beta, desc.c[c_index]));
            }
        }
    }
}

pub fn gemmEx(desc: anytype) !void {
    const Desc = @TypeOf(desc);
    if (comptime @hasField(Desc, "input")) {
        const legacy: GemmExDesc = .{
            .op_a = desc.op_a,
            .op_b = desc.op_b,
            .m = desc.m,
            .n = desc.n,
            .k = desc.k,
            .a = desc.a,
            .b = desc.b,
            .c = desc.c,
            .a_layout = .{ .offset = if (comptime @hasField(@TypeOf(desc.a_layout), "offset")) desc.a_layout.offset else 0, .row_stride = desc.a_layout.row_stride, .col_stride = desc.a_layout.col_stride },
            .b_layout = .{ .offset = if (comptime @hasField(@TypeOf(desc.b_layout), "offset")) desc.b_layout.offset else 0, .row_stride = desc.b_layout.row_stride, .col_stride = desc.b_layout.col_stride },
            .c_layout = .{ .offset = if (comptime @hasField(@TypeOf(desc.c_layout), "offset")) desc.c_layout.offset else 0, .row_stride = desc.c_layout.row_stride, .col_stride = desc.c_layout.col_stride },
            .alpha = desc.alpha,
            .beta = desc.beta,
            .input = desc.input,
            .batch_count = if (comptime @hasField(Desc, "batch_count")) desc.batch_count else 1,
            .stride_a = if (comptime @hasField(Desc, "stride_a")) desc.stride_a else 0,
            .stride_b = if (comptime @hasField(Desc, "stride_b")) desc.stride_b else 0,
            .stride_c = if (comptime @hasField(Desc, "stride_c")) desc.stride_c else 0,
        };
        return gemmExLegacy(legacy);
    }
    if (comptime @hasField(Desc, "a_type")) {
        const ex = normalizeExDesc(desc);
        return gemmExCuda(ex);
    }
    @compileError("unsupported gemmEx descriptor");
}

fn normalizeExDesc(desc: anytype) ExDesc {
    const Desc = @TypeOf(desc);
    return .{
        .op_a = desc.op_a,
        .op_b = desc.op_b,
        .m = desc.m,
        .n = desc.n,
        .k = desc.k,
        .alpha = desc.alpha,
        .a = desc.a,
        .a_type = desc.a_type,
        .lda = desc.lda,
        .b = desc.b,
        .b_type = desc.b_type,
        .ldb = desc.ldb,
        .beta = desc.beta,
        .c = desc.c,
        .c_type = desc.c_type,
        .ldc = desc.ldc,
        .compute_type = desc.compute_type,
        .layout = if (comptime @hasField(Desc, "layout")) desc.layout else .col_major,
        .batch_count = if (comptime @hasField(Desc, "batch_count")) desc.batch_count else 1,
        .stride_a = if (comptime @hasField(Desc, "stride_a")) desc.stride_a else 0,
        .stride_b = if (comptime @hasField(Desc, "stride_b")) desc.stride_b else 0,
        .stride_c = if (comptime @hasField(Desc, "stride_c")) desc.stride_c else 0,
        .algo = if (comptime @hasField(Desc, "algo")) desc.algo else .default,
        .three_m = if (comptime @hasField(Desc, "three_m")) desc.three_m else false,
    };
}

fn gemmExLegacy(desc: GemmExDesc) !void {
    if (desc.input != .f16_f32 and desc.input != .bf16_f32) return error.InvalidType;
    try validateGemmExDesc(desc);
    const stride_a = if (desc.stride_a == 0) matrixExtent(desc.a_layout, storedRows(desc.op_a, desc.m, desc.k), storedCols(desc.op_a, desc.m, desc.k)) else desc.stride_a;
    const stride_b = if (desc.stride_b == 0) matrixExtent(desc.b_layout, storedRows(desc.op_b, desc.k, desc.n), storedCols(desc.op_b, desc.k, desc.n)) else desc.stride_b;
    const stride_c = if (desc.stride_c == 0) matrixExtent(desc.c_layout, desc.m, desc.n) else desc.stride_c;

    for (0..desc.batch_count) |batch| {
        const a_base = batch * stride_a;
        const b_base = batch * stride_b;
        const c_base = batch * stride_c;
        for (0..desc.m) |row| {
            for (0..desc.n) |col| {
                var sum: f32 = 0;
                for (0..desc.k) |kk| {
                    sum += loadEx(desc.input, desc.a, desc.a_layout, desc.op_a, a_base, row, kk) * loadEx(desc.input, desc.b, desc.b_layout, desc.op_b, b_base, kk, col);
                }
                const c_index = c_base + desc.c_layout.index(row, col);
                desc.c[c_index] = desc.alpha * sum + desc.beta * desc.c[c_index];
            }
        }
    }
}

pub fn gemmStridedBatchedEx(desc: ExDesc) !void {
    try gemmExCuda(desc);
}

pub fn gemmBatchedEx(desc: PointerBatchedExDesc) !void {
    if (desc.a.len != desc.b.len or desc.a.len != desc.c.len) return error.InvalidBatchCount;
    for (0..desc.a.len) |batch| {
        const one: ExDesc = .{
            .op_a = desc.op_a,
            .op_b = desc.op_b,
            .m = desc.m,
            .n = desc.n,
            .k = desc.k,
            .alpha = desc.alpha,
            .a = desc.a[batch],
            .a_type = desc.a_type,
            .lda = desc.lda,
            .b = desc.b[batch],
            .b_type = desc.b_type,
            .ldb = desc.ldb,
            .beta = desc.beta,
            .c = desc.c[batch],
            .c_type = desc.c_type,
            .ldc = desc.ldc,
            .compute_type = desc.compute_type,
            .layout = desc.layout,
            .algo = desc.algo,
            .three_m = desc.three_m,
        };
        try gemmExCuda(one);
    }
}

pub fn cgemm3mEx(desc: ExDesc) !void {
    var three_m_desc = desc;
    three_m_desc.three_m = true;
    try gemmExCuda(three_m_desc);
}

fn gemmExCuda(desc: ExDesc) !void {
    if (desc.batch_count == 0) return error.InvalidBatchCount;
    const a_layout = matrixLayout(desc.layout, desc.lda);
    const b_layout = matrixLayout(desc.layout, desc.ldb);
    const c_layout = matrixLayout(desc.layout, desc.ldc);
    const a_extent = matrixExtent(a_layout, storedRows(desc.op_a, desc.m, desc.k), storedCols(desc.op_a, desc.m, desc.k));
    const b_extent = matrixExtent(b_layout, storedRows(desc.op_b, desc.k, desc.n), storedCols(desc.op_b, desc.k, desc.n));
    const c_extent = matrixExtent(c_layout, desc.m, desc.n);
    const stride_a = if (desc.stride_a == 0) a_extent else desc.stride_a;
    const stride_b = if (desc.stride_b == 0) b_extent else desc.stride_b;
    const stride_c = if (desc.stride_c == 0) c_extent else desc.stride_c;
    try validateByteSliceLen(desc.a, desc.a_type, a_extent, stride_a, desc.batch_count);
    try validateByteSliceLen(desc.b, desc.b_type, b_extent, stride_b, desc.batch_count);
    try validateByteSliceLen(desc.c, desc.c_type, c_extent, stride_c, desc.batch_count);

    _ = desc.compute_type;
    _ = desc.algo;
    for (0..desc.batch_count) |batch| {
        const a_base = batch * stride_a;
        const b_base = batch * stride_b;
        const c_base = batch * stride_c;
        for (0..desc.m) |row| {
            for (0..desc.n) |col| {
                var sum = Scalar.real(0);
                for (0..desc.k) |kk| {
                    const av = loadExScalar(desc.a, desc.a_type, a_layout, desc.op_a, a_base, row, kk);
                    const bv = loadExScalar(desc.b, desc.b_type, b_layout, desc.op_b, b_base, kk, col);
                    sum = scalarAdd(sum, scalarMul(av, bv));
                }
                const c_index = c_base + c_layout.index(row, col);
                const old = readScalar(desc.c, desc.c_type, c_index);
                writeScalar(desc.c, desc.c_type, c_index, scalarAdd(scalarMul(desc.alpha, sum), scalarMul(desc.beta, old)));
            }
        }
    }
}

pub fn matrixExtent(layout: MatrixLayout, rows: usize, cols: usize) usize {
    if (rows == 0 or cols == 0) return 0;
    return layout.index(rows - 1, cols - 1) + 1;
}

fn validateScalarDesc(comptime T: type, desc: ScalarDesc(T)) !void {
    if (desc.batch_count == 0) return error.InvalidBatchCount;
    const a_extent = matrixExtent(desc.a_layout, storedRows(desc.op_a, desc.m, desc.k), storedCols(desc.op_a, desc.m, desc.k));
    const b_extent = matrixExtent(desc.b_layout, storedRows(desc.op_b, desc.k, desc.n), storedCols(desc.op_b, desc.k, desc.n));
    const c_extent = matrixExtent(desc.c_layout, desc.m, desc.n);
    try validateSliceLen(desc.a.len, a_extent, desc.stride_a, desc.batch_count);
    try validateSliceLen(desc.b.len, b_extent, desc.stride_b, desc.batch_count);
    try validateSliceLen(desc.c.len, c_extent, desc.stride_c, desc.batch_count);
}

fn validateGemmExDesc(desc: GemmExDesc) !void {
    if (desc.batch_count == 0) return error.InvalidBatchCount;
    const a_extent = matrixExtent(desc.a_layout, storedRows(desc.op_a, desc.m, desc.k), storedCols(desc.op_a, desc.m, desc.k));
    const b_extent = matrixExtent(desc.b_layout, storedRows(desc.op_b, desc.k, desc.n), storedCols(desc.op_b, desc.k, desc.n));
    const c_extent = matrixExtent(desc.c_layout, desc.m, desc.n);
    try validateSliceLen(desc.a.len, a_extent, desc.stride_a, desc.batch_count);
    try validateSliceLen(desc.b.len, b_extent, desc.stride_b, desc.batch_count);
    try validateSliceLen(desc.c.len, c_extent, desc.stride_c, desc.batch_count);
}

fn validateSliceLen(len: usize, extent: usize, stride: usize, batch_count: usize) !void {
    if (extent == 0) return;
    const used_stride = if (stride == 0) extent else stride;
    if (used_stride < extent) return error.InvalidStride;
    const required = (batch_count - 1) * used_stride + extent;
    if (len < required) return error.BufferTooSmall;
}

fn checkedUsize(value: u64) usize {
    return std.math.cast(usize, value) orelse @panic("GEMM dimension does not fit usize");
}

fn loadOp(comptime T: type, data: []const T, layout: MatrixLayout, op: Op, base: usize, row: usize, col: usize) T {
    return switch (op) {
        .no_trans => data[base + layout.index(row, col)],
        .trans => data[base + layout.index(col, row)],
        .conj_trans => conj(T, data[base + layout.index(col, row)]),
    };
}

fn loadEx(input: GemmType, data: []const u16, layout: MatrixLayout, op: Op, base: usize, row: usize, col: usize) f32 {
    const index = switch (op) {
        .no_trans => layout.index(row, col),
        .trans, .conj_trans => layout.index(col, row),
    };
    return decodeInput(data[base + index], input);
}

fn loadExScalar(data: []const u8, data_type: DataType, layout: MatrixLayout, op: Op, base: usize, row: usize, col: usize) Scalar {
    return switch (op) {
        .no_trans => readScalar(data, data_type, base + layout.index(row, col)),
        .trans => readScalar(data, data_type, base + layout.index(col, row)),
        .conj_trans => scalarConj(readScalar(data, data_type, base + layout.index(col, row))),
    };
}

fn validateByteSliceLen(data: []const u8, data_type: DataType, extent: usize, stride: usize, batch_count: usize) !void {
    if (extent == 0) return;
    if (stride < extent) return error.InvalidStride;
    const required_elements = (batch_count - 1) * stride + extent;
    const required_bytes = required_elements * elementSize(data_type);
    if (data.len < required_bytes) return error.BufferTooSmall;
}

pub fn elementSize(data_type: DataType) usize {
    return switch (data_type) {
        .r_16f, .r_16bf, .r_16i, .r_16u => 2,
        .c_16f, .c_16bf, .c_16i, .c_16u => 4,
        .r_32f, .r_32i, .r_32u => 4,
        .c_32f, .c_32i, .c_32u => 8,
        .r_64f, .r_64i, .r_64u => 8,
        .c_64f, .c_64i, .c_64u => 16,
        .r_4i, .r_4u, .r_8i, .r_8u, .r_8f_e4m3, .r_8f_e5m2, .r_8f_ue8m0, .r_6f_e2m3, .r_6f_e3m2, .r_4f_e2m1 => 1,
        .c_4i, .c_4u, .c_8i, .c_8u => 2,
    };
}

pub fn isComplexType(data_type: DataType) bool {
    return switch (data_type) {
        .c_16f, .c_16bf, .c_32f, .c_64f, .c_4i, .c_4u, .c_8i, .c_8u, .c_16i, .c_16u, .c_32i, .c_32u, .c_64i, .c_64u => true,
        else => false,
    };
}

pub fn readScalar(data: []const u8, data_type: DataType, element_index: usize) Scalar {
    const offset = element_index * elementSize(data_type);
    return switch (data_type) {
        .r_16f => Scalar.real(f16BitsToF32(readU16(data, offset))),
        .c_16f => Scalar.complex(f16BitsToF32(readU16(data, offset)), f16BitsToF32(readU16(data, offset + 2))),
        .r_16bf => Scalar.real(bf16BitsToF32(readU16(data, offset))),
        .c_16bf => Scalar.complex(bf16BitsToF32(readU16(data, offset)), bf16BitsToF32(readU16(data, offset + 2))),
        .r_32f => Scalar.real(readF32(data, offset)),
        .c_32f => Scalar.complex(readF32(data, offset), readF32(data, offset + 4)),
        .r_64f => Scalar.real(readF64(data, offset)),
        .c_64f => Scalar.complex(readF64(data, offset), readF64(data, offset + 8)),
        .r_4i => Scalar.real(@floatFromInt(signExtend4(data[offset] & 0x0f))),
        .c_4i => Scalar.complex(@floatFromInt(signExtend4(data[offset] & 0x0f)), @floatFromInt(signExtend4(data[offset + 1] & 0x0f))),
        .r_4u => Scalar.real(@floatFromInt(data[offset] & 0x0f)),
        .c_4u => Scalar.complex(@floatFromInt(data[offset] & 0x0f), @floatFromInt(data[offset + 1] & 0x0f)),
        .r_8i => Scalar.real(@floatFromInt(@as(i8, @bitCast(data[offset])))),
        .c_8i => Scalar.complex(@floatFromInt(@as(i8, @bitCast(data[offset]))), @floatFromInt(@as(i8, @bitCast(data[offset + 1])))),
        .r_8u => Scalar.real(@floatFromInt(data[offset])),
        .c_8u => Scalar.complex(@floatFromInt(data[offset]), @floatFromInt(data[offset + 1])),
        .r_16i => Scalar.real(@floatFromInt(@as(i16, @bitCast(readU16(data, offset))))),
        .c_16i => Scalar.complex(@floatFromInt(@as(i16, @bitCast(readU16(data, offset)))), @floatFromInt(@as(i16, @bitCast(readU16(data, offset + 2))))),
        .r_16u => Scalar.real(@floatFromInt(readU16(data, offset))),
        .c_16u => Scalar.complex(@floatFromInt(readU16(data, offset)), @floatFromInt(readU16(data, offset + 2))),
        .r_32i => Scalar.real(@floatFromInt(@as(i32, @bitCast(readU32(data, offset))))),
        .c_32i => Scalar.complex(@floatFromInt(@as(i32, @bitCast(readU32(data, offset)))), @floatFromInt(@as(i32, @bitCast(readU32(data, offset + 4))))),
        .r_32u => Scalar.real(@floatFromInt(readU32(data, offset))),
        .c_32u => Scalar.complex(@floatFromInt(readU32(data, offset)), @floatFromInt(readU32(data, offset + 4))),
        .r_64i => Scalar.real(@floatFromInt(@as(i64, @bitCast(readU64(data, offset))))),
        .c_64i => Scalar.complex(@floatFromInt(@as(i64, @bitCast(readU64(data, offset)))), @floatFromInt(@as(i64, @bitCast(readU64(data, offset + 8))))),
        .r_64u => Scalar.real(@floatFromInt(readU64(data, offset))),
        .c_64u => Scalar.complex(@floatFromInt(readU64(data, offset)), @floatFromInt(readU64(data, offset + 8))),
        .r_8f_e4m3 => Scalar.real(fp8E4M3ToF32(data[offset])),
        .r_8f_e5m2 => Scalar.real(fp8E5M2ToF32(data[offset])),
        .r_8f_ue8m0 => Scalar.real(std.math.exp2(@as(f32, @floatFromInt(data[offset])) - 127.0)),
        .r_6f_e2m3 => Scalar.real(tinyFloatToF32(data[offset] & 0x3f, 2, 3, 1)),
        .r_6f_e3m2 => Scalar.real(tinyFloatToF32(data[offset] & 0x3f, 3, 2, 3)),
        .r_4f_e2m1 => Scalar.real(tinyFloatToF32(data[offset] & 0x0f, 2, 1, 1)),
    };
}

pub fn writeScalar(data: []u8, data_type: DataType, element_index: usize, value: Scalar) void {
    const offset = element_index * elementSize(data_type);
    switch (data_type) {
        .r_16f => writeU16(data, offset, f32ToF16Bits(@floatCast(value.re))),
        .c_16f => {
            writeU16(data, offset, f32ToF16Bits(@floatCast(value.re)));
            writeU16(data, offset + 2, f32ToF16Bits(@floatCast(value.im)));
        },
        .r_16bf => writeU16(data, offset, f32ToBf16Bits(@floatCast(value.re))),
        .c_16bf => {
            writeU16(data, offset, f32ToBf16Bits(@floatCast(value.re)));
            writeU16(data, offset + 2, f32ToBf16Bits(@floatCast(value.im)));
        },
        .r_32f => writeF32(data, offset, @floatCast(value.re)),
        .c_32f => {
            writeF32(data, offset, @floatCast(value.re));
            writeF32(data, offset + 4, @floatCast(value.im));
        },
        .r_64f => writeF64(data, offset, value.re),
        .c_64f => {
            writeF64(data, offset, value.re);
            writeF64(data, offset + 8, value.im);
        },
        .r_4i, .r_4u, .r_8i, .r_8u, .r_8f_e4m3, .r_8f_e5m2, .r_8f_ue8m0, .r_6f_e2m3, .r_6f_e3m2, .r_4f_e2m1 => data[offset] = quantizeByte(data_type, value.re),
        .c_4i, .c_4u, .c_8i, .c_8u => {
            data[offset] = quantizeByte(componentType(data_type), value.re);
            data[offset + 1] = quantizeByte(componentType(data_type), value.im);
        },
        .r_16i => writeU16(data, offset, @bitCast(clampInt(i16, value.re))),
        .c_16i => {
            writeU16(data, offset, @bitCast(clampInt(i16, value.re)));
            writeU16(data, offset + 2, @bitCast(clampInt(i16, value.im)));
        },
        .r_16u => writeU16(data, offset, clampUnsigned(u16, value.re)),
        .c_16u => {
            writeU16(data, offset, clampUnsigned(u16, value.re));
            writeU16(data, offset + 2, clampUnsigned(u16, value.im));
        },
        .r_32i => writeU32(data, offset, @bitCast(clampInt(i32, value.re))),
        .c_32i => {
            writeU32(data, offset, @bitCast(clampInt(i32, value.re)));
            writeU32(data, offset + 4, @bitCast(clampInt(i32, value.im)));
        },
        .r_32u => writeU32(data, offset, clampUnsigned(u32, value.re)),
        .c_32u => {
            writeU32(data, offset, clampUnsigned(u32, value.re));
            writeU32(data, offset + 4, clampUnsigned(u32, value.im));
        },
        .r_64i => writeU64(data, offset, @bitCast(clampInt(i64, value.re))),
        .c_64i => {
            writeU64(data, offset, @bitCast(clampInt(i64, value.re)));
            writeU64(data, offset + 8, @bitCast(clampInt(i64, value.im)));
        },
        .r_64u => writeU64(data, offset, clampUnsigned(u64, value.re)),
        .c_64u => {
            writeU64(data, offset, clampUnsigned(u64, value.re));
            writeU64(data, offset + 8, clampUnsigned(u64, value.im));
        },
    }
}

pub fn encodeInput(value: f32, input: GemmType) u16 {
    return switch (input) {
        .f16_f32 => f32ToF16Bits(value),
        .bf16_f32 => f32ToBf16Bits(value),
        else => unreachable,
    };
}

pub fn decodeInput(bits: u16, input: GemmType) f32 {
    return switch (input) {
        .f16_f32 => f16BitsToF32(bits),
        .bf16_f32 => bf16BitsToF32(bits),
        else => unreachable,
    };
}

pub fn f32ToF16Bits(value: f32) u16 {
    const half: f16 = @floatCast(value);
    return @bitCast(half);
}

pub fn f16BitsToF32(bits: u16) f32 {
    const half: f16 = @bitCast(bits);
    return @floatCast(half);
}

pub fn f32ToBf16Bits(value: f32) u16 {
    const raw: u32 = @bitCast(value);
    const rounded = raw + 0x7fff + ((raw >> 16) & 1);
    return @truncate(rounded >> 16);
}

pub fn bf16BitsToF32(bits: u16) f32 {
    const raw = @as(u32, bits) << 16;
    return @bitCast(raw);
}

pub fn fp8E4M3ToF32(bits: u8) f32 {
    return tinyFloatToF32(bits, 4, 3, 7);
}

pub fn fp8E5M2ToF32(bits: u8) f32 {
    return tinyFloatToF32(bits, 5, 2, 15);
}

fn tinyFloatToF32(bits: u8, exp_bits: u5, mant_bits: u5, bias: i32) f32 {
    const sign_shift: u3 = @intCast(exp_bits + mant_bits);
    const sign = (bits >> sign_shift) & 1;
    const exp_shift: u3 = @intCast(exp_bits);
    const mant_shift: u3 = @intCast(mant_bits);
    const exp_mask: u8 = (@as(u8, 1) << exp_shift) - 1;
    const mant_mask: u8 = (@as(u8, 1) << mant_shift) - 1;
    const exp_raw = (bits >> mant_shift) & exp_mask;
    const mant_raw = bits & mant_mask;
    const sign_scale: f32 = if (sign == 0) 1.0 else -1.0;
    if (exp_raw == 0 and mant_raw == 0) return if (sign == 0) 0.0 else -0.0;
    const mant_den = @as(f32, @floatFromInt(@as(u32, 1) << mant_bits));
    if (exp_raw == 0) {
        const exponent = 1 - bias;
        return sign_scale * (@as(f32, @floatFromInt(mant_raw)) / mant_den) * std.math.exp2(@as(f32, @floatFromInt(exponent)));
    }
    const exponent = @as(i32, @intCast(exp_raw)) - bias;
    const mant = 1.0 + @as(f32, @floatFromInt(mant_raw)) / mant_den;
    return sign_scale * mant * std.math.exp2(@as(f32, @floatFromInt(exponent)));
}

fn scalarAdd(a: Scalar, b: Scalar) Scalar {
    return .{ .re = a.re + b.re, .im = a.im + b.im };
}

fn scalarMul(a: Scalar, b: Scalar) Scalar {
    return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
}

fn scalarConj(value: Scalar) Scalar {
    return .{ .re = value.re, .im = -value.im };
}

fn componentType(data_type: DataType) DataType {
    return switch (data_type) {
        .c_4i => .r_4i,
        .c_4u => .r_4u,
        .c_8i => .r_8i,
        .c_8u => .r_8u,
        else => data_type,
    };
}

fn quantizeByte(data_type: DataType, value: f64) u8 {
    return switch (data_type) {
        .r_4i => @as(u8, @bitCast(@as(i8, @intCast(clampInt(i4, value))))) & 0x0f,
        .r_4u => @intCast(@min(@as(u64, 15), @as(u64, @intFromFloat(@max(0.0, @round(value)))))),
        .r_8i => @bitCast(clampInt(i8, value)),
        .r_8u => clampUnsigned(u8, value),
        .r_8f_e4m3, .r_8f_e5m2, .r_8f_ue8m0, .r_6f_e2m3, .r_6f_e3m2, .r_4f_e2m1 => clampUnsigned(u8, value),
        else => clampUnsigned(u8, value),
    };
}

fn signExtend4(value: u8) i8 {
    const low = value & 0x0f;
    return if ((low & 0x08) == 0) @intCast(low) else @as(i8, @intCast(low)) - 16;
}

fn clampInt(comptime T: type, value: f64) T {
    const info = @typeInfo(T).int;
    const min_value = if (info.signedness == .signed) -(@as(f64, @floatFromInt(@as(i128, 1) << @intCast(info.bits - 1)))) else 0;
    const max_value = if (info.signedness == .signed)
        @as(f64, @floatFromInt((@as(i128, 1) << @intCast(info.bits - 1)) - 1))
    else
        @as(f64, @floatFromInt((@as(u128, 1) << @intCast(info.bits)) - 1));
    return @intFromFloat(@min(max_value, @max(min_value, @round(value))));
}

fn clampUnsigned(comptime T: type, value: f64) T {
    const info = @typeInfo(T).int;
    const max_value = @as(f64, @floatFromInt((@as(u128, 1) << @intCast(info.bits)) - 1));
    return @intFromFloat(@min(max_value, @max(0.0, @round(value))));
}

fn readU16(data: []const u8, offset: usize) u16 {
    return std.mem.readInt(u16, data[offset..][0..2], .little);
}

fn readU32(data: []const u8, offset: usize) u32 {
    return std.mem.readInt(u32, data[offset..][0..4], .little);
}

fn readU64(data: []const u8, offset: usize) u64 {
    return std.mem.readInt(u64, data[offset..][0..8], .little);
}

fn readF32(data: []const u8, offset: usize) f32 {
    return @bitCast(readU32(data, offset));
}

fn readF64(data: []const u8, offset: usize) f64 {
    return @bitCast(readU64(data, offset));
}

fn writeU16(data: []u8, offset: usize, value: u16) void {
    std.mem.writeInt(u16, data[offset..][0..2], value, .little);
}

fn writeU32(data: []u8, offset: usize, value: u32) void {
    std.mem.writeInt(u32, data[offset..][0..4], value, .little);
}

fn writeU64(data: []u8, offset: usize, value: u64) void {
    std.mem.writeInt(u64, data[offset..][0..8], value, .little);
}

fn writeF32(data: []u8, offset: usize, value: f32) void {
    writeU32(data, offset, @bitCast(value));
}

fn writeF64(data: []u8, offset: usize, value: f64) void {
    writeU64(data, offset, @bitCast(value));
}

fn zero(comptime T: type) T {
    if (T == f32 or T == f64) return 0;
    if (T == Complex32) return .{ .re = 0, .im = 0 };
    if (T == Complex64) return .{ .re = 0, .im = 0 };
    @compileError("unsupported GEMM scalar type");
}

fn add(comptime T: type, a: T, b: T) T {
    if (T == f32 or T == f64) return a + b;
    if (T == Complex32 or T == Complex64) return .{ .re = a.re + b.re, .im = a.im + b.im };
    @compileError("unsupported GEMM scalar type");
}

fn mul(comptime T: type, a: T, b: T) T {
    if (T == f32 or T == f64) return a * b;
    if (T == Complex32 or T == Complex64) return .{ .re = a.re * b.re - a.im * b.im, .im = a.re * b.im + a.im * b.re };
    @compileError("unsupported GEMM scalar type");
}

fn conj(comptime T: type, value: T) T {
    if (T == f32 or T == f64) return value;
    if (T == Complex32 or T == Complex64) return .{ .re = value.re, .im = -value.im };
    @compileError("unsupported GEMM scalar type");
}
