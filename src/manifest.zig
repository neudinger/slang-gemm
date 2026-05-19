const std = @import("std");
const gemm = @import("gemm");

pub const Api = enum {
    regular,
    regular_64,
    batched,
    strided_batched,
    complex_3m,
    gemm_ex,
    gemm_batched_ex,
    gemm_strided_batched_ex,

    pub fn label(self: Api) []const u8 {
        return switch (self) {
            .regular => "regular",
            .regular_64 => "regular_64",
            .batched => "batched",
            .strided_batched => "strided_batched",
            .complex_3m => "complex_3m",
            .gemm_ex => "gemm_ex",
            .gemm_batched_ex => "gemm_batched_ex",
            .gemm_strided_batched_ex => "gemm_strided_batched_ex",
        };
    }
};

pub const Status = enum {
    ok,
    cublas_unsupported,
    shader_unavailable,
    validation_failed,
    perf_regression,
    invalid,

    pub fn label(self: Status) []const u8 {
        return switch (self) {
            .ok => "ok",
            .cublas_unsupported => "cuBLAS_unsupported",
            .shader_unavailable => "shader_unavailable",
            .validation_failed => "validation_failed",
            .perf_regression => "perf_regression",
            .invalid => "invalid",
        };
    }
};

pub const ShaderFamily = enum {
    none,
    f16_scalar,
    f16_bf16_tensor,
    f32_tf32,
    f64,
    complex_f32,
    complex_f64,
    int8_int32,
    low_precision_f32,

    pub fn label(self: ShaderFamily) []const u8 {
        return switch (self) {
            .none => "none",
            .f16_scalar => "gemm_family_f16_scalar",
            .f16_bf16_tensor => "gemm_family_f16_bf16_tensor",
            .f32_tf32 => "gemm_family_f32_tf32",
            .f64 => "gemm_family_f64",
            .complex_f32 => "gemm_family_complex_f32",
            .complex_f64 => "gemm_family_complex_f64",
            .int8_int32 => "gemm_family_int8_int32",
            .low_precision_f32 => "gemm_family_low_precision_packed",
        };
    }
};

pub const Shape = struct {
    m: usize,
    n: usize,
    k: usize,
};

pub const Case = struct {
    api: Api,
    op_a: gemm.Op,
    op_b: gemm.Op,
    a_type: gemm.DataType,
    b_type: gemm.DataType,
    c_type: gemm.DataType,
    compute_type: gemm.ComputeType,
    shape: Shape,
    batched: bool = false,

    pub fn family(self: Case) []const u8 {
        if (self.api == .regular and self.a_type == .r_32f and self.b_type == .r_32f and self.c_type == .r_32f) return "classic_sgemm";
        if (self.api == .regular and self.a_type == .r_64f and self.b_type == .r_64f and self.c_type == .r_64f) return "classic_dgemm";
        if (self.api == .regular and self.a_type == .c_32f and self.b_type == .c_32f and self.c_type == .c_32f) return "classic_cgemm";
        if (self.api == .regular and self.a_type == .c_64f and self.b_type == .c_64f and self.c_type == .c_64f) return "classic_zgemm";
        if (isTensorEx(self)) return "tensor_ex";
        if (isInt8Ex(self)) return "int8_ex";
        return "generic";
    }

    pub fn cublasSupported(self: Case) bool {
        return switch (self.api) {
            .regular, .regular_64, .batched, .strided_batched => classicType(self.a_type, self.b_type, self.c_type) and classicComputeType(self.a_type, self.compute_type),
            .complex_3m => ((self.a_type == .c_32f and self.b_type == .c_32f and self.c_type == .c_32f and self.compute_type == .f32) or (self.a_type == .c_64f and self.b_type == .c_64f and self.c_type == .c_64f and self.compute_type == .f64)),
            .gemm_ex, .gemm_batched_ex, .gemm_strided_batched_ex => exTypeSupported(self),
        };
    }

    pub fn shaderFamily(self: Case) ShaderFamily {
        if (!self.cublasSupported()) return .none;
        return switch (self.api) {
            .regular, .regular_64, .batched, .strided_batched => classicShaderFamily(self.a_type, self.b_type, self.c_type),
            .complex_3m => if (self.a_type == .c_32f) .complex_f32 else .complex_f64,
            .gemm_ex, .gemm_batched_ex, .gemm_strided_batched_ex => exShaderFamily(self),
        };
    }

    pub fn shaderImplemented(self: Case) bool {
        return self.shaderFamily() != .none;
    }

    pub fn dispatchAvailable(self: Case) bool {
        if (self.api == .regular or self.api == .regular_64 or self.api == .batched or self.api == .strided_batched) {
            return (self.a_type == .r_32f and self.b_type == .r_32f and self.c_type == .r_32f and self.compute_type == .f32) or
                (self.a_type == .r_64f and self.b_type == .r_64f and self.c_type == .r_64f and self.compute_type == .f64) or
                (self.a_type == .c_32f and self.b_type == .c_32f and self.c_type == .c_32f and self.compute_type == .f32) or
                (self.a_type == .c_64f and self.b_type == .c_64f and self.c_type == .c_64f and self.compute_type == .f64);
        }
        if (self.api == .complex_3m) {
            return (self.a_type == .c_32f and self.b_type == .c_32f and self.c_type == .c_32f and self.compute_type == .f32) or
                (self.a_type == .c_64f and self.b_type == .c_64f and self.c_type == .c_64f and self.compute_type == .f64);
        }
        if (self.api == .gemm_ex or self.api == .gemm_batched_ex or self.api == .gemm_strided_batched_ex) {
            if (self.a_type == .r_32f and self.b_type == .r_32f and self.c_type == .r_32f and (self.compute_type == .f32 or self.compute_type == .f32_pedantic or self.compute_type == .f32_fast_tf32)) return true;
            return (self.a_type == .r_16f and self.b_type == .r_16f and self.c_type == .r_32f and (self.compute_type == .f32 or self.compute_type == .f32_fast_16f)) or
                (self.a_type == .r_16f and self.b_type == .r_16f and self.c_type == .r_16f and (self.compute_type == .f16 or self.compute_type == .f32 or self.compute_type == .f32_fast_16f)) or
                (self.a_type == .r_16bf and self.b_type == .r_16bf and self.c_type == .r_32f and (self.compute_type == .f32 or self.compute_type == .f32_fast_16bf)) or
                (self.a_type == .r_8i and self.b_type == .r_8i and self.c_type == .r_32i and self.compute_type == .i32);
        }
        return false;
    }

    pub fn shaderAvailable(self: Case) bool {
        return self.dispatchAvailable();
    }

    pub fn hardwareAccelerated(self: Case) bool {
        // Performance-gate only rows that are routed to actual optimized
        // tensor-core shaders today. TF32, FP64, complex, non-N/N, and scalar
        // fallback rows are correctness/reporting paths until dedicated
        // optimized kernels are wired.
        if ((self.api == .gemm_ex or self.api == .gemm_batched_ex or self.api == .gemm_strided_batched_ex) and
            self.c_type == .r_32f and
            self.shape.k % 16 == 0)
        {
            if (self.a_type == .r_16f and self.b_type == .r_16f and
                (self.compute_type == .f32 or self.compute_type == .f32_fast_16f))
            {
                return self.shape.m % 128 == 0 and self.shape.n % 256 == 0;
            }
            if (self.a_type == .r_16bf and self.b_type == .r_16bf and
                (self.compute_type == .f32 or self.compute_type == .f32_fast_16bf))
            {
                return self.op_a == .no_trans and self.op_b == .no_trans and self.shape.m % 64 == 0 and self.shape.n % 128 == 0;
            }
        }
        if ((self.api == .gemm_ex or self.api == .gemm_batched_ex or self.api == .gemm_strided_batched_ex) and
            self.op_a == .no_trans and
            self.op_b == .no_trans and
            self.a_type == .r_8i and
            self.b_type == .r_8i and
            self.c_type == .r_32i and
            self.compute_type == .i32)
        {
            return self.shape.m % 128 == 0 and self.shape.n % 256 == 0 and self.shape.k % 32 == 0;
        }
        return false;
    }

    pub fn tileCompatible(self: Case) bool {
        if (!self.cublasSupported()) return false;
        if (self.a_type == .r_16f and self.b_type == .r_16f and self.c_type == .r_32f) {
            return self.shape.m % 128 == 0 and self.shape.n % 256 == 0 and self.shape.k % 16 == 0;
        }
        if (self.a_type == .r_16bf and self.b_type == .r_16bf and self.c_type == .r_32f) {
            return self.shape.m % 64 == 0 and self.shape.n % 128 == 0 and self.shape.k % 16 == 0;
        }
        if (self.a_type == .r_8i and self.b_type == .r_8i and self.c_type == .r_32i) {
            return self.shape.m % 128 == 0 and self.shape.n % 256 == 0 and self.shape.k % 32 == 0;
        }
        if (self.a_type == .r_32f or self.a_type == .c_32f) {
            return self.shape.m % 64 == 0 and self.shape.n % 64 == 0 and self.shape.k % 16 == 0;
        }
        if (self.a_type == .r_64f or self.a_type == .c_64f) {
            return self.shape.m % 32 == 0 and self.shape.n % 32 == 0 and self.shape.k % 16 == 0;
        }
        return self.shape.m % 16 == 0 and self.shape.n % 16 == 0 and self.shape.k % 16 == 0;
    }

    pub fn baseStatus(self: Case) Status {
        if (!self.cublasSupported()) return .cublas_unsupported;
        if (!self.dispatchAvailable()) return .shader_unavailable;
        return .ok;
    }
};

pub const Options = struct {
    manifest_all: bool = false,
    ops_all: bool = false,
    types_all: bool = false,
    suite_all: bool = false,
};

pub const apis = [_]Api{
    .regular,
    .regular_64,
    .batched,
    .strided_batched,
    .complex_3m,
    .gemm_ex,
    .gemm_batched_ex,
    .gemm_strided_batched_ex,
};

pub const ops = [_]gemm.Op{ .no_trans, .trans, .conj_trans };

pub const data_types = [_]gemm.DataType{
    .r_16f,
    .c_16f,
    .r_16bf,
    .c_16bf,
    .r_32f,
    .c_32f,
    .r_64f,
    .c_64f,
    .r_4i,
    .c_4i,
    .r_4u,
    .c_4u,
    .r_8i,
    .c_8i,
    .r_8u,
    .c_8u,
    .r_16i,
    .c_16i,
    .r_16u,
    .c_16u,
    .r_32i,
    .c_32i,
    .r_32u,
    .c_32u,
    .r_64i,
    .c_64i,
    .r_64u,
    .c_64u,
    .r_8f_e4m3,
    .r_8f_e5m2,
    .r_8f_ue8m0,
    .r_6f_e2m3,
    .r_6f_e3m2,
    .r_4f_e2m1,
};

pub const compute_types = [_]gemm.ComputeType{
    .f16,
    .f16_pedantic,
    .f32,
    .f32_pedantic,
    .f32_fast_16f,
    .f32_fast_16bf,
    .f32_fast_tf32,
    .f32_emulated_16bfx9,
    .f64,
    .f64_pedantic,
    .f64_emulated_fixedpoint,
    .i32,
    .i32_pedantic,
};

pub fn opLabel(op: gemm.Op) []const u8 {
    return switch (op) {
        .no_trans => "N",
        .trans => "T",
        .conj_trans => "C",
    };
}

pub fn dataTypeLabel(value: gemm.DataType) []const u8 {
    return switch (value) {
        .r_16f => "CUDA_R_16F",
        .c_16f => "CUDA_C_16F",
        .r_16bf => "CUDA_R_16BF",
        .c_16bf => "CUDA_C_16BF",
        .r_32f => "CUDA_R_32F",
        .c_32f => "CUDA_C_32F",
        .r_64f => "CUDA_R_64F",
        .c_64f => "CUDA_C_64F",
        .r_4i => "CUDA_R_4I",
        .c_4i => "CUDA_C_4I",
        .r_4u => "CUDA_R_4U",
        .c_4u => "CUDA_C_4U",
        .r_8i => "CUDA_R_8I",
        .c_8i => "CUDA_C_8I",
        .r_8u => "CUDA_R_8U",
        .c_8u => "CUDA_C_8U",
        .r_16i => "CUDA_R_16I",
        .c_16i => "CUDA_C_16I",
        .r_16u => "CUDA_R_16U",
        .c_16u => "CUDA_C_16U",
        .r_32i => "CUDA_R_32I",
        .c_32i => "CUDA_C_32I",
        .r_32u => "CUDA_R_32U",
        .c_32u => "CUDA_C_32U",
        .r_64i => "CUDA_R_64I",
        .c_64i => "CUDA_C_64I",
        .r_64u => "CUDA_R_64U",
        .c_64u => "CUDA_C_64U",
        .r_8f_e4m3 => "CUDA_R_8F_E4M3",
        .r_8f_e5m2 => "CUDA_R_8F_E5M2",
        .r_8f_ue8m0 => "CUDA_R_8F_UE8M0",
        .r_6f_e2m3 => "CUDA_R_6F_E2M3",
        .r_6f_e3m2 => "CUDA_R_6F_E3M2",
        .r_4f_e2m1 => "CUDA_R_4F_E2M1",
    };
}

pub fn computeTypeLabel(value: gemm.ComputeType) []const u8 {
    return switch (value) {
        .f16 => "CUBLAS_COMPUTE_16F",
        .f16_pedantic => "CUBLAS_COMPUTE_16F_PEDANTIC",
        .f32 => "CUBLAS_COMPUTE_32F",
        .f32_pedantic => "CUBLAS_COMPUTE_32F_PEDANTIC",
        .f32_fast_16f => "CUBLAS_COMPUTE_32F_FAST_16F",
        .f32_fast_16bf => "CUBLAS_COMPUTE_32F_FAST_16BF",
        .f32_fast_tf32 => "CUBLAS_COMPUTE_32F_FAST_TF32",
        .f32_emulated_16bfx9 => "CUBLAS_COMPUTE_32F_EMULATED_16BFX9",
        .f64 => "CUBLAS_COMPUTE_64F",
        .f64_pedantic => "CUBLAS_COMPUTE_64F_PEDANTIC",
        .f64_emulated_fixedpoint => "CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT",
        .i32 => "CUBLAS_COMPUTE_32I",
        .i32_pedantic => "CUBLAS_COMPUTE_32I_PEDANTIC",
    };
}

pub fn defaultShapes() []const Shape {
    return &[_]Shape{
        .{ .m = 128, .n = 128, .k = 128 },
        .{ .m = 256, .n = 256, .k = 256 },
        .{ .m = 512, .n = 512, .k = 512 },
        .{ .m = 1024, .n = 1024, .k = 1024 },
        .{ .m = 2048, .n = 2048, .k = 2048 },
        .{ .m = 4096, .n = 4096, .k = 4096 },
        .{ .m = 256, .n = 512, .k = 128 },
        .{ .m = 512, .n = 256, .k = 1024 },
        .{ .m = 1024, .n = 4096, .k = 512 },
        .{ .m = 4096, .n = 1024, .k = 2048 },
    };
}

pub fn defaultRunnableCases(shape: Shape) []const Case {
    return &[_]Case{
        .{ .api = .regular, .op_a = .no_trans, .op_b = .no_trans, .a_type = .r_32f, .b_type = .r_32f, .c_type = .r_32f, .compute_type = .f32, .shape = shape },
    };
}

pub fn compactAllCases(shape: Shape) []const Case {
    return &[_]Case{
        .{ .api = .regular, .op_a = .no_trans, .op_b = .no_trans, .a_type = .r_32f, .b_type = .r_32f, .c_type = .r_32f, .compute_type = .f32, .shape = shape },
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
}

fn classicType(a_type: gemm.DataType, b_type: gemm.DataType, c_type: gemm.DataType) bool {
    return (a_type == .r_32f and b_type == .r_32f and c_type == .r_32f) or
        (a_type == .r_64f and b_type == .r_64f and c_type == .r_64f) or
        (a_type == .c_32f and b_type == .c_32f and c_type == .c_32f) or
        (a_type == .c_64f and b_type == .c_64f and c_type == .c_64f);
}

fn classicComputeType(data_type: gemm.DataType, compute_type: gemm.ComputeType) bool {
    return switch (data_type) {
        .r_32f, .c_32f => compute_type == .f32,
        .r_64f, .c_64f => compute_type == .f64,
        else => false,
    };
}

fn classicShaderFamily(a_type: gemm.DataType, b_type: gemm.DataType, c_type: gemm.DataType) ShaderFamily {
    if (a_type == .r_32f and b_type == .r_32f and c_type == .r_32f) return .f32_tf32;
    if (a_type == .r_64f and b_type == .r_64f and c_type == .r_64f) return .f64;
    if (a_type == .c_32f and b_type == .c_32f and c_type == .c_32f) return .complex_f32;
    if (a_type == .c_64f and b_type == .c_64f and c_type == .c_64f) return .complex_f64;
    return .none;
}

fn exTypeSupported(case: Case) bool {
    if (case.a_type == .r_32f and case.b_type == .r_32f and case.c_type == .r_32f and (case.compute_type == .f32 or case.compute_type == .f32_fast_tf32 or case.compute_type == .f32_pedantic)) return true;
    if (case.a_type == .r_16f and case.b_type == .r_16f and (case.c_type == .r_16f or case.c_type == .r_32f) and (case.compute_type == .f16 or case.compute_type == .f32 or case.compute_type == .f32_fast_16f)) return true;
    if (case.a_type == .r_16bf and case.b_type == .r_16bf and case.c_type == .r_32f and (case.compute_type == .f32 or case.compute_type == .f32_fast_16bf)) return true;
    if (case.a_type == .r_8i and case.b_type == .r_8i and case.c_type == .r_32i and (case.compute_type == .i32 or case.compute_type == .i32_pedantic)) return true;
    return false;
}

fn exShaderFamily(case: Case) ShaderFamily {
    if (!exTypeSupported(case)) return .none;
    if (case.a_type == .r_32f and case.b_type == .r_32f and case.c_type == .r_32f) return .f32_tf32;
    if (case.a_type == .r_16f and case.b_type == .r_16f and case.c_type == .r_16f) return .f16_scalar;
    if (case.a_type == .r_16f and case.b_type == .r_16f and case.c_type == .r_32f) return .f16_bf16_tensor;
    if (case.a_type == .r_16bf and case.b_type == .r_16bf and case.c_type == .r_32f) return .f16_bf16_tensor;
    if (case.a_type == .r_8i and case.b_type == .r_8i and case.c_type == .r_32i) return .int8_int32;
    if (isLowPrecisionFloat(case.a_type) and isLowPrecisionFloat(case.b_type) and case.c_type == .r_32f) return .low_precision_f32;
    return .none;
}

fn isLowPrecisionFloat(data_type: gemm.DataType) bool {
    return switch (data_type) {
        .r_16f, .r_16bf, .r_8f_e4m3, .r_8f_e5m2, .r_8f_ue8m0, .r_6f_e2m3, .r_6f_e3m2, .r_4f_e2m1 => true,
        else => false,
    };
}

fn isTensorEx(case: Case) bool {
    return switch (case.api) {
        .gemm_ex, .gemm_batched_ex, .gemm_strided_batched_ex => (case.a_type == .r_16f and case.b_type == .r_16f) or (case.a_type == .r_16bf and case.b_type == .r_16bf) or (case.a_type == .r_32f and case.b_type == .r_32f and case.compute_type == .f32_fast_tf32),
        else => false,
    };
}

fn isInt8Ex(case: Case) bool {
    return switch (case.api) {
        .gemm_ex, .gemm_batched_ex, .gemm_strided_batched_ex => case.a_type == .r_8i and case.b_type == .r_8i and case.compute_type == .i32,
        else => false,
    };
}
