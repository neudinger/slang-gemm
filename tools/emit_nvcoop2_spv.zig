const std = @import("std");

const Mode = enum {
    bf16,
    bf16_wide,
    f16,
    f16_wide,
    f16_square,

    fn importName(self: Mode) []const u8 {
        return switch (self) {
            .bf16, .bf16_wide => "gemm_nvcoop2_bf16_spv",
            .f16 => "gemm_nvcoop2_generated_f16_spv",
            .f16_wide => "gemm_nvcoop2_wide_f16_spv",
            .f16_square => "gemm_nvcoop2_square_f16_spv",
        };
    }

    fn componentLabel(self: Mode) []const u8 {
        return switch (self) {
            .bf16, .bf16_wide => "bf16",
            .f16, .f16_wide, .f16_square => "f16",
        };
    }

    fn isF16(self: Mode) bool {
        return switch (self) {
            .f16, .f16_wide, .f16_square => true,
            .bf16, .bf16_wide => false,
        };
    }

    fn tileM(self: Mode) u32 {
        return switch (self) {
            .bf16, .bf16_wide, .f16, .f16_wide => 64,
            .f16_square => 128,
        };
    }

    fn tileN(self: Mode) u32 {
        return switch (self) {
            .bf16, .f16 => 64,
            .bf16_wide => 128,
            .f16_wide, .f16_square => 128,
        };
    }
};

fn modeFromArg(arg: []const u8) !Mode {
    if (std.mem.eql(u8, arg, "--mode=bf16")) return .bf16;
    if (std.mem.eql(u8, arg, "--mode=bf16-wide")) return .bf16_wide;
    if (std.mem.eql(u8, arg, "--mode=f16")) return .f16;
    if (std.mem.eql(u8, arg, "--mode=f16-wide")) return .f16_wide;
    if (std.mem.eql(u8, arg, "--mode=f16-square")) return .f16_square;
    return error.InvalidMode;
}

fn constForDim(ids: Ids, value: u32) u32 {
    return switch (value) {
        64 => ids.u32_64,
        128 => ids.u32_128,
        else => unreachable,
    };
}

const OpName: u16 = 5;
const OpMemberName: u16 = 6;
const OpExtension: u16 = 10;
const OpMemoryModel: u16 = 14;
const OpEntryPoint: u16 = 15;
const OpExecutionMode: u16 = 16;
const OpCapability: u16 = 17;
const OpTypeVoid: u16 = 19;
const OpTypeBool: u16 = 20;
const OpTypeInt: u16 = 21;
const OpTypeFloat: u16 = 22;
const OpTypeVector: u16 = 23;
const OpTypeRuntimeArray: u16 = 29;
const OpTypeStruct: u16 = 30;
const OpTypePointer: u16 = 32;
const OpTypeFunction: u16 = 33;
const OpConstant: u16 = 43;
const OpConstantComposite: u16 = 44;
const OpFunction: u16 = 54;
const OpFunctionEnd: u16 = 56;
const OpVariable: u16 = 59;
const OpLoad: u16 = 61;
const OpStore: u16 = 62;
const OpAccessChain: u16 = 65;
const OpDecorate: u16 = 71;
const OpMemberDecorate: u16 = 72;
const OpIAdd: u16 = 128;
const OpIMul: u16 = 132;
const OpULessThan: u16 = 176;
const OpLoopMerge: u16 = 246;
const OpLabel: u16 = 248;
const OpBranch: u16 = 249;
const OpBranchConditional: u16 = 250;
const OpReturn: u16 = 253;
const OpTypeCooperativeMatrixKHR: u16 = 4456;
const OpCooperativeMatrixMulAddKHR: u16 = 4459;
const OpCooperativeMatrixLoadTensorNV: u16 = 5367;
const OpCooperativeMatrixStoreTensorNV: u16 = 5368;
const OpTypeTensorLayoutNV: u16 = 5370;
const OpCreateTensorLayoutNV: u16 = 5372;
const OpTensorLayoutSetDimensionNV: u16 = 5373;
const OpTensorLayoutSliceNV: u16 = 5375;

const CapabilityShader: u32 = 1;
const CapabilityFloat16: u32 = 9;
const CapabilityStorageBuffer16BitAccess: u32 = 4433;
const CapabilityBFloat16TypeKHR: u32 = 5116;
const CapabilityBFloat16CooperativeMatrixKHR: u32 = 5118;
const CapabilityVulkanMemoryModel: u32 = 5345;
const CapabilityCooperativeMatrixWorkgroupScopeNV: u32 = 5430;
const CapabilityCooperativeMatrixFlexibleDimensionsNV: u32 = 5431;
const CapabilityCooperativeMatrixTensorAddressingNV: u32 = 5433;
const CapabilityTensorAddressingNV: u32 = 5439;
const CapabilityCooperativeMatrixKHR: u32 = 6022;

const AddressingModelLogical: u32 = 0;
const MemoryModelVulkan: u32 = 3;
const ExecutionModelGLCompute: u32 = 5;
const ExecutionModeLocalSize: u32 = 17;

const StorageClassInput: u32 = 1;
const StorageClassFunction: u32 = 7;
const StorageClassPushConstant: u32 = 9;
const StorageClassStorageBuffer: u32 = 12;

const DecorationBlock: u32 = 2;
const DecorationArrayStride: u32 = 6;
const DecorationBuiltIn: u32 = 11;
const DecorationNonWritable: u32 = 24;
const DecorationNonReadable: u32 = 25;
const DecorationBinding: u32 = 33;
const DecorationDescriptorSet: u32 = 34;
const DecorationOffset: u32 = 35;
const BuiltInWorkgroupId: u32 = 26;

const MatrixUseA: u32 = 0;
const MatrixUseB: u32 = 1;
const MatrixUseAccumulator: u32 = 2;
const ScopeWorkgroup: u32 = 2;
const MemoryAccessNone: u32 = 0;
const TensorAddressingNone: u32 = 0;
const FunctionControlNone: u32 = 0;
const LoopControlNone: u32 = 0;
const BFloat16EncodingKHR: u32 = 0;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(gpa);
    defer gpa.free(args);

    if (args.len != 4) return error.InvalidArguments;
    const mode = try modeFromArg(args[1]);

    var builder = Builder{};
    const words = try emitModule(&builder, mode);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[2], .data = std.mem.sliceAsBytes(words) });

    const zig = try renderZigWords(gpa, mode, words);
    defer gpa.free(zig);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[3], .data = zig });
}

const Ids = struct {
    main: u32,
    wg: u32,
    a_var: u32,
    b_var: u32,
    c_var: u32,
    pc_var: u32,

    void_ty: u32,
    bool_ty: u32,
    u32_ty: u32,
    i32_ty: u32,
    f32_ty: u32,
    in_ty: u32,
    vec3_u32_ty: u32,
    fn_void_ty: u32,

    u32_0: u32,
    u32_1: u32,
    u32_2: u32,
    u32_16: u32,
    u32_64: u32,
    u32_128: u32,
    i32_0: u32,
    i32_1: u32,
    i32_2: u32,
    i32_3: u32,
    i32_4: u32,
    i32_5: u32,
    f32_0: u32,
    in_zero: u32,

    acc_ty: u32,
    a_ty: u32,
    b_ty: u32,
    layout_ty: u32,
    acc_zero: u32,
    a_zero: u32,
    b_zero: u32,

    ptr_input_vec3_u32: u32,
    ptr_input_u32: u32,
    ptr_fn_u32: u32,
    ptr_fn_acc: u32,
    ptr_pc_struct: u32,
    ptr_pc_u32: u32,
    ptr_sb_f32: u32,
    ptr_sb_in: u32,

    array_a: u32,
    array_b: u32,
    array_c: u32,
    struct_a: u32,
    struct_b: u32,
    struct_c: u32,
    ptr_sb_struct_a: u32,
    ptr_sb_struct_b: u32,
    ptr_sb_struct_c: u32,
    pc_struct: u32,

    acc_var: u32,
    kk_var: u32,
};

const Builder = struct {
    words: [8192]u32 = undefined,
    len: usize = 0,
    bound: u32 = 1,

    fn nextId(self: *Builder) u32 {
        const id = self.bound;
        self.bound += 1;
        return id;
    }

    fn emit(self: *Builder, opcode: u16, operands: []const u32) !void {
        const wc = operands.len + 1;
        if (self.len + wc > self.words.len) return error.ModuleTooLarge;
        self.words[self.len] = (@as(u32, @intCast(wc)) << 16) | opcode;
        self.len += 1;
        @memcpy(self.words[self.len..][0..operands.len], operands);
        self.len += operands.len;
    }

    fn emitString(self: *Builder, opcode: u16, operands: []const u32, text: []const u8, trailing: []const u32) !void {
        const string_words = (text.len + 1 + 3) / 4;
        const wc = 1 + operands.len + string_words + trailing.len;
        if (self.len + wc > self.words.len) return error.ModuleTooLarge;
        self.words[self.len] = (@as(u32, @intCast(wc)) << 16) | opcode;
        self.len += 1;
        @memcpy(self.words[self.len..][0..operands.len], operands);
        self.len += operands.len;

        var i: usize = 0;
        while (i < string_words) : (i += 1) {
            var word: u32 = 0;
            for (0..4) |byte_i| {
                const text_i = i * 4 + byte_i;
                const byte = if (text_i < text.len) text[text_i] else 0;
                word |= @as(u32, byte) << @intCast(byte_i * 8);
            }
            self.words[self.len] = word;
            self.len += 1;
        }

        @memcpy(self.words[self.len..][0..trailing.len], trailing);
        self.len += trailing.len;
    }

    fn reserveIds(self: *Builder) Ids {
        return .{
            .main = self.nextId(),
            .wg = self.nextId(),
            .a_var = self.nextId(),
            .b_var = self.nextId(),
            .c_var = self.nextId(),
            .pc_var = self.nextId(),

            .void_ty = self.nextId(),
            .bool_ty = self.nextId(),
            .u32_ty = self.nextId(),
            .i32_ty = self.nextId(),
            .f32_ty = self.nextId(),
            .in_ty = self.nextId(),
            .vec3_u32_ty = self.nextId(),
            .fn_void_ty = self.nextId(),

            .u32_0 = self.nextId(),
            .u32_1 = self.nextId(),
            .u32_2 = self.nextId(),
            .u32_16 = self.nextId(),
            .u32_64 = self.nextId(),
            .u32_128 = self.nextId(),
            .i32_0 = self.nextId(),
            .i32_1 = self.nextId(),
            .i32_2 = self.nextId(),
            .i32_3 = self.nextId(),
            .i32_4 = self.nextId(),
            .i32_5 = self.nextId(),
            .f32_0 = self.nextId(),
            .in_zero = self.nextId(),

            .acc_ty = self.nextId(),
            .a_ty = self.nextId(),
            .b_ty = self.nextId(),
            .layout_ty = self.nextId(),
            .acc_zero = self.nextId(),
            .a_zero = self.nextId(),
            .b_zero = self.nextId(),

            .ptr_input_vec3_u32 = self.nextId(),
            .ptr_input_u32 = self.nextId(),
            .ptr_fn_u32 = self.nextId(),
            .ptr_fn_acc = self.nextId(),
            .ptr_pc_struct = self.nextId(),
            .ptr_pc_u32 = self.nextId(),
            .ptr_sb_f32 = self.nextId(),
            .ptr_sb_in = self.nextId(),

            .array_a = self.nextId(),
            .array_b = self.nextId(),
            .array_c = self.nextId(),
            .struct_a = self.nextId(),
            .struct_b = self.nextId(),
            .struct_c = self.nextId(),
            .ptr_sb_struct_a = self.nextId(),
            .ptr_sb_struct_b = self.nextId(),
            .ptr_sb_struct_c = self.nextId(),
            .pc_struct = self.nextId(),

            .acc_var = self.nextId(),
            .kk_var = self.nextId(),
        };
    }
};

fn emitModule(builder: *Builder, mode: Mode) ![]const u32 {
    builder.len = 0;
    builder.bound = 1;
    const ids = builder.reserveIds();

    try emitHeader(builder);
    try emitCapabilities(builder, mode);
    try emitEntry(builder, ids);
    try emitNames(builder, ids, mode);
    try emitDecorations(builder, ids);
    try emitTypes(builder, ids, mode);
    try emitFunction(builder, ids, mode);

    builder.words[3] = builder.bound;
    return builder.words[0..builder.len];
}

fn emitHeader(builder: *Builder) !void {
    if (builder.words.len < 5) return error.ModuleTooLarge;
    builder.words[0] = 0x07230203;
    builder.words[1] = 0x00010600;
    builder.words[2] = 0;
    builder.words[3] = 0;
    builder.words[4] = 0;
    builder.len = 5;
}

fn emitCapabilities(builder: *Builder, mode: Mode) !void {
    try builder.emit(OpCapability, &.{CapabilityShader});
    try builder.emit(OpCapability, &.{CapabilityStorageBuffer16BitAccess});
    try builder.emit(OpCapability, &.{CapabilityVulkanMemoryModel});
    try builder.emit(OpCapability, &.{CapabilityCooperativeMatrixKHR});
    try builder.emit(OpCapability, &.{CapabilityCooperativeMatrixWorkgroupScopeNV});
    try builder.emit(OpCapability, &.{CapabilityCooperativeMatrixFlexibleDimensionsNV});
    try builder.emit(OpCapability, &.{CapabilityCooperativeMatrixTensorAddressingNV});
    try builder.emit(OpCapability, &.{CapabilityTensorAddressingNV});
    if (mode.isF16()) {
        try builder.emit(OpCapability, &.{CapabilityFloat16});
    } else {
        try builder.emit(OpCapability, &.{CapabilityBFloat16TypeKHR});
        try builder.emit(OpCapability, &.{CapabilityBFloat16CooperativeMatrixKHR});
        try builder.emitString(OpExtension, &.{}, "SPV_KHR_bfloat16", &.{});
    }
    try builder.emitString(OpExtension, &.{}, "SPV_KHR_cooperative_matrix", &.{});
    try builder.emitString(OpExtension, &.{}, "SPV_NV_tensor_addressing", &.{});
    try builder.emitString(OpExtension, &.{}, "SPV_NV_cooperative_matrix2", &.{});
    try builder.emit(OpMemoryModel, &.{ AddressingModelLogical, MemoryModelVulkan });
}

fn emitEntry(builder: *Builder, ids: Ids) !void {
    try builder.emitString(OpEntryPoint, &.{ ExecutionModelGLCompute, ids.main }, "main", &.{
        ids.wg,
        ids.a_var,
        ids.b_var,
        ids.c_var,
        ids.pc_var,
    });
    try builder.emit(OpExecutionMode, &.{ ids.main, ExecutionModeLocalSize, 256, 1, 1 });
}

fn emitNames(builder: *Builder, ids: Ids, mode: Mode) !void {
    try builder.emitString(OpName, &.{ids.main}, "main", &.{});
    try builder.emitString(OpName, &.{ids.wg}, "gl_WorkGroupID", &.{});
    try builder.emitString(OpName, &.{ids.a_var}, "a_buffer", &.{});
    try builder.emitString(OpName, &.{ids.b_var}, "b_buffer", &.{});
    try builder.emitString(OpName, &.{ids.c_var}, "c_buffer", &.{});
    try builder.emitString(OpName, &.{ids.pc_var}, "push_constants", &.{});
    try builder.emitString(OpName, &.{ids.struct_a}, "A", &.{});
    try builder.emitString(OpMemberName, &.{ ids.struct_a, 0 }, "data", &.{});
    try builder.emitString(OpName, &.{ids.struct_b}, "B", &.{});
    try builder.emitString(OpMemberName, &.{ ids.struct_b, 0 }, "data", &.{});
    try builder.emitString(OpName, &.{ids.struct_c}, "C", &.{});
    try builder.emitString(OpMemberName, &.{ ids.struct_c, 0 }, "data", &.{});
    try builder.emitString(OpName, &.{ids.pc_struct}, "PushConstants", &.{});
    try builder.emitString(OpName, &.{ids.in_ty}, mode.componentLabel(), &.{});
}

fn emitDecorations(builder: *Builder, ids: Ids) !void {
    try builder.emit(OpDecorate, &.{ ids.wg, DecorationBuiltIn, BuiltInWorkgroupId });

    try builder.emit(OpDecorate, &.{ ids.array_a, DecorationArrayStride, 2 });
    try builder.emit(OpDecorate, &.{ ids.array_b, DecorationArrayStride, 2 });
    try builder.emit(OpDecorate, &.{ ids.array_c, DecorationArrayStride, 4 });

    try builder.emit(OpDecorate, &.{ ids.struct_a, DecorationBlock });
    try builder.emit(OpMemberDecorate, &.{ ids.struct_a, 0, DecorationNonWritable });
    try builder.emit(OpMemberDecorate, &.{ ids.struct_a, 0, DecorationOffset, 0 });
    try builder.emit(OpDecorate, &.{ ids.a_var, DecorationNonWritable });
    try builder.emit(OpDecorate, &.{ ids.a_var, DecorationBinding, 0 });
    try builder.emit(OpDecorate, &.{ ids.a_var, DecorationDescriptorSet, 0 });

    try builder.emit(OpDecorate, &.{ ids.struct_b, DecorationBlock });
    try builder.emit(OpMemberDecorate, &.{ ids.struct_b, 0, DecorationNonWritable });
    try builder.emit(OpMemberDecorate, &.{ ids.struct_b, 0, DecorationOffset, 0 });
    try builder.emit(OpDecorate, &.{ ids.b_var, DecorationNonWritable });
    try builder.emit(OpDecorate, &.{ ids.b_var, DecorationBinding, 1 });
    try builder.emit(OpDecorate, &.{ ids.b_var, DecorationDescriptorSet, 0 });

    try builder.emit(OpDecorate, &.{ ids.struct_c, DecorationBlock });
    try builder.emit(OpMemberDecorate, &.{ ids.struct_c, 0, DecorationNonReadable });
    try builder.emit(OpMemberDecorate, &.{ ids.struct_c, 0, DecorationOffset, 0 });
    try builder.emit(OpDecorate, &.{ ids.c_var, DecorationNonReadable });
    try builder.emit(OpDecorate, &.{ ids.c_var, DecorationBinding, 2 });
    try builder.emit(OpDecorate, &.{ ids.c_var, DecorationDescriptorSet, 0 });

    try builder.emit(OpDecorate, &.{ ids.pc_struct, DecorationBlock });
    inline for (0..6) |i| {
        try builder.emit(OpMemberDecorate, &.{ ids.pc_struct, i, DecorationOffset, i * 4 });
    }
}

fn emitTypes(builder: *Builder, ids: Ids, mode: Mode) !void {
    try builder.emit(OpTypeVoid, &.{ids.void_ty});
    try builder.emit(OpTypeBool, &.{ids.bool_ty});
    try builder.emit(OpTypeInt, &.{ ids.u32_ty, 32, 0 });
    try builder.emit(OpTypeInt, &.{ ids.i32_ty, 32, 1 });
    try builder.emit(OpTypeFloat, &.{ ids.f32_ty, 32 });
    if (mode.isF16()) {
        try builder.emit(OpTypeFloat, &.{ ids.in_ty, 16 });
    } else {
        try builder.emit(OpTypeFloat, &.{ ids.in_ty, 16, BFloat16EncodingKHR });
    }
    try builder.emit(OpTypeVector, &.{ ids.vec3_u32_ty, ids.u32_ty, 3 });
    try builder.emit(OpTypeFunction, &.{ ids.fn_void_ty, ids.void_ty });

    try builder.emit(OpConstant, &.{ ids.u32_ty, ids.u32_0, 0 });
    try builder.emit(OpConstant, &.{ ids.u32_ty, ids.u32_1, 1 });
    try builder.emit(OpConstant, &.{ ids.u32_ty, ids.u32_2, 2 });
    try builder.emit(OpConstant, &.{ ids.u32_ty, ids.u32_16, 16 });
    try builder.emit(OpConstant, &.{ ids.u32_ty, ids.u32_64, 64 });
    try builder.emit(OpConstant, &.{ ids.u32_ty, ids.u32_128, 128 });
    try builder.emit(OpConstant, &.{ ids.i32_ty, ids.i32_0, 0 });
    try builder.emit(OpConstant, &.{ ids.i32_ty, ids.i32_1, 1 });
    try builder.emit(OpConstant, &.{ ids.i32_ty, ids.i32_2, 2 });
    try builder.emit(OpConstant, &.{ ids.i32_ty, ids.i32_3, 3 });
    try builder.emit(OpConstant, &.{ ids.i32_ty, ids.i32_4, 4 });
    try builder.emit(OpConstant, &.{ ids.i32_ty, ids.i32_5, 5 });
    try builder.emit(OpConstant, &.{ ids.f32_ty, ids.f32_0, 0 });
    try builder.emit(OpConstant, &.{ ids.in_ty, ids.in_zero, 0 });

    const tile_m = constForDim(ids, mode.tileM());
    const tile_n = constForDim(ids, mode.tileN());
    try builder.emit(OpTypeCooperativeMatrixKHR, &.{ ids.acc_ty, ids.f32_ty, ids.u32_2, tile_m, tile_n, ids.u32_2 });
    try builder.emit(OpTypeCooperativeMatrixKHR, &.{ ids.a_ty, ids.in_ty, ids.u32_2, tile_m, ids.u32_16, ids.u32_0 });
    try builder.emit(OpTypeCooperativeMatrixKHR, &.{ ids.b_ty, ids.in_ty, ids.u32_2, ids.u32_16, tile_n, ids.u32_1 });
    try builder.emit(OpTypeTensorLayoutNV, &.{ ids.layout_ty, ids.u32_2, ids.u32_0 });
    try builder.emit(OpConstantComposite, &.{ ids.acc_ty, ids.acc_zero, ids.f32_0 });
    try builder.emit(OpConstantComposite, &.{ ids.a_ty, ids.a_zero, ids.in_zero });
    try builder.emit(OpConstantComposite, &.{ ids.b_ty, ids.b_zero, ids.in_zero });

    try builder.emit(OpTypePointer, &.{ ids.ptr_input_vec3_u32, StorageClassInput, ids.vec3_u32_ty });
    try builder.emit(OpTypePointer, &.{ ids.ptr_input_u32, StorageClassInput, ids.u32_ty });
    try builder.emit(OpTypePointer, &.{ ids.ptr_fn_u32, StorageClassFunction, ids.u32_ty });
    try builder.emit(OpTypePointer, &.{ ids.ptr_fn_acc, StorageClassFunction, ids.acc_ty });

    try builder.emit(OpTypeRuntimeArray, &.{ ids.array_a, ids.in_ty });
    try builder.emit(OpTypeRuntimeArray, &.{ ids.array_b, ids.in_ty });
    try builder.emit(OpTypeRuntimeArray, &.{ ids.array_c, ids.f32_ty });
    try builder.emit(OpTypeStruct, &.{ ids.struct_a, ids.array_a });
    try builder.emit(OpTypeStruct, &.{ ids.struct_b, ids.array_b });
    try builder.emit(OpTypeStruct, &.{ ids.struct_c, ids.array_c });
    try builder.emit(OpTypePointer, &.{ ids.ptr_sb_struct_a, StorageClassStorageBuffer, ids.struct_a });
    try builder.emit(OpTypePointer, &.{ ids.ptr_sb_struct_b, StorageClassStorageBuffer, ids.struct_b });
    try builder.emit(OpTypePointer, &.{ ids.ptr_sb_struct_c, StorageClassStorageBuffer, ids.struct_c });
    try builder.emit(OpTypePointer, &.{ ids.ptr_sb_in, StorageClassStorageBuffer, ids.in_ty });
    try builder.emit(OpTypePointer, &.{ ids.ptr_sb_f32, StorageClassStorageBuffer, ids.f32_ty });

    try builder.emit(OpTypeStruct, &.{ ids.pc_struct, ids.u32_ty, ids.u32_ty, ids.u32_ty, ids.u32_ty, ids.u32_ty, ids.u32_ty });
    try builder.emit(OpTypePointer, &.{ ids.ptr_pc_struct, StorageClassPushConstant, ids.pc_struct });
    try builder.emit(OpTypePointer, &.{ ids.ptr_pc_u32, StorageClassPushConstant, ids.u32_ty });

    try builder.emit(OpVariable, &.{ ids.ptr_input_vec3_u32, ids.wg, StorageClassInput });
    try builder.emit(OpVariable, &.{ ids.ptr_sb_struct_a, ids.a_var, StorageClassStorageBuffer });
    try builder.emit(OpVariable, &.{ ids.ptr_sb_struct_b, ids.b_var, StorageClassStorageBuffer });
    try builder.emit(OpVariable, &.{ ids.ptr_sb_struct_c, ids.c_var, StorageClassStorageBuffer });
    try builder.emit(OpVariable, &.{ ids.ptr_pc_struct, ids.pc_var, StorageClassPushConstant });
}

fn emitFunction(builder: *Builder, ids: Ids, mode: Mode) !void {
    try builder.emit(OpFunction, &.{ ids.void_ty, ids.main, FunctionControlNone, ids.fn_void_ty });
    const entry_label = builder.nextId();
    try builder.emit(OpLabel, &.{entry_label});
    try builder.emit(OpVariable, &.{ ids.ptr_fn_acc, ids.acc_var, StorageClassFunction });
    try builder.emit(OpVariable, &.{ ids.ptr_fn_u32, ids.kk_var, StorageClassFunction });

    const wg_x_ptr = try accessChain(builder, ids.ptr_input_u32, ids.wg, &.{ids.u32_0});
    const tile_col = try load(builder, ids.u32_ty, wg_x_ptr);
    const wg_y_ptr = try accessChain(builder, ids.ptr_input_u32, ids.wg, &.{ids.u32_1});
    const tile_row = try load(builder, ids.u32_ty, wg_y_ptr);
    const row = try binary(builder, OpIMul, ids.u32_ty, tile_row, constForDim(ids, mode.tileM()));
    const col = try binary(builder, OpIMul, ids.u32_ty, tile_col, constForDim(ids, mode.tileN()));

    const m_value = try loadPush(builder, ids, ids.i32_0);
    const n_value = try loadPush(builder, ids, ids.i32_1);
    const k_value = try loadPush(builder, ids, ids.i32_2);

    const layout_base = try createTensorLayout(builder, ids);
    const a_layout = try tensorSetDimension(builder, ids, layout_base, m_value, k_value);
    const b_layout = try tensorSetDimension(builder, ids, layout_base, k_value, n_value);
    const c_layout = try tensorSetDimension(builder, ids, layout_base, m_value, n_value);

    const a_base_ptr = try accessChain(builder, ids.ptr_sb_in, ids.a_var, &.{ ids.i32_0, ids.u32_0 });
    const b_base_ptr = try accessChain(builder, ids.ptr_sb_in, ids.b_var, &.{ ids.i32_0, ids.u32_0 });
    const c_base_ptr = try accessChain(builder, ids.ptr_sb_f32, ids.c_var, &.{ ids.i32_0, ids.u32_0 });

    try builder.emit(OpStore, &.{ ids.acc_var, ids.acc_zero });
    try builder.emit(OpStore, &.{ ids.kk_var, ids.u32_0 });

    const loop_header = builder.nextId();
    const loop_cond = builder.nextId();
    const loop_body = builder.nextId();
    const loop_merge = builder.nextId();
    const loop_continue = builder.nextId();
    try builder.emit(OpBranch, &.{loop_header});

    try builder.emit(OpLabel, &.{loop_header});
    try builder.emit(OpLoopMerge, &.{ loop_merge, loop_continue, LoopControlNone });
    try builder.emit(OpBranch, &.{loop_cond});

    try builder.emit(OpLabel, &.{loop_cond});
    const kk_for_cond = try load(builder, ids.u32_ty, ids.kk_var);
    const cond = try binary(builder, OpULessThan, ids.bool_ty, kk_for_cond, k_value);
    try builder.emit(OpBranchConditional, &.{ cond, loop_body, loop_merge });

    try builder.emit(OpLabel, &.{loop_body});
    const kk = try load(builder, ids.u32_ty, ids.kk_var);
    const a_slice = try tensorSlice(builder, ids, a_layout, row, constForDim(ids, mode.tileM()), kk, ids.u32_16);
    const b_slice = try tensorSlice(builder, ids, b_layout, kk, ids.u32_16, col, constForDim(ids, mode.tileN()));
    const a_mat = try coopTensorLoad(builder, ids.a_ty, a_base_ptr, ids.a_zero, a_slice);
    const b_mat = try coopTensorLoad(builder, ids.b_ty, b_base_ptr, ids.b_zero, b_slice);
    const acc_in = try load(builder, ids.acc_ty, ids.acc_var);
    const acc_out = builder.nextId();
    try builder.emit(OpCooperativeMatrixMulAddKHR, &.{ ids.acc_ty, acc_out, a_mat, b_mat, acc_in });
    try builder.emit(OpStore, &.{ ids.acc_var, acc_out });
    try builder.emit(OpBranch, &.{loop_continue});

    try builder.emit(OpLabel, &.{loop_continue});
    const kk_old = try load(builder, ids.u32_ty, ids.kk_var);
    const kk_next = try binary(builder, OpIAdd, ids.u32_ty, kk_old, ids.u32_16);
    try builder.emit(OpStore, &.{ ids.kk_var, kk_next });
    try builder.emit(OpBranch, &.{loop_header});

    try builder.emit(OpLabel, &.{loop_merge});
    const c_slice = try tensorSlice(builder, ids, c_layout, row, constForDim(ids, mode.tileM()), col, constForDim(ids, mode.tileN()));
    const acc_final = try load(builder, ids.acc_ty, ids.acc_var);
    try coopTensorStore(builder, c_base_ptr, acc_final, c_slice);
    try builder.emit(OpReturn, &.{});
    try builder.emit(OpFunctionEnd, &.{});
}

fn loadPush(builder: *Builder, ids: Ids, member: u32) !u32 {
    const ptr = try accessChain(builder, ids.ptr_pc_u32, ids.pc_var, &.{member});
    return load(builder, ids.u32_ty, ptr);
}

fn accessChain(builder: *Builder, result_ty: u32, base: u32, indexes: []const u32) !u32 {
    const result = builder.nextId();
    var operands: [8]u32 = undefined;
    if (indexes.len + 3 > operands.len) return error.TooManyIndexes;
    operands[0] = result_ty;
    operands[1] = result;
    operands[2] = base;
    @memcpy(operands[3..][0..indexes.len], indexes);
    try builder.emit(OpAccessChain, operands[0 .. indexes.len + 3]);
    return result;
}

fn load(builder: *Builder, result_ty: u32, ptr: u32) !u32 {
    const result = builder.nextId();
    try builder.emit(OpLoad, &.{ result_ty, result, ptr });
    return result;
}

fn binary(builder: *Builder, opcode: u16, result_ty: u32, lhs: u32, rhs: u32) !u32 {
    const result = builder.nextId();
    try builder.emit(opcode, &.{ result_ty, result, lhs, rhs });
    return result;
}

fn createTensorLayout(builder: *Builder, ids: Ids) !u32 {
    const result = builder.nextId();
    try builder.emit(OpCreateTensorLayoutNV, &.{ ids.layout_ty, result });
    return result;
}

fn tensorSetDimension(builder: *Builder, ids: Ids, layout: u32, dim0: u32, dim1: u32) !u32 {
    const result = builder.nextId();
    try builder.emit(OpTensorLayoutSetDimensionNV, &.{ ids.layout_ty, result, layout, dim0, dim1 });
    return result;
}

fn tensorSlice(builder: *Builder, ids: Ids, layout: u32, offset0: u32, span0: u32, offset1: u32, span1: u32) !u32 {
    const result = builder.nextId();
    try builder.emit(OpTensorLayoutSliceNV, &.{ ids.layout_ty, result, layout, offset0, span0, offset1, span1 });
    return result;
}

fn coopTensorLoad(builder: *Builder, result_ty: u32, ptr: u32, object: u32, layout: u32) !u32 {
    const result = builder.nextId();
    try builder.emit(OpCooperativeMatrixLoadTensorNV, &.{ result_ty, result, ptr, object, layout, MemoryAccessNone, TensorAddressingNone });
    return result;
}

fn coopTensorStore(builder: *Builder, ptr: u32, object: u32, layout: u32) !void {
    try builder.emit(OpCooperativeMatrixStoreTensorNV, &.{ ptr, object, layout, MemoryAccessNone, TensorAddressingNone });
}

fn renderZigWords(gpa: std.mem.Allocator, mode: Mode, words: []const u32) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(gpa);
    const writer = &out.writer;
    defer out.deinit();

    try writer.print(
        \\//! Generated by `//tools:emit_nvcoop2_spv`.
        \\//! Do not edit by hand.
        \\
        \\pub const import_name = "{s}";
        \\pub const words = [_]u32{{
        \\
    , .{mode.importName()});

    var i: usize = 0;
    while (i < words.len) : (i += 8) {
        try writer.writeAll("    ");
        const end = @min(i + 8, words.len);
        for (words[i..end]) |word| try writer.print("0x{x:0>8}, ", .{word});
        try writer.writeByte('\n');
    }
    try writer.writeAll("};\n");

    return out.toOwnedSlice();
}
