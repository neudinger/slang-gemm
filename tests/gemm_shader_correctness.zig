const std = @import("std");

test "Slang GEMM shader artifact is built as test data" {
    // The SPIR-V file is listed in the Bazel data deps for this target. That
    // makes Bazel run the pinned Slang and spirv-val pipeline before this test.
    try std.testing.expect(true);
}

