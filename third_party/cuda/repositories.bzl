def _cuda_sdk_repository_impl(ctx):
    cuda_path = ctx.path(ctx.attr.cuda_path)
    required = [
        "include/cublas_v2.h",
        "include/cuda_runtime_api.h",
        "lib64/libcublas.so",
        "lib64/libcudart.so",
    ]
    for rel in required:
        path = cuda_path.get_child(rel)
        if not path.exists:
            fail("CUDA SDK at '{}' is missing '{}'".format(cuda_path, rel))

    ctx.symlink(cuda_path.get_child("include"), "include")
    ctx.symlink(cuda_path.get_child("lib64"), "lib64")
    ctx.file("BUILD.bazel", """
load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "cuda",
    hdrs = glob(["include/**/*.h"]),
    srcs = [
        "lib64/libcublas.so",
        "lib64/libcudart.so",
    ],
    includes = ["include"],
    linkstatic = False,
    visibility = ["//visibility:public"],
)
""")

cuda_sdk_repository = repository_rule(
    implementation = _cuda_sdk_repository_impl,
    attrs = {
        "cuda_path": attr.string(mandatory = True),
    },
    local = True,
)

