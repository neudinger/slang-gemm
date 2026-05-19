const std = @import("std");

pub fn main(init: std.process.Init) !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const args = try init.minimal.args.toSlice(allocator);
    if (args.len != 4) return error.InvalidArgument;

    const input_path = args[1];
    const output_path = args[2];
    const symbol = args[3];

    var input = try std.Io.Dir.cwd().openFile(init.io, input_path, .{});
    defer input.close(init.io);
    const len: usize = @intCast(try input.length(init.io));
    if (len > 256 * 1024 * 1024) return error.FileTooLarge;
    const bytes = try allocator.alloc(u8, len);
    var offset: usize = 0;
    while (offset < bytes.len) {
        const n = try input.readPositional(init.io, &.{bytes[offset..]}, offset);
        if (n == 0) return error.EndOfStream;
        offset += n;
    }
    if (bytes.len % 4 != 0) return error.InvalidSpirvSize;

    var out = try std.Io.Dir.cwd().createFile(init.io, output_path, .{});
    defer out.close(init.io);
    var out_buffer: [4096]u8 = undefined;
    var writer = out.writer(init.io, &out_buffer);
    try writer.interface.print("pub const words = [_]u32{{\n", .{});
    var i: usize = 0;
    while (i < bytes.len) : (i += 4) {
        const word = std.mem.readInt(u32, bytes[i..][0..4], .little);
        try writer.interface.print("    0x{x:0>8},\n", .{word});
    }
    try writer.interface.print("}};\npub const name = \"{s}\";\n", .{symbol});
    try writer.interface.flush();
}
