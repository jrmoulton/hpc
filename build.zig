const std = @import("std");
const Build = std.Build;
const Target = std.Target;
const fs = std.fs;
const zcc = @import("build/compile_commands.zig");

pub fn main() void {
    std.build.run(build);
}

pub fn build(b: *Build) void {
    const CFilesList = std.ArrayList([]const u8);
    var src_dir = fs.cwd().openDir("src/", .{ .iterate = true }) catch unreachable;
    var cpp_files: CFilesList = CFilesList.init(b.allocator);
    defer cpp_files.deinit();

    var src_iter = src_dir.iterate();
    while (src_iter.next() catch unreachable) |entry| {
        if (std.mem.endsWith(u8, entry.name, ".cpp")) {
            const c_file_path = std.fmt.allocPrint(b.allocator, "src/{s}", .{entry.name}) catch unreachable;
            cpp_files.append(c_file_path) catch unreachable;
        }
    }
    src_dir.close();

    var targets = std.ArrayList(*std.Build.Step.Compile).init(b.allocator);

    const install_all = b.step("all", "Build and install all targets");

    const mode = b.standardOptimizeOption(.{});

    var bin_dir = fs.cwd().openDir("bin/", .{ .iterate = true }) catch unreachable;
    defer bin_dir.close();
    var bin_iter = bin_dir.iterate();

    while (bin_iter.next() catch unreachable) |entry| {
        const ends_with_cpp = std.mem.endsWith(u8, entry.name, ".cpp");
        const ends_with_c = std.mem.endsWith(u8, entry.name, ".c");

        if (entry.kind == .file and (ends_with_cpp or ends_with_c)) {
            const ext = if (ends_with_cpp) ".cpp" else ".c";
            const exe_name = std.mem.trimRight(u8, entry.name, ext);

            addExecutable(b, &targets, mode, install_all, exe_name, std.fmt.allocPrint(b.allocator, "bin/{s}", .{entry.name}) catch unreachable, cpp_files.items);
        }
    }

    const targets_clone = targets.clone() catch unreachable;

    const cdb_step = zcc.createStep(b, "cdb", targets.toOwnedSlice() catch unreachable);
    for (targets_clone.items) |target| {
        target.step.dependOn(cdb_step);
    }
    b.default_step.dependOn(cdb_step);
}

fn addExecutable(
    b: *Build,
    targets: *std.ArrayList(*std.Build.Step.Compile),
    mode: std.builtin.OptimizeMode,
    install_all: *Build.Step,
    name: []const u8,
    src_path: []const u8,
    c_files: []const []const u8,
) void {
    const exe = b.addExecutable(.{
        .name = name,
        .optimize = mode,
        .target = b.host,
    });

    // const gtest_dep = b.dependency("gtest", .{});
    // exe.addCSourceFiles(.{
    //     .root = gtest_dep.path("googletest/src"),
    //     .files = &.{
    //         "gtest-all.cc",
    //     },
    // });

    // exe.addIncludePath(gtest_dep.path("googletest"));
    // exe.addIncludePath(gtest_dep.path("googletest/include"));

    const expected_dep = b.dependency("tl_expected", .{});
    exe.addIncludePath(expected_dep.path("include"));

    exe.linkSystemLibrary("c");
    exe.linkSystemLibrary("c++");
    exe.addLibraryPath(.{ .src_path = .{ .sub_path = "/opt/homebrew/opt/libomp/lib", .owner = b } });
    exe.addIncludePath(.{ .src_path = .{ .sub_path = "/opt/homebrew/opt/libomp/include", .owner = b } });

    exe.linkSystemLibrary("omp");

    const flags = .{
        "-Wall",
        "-Wextra",
        "-Wconversion",
        "-Wsign-conversion",
        "-Wtype-limits",
        // "-fsanitize=undefined",
        "-Werror",
        "-std=c++11",
        "-fopenmp",
    };

    exe.addCSourceFile(.{
        .file = .{ .src_path = .{ .owner = b, .sub_path = src_path } },
        .flags = &flags,
    });

    exe.addCSourceFiles(.{
        .files = c_files,
        .flags = &flags,
    });

    exe.addIncludePath(.{ .src_path = .{ .owner = b, .sub_path = "include/" } });

    targets.append(exe) catch @panic("OOM");

    const install_artifact = b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "../target" } } });
    install_all.dependOn(&install_artifact.step);

    var build_step_name_buffer: [64]u8 = undefined;
    const build_step_name = std.fmt.bufPrint(&build_step_name_buffer, "{s}", .{name}) catch @panic("OOM");
    var build_step_description_buffer: [64]u8 = undefined;
    const build_description = std.fmt.bufPrint(&build_step_description_buffer, "Build the {s} program", .{build_step_name}) catch @panic("OOM");
    const build_step = b.step(build_step_name, build_description);

    build_step.dependOn(&install_artifact.step);
}
