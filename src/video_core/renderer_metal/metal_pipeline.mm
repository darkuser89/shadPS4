// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <exception>
#include <memory>
#include <stdexcept>
#include <utility>
#include <fmt/format.h>

#include "common/assert.h"
#include "video_core/renderer_metal/metal_instance.h"
#include "video_core/renderer_metal/metal_pipeline.h"
#include "video_core/renderer_metal/metal_pipeline_archive.h"

namespace Metal {

namespace {

template <typename Result, typename Descriptor>
struct Compilation {
    void Complete(Result result, std::exception_ptr error = {}) {
        __attribute__((objc_precise_lifetime)) id<MTL4CompilerTask> completed_task;
        std::shared_ptr<void> completed_lifetime;
        {
            std::scoped_lock lock{mutex};
            if (completed) {
                return;
            }
            if (error) {
                promise.set_exception(std::move(error));
            } else {
                promise.set_value(result);
            }
            completed = true;
            completed_task = task;
            task = nil;
            completed_lifetime = std::move(lifetime);
        }
    }

    void Fail(std::exception_ptr error) {
        Complete(nil, std::move(error));
    }

    void SetTask(id<MTL4CompilerTask> pending_task) {
        std::scoped_lock lock{mutex};
        if (!completed) {
            task = pending_task;
        }
    }

    std::promise<Result> promise;
    std::mutex mutex;
    id<MTL4Compiler> compiler;
    Descriptor* descriptor;
    id<MTL4CompilerTask> task;
    bool completed{};
    std::shared_ptr<void> lifetime;
    MTL4CompilerTaskOptions* options;
};

} // namespace

PipelineCompiler::PipelineCompiler(const Instance& instance_) : instance{instance_} {}

size_t PipelineCompiler::LibrarySourceHash::operator()(std::string_view source) const noexcept {
    return std::hash<std::string_view>{}(source);
}

id<MTLLibrary> PipelineCompiler::CompileLibrary(std::string_view source, std::string_view name) {
    return CompileLibraryAsync(source, name).get();
}

std::shared_future<id<MTLLibrary>> PipelineCompiler::CompileLibraryAsync(std::string_view source,
                                                                         std::string_view name) {
    std::shared_ptr<Compilation<id<MTLLibrary>, MTL4LibraryDescriptor>> compilation;
    std::shared_future<id<MTLLibrary>> result;
    {
        std::scoped_lock lock{library_mutex};
        const auto it = libraries.find(source);
        if (it != libraries.end()) {
            return it->second;
        } else {
            compilation = std::make_shared<Compilation<id<MTLLibrary>, MTL4LibraryDescriptor>>();
            result = compilation->promise.get_future().share();
            libraries.emplace(std::string{source}, result);
        }
    }
    try {
        @autoreleasepool {
            MTL4LibraryDescriptor* descriptor = [MTL4LibraryDescriptor new];
            descriptor.source = [[NSString alloc] initWithBytes:source.data()
                                                         length:source.size()
                                                       encoding:NSUTF8StringEncoding];
            descriptor.name = [[NSString alloc] initWithBytes:name.data()
                                                       length:name.size()
                                                     encoding:NSUTF8StringEncoding];
            if (!descriptor.source || !descriptor.name) {
                throw std::runtime_error("Invalid UTF-8 in Metal shader source or name");
            }
            MTLCompileOptions* options = [MTLCompileOptions new];
            options.languageVersion = MTLLanguageVersion4_0;
            options.mathMode = MTLMathModeSafe;
            options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
            descriptor.options = options;
            compilation->compiler = instance.GetCompiler();
            compilation->descriptor = descriptor;
            compilation->lifetime = instance.GetPipelineArchive().TrackCompilation(false);
            const std::string shader_name{name};
            id<MTL4CompilerTask> task = [compilation->compiler
                newLibraryWithDescriptor:descriptor
                       completionHandler:^(id<MTLLibrary> library, NSError* error) {
                         @autoreleasepool {
                             try {
                                 if (!library) {
                                     throw std::runtime_error(
                                         fmt::format("Failed to compile Metal shader {}: {}",
                                                     shader_name, ErrorDescription(error)));
                                 }
                                 compilation->Complete(library);
                             } catch (...) {
                                 compilation->Fail(std::current_exception());
                             }
                         }
                       }];
            if (!task) {
                throw std::runtime_error(
                    fmt::format("Failed to schedule Metal shader compilation: {}", name));
            }
            compilation->SetTask(task);
        }
    } catch (...) {
        compilation->Fail(std::current_exception());
    }
    return result;
}

id<MTLComputePipelineState> PipelineCompiler::CompileComputePipeline(
    MTL4ComputePipelineDescriptor* descriptor) {
    return CompileComputePipelineAsync(descriptor).get();
}

std::shared_future<id<MTLComputePipelineState>> PipelineCompiler::CompileComputePipelineAsync(
    MTL4ComputePipelineDescriptor* descriptor) {
    ASSERT(descriptor);
    auto compilation =
        std::make_shared<Compilation<id<MTLComputePipelineState>, MTL4ComputePipelineDescriptor>>();
    auto result = compilation->promise.get_future().share();
    try {
        @autoreleasepool {
            compilation->compiler = instance.GetCompiler();
            compilation->descriptor = [descriptor copy];
            compilation->options = instance.GetPipelineArchive().TaskOptions();
            compilation->lifetime = instance.GetPipelineArchive().TrackCompilation(true);
            dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
              @autoreleasepool {
                  try {
                      NSError* error = nil;
                      auto pipeline = [compilation->compiler
                          newComputePipelineStateWithDescriptor:compilation->descriptor
                                            compilerTaskOptions:compilation->options
                                                          error:&error];
                      if (!pipeline) {
                          throw std::runtime_error(
                              fmt::format("Failed to compile Metal 4 compute pipeline: {}",
                                          ErrorDescription(error)));
                      }
                      compilation->Complete(pipeline);
                  } catch (...) {
                      compilation->Fail(std::current_exception());
                  }
              }
            });
        }
    } catch (...) {
        compilation->Fail(std::current_exception());
    }
    return result;
}

id<MTLRenderPipelineState> PipelineCompiler::CompileRenderPipeline(
    MTL4PipelineDescriptor* descriptor) {
    return CompileRenderPipelineAsync(descriptor).get();
}

std::shared_future<id<MTLRenderPipelineState>> PipelineCompiler::CompileRenderPipelineAsync(
    MTL4PipelineDescriptor* descriptor) {
    ASSERT(descriptor);
    auto compilation =
        std::make_shared<Compilation<id<MTLRenderPipelineState>, MTL4PipelineDescriptor>>();
    auto result = compilation->promise.get_future().share();
    try {
        @autoreleasepool {
            compilation->compiler = instance.GetCompiler();
            compilation->descriptor = [descriptor copy];
            compilation->options = instance.GetPipelineArchive().TaskOptions();
            compilation->lifetime = instance.GetPipelineArchive().TrackCompilation(true);
            dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
              @autoreleasepool {
                  try {
                      NSError* error = nil;
                      auto pipeline = [compilation->compiler
                          newRenderPipelineStateWithDescriptor:compilation->descriptor
                                           compilerTaskOptions:compilation->options
                                                         error:&error];
                      if (!pipeline) {
                          throw std::runtime_error(
                              fmt::format("Failed to compile Metal 4 render pipeline: {}",
                                          ErrorDescription(error)));
                      }
                      compilation->Complete(pipeline);
                  } catch (...) {
                      compilation->Fail(std::current_exception());
                  }
              }
            });
        }
    } catch (...) {
        compilation->Fail(std::current_exception());
    }
    return result;
}

} // namespace Metal
