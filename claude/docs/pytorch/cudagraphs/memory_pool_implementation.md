# PyTorch CUDA Graphs: Private Memory Pool Implementation

## Table of Contents
1. [Overview](#overview)
2. [Frame-by-Frame Trace: torch.cuda.CUDAGraph](#frame-by-frame-trace-torchcudacudagraph)
3. [Frame-by-Frame Trace: make_graphed_callables](#frame-by-frame-trace-make_graphed_callables)
4. [Key Data Structures](#key-data-structures)
5. [Memory Pool Lifecycle](#memory-pool-lifecycle)
6. [Design Rationale](#design-rationale)

---

## Overview

PyTorch's CUDA graph support provides a convenience wrapper around NVIDIA's CUDA graphs API that handles memory allocation through a **separate private memory pool** for each graph capture. This design ensures that:

1. **Memory addresses remain stable** across graph replays (required by CUDA graphs)
2. **No interference** between eager allocations and graph allocations
3. **Safe replay** without risk of memory corruption

The key insight from the PyTorch documentation:

> "The CachingAllocator uses a separate memory pool for all the new allocations. During CUDAGraph recording, memory is accounted for, allocated, and freed exactly as during eager run. On replay, just the kernels are invoked, and there are no changes to the allocator."

---

## Frame-by-Frame Trace: torch.cuda.CUDAGraph

This section traces the execution from Python through to C++/CUDA, showing how the private memory pool is created and managed.

### 1. Python Level Entry Point

**File**: `pytorch/torch/cuda/graphs.py`

#### User Code Example:
```python
g = torch.cuda.CUDAGraph()
static_input = torch.empty((5,), device="cuda")

# Warmup on a side stream
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        static_output = static_input * 2
torch.cuda.current_stream().wait_stream(s)

# Capture the graph
with torch.cuda.graph(g):
    static_output = static_input * 2
```

#### Context Manager: `torch.cuda.graph` Class

**Location**: [pytorch/torch/cuda/graphs.py:185-270](pytorch/torch/cuda/graphs.py#L185-L270)

```python
class graph:
    def __init__(
        self,
        cuda_graph: CUDAGraph,
        pool: Optional[_POOL_HANDLE] = None,
        stream: Optional[torch.cuda.Stream] = None,
        capture_error_mode: str = "global",
    ):
        # Create or reuse capture stream
        if self.__class__.default_capture_stream is None:
            self.__class__.default_capture_stream = torch.cuda.Stream()

        self.pool = () if pool is None else (pool,)
        self.capture_stream = stream if stream is not None else self.__class__.default_capture_stream
        self.stream_ctx = torch.cuda.stream(self.capture_stream)
        self.cuda_graph = cuda_graph
        self.capture_error_mode = capture_error_mode
```

**Key Points**:
- Always uses a non-default stream for capture (requirement from CUDA)
- Can optionally share memory pool with other graphs via `pool` parameter
- Default capture mode is `"global"` (most conservative)

---

### 2. Capture Begin

#### Python Side: `__enter__` Method

**Location**: [pytorch/torch/cuda/graphs.py:241-264](pytorch/torch/cuda/graphs.py#L241-L264)

```python
def __enter__(self) -> None:
    # Free as much memory as possible
    torch.cuda.synchronize()

    # Optional garbage collection (expensive)
    if torch.compiler.config.force_cudagraph_gc:
        gc.collect()

    torch.cuda.empty_cache()

    # Set capture stream as current
    self.stream_ctx.__enter__()

    # Begin capture - THIS CREATES THE PRIVATE POOL
    self.cuda_graph.capture_begin(
        *self.pool,
        capture_error_mode=self.capture_error_mode,
    )
```

**Key Actions**:
1. Synchronize and free cached memory to maximize available memory
2. Switch to side stream (non-default required by CUDA)
3. Call `capture_begin` on the C++ CUDAGraph object

#### Python Wrapper to C++: `CUDAGraph.capture_begin`

**Location**: [pytorch/torch/cuda/graphs.py:99-118](pytorch/torch/cuda/graphs.py#L99-L118)

```python
def capture_begin(
    self, pool: Optional[_POOL_HANDLE] = None, capture_error_mode: str = "global"
) -> None:
    """Begin capturing CUDA work on the current stream."""
    super().capture_begin(pool=pool, capture_error_mode=capture_error_mode)
```

This calls into the C++ implementation (`torch._C._CUDAGraph`).

---

### 3. C++ Level: Graph Capture Begin

**File**: `pytorch/aten/src/ATen/cuda/CUDAGraph.cpp`

#### CUDAGraph::capture_begin

**Location**: [pytorch/aten/src/ATen/cuda/CUDAGraph.cpp:58-116](pytorch/aten/src/ATen/cuda/CUDAGraph.cpp#L58-L116)

```cpp
void CUDAGraph::capture_begin(MempoolId_t pool/*=0*/, cudaStreamCaptureMode capture_mode) {
  TORCH_CHECK(!has_graph_exec_,
              "This CUDAGraph instance already owns a captured graph.");

  // Register default RNG generator for the graph
  auto* gen = get_generator_or_default<CUDAGeneratorImpl>(
      std::nullopt, cuda::detail::getDefaultCUDAGenerator());
  gen->register_graph(this);

  // Prepare all registered generators for capture
  for (auto& [generator_state, wholegraph_increments] : captured_generator_states_) {
    generator_state->capture_prologue();
  }

  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream != at::cuda::getDefaultCUDAStream(),
              "CUDA graphs must be captured on a non-default stream.");

  capture_stream_ = stream;
  capture_dev_ = c10::cuda::current_device();

  // CREATE OR REUSE MEMORY POOL ID
  if (pool.first != 0 || pool.second != 0) {
    // User wants to share an existing pool
    TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
    mempool_id_ = pool;
  } else {
    // Create a new unique pool ID
    // Format: {unique_id, 0} distinguishes from user-created handles
    mempool_id_ = c10::cuda::MemPool::graph_pool_handle(false);
    TORCH_INTERNAL_ASSERT(mempool_id_.first > 0);
  }

  // *** CRITICAL: Tell allocator to start using private pool ***
  c10::cuda::CUDACachingAllocator::beginAllocateToPool(
      capture_dev_,
      mempool_id_,
      [this](cudaStream_t stream) {
          cudaStreamCaptureStatus status{};
          CaptureId_t stream_capture_id = 0;
          AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &stream_capture_id));
          return status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive
                 && stream_capture_id == capture_id_;
      }
  );

  // Begin CUDA graph capture
  AT_CUDA_CHECK(cudaStreamBeginCapture(capture_stream_, capture_mode));

  // Get the capture ID assigned by CUDA
  cudaStreamCaptureStatus status{};
  AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &capture_id_));
  TORCH_INTERNAL_ASSERT(status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive);
}
```

**Key Steps**:
1. **Create/reuse mempool_id**: Unique identifier for this graph's memory pool
2. **`beginAllocateToPool`**: Tells caching allocator to redirect allocations
3. **`cudaStreamBeginCapture`**: Start CUDA graph recording
4. **Store capture_id**: Used to identify if a stream is participating in this capture

**Critical Ordering**: Note that `beginAllocateToPool` is called **before** `cudaStreamBeginCapture`. This prevents race conditions where an autograd thread might try to free memory after capture starts but before the allocator knows about it.

---

### 4. Allocator Level: Creating the Private Pool

**File**: `pytorch/c10/cuda/CUDACachingAllocator.cpp`

#### beginAllocateToPool

**Location**: [pytorch/c10/cuda/CUDACachingAllocator.cpp:2529-2541](pytorch/c10/cuda/CUDACachingAllocator.cpp#L2529-L2541)

```cpp
void beginAllocateToPool(
    MempoolId_t mempool_id,
    std::function<bool(cudaStream_t)> filter) {
  std::lock_guard<std::recursive_mutex> lock(mutex);

  // Create the private pool or increment ref count if it exists
  create_or_incref_pool(mempool_id);

  // Verify we're not already capturing to this pool
  for (auto it2 = captures_underway.begin(); it2 != captures_underway.end(); ++it2) {
    TORCH_CHECK(
        it2->first != mempool_id,
        "beginAllocateToPool: already recording to mempool_id");
  }

  // Register this capture with its pool ID and stream filter
  captures_underway.emplace_back(mempool_id, std::move(filter));
}
```

**Key Data Structure**: `captures_underway`
```cpp
std::vector<std::pair<MempoolId_t, std::function<bool(cudaStream_t)>>>
    captures_underway;
```

This vector tracks all active captures. Each entry has:
- **MempoolId_t**: Identifies the private pool
- **Filter function**: Determines if a stream belongs to this capture

#### create_or_incref_pool

**Location**: [pytorch/c10/cuda/CUDACachingAllocator.cpp:2670-2689](pytorch/c10/cuda/CUDACachingAllocator.cpp#L2670-L2689)

```cpp
void create_or_incref_pool(
    MempoolId_t mempool_id,
    CUDAAllocator* allocator = nullptr) {
  auto it = graph_pools.find(mempool_id);
  if (it == graph_pools.end()) {
    // CREATE NEW PRIVATE POOL
    // use_count starts at 1 (one graph using it)
    graph_pools.emplace(
        mempool_id,
        std::make_unique<PrivatePool>(mempool_id, allocator)
    );
  } else {
    // REUSE EXISTING POOL (pool sharing)
    TORCH_INTERNAL_ASSERT(it->second->use_count > 0);
    TORCH_INTERNAL_ASSERT(allocator == nullptr);
    it->second->use_count++;
  }
}
```

**PrivatePool Structure**:

**Location**: [pytorch/c10/cuda/CUDACachingAllocator.cpp:942-974](pytorch/c10/cuda/CUDACachingAllocator.cpp#L942-L974)

```cpp
struct PrivatePool {
  explicit PrivatePool(MempoolId_t id, CUDAAllocator* allocator = nullptr)
      : id(std::move(id)),
        allocator_(allocator),
        large_blocks(/*small=*/false, this),  // Pool for allocations > 1MB
        small_blocks(/*small=*/true, this) {}  // Pool for allocations <= 1MB

  MempoolId_t id{0, 0};
  int use_count{1};           // Number of graphs using this pool
  int cudaMalloc_count{0};    // Number of unfreed cudaMallocs

  CUDAAllocator* allocator_;
  BlockPool large_blocks;     // Blocks > 1MB
  BlockPool small_blocks;     // Blocks <= 1MB
};
```

**Key Insight**: Each `PrivatePool` has its own `BlockPool` structures (small and large) that are completely separate from the default allocator's pools.

---

### 5. Allocation During Capture

When any CUDA memory allocation happens during capture (e.g., creating a tensor), the allocator redirects it to the private pool.

#### malloc (Main Allocation Function)

**Location**: [pytorch/c10/cuda/CUDACachingAllocator.cpp:1360-1400](pytorch/c10/cuda/CUDACachingAllocator.cpp#L1360-L1400)

```cpp
void malloc(
    void** devPtr,
    c10::DeviceIndex device_id,
    size_t orig_size,
    cudaStream_t stream) {
  // ... context gathering ...

  std::unique_lock<std::recursive_mutex> lock(mutex);

  if (C10_LIKELY(captures_underway.empty())) {
    // NORMAL PATH: No capture happening
    process_events(context);
  } else {
    // CAPTURE PATH: Skip event processing
    // (cudaEventQueries are illegal during capture)
    if (CUDAAllocatorConfig::graph_capture_record_stream_reuse()) {
      free_safe_blocks_in_capture(context, stream);
    }
  }

  size_t size = round_size(orig_size);

  // *** CRITICAL: get_pool checks if we're capturing ***
  auto& pool = get_pool(size, stream);

  const size_t alloc_size = get_allocation_size(size);
  AllocParams params(device_id, size, stream, &pool, alloc_size);
  params.stat_types = get_stat_types_for_pool(pool);

  // Try to get a block from the pool
  bool block_found = get_free_block(params)
      || (trigger_free_memory_callbacks(params) && get_free_block(params));

  // ... rest of allocation logic ...
}
```

#### get_pool (Pool Selection Logic)

**Location**: [pytorch/c10/cuda/CUDACachingAllocator.cpp:2960-2983](pytorch/c10/cuda/CUDACachingAllocator.cpp#L2960-L2983)

```cpp
BlockPool& get_pool(size_t size, cudaStream_t stream) {
  // Fast path: no captures underway
  if (C10_UNLIKELY(!captures_underway.empty())) {
    // Check each active capture
    for (auto& entry : captures_underway) {
      // entry.first = mempool_id
      // entry.second = filter function

      if (entry.second(stream)) {
        // This stream is participating in this capture!
        // Find the private pool
        auto it1 = graph_pools.find(entry.first);
        TORCH_INTERNAL_ASSERT(it1 != graph_pools.end());

        // Return private pool (small or large)
        if (size <= kSmallSize) {
          return it1->second->small_blocks;  // Private small pool
        } else {
          return it1->second->large_blocks;  // Private large pool
        }
      }
    }
  }

  // DEFAULT PATH: Use global pools
  if (size <= kSmallSize) {
    return small_blocks;  // Global small pool
  } else {
    return large_blocks;  // Global large pool
  }
}
```

**The Magic**:
1. Check if any captures are active (`captures_underway`)
2. For each capture, run its filter function with the current stream
3. If filter returns true → use the capture's **private pool**
4. Otherwise → use **global pool**

**Filter Function** (from capture_begin):
```cpp
[this](cudaStream_t stream) {
    cudaStreamCaptureStatus status{};
    CaptureId_t stream_capture_id = 0;
    AT_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, &stream_capture_id));
    return status == cudaStreamCaptureStatus::cudaStreamCaptureStatusActive
           && stream_capture_id == capture_id_;
}
```

This checks:
1. Is the stream currently capturing? (`cudaStreamCaptureStatusActive`)
2. Is it capturing for **this** specific graph? (`stream_capture_id == capture_id_`)

---

### 6. Capture End

**Location**: [pytorch/aten/src/ATen/cuda/CUDAGraph.cpp:118-151](pytorch/aten/src/ATen/cuda/CUDAGraph.cpp#L118-L151)

```cpp
void CUDAGraph::capture_end() {
  auto stream = at::cuda::getCurrentCUDAStream();

  TORCH_CHECK(stream == capture_stream_,
              "Capture must end on the same stream it began on.");

  // End CUDA graph capture
  AT_CUDA_CHECK(cudaStreamEndCapture(capture_stream_, &graph_));

  // Tell allocator we're done capturing
  c10::cuda::CUDACachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);

  TORCH_CHECK(graph_ != nullptr, "Invalid capture.");

  // Finalize generator states
  for (auto& [generator_state, wholegraph_increments] : captured_generator_states_) {
    wholegraph_increments = generator_state->capture_epilogue();
  }

  size_t numCUDAGraphNodes = 0;
  AT_CUDA_CHECK(cudaGraphGetNodes(graph_, nullptr, &numCUDAGraphNodes));
  if (numCUDAGraphNodes == 0) {
      TORCH_WARN("The CUDA Graph is empty...");
  }

  capture_ended_ = true;
  has_graph_ = true;

  // Instantiate immediately if keep_graph=False
  if (!keep_graph_) {
    instantiate();
    if (!_cuda_graphs_debug) {
      AT_CUDA_CHECK(cudaGraphDestroy(graph_));
    }
    has_graph_ = false;
  }
}
```

#### endAllocateToPool

**Location**: [pytorch/c10/cuda/CUDACachingAllocator.cpp:2544-2570](pytorch/c10/cuda/CUDACachingAllocator.cpp#L2544-L2570)

```cpp
void endAllocateToPool(MempoolId_t mempool_id) {
  std::lock_guard<std::recursive_mutex> lock(mutex);

  if (CUDAAllocatorConfig::graph_capture_record_stream_reuse() &&
      !graph_reuse_context.empty()) {
    auto capture_id = mempool_to_capture_id[mempool_id];
    auto graph_context = graph_reuse_context[capture_id];
    for (auto& [stream, _] : graph_context.visited) {
      TORCH_INTERNAL_ASSERT(
          stream_get_capture_info(stream).status ==
              cudaStreamCaptureStatusNone,
          "This stream should not be capturing when the capture is ended");
    }
    graph_reuse_context.erase(capture_id);
    mempool_to_capture_id.erase(mempool_id);
  }

  // Remove this capture from active captures
  for (auto it = captures_underway.begin(); it != captures_underway.end(); ++it) {
    if (it->first == mempool_id) {
      captures_underway.erase(it);
      return;
    }
  }
  TORCH_CHECK(false, "endAllocatePool: not currently recording to mempool_id");
}
```

**Key Action**: Remove the capture from `captures_underway`, so future allocations use global pools again.

---

### 7. Graph Replay

**Location**: [pytorch/aten/src/ATen/cuda/CUDAGraph.cpp:192-219](pytorch/aten/src/ATen/cuda/CUDAGraph.cpp#L192-L219)

```cpp
void CUDAGraph::replay() {
  TORCH_CHECK(capture_ended_,
              "Called CUDAGraph::replay without a preceding successful capture.");

  if (!has_graph_exec_) {
    TORCH_INTERNAL_ASSERT(keep_graph_);
    instantiate();
  }

  c10::OptionalDeviceGuard device_guard{capture_stream_.device()};

  // Update generator states for this replay
  for (auto& [generator_state, wholegraph_increments] : captured_generator_states_) {
    generator_state->replay_prologue(wholegraph_increments);
  }

  // REPLAY THE GRAPH
  AT_CUDA_CHECK(cudaGraphLaunch(graph_exec_, at::cuda::getCurrentCUDAStream()));

  // Workaround for CUDA < 11.4 bug
  int version = 0;
  AT_CUDA_CHECK(cudaDriverGetVersion(&version));
  if (version < 11040) {
    AT_CUDA_CHECK(cudaDeviceSynchronize());
  }
}
```

**During Replay**:
- **NO** allocator calls happen
- **NO** memory management changes
- Only kernels execute, using the exact same memory addresses as during capture
- The private pool remains intact, holding all the memory used by the graph

---

### 8. Graph Cleanup

#### releasePool (Called on graph destruction or reset)

**Location**: [pytorch/c10/cuda/CUDACachingAllocator.cpp:2573-2594](pytorch/c10/cuda/CUDACachingAllocator.cpp#L2573-L2594)

```cpp
void releasePool(MempoolId_t mempool_id) {
  std::lock_guard<std::recursive_mutex> lock(mutex);

  // The cudaGraphExec_t has been destroyed
  // But we can't immediately free the memory because:
  //   1. Other graphs might share this pool
  //   2. User might still hold references to output tensors

  auto pp = get_private_pool(mempool_id);
  auto uc = --(pp->use_count);
  TORCH_INTERNAL_ASSERT(uc >= 0);

  if (uc == 0) {
    // No more graphs using this pool
    // Mark it as freeable (cudaFree can now reclaim blocks)
    bool inserted = graph_pools_freeable.insert({mempool_id, pp}).second;
    TORCH_INTERNAL_ASSERT(inserted);
  }
}
```

**Lifecycle**:
1. Graph created → `use_count = 1`
2. Another graph shares pool → `use_count++`
3. Graph destroyed → `use_count--`
4. When `use_count == 0` → memory can be reclaimed by `free_cached_blocks()`

---

## Frame-by-Frame Trace: make_graphed_callables

This function provides a higher-level API that captures both forward and backward passes, manages memory more automatically, and supports multiple callables sharing a pool.

### 1. Python API Entry

**Location**: [pytorch/torch/cuda/graphs.py:295-367](pytorch/torch/cuda/graphs.py#L295-L367)

```python
def make_graphed_callables(
    callables: Union[_ModuleOrCallable, tuple[_ModuleOrCallable, ...]],
    sample_args: Union[tuple[Tensor, ...], tuple[tuple[Tensor, ...], ...]],
    num_warmup_iters: int = 3,
    allow_unused_input: bool = False,
    pool: Optional[_POOL_HANDLE] = None,
) -> Union[_ModuleOrCallable, tuple[_ModuleOrCallable, ...]]:
    """
    Accept callables (functions or nn.Modules) and return graphed versions.

    Each graphed callable's forward pass runs its source callable's
    forward CUDA work as a CUDA graph inside a single autograd node.
    """
```

#### Example Usage:
```python
module1 = torch.nn.Linear(D_in, H).cuda()
module2 = torch.nn.Linear(H, D_out).cuda()
loss_fn = torch.nn.MSELoss()

# Sample inputs for capture
x = torch.randn(N, D_in, device='cuda')
h = torch.randn(N, H, device='cuda', requires_grad=True)

# Graph the modules
module1 = torch.cuda.make_graphed_callables(module1, (x,))
module2 = torch.cuda.make_graphed_callables(module2, (h,))

# Use normally in training loop
for data in dataloader:
    optimizer.zero_grad()
    tmp = module1(data)      # Runs as a graph
    loss = loss_fn(tmp, target)
    loss.backward()          # Backward also runs as a graph!
    optimizer.step()
```

---

### 2. Setup and Validation

**Location**: [pytorch/torch/cuda/graphs.py:368-417](pytorch/torch/cuda/graphs.py#L368-L417)

```python
# Check autocast compatibility
if torch.is_autocast_enabled() and torch.is_autocast_cache_enabled():
    raise RuntimeError(
        "make_graphed_callables does not support autocast caching. "
        "Please set `cache_enabled=False`."
    )

just_one_callable = False

# Normalize input to always be tuples
if not isinstance(callables, tuple):
    just_one_callable = True
    callables = (callables,)
    _sample_args = (typing.cast(tuple[Tensor, ...], sample_args),)
else:
    _sample_args = typing.cast(tuple[tuple[Tensor, ...], ...], sample_args)

# Validate and prepare
flatten_sample_args = []

for c, args in zip(callables, _sample_args):
    if isinstance(c, torch.nn.Module):
        # Validate no hooks
        assert (
            len(c._backward_hooks) == 0
            and len(c._forward_hooks) == 0
            and len(c._forward_pre_hooks) == 0
        ), "Modules must not have hooks registered..."

        # Validate buffers don't require grad
        assert all(b.requires_grad is False for b in c.buffers()), \
            "Only parameters may be trainable. All buffers must have requires_grad=False."

    # Flatten arguments using pytree
    flatten_arg = torch.utils._pytree.arg_tree_leaves(*args)
    flatten_sample_args.append(tuple(flatten_arg))

    # Validate all args are tensors
    assert all(isinstance(arg, torch.Tensor) for arg in flatten_arg), \
        "sample_args must contain only Tensors."
```

**Validation Rules**:
1. Autocast caching must be disabled
2. Modules cannot have hooks (but can register them *after* graphing)
3. Module buffers must not require gradients (only parameters can)
4. All arguments must be tensors

---

### 3. Determine Input Surfaces

**Location**: [pytorch/torch/cuda/graphs.py:407-417](pytorch/torch/cuda/graphs.py#L407-L417)

```python
# For nn.Module, the graph's full input surface includes:
#   1. User-provided args (sample_args)
#   2. Module parameters
per_callable_len_user_args = [len(args) for args in flatten_sample_args]

per_callable_module_params = [
    tuple(c.parameters()) if isinstance(c, torch.nn.Module) else ()
    for c in callables
]

per_callable_static_input_surfaces = [
    flatten_sample_args[i] + per_callable_module_params[i]
    for i in range(len(callables))
]
```

**Key Concept**: For modules, the "input surface" includes both:
- Explicit args passed to `forward()`
- Model parameters (since they're accessed during forward)

---

### 4. Create Graph Objects and Memory Pool

**Location**: [pytorch/torch/cuda/graphs.py:419-422](pytorch/torch/cuda/graphs.py#L419-L422)

```python
# Create separate CUDAGraph objects for each callable's fwd and bwd
fwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(callables))]
bwd_graphs = [torch.cuda.CUDAGraph() for _ in range(len(callables))]

# Create shared memory pool
mempool = graph_pool_handle() if pool is None else pool
```

**Memory Sharing Strategy**:
- If `pool=None`: Create a **new** pool that all callables will share
- If `pool` provided: Use **existing** pool (share with other graphs)

**Why share?** If callables always run in the same order, they can safely reuse the same memory pool, reducing total memory usage.

---

### 5. Warmup Phase

**Location**: [pytorch/torch/cuda/graphs.py:424-451](pytorch/torch/cuda/graphs.py#L424-L451)

```python
torch.cuda.synchronize()

# Warmup on a side stream
with torch.cuda.stream(torch.cuda.Stream()):
    for func, args, static_input_surface in zip(
        callables, _sample_args, per_callable_static_input_surfaces
    ):
        grad_inputs, outputs, outputs_grad = None, None, None

        # Run num_warmup_iters times
        for _ in range(num_warmup_iters):
            # Forward pass
            outputs = torch.utils._pytree.tree_leaves(func(*args))
            outputs_grad = tuple(o for o in outputs if o.requires_grad)

            # Backward pass (if any outputs require grad)
            if len(outputs_grad) > 0:
                grad_inputs = torch.autograd.grad(
                    outputs=outputs_grad,
                    inputs=tuple(i for i in static_input_surface if i.requires_grad),
                    grad_outputs=tuple(torch.empty_like(o) for o in outputs if o.requires_grad),
                    only_inputs=True,
                    allow_unused=allow_unused_input,
                )

        # Clean up intermediate results
        for v in [outputs, outputs_grad, grad_inputs]:
            del v

torch.cuda.synchronize()
```

**Purpose of Warmup**:
1. **CuDNN benchmarking**: Let CuDNN find the fastest algorithms
2. **Lazy initialization**: Initialize any CUDA resources
3. **Memory layout**: Establish stable memory access patterns

**Why side stream?** Capture must happen on a non-default stream, and warmup should match capture conditions.

---

### 6. Capture Forward Passes

**Location**: [pytorch/torch/cuda/graphs.py:453-466](pytorch/torch/cuda/graphs.py#L453-L466)

```python
# All captures share the same mempool
# Capture order: fwd1, fwd2, ..., fwdN, then bwdN, ..., bwd1

per_callable_static_outputs = []
per_callable_output_unflatten_spec = []

for func, args, fwd_graph in zip(callables, _sample_args, fwd_graphs):
    # Capture forward pass
    with torch.cuda.graph(fwd_graph, pool=mempool):
        func_outputs = func(*args)

    # Store static outputs (memory addresses frozen)
    flatten_outputs, spec = torch.utils._pytree.tree_flatten(func_outputs)
    per_callable_static_outputs.append(tuple(flatten_outputs))
    per_callable_output_unflatten_spec.append(spec)
```

**Key Points**:
- All forward captures use `pool=mempool` (shared pool)
- Outputs are stored as "static" tensors (fixed memory addresses)
- PyTree flattening handles complex output structures (tuples, dicts, etc.)

---

### 7. Capture Backward Passes

**Location**: [pytorch/torch/cuda/graphs.py:468-512](pytorch/torch/cuda/graphs.py#L468-L512)

```python
# Capture backward graphs in REVERSE order
per_callable_static_grad_outputs = []
per_callable_static_grad_inputs = []

for static_input_surface, static_outputs, bwd_graph in zip(
    reversed(per_callable_static_input_surfaces),
    reversed(per_callable_static_outputs),
    reversed(bwd_graphs),
):
    # Allocate static gradient placeholders
    static_grad_outputs = tuple(
        torch.empty_like(o) if o.requires_grad else None
        for o in static_outputs
    )

    outputs_grad = tuple(o for o in static_outputs if o.requires_grad)
    grad_inputs = None

    if len(outputs_grad) > 0:
        # Capture backward pass
        with torch.cuda.graph(bwd_graph, pool=mempool):
            grad_inputs = torch.autograd.grad(
                outputs=outputs_grad,
                inputs=tuple(i for i in static_input_surface if i.requires_grad),
                grad_outputs=tuple(o for o in static_grad_outputs if o is not None),
                only_inputs=True,
                allow_unused=allow_unused_input,
            )

    # Pad grad_inputs with None for non-differentiable inputs
    static_grad_inputs = []
    grad_idx = 0
    for arg in static_input_surface:
        if arg.requires_grad and grad_inputs is not None:
            static_grad_inputs.append(grad_inputs[grad_idx])
            grad_idx += 1
        else:
            static_grad_inputs.append(None)
    static_grad_inputs = tuple(static_grad_inputs)

    per_callable_static_grad_outputs.append(static_grad_outputs)
    per_callable_static_grad_inputs.append(static_grad_inputs)

# Reverse back to match forward order
per_callable_static_grad_outputs.reverse()
per_callable_static_grad_inputs.reverse()
```

**Reverse Order**: Backward captures happen in reverse because that's the order they'll execute during training (last forward's backward runs first).

**Memory Sharing**: Because we capture in execution order (fwd1, fwd2, ..., bwdN, bwd2, bwd1), memory used by fwd1 can be reused by bwd1 if fwd1's outputs are no longer live.

---

### 8. Create Autograd Function Wrapper

**Location**: [pytorch/torch/cuda/graphs.py:515-567](pytorch/torch/cuda/graphs.py#L515-L567)

```python
def make_graphed_autograd_function(
    fwd_graph: CUDAGraph,
    bwd_graph: CUDAGraph,
    module_params: tuple[torch.nn.Parameter, ...],
    len_user_args: int,
    output_unflatten_spec: torch.utils._pytree.TreeSpec,
    static_input_surface: tuple[Tensor, ...],
    static_outputs: tuple[Tensor, ...],
    static_grad_outputs: tuple[Optional[Tensor], ...],
    static_grad_inputs: tuple[Tensor, ...],
) -> Callable[..., object]:

    class Graphed(torch.autograd.Function):
        @staticmethod
        def forward(ctx: object, *inputs: Tensor) -> tuple[Tensor, ...]:
            # Copy new inputs to static memory locations
            for i in range(len_user_args):
                if static_input_surface[i].data_ptr() != inputs[i].data_ptr():
                    static_input_surface[i].copy_(inputs[i])

            # Replay forward graph
            fwd_graph.replay()

            # Return detached static outputs
            return tuple(o.detach() for o in static_outputs)

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx: object, *grads: Tensor) -> tuple[Tensor, ...]:
            # Copy incoming gradients to static memory
            for g, grad in zip(static_grad_outputs, grads):
                if g is not None:
                    if g.data_ptr() != grad.data_ptr():
                        g.copy_(grad)

            # Replay backward graph
            bwd_graph.replay()

            # Return detached static grad inputs
            return tuple(
                b.detach() if b is not None else b
                for b in static_grad_inputs
            )

    def functionalized(*user_args: object) -> object:
        # Run autograd function with user args + module params
        flatten_user_args = torch.utils._pytree.arg_tree_leaves(*user_args)
        out = Graphed.apply(*(tuple(flatten_user_args) + module_params))
        return torch.utils._pytree.tree_unflatten(out, output_unflatten_spec)

    return functionalized
```

**How It Works**:

1. **Forward**:
   - Copy new input data to static input tensors (fixed addresses)
   - Replay forward graph
   - Return outputs (static tensors, just detached)

2. **Backward**:
   - Copy incoming gradients to static grad output tensors
   - Replay backward graph
   - Return gradients w.r.t. inputs

**Copy Optimization**: The code checks `data_ptr()` to avoid unnecessary copies if autograd happens to use the same memory location.

---

### 9. Wrap Modules or Return Functions

**Location**: [pytorch/torch/cuda/graphs.py:569-612](pytorch/torch/cuda/graphs.py#L569-L612)

```python
ret: list[_ModuleOrCallable] = []

for i, func in enumerate(callables):
    graphed = make_graphed_autograd_function(
        fwd_graphs[i],
        bwd_graphs[i],
        per_callable_module_params[i],
        per_callable_len_user_args[i],
        per_callable_output_unflatten_spec[i],
        per_callable_static_input_surfaces[i],
        per_callable_static_outputs[i],
        per_callable_static_grad_outputs[i],
        per_callable_static_grad_inputs[i],
    )

    if isinstance(func, torch.nn.Module):
        # For modules, replace the forward method
        def make_graphed_forward(
            func: torch.nn.Module,
            graph_training_state: bool,
            graphed: Callable[_P, _R],
            orig_fwd: Callable[_P, _R],
        ) -> Callable[_P, _R]:
            def new_fwd(*user_args: _P.args, **user_kwargs: _P.kwargs) -> _R:
                # Only use graph if training state matches
                if func.training == graph_training_state:
                    return graphed(*user_args, **user_kwargs)
                else:
                    return orig_fwd(*user_args, **user_kwargs)
            return new_fwd

        func.forward = make_graphed_forward(
            func, func.training, graphed, func.forward
        )
        ret.append(func)
    else:
        # For functions, just return the graphed version
        ret.append(graphed)

if just_one_callable:
    return ret[0]

return tuple(ret)
```

**Training/Eval Handling**: For modules, the graphed forward only runs when the module's training state matches what was captured. If you switch between train/eval mode, it falls back to the original forward.

---

## Key Data Structures

### 1. MempoolId_t

**Definition**: [pytorch/c10/cuda/CUDAGraphsC10Utils.h](pytorch/c10/cuda/CUDAGraphsC10Utils.h)

```cpp
using MempoolId_t = std::pair<int, int>;
```

**Format**:
- `{uuid, 0}`: Created by `CUDAGraph.capture_begin()` (automatic)
- `{0, uid}`: Created by `graph_pool_handle()` (user-requested)

**Purpose**: Uniquely identifies a private memory pool. The format distinguishes between:
- Graphs that create their own pool (first component non-zero)
- User-created pool handles for explicit sharing (second component non-zero)

### 2. PrivatePool

```cpp
struct PrivatePool {
  MempoolId_t id{0, 0};
  int use_count{1};           // Number of graphs using this pool
  int cudaMalloc_count{0};    // Number of unfreed cudaMallocs
  CUDAAllocator* allocator_;
  BlockPool large_blocks;     // Pool for allocations > 1MB
  BlockPool small_blocks;     // Pool for allocations <= 1MB
};
```

**Lifecycle**:
1. Created when first graph captures with a new mempool_id
2. `use_count` incremented when another graph shares the pool
3. `use_count` decremented when a graph is destroyed
4. When `use_count` reaches 0, moved to `graph_pools_freeable`
5. Memory eventually reclaimed by `free_cached_blocks()`

### 3. BlockPool

```cpp
struct BlockPool {
  std::set<Block*, Comparison> blocks;      // Free blocks, sorted by size
  std::set<Block*, Comparison> unmapped;    // Unmapped blocks (expandable segments)
  const bool is_small;                       // True if this is the small pool
  PrivatePool* owner_PrivatePool;           // Non-null if this is a private pool
  int64_t get_free_blocks_call_count{0};    // For garbage collection heuristics
};
```

**Key Methods**:
- `insert_into_blocks()`: Add a free block (with GC tracking)
- `owner_MempoolId()`: Get the owning pool's ID (or {0,0} for global pool)

### 4. Block

```cpp
struct Block {
  c10::DeviceIndex device;
  cudaStream_t stream;              // Allocation stream
  stream_set stream_uses;           // Streams that have used this block
  size_t size;                      // Block size in bytes
  size_t requested_size;            // Original request size
  BlockPool* pool;                  // Owning pool
  void* ptr;                        // CUDA memory address
  bool allocated;                   // In-use flag
  bool mapped;                      // Is memory physically mapped?
  Block* prev;                      // Previous block if split
  Block* next;                      // Next block if split
  int event_count;                  // Outstanding CUDA events
  int64_t gc_count_base;            // GC counter baseline
  std::shared_ptr<GatheredContext> context_when_allocated;
  std::shared_ptr<GatheredContext> context_when_segment_allocated;
  ExpandableSegment* expandable_segment_;
};
```

### 5. captures_underway

```cpp
std::vector<std::pair<MempoolId_t, std::function<bool(cudaStream_t)>>>
    captures_underway;
```

**Purpose**: Tracks all active graph captures. Each entry contains:
- **MempoolId_t**: Which private pool to use
- **Filter function**: Returns true if a stream is participating in this capture

**Checked on every allocation** to determine if we should use a private pool.

---

## Memory Pool Lifecycle

### Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Graph Capture Begin                                          │
│    g.capture_begin(pool=...)                                    │
│      ├─> Create/reuse MempoolId_t                              │
│      ├─> beginAllocateToPool(mempool_id, filter)               │
│      │     ├─> create_or_incref_pool(mempool_id)              │
│      │     │     ├─> graph_pools.emplace(id, new PrivatePool) │
│      │     │     │   (or increment use_count if exists)        │
│      │     │     └─> PrivatePool.use_count = 1               │
│      │     └─> captures_underway.push_back({id, filter})      │
│      └─> cudaStreamBeginCapture(stream, mode)                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Allocations During Capture                                   │
│    Any tensor creation or CUDA malloc                           │
│      └─> malloc(ptr, device, size, stream)                     │
│            ├─> get_pool(size, stream)                          │
│            │     ├─> Check captures_underway                   │
│            │     ├─> Run filter(stream) for each capture       │
│            │     ├─> If match: return PrivatePool->small/large │
│            │     └─> Else: return global small/large_blocks    │
│            ├─> get_free_block(params) from the selected pool   │
│            │   OR alloc_block() if no free block available     │
│            └─> Memory allocated from private pool!             │
│                                                                 │
│    All allocations during capture go to PrivatePool            │
│    Memory addresses are recorded by CUDA graph                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Capture End                                                  │
│    g.capture_end()                                              │
│      ├─> cudaStreamEndCapture(stream, &graph)                  │
│      ├─> endAllocateToPool(mempool_id)                         │
│      │     └─> captures_underway.erase(mempool_id)            │
│      │         (Future allocations use global pools again)     │
│      ├─> instantiate() → cudaGraphInstantiate(&graph_exec, ..)│
│      └─> Graph now owns cudaGraphExec_t                        │
│                                                                 │
│    PrivatePool remains alive, holding all captured memory      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Graph Replay (can repeat many times)                         │
│    g.replay()                                                   │
│      └─> cudaGraphLaunch(graph_exec, stream)                   │
│            ├─> Kernels execute with captured memory addresses  │
│            ├─> NO allocator involvement                        │
│            └─> PrivatePool memory addresses reused             │
│                                                                 │
│    Memory pool unchanged during replay!                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. Graph Destruction or Pool Sharing                            │
│    g.reset() or ~CUDAGraph()                                    │
│      └─> releasePool(mempool_id)                               │
│            ├─> PrivatePool->use_count--                        │
│            └─> if use_count == 0:                              │
│                  ├─> graph_pools_freeable.insert({id, pool})  │
│                  └─> free_cached_blocks() can now reclaim     │
│                                                                 │
│    If another graph shares the pool, use_count > 0             │
│    Memory only freed when ALL users are done                   │
└─────────────────────────────────────────────────────────────────┘
```

### Pool Sharing Example

```python
g1 = torch.cuda.CUDAGraph()
g2 = torch.cuda.CUDAGraph()

# Capture g1 (creates new pool)
with torch.cuda.graph(g1):
    out1 = workload1(in1)
# g1's pool use_count = 1

# Capture g2, sharing g1's pool
with torch.cuda.graph(g2, pool=g1.pool()):
    out2 = workload2(in2)
# Same pool use_count = 2

# Replay in order
g1.replay()
g2.replay()
# Safe because execution order matches capture order

# Cleanup
del g1  # use_count = 2 -> 1
del g2  # use_count = 1 -> 0
        # Pool moved to freeable
        # Memory can be reclaimed
```

---

## Design Rationale

### Why Separate Memory Pools?

**Problem**: CUDA graphs require stable memory addresses across replays.

**Without Private Pools**:
```
Time:      Capture              Replay 1             Replay 2
──────────────────────────────────────────────────────────────
Tensor A:  addr 0x1000          addr 0x1000          ???
           (captured)           (works)              (different tensor
                                                      might use 0x1000)
```

If eager execution allocates and frees memory between replays, addresses can change, causing:
1. **Segfaults** (accessing freed memory)
2. **Data corruption** (overwriting another tensor's data)
3. **Incorrect results** (reading wrong data)

**With Private Pools**:
```
Private Pool:  Reserved for graph, never reused by eager
Global Pool:   Used by eager execution

Time:      Capture              Replay 1             Replay 2
──────────────────────────────────────────────────────────────
Graph:     addr 0x1000          addr 0x1000          addr 0x1000
           (captured in         (stable)             (stable)
            private pool)

Eager:     addr 0x5000          addr 0x5000          addr 0x5000
           (uses global pool)   (independent)        (independent)
```

### Memory Overhead Trade-off

**Cost**: Private pools reserve memory even when graphs aren't replaying.

```
┌─────────────────────────────┐
│  Total GPU Memory           │
│                             │
│  ┌──────────────────────┐  │
│  │  Global Pool         │  │  ← Eager execution
│  │  (eager tensors)     │  │
│  └──────────────────────┘  │
│                             │
│  ┌──────────────────────┐  │
│  │  Private Pool 1      │  │  ← Graph 1
│  │  (graph 1 memory)    │  │
│  └──────────────────────┘  │
│                             │
│  ┌──────────────────────┐  │
│  │  Private Pool 2      │  │  ← Graph 2
│  │  (graph 2 memory)    │  │
│  └──────────────────────┘  │
└─────────────────────────────┘
```

**Benefit**: Safety and correctness. The memory usage increase is typically acceptable because:
1. Graphs are most useful when memory is NOT the bottleneck
2. Pool sharing mitigates overhead for multiple graphs
3. Alternative (no graphs) has higher CPU overhead

### Why Capture on Side Stream?

**CUDA Requirement**: Default stream has implicit synchronization semantics that interfere with graph capture.

**Side Stream Benefits**:
1. Isolated from main execution stream
2. Can capture async operations correctly
3. Prevents accidental synchronization during capture

**Replay Flexibility**: Graph can replay on ANY stream (including default).

### Allocation Tracking During Capture

The allocator tracks **exactly** what was allocated during capture:

```python
# During capture:
x = torch.randn(10, device='cuda')  # Allocated from private pool
y = x * 2                           # Allocated from private pool
del x                               # Freed in private pool
# Final state: only 'y' is live

# During replay:
# Replay uses same addresses:
#   - Allocates to same address as 'x' (then "frees" it)
#   - Allocates to same address as 'y'
#   - 'y' remains live after replay
```

The allocator's tracking ensures that:
1. Allocations happen in the same order
2. Frees happen in the same order
3. Memory layout is identical between capture and replay

### Pool Sharing Safety Rules

**Safe to share when**:
1. Graphs replay in the **same order** as captured
2. No **concurrent** replays of graphs sharing a pool

**Example of UNSAFE sharing**:
```python
g1 = torch.cuda.CUDAGraph()
g2 = torch.cuda.CUDAGraph()

with torch.cuda.graph(g1):
    out1 = model1(in1)

with torch.cuda.graph(g2, pool=g1.pool()):
    out2 = model2(in2)

# UNSAFE: Reverse replay order
g2.replay()  # Uses memory from pool
g1.replay()  # Overwrites g2's output!
```

**Safe pattern**:
```python
# Always replay in capture order
g1.replay()
result1 = out1.clone()  # Clone if you need to preserve

g2.replay()
result2 = out2.clone()
```

---

## Summary

### Key Takeaways

1. **Private Memory Pools**: Each graph capture creates or reuses a separate memory pool that persists across replays, ensuring stable addresses.

2. **Allocation Redirection**: During capture, the allocator checks `captures_underway` and routes allocations to the appropriate private pool based on the stream's capture state.

3. **Replay Efficiency**: Graph replay involves **no allocator calls**—only kernel execution. This dramatically reduces CPU overhead.

4. **make_graphed_callables**: Provides an autograd-aware wrapper that:
   - Captures both forward and backward passes
   - Manages static tensors automatically
   - Handles module parameters correctly
   - Supports pool sharing across multiple callables

5. **Memory vs. Performance Trade-off**: Private pools increase memory usage (keeping captured memory reserved) but enable significant performance gains by reducing CPU overhead and kernel launch latency.

6. **Safety Guarantees**: The separate pool design prevents memory corruption and ensures correct numerics on replay.

### Performance Implications

**Benefits**:
- **Reduced CPU overhead**: Single `cudaGraphLaunch` vs. thousands of kernel launches
- **Reduced kernel launch latency**: GPU-side synchronization only
- **Better GPU utilization**: More time executing kernels, less time waiting for CPU

**Costs**:
- **Increased memory**: Private pools remain allocated while graphs exist
- **Inflexibility**: Can't change sizes, control flow, or memory addresses
- **Capture overhead**: Initial capture slower than eager due to tracking

**Best Use Cases**:
- Inference with fixed batch sizes
- Training with static architectures
- CPU-bound models (where kernel launch overhead dominates)
- Repeated execution of identical operations

### Implementation Highlights

**Python Layer** ([graphs.py](pytorch/torch/cuda/graphs.py)):
- `graph` context manager: User-friendly API
- `make_graphed_callables`: Autograd integration
- `CUDAGraph` class: Thin Python wrapper around C++ class

**C++ Layer** ([CUDAGraph.cpp](pytorch/aten/src/ATen/cuda/CUDAGraph.cpp)):
- `capture_begin()`: Initiates capture, creates pool ID, registers with allocator
- `capture_end()`: Ends capture, instantiates graph
- `replay()`: Launches graph
- `pool()`: Returns pool ID for sharing

**Allocator Layer** ([CUDACachingAllocator.cpp](pytorch/c10/cuda/CUDACachingAllocator.cpp)):
- `beginAllocateToPool()`: Registers capture, creates private pool
- `get_pool()`: Routes allocations to private or global pool
- `endAllocateToPool()`: Deregisters capture
- `releasePool()`: Manages pool lifecycle

This architecture cleanly separates concerns:
- Python: User-facing API and convenience
- C++: CUDA graph management and integration
- Allocator: Memory pool management and allocation routing

---

## Code References

### Python Files
- [torch/cuda/graphs.py](pytorch/torch/cuda/graphs.py)
  - Line 185-270: `graph` context manager
  - Line 295-612: `make_graphed_callables` implementation

### C++ Files
- [aten/src/ATen/cuda/CUDAGraph.cpp](pytorch/aten/src/ATen/cuda/CUDAGraph.cpp)
  - Line 58-116: `capture_begin()`
  - Line 118-151: `capture_end()`
  - Line 192-219: `replay()`

- [aten/src/ATen/cuda/CUDAGraph.h](pytorch/aten/src/ATen/cuda/CUDAGraph.h)
  - Line 22-92: `CUDAGraph` class definition

- [c10/cuda/CUDACachingAllocator.cpp](pytorch/c10/cuda/CUDACachingAllocator.cpp)
  - Line 942-974: `PrivatePool` structure
  - Line 2529-2541: `beginAllocateToPool()`
  - Line 2544-2570: `endAllocateToPool()`
  - Line 2573-2594: `releasePool()`
  - Line 2670-2689: `create_or_incref_pool()`
  - Line 2960-2983: `get_pool()` (allocation routing)

### Documentation Files
- [docs/source/notes/cuda.rst](pytorch/docs/source/notes/cuda.rst)
  - Line 1289-1747: CUDA Graphs documentation

- [docs/source/torch.compiler_cudagraph_trees.md](pytorch/docs/source/torch.compiler_cudagraph_trees.md)
  - Complete guide to CUDAGraph Trees for torch.compile

---

*This documentation was created by tracing the PyTorch source code from Python entry points through C++ implementation down to CUDA driver API calls.*
