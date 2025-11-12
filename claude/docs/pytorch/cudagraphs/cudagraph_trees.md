# PyTorch CUDAGraph Trees: torch.compile with "reduce-overhead" Mode

## Table of Contents
1. [Overview](#overview)
2. [Entry Point: torch.compile](#entry-point-torchcompile)
3. [Dynamo to Inductor Pipeline](#dynamo-to-inductor-pipeline)
4. [Inductor Compilation with CUDAGraphs](#inductor-compilation-with-cudagraphs)
5. [CUDAGraph Trees Implementation](#cudagraph-trees-implementation)
6. [Memory Pool Management](#memory-pool-management)
7. [Recording and Replay](#recording-and-replay)
8. [Graph Tree Structure](#graph-tree-structure)
9. [Checkpointing and Branching](#checkpointing-and-branching)
10. [Complete Example Trace](#complete-example-trace)

---

## Overview

**CUDAGraph Trees** is PyTorch's advanced memory management system for CUDA graphs that extends the basic `torch.cuda.CUDAGraph` API to support:

1. **Dynamic execution paths**: Unlike basic CUDA graphs that must replay in the same order they were captured, CUDA graph trees support branching (e.g., `if` statements with different branches)
2. **Memory pool sharing across graphs**: All graphs in a tree share a single memory pool, dramatically reducing memory overhead
3. **Live tensor tracking**: Automatically tracks which tensors from previous graphs are still alive and prevents memory overwrites
4. **Integration with torch.compile**: Seamlessly works with Dynamo and Inductor to cudagraph entire models

### Key Innovation

The core innovation is **checkpointing the allocator state**. When you replay a CUDA graph, the GPU executes kernels but the CPU-side allocator state doesn't update. CUDAGraph Trees checkpoint the allocator state after each graph recording, allowing us to:
- Resume recording new graphs after replaying old ones
- Build tree structures where each path represents a different execution trace
- Share memory efficiently across all paths in the tree

**From the documentation**:
> "CUDA graph trees are flexible enough to be used in Dynamo across graph breaks, which is their primary use case."

---

## Entry Point: torch.compile

### User Code
```python
import torch

@torch.compile(mode="reduce-overhead")
def my_model(x):
    y = x * x * x
    if y.sum() > 0:
        return y + 10
    else:
        return y - 10

input_tensor = torch.randn(10, 10, device="cuda")
output = my_model(input_tensor)  # First run: Dynamo tracing
output = my_model(input_tensor)  # Second run: Inductor compilation + warmup
output = my_model(input_tensor)  # Third run: CUDAGraph recording
output = my_model(input_tensor)  # Fourth+ run: CUDAGraph replay (fast!)
```

### Mode Configuration

**Location**: [pytorch/torch/_inductor/__init__.py:357-359](pytorch/torch/_inductor/__init__.py#L357-L359)

```python
mode_options: dict[str, dict[str, bool]] = {
    "default": {},
    "lite": lite_mode_options,
    "reduce-overhead": {
        "triton.cudagraphs": True,  # ← ENABLES CUDAGRAPHS
    },
    "max-autotune-no-cudagraphs": {
        "max_autotune": True,
        "coordinate_descent_tuning": True,
    },
    "max-autotune": {
        "max_autotune": True,
        "triton.cudagraphs": True,
        "coordinate_descent_tuning": True,
    },
}
```

**Effect**: Setting `mode="reduce-overhead"` automatically sets:
```python
config.triton.cudagraphs = True
config.triton.cudagraph_trees = True  # (default, see config.py:1356)
```

---

## Dynamo to Inductor Pipeline

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. User calls @torch.compile(mode="reduce-overhead")(func)         │
│    torch.__init__.py creates _TorchCompileInductorWrapper          │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 2. TorchDynamo (torch._dynamo)                                      │
│    - Trace Python bytecode                                          │
│    - Capture FX graph                                               │
│    - Handle graph breaks (control flow, etc.)                       │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 3. AOTAutograd (torch._functorch.aot_autograd)                      │
│    - Partition into forward/backward graphs                         │
│    - Generate autograd metadata                                     │
│    - Pass graphs to compiler backend                                │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 4. TorchInductor (torch._inductor)                                  │
│    - Optimize FX graph                                              │
│    - Generate Triton/C++ code                                       │
│    - Wrap in CUDAGraph if config.triton.cudagraphs == True         │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 5. Compiled Function with CUDAGraph Trees                           │
│    - Returns callable that manages graph recording/replay           │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Files in Pipeline

1. **torch/__init__.py**: Defines `_TorchCompileInductorWrapper` and applies mode settings
2. **torch/_dynamo**: Bytecode tracing and FX graph generation
3. **torch/_functorch/aot_autograd**: Forward/backward partitioning
4. **torch/_inductor/compile_fx.py**: Main compilation entry point
5. **torch/_inductor/cudagraph_trees.py**: CUDAGraph Trees implementation

---

## Inductor Compilation with CUDAGraphs

### compile_fx_inner: Main Entry Point

**Location**: [pytorch/torch/_inductor/compile_fx.py](pytorch/torch/_inductor/compile_fx.py)

The main compilation function in Inductor is `compile_fx_inner`, which:
1. Optimizes the FX graph
2. Generates kernel code (Triton/C++)
3. **Wraps the compiled function with `cudagraphify` if enabled**

### cudagraphify Wrapper

**Location**: [pytorch/torch/_inductor/compile_fx.py:1789-1830](pytorch/torch/_inductor/compile_fx.py#L1789-L1830)

```python
def cudagraphify(
    model: Callable[..., Any],
    static_input_idxs: Sequence[int] = (),
    *,
    device_index: int,
    stack_traces: list[Optional[str]],
    is_backward: bool,
    is_inference: bool,
    constants: tuple[torch.Tensor, ...] = (),
    placeholders: Sequence[PlaceholderInfo] = (),
    mutated_input_idxs: tuple[int, ...] = (),
) -> Callable[..., Any]:
    from torch._inductor.cudagraph_trees import (
        cudagraphify_impl as new_cudagraphify_impl,
    )

    cudagraphify_fn: Callable[..., Any]
    if config.triton.cudagraph_trees:
        # USE CUDA GRAPH TREES
        cudagraphify_fn = functools.partial(
            new_cudagraphify_impl,
            device_index=device_index,
            stack_traces=stack_traces,
            is_backward=is_backward,
            is_inference=is_inference,
            constants=constants,
            placeholders=placeholders,
            mutated_input_idxs=mutated_input_idxs,
            compile_id=torch._guards.CompileContext.current_compile_id(),
        )
    else:
        # USE BASIC CUDAGRAPH (no trees)
        cudagraphify_fn = cudagraphify_impl

    compiled_fn = None

    def run(new_inputs: Sequence[InputType]) -> Any:
        nonlocal compiled_fn
        if compiled_fn is None:
            # LAZY INITIALIZATION: First call records the graph
            with dynamo_utils.preserve_rng_state():
                compiled_fn = cudagraphify_fn(model, new_inputs, static_input_idxs)
        return compiled_fn(new_inputs)

    return run
```

**Key Points**:
1. **Lazy recording**: The graph isn't recorded until the first actual invocation
2. **Preserves RNG state**: Ensures deterministic execution
3. **Routes to trees or basic**: Based on `config.triton.cudagraph_trees`

### Basic vs. Trees Implementation

#### Basic CUDAGraph (cudagraphify_impl)

**Location**: [pytorch/torch/_inductor/compile_fx.py:1851-1929](pytorch/torch/_inductor/compile_fx.py#L1851-L1929)

```python
def cudagraphify_impl(
    model: Callable[..., Any],
    inputs: list[torch.Tensor],
    static_input_idxs: Sequence[int] = (),
) -> Callable[[list[InputType]], Any]:
    """
    Basic CUDA graph implementation (no trees)
    Assumes inputs[static_input_idxs[i]] are always the same memory address
    """
    # 1. Allocate static tensor inputs
    static_inputs = [
        x if not isinstance(x, torch.Tensor)
        else static_input(x) if idx not in static_input_idxs
        else x.detach()
        for idx, x in enumerate(inputs)
    ]

    # 2. Warmup run (initializes CuDNN, etc.)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        model(list(static_inputs))
    stream.synchronize()
    torch.cuda.synchronize()

    # 3. Record CUDA graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
        static_outputs = model(list(static_inputs))

    # 4. Return replay function
    def run(new_inputs: list[InputType]) -> Any:
        # Copy new input data to static locations
        for idx, (dst, src) in enumerate(zip(static_inputs, new_inputs)):
            if isinstance(dst, torch.Tensor):
                if idx in static_input_idxs:
                    assert dst.data_ptr() == src.data_ptr()
                else:
                    dst.copy_(src)
        new_inputs.clear()
        graph.replay()  # Replay the captured graph
        return static_outputs

    return run
```

**Limitations**:
- Can't handle branching (different execution paths)
- Can't track live tensors from previous graphs
- Each graph uses its own memory pool (high memory usage with multiple graphs)

#### CUDAGraph Trees (new_cudagraphify_impl)

**Location**: [pytorch/torch/_inductor/cudagraph_trees.py:361-415](pytorch/torch/_inductor/cudagraph_trees.py#L361-L415)

```python
def cudagraphify_impl(
    model: ModelType,
    inputs: list[InputType],
    static_input_idxs: Sequence[int],
    *args: Any,
    **kwargs: Any,
) -> ModelType:
    fn_cache: dict[tuple[int, ...], Callable[..., Any]] = {}

    # Detect int inputs (symints for dynamic shapes)
    int_key = [i for i, v in enumerate(inputs) if isinstance(v, int)]
    get_ints: Any = operator.itemgetter(*int_key) if int_key else lambda _: None

    has_warn = False
    del inputs

    def deferred_cudagraphify(inputs: list[InputType]) -> OutputType:
        nonlocal has_warn

        # Extract int key for caching (e.g., dynamic shapes)
        int_key = get_ints(inputs)

        # Check if we should use CUDAGraphs for this shape
        if not is_cudagraph_capture_sizes(int_key):
            return model(inputs)

        # Check cache for existing graph
        fn = fn_cache.get(int_key)
        if fn is not None:
            return fn(inputs)

        # RECORD NEW GRAPH
        log.info("recording cudagraph tree for symint key %s", int_key)

        # Handle unaligned inputs
        check_input_idxs = get_input_idxs_to_check(inputs, static_input_idxs)
        new_static_input_idxs = remove_unaligned_input_idxs(inputs, static_input_idxs)
        copy_misaligned_inputs(inputs, check_input_idxs)

        # *** CALL INTO CUDAGRAPH TREE MANAGER ***
        fn, out = cudagraphify(model, inputs, new_static_input_idxs, *args, **kwargs)

        # Align inputs for subsequent calls
        mutated_input_idxs: OrderedSet[int] = OrderedSet()
        fn = align_inputs_from_check_idxs(
            fn, inputs_to_check=check_input_idxs, mutated_input_idxs=mutated_input_idxs
        )

        # Cache the function
        fn_cache[int_key] = fn

        return out

    return deferred_cudagraphify
```

**Key Difference**: Instead of recording directly, this:
1. **Caches graphs per shape** (`int_key` from dynamic shapes)
2. **Delegates to tree manager** via `cudagraphify()`
3. **Handles alignment** for unaligned inputs

---

## CUDAGraph Trees Implementation

### Core Components

**Location**: [pytorch/torch/_inductor/cudagraph_trees.py](pytorch/torch/_inductor/cudagraph_trees.py)

#### 1. TreeManagerContainer

**Location**: [pytorch/torch/_inductor/cudagraph_trees.py:184-276](pytorch/torch/_inductor/cudagraph_trees.py#L184-L276)

```python
class TreeManagerContainer:
    """
    Manages the lifetime of the tree manager. There is one per device.

    Lifecycle:
    1. Tree manager is fetched → allocated
    2. Functions are generated → add_strong_reference()
    3. Functions die → finalize_reference()
    4. All functions dead → finalize_tree_manager()
    """

    def __init__(self, device_index: int) -> None:
        self.tree_manager: Optional[CUDAGraphTreeManager] = None
        self.live_cudagraphify_fns = 0
        self.device_index = device_index
        self.live_storages_count = 0
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.lock = threading.Lock()

    def get_tree_manager(self) -> CUDAGraphTreeManager:
        with self.lock:
            if self.tree_manager is None:
                self.tree_manager = CUDAGraphTreeManager(self.device_index)
            return self.tree_manager
```

**Purpose**:
- One container per CUDA device
- Keeps tree manager alive while any compiled functions exist
- Thread-safe access via locks

#### 2. CUDAGraphTreeManager

**Location**: [pytorch/torch/_inductor/cudagraph_trees.py](pytorch/torch/_inductor/cudagraph_trees.py) (class definition spans ~2000 lines)

```python
class CUDAGraphTreeManager:
    """
    Manages a tree of CUDAGraphNode objects.

    Key responsibilities:
    - Track execution path through the tree
    - Create new nodes for new execution paths
    - Manage memory pool checkpointing
    - Track live tensors across graph boundaries
    """

    def __init__(self, device_index: int):
        self.device_index = device_index
        self.roots: list[CUDAGraphNode] = []  # Root nodes
        self.current_node: Optional[CUDAGraphNode] = None  # Current position in tree
        self.ids = count(0)  # Unique graph IDs
        self.func_ids = count(0)  # Unique function IDs

        # Tracking live tensors
        self.path_live_weakrefs: list[list[Optional[StorageWeakRefWrapper]]] = []

        # Generation tracking (for mark_step_begin)
        self.current_gen = itertools.count(0)
        self.running_forwards_with_pending_backwards = False
```

**Key Methods**:
- `add_function()`: Main entry point, adds a function to the tree
- `get_roots()`: Returns root nodes
- `set_to_running_backward()`: Marks transition to backward pass
- `_check_liveness()`: Checks if tensors from previous graphs are still alive

#### 3. CUDAGraphNode

**Location**: [pytorch/torch/_inductor/cudagraph_trees.py:610+](pytorch/torch/_inductor/cudagraph_trees.py#L610)

```python
class CUDAGraphNode:
    """
    Represents a single recorded CUDA graph in the tree.

    Each node:
    - Owns a torch.cuda.CUDAGraph
    - Tracks parent/children relationships
    - Stores input/output metadata
    - Manages memory pool checkpointing
    """

    def __init__(
        self,
        wrapped_function: WrappedFunction,
        parent: Optional[CUDAGraphNode],
        inputs: list[InputType],
        cuda_graphs_pool: tuple[int, int],
        device: int,
        stack_traces: Optional[list[Optional[str]]],
        stream: torch.cuda.Stream,
        already_warm: bool,
    ):
        self.wrapped_function = wrapped_function
        self.parent = parent
        self.children: dict[PathLiveness, CUDAGraphNode] = {}

        # The actual CUDA graph
        self.graph: Optional[torch.cuda.CUDAGraph] = torch.cuda.CUDAGraph()

        # Reconstructed inputs/outputs (non-owning references)
        self.reconstructed_inputs: list[InputType] = []
        self.outputs_metadata: OutputList[Union[dict[str, Any], int, None]] = []

        # Checkpointed allocator state
        self.checkpointed_caching_state: Optional[AllocatorState] = None

        # Recording happens in __init__!
        self.recording_outputs = self._record(wrapped_function.model, recording_inputs)
```

**Key Responsibilities**:
1. **Recording**: Captures CUDA operations into a graph
2. **Replay**: Executes the captured graph
3. **Memory management**: Reconstructs inputs/outputs from metadata
4. **Liveness tracking**: Monitors which outputs are still alive

---

## Memory Pool Management

### Shared Memory Pool

Unlike basic CUDA graphs where each graph has its own pool, **CUDAGraph Trees share a single memory pool across all nodes in the tree**.

```
┌─────────────────────────────────────────────────────────────┐
│  CUDA Graph Trees - Single Shared Memory Pool              │
│                                                              │
│  ┌──────────────┐                                          │
│  │ Root Node 1  │  Allocates at addr 0x1000-0x2000         │
│  └──────────────┘                                          │
│         │                                                   │
│         ├── ┌──────────────┐                               │
│         │   │  Child 1A    │  Reuses 0x1000, allocs 0x2000│
│         │   └──────────────┘                               │
│         │                                                   │
│         └── ┌──────────────┐                               │
│             │  Child 1B    │  Reuses 0x1000, allocs 0x2500│
│             └──────────────┘                               │
│                                                              │
│  All nodes share memory pool: [0x1000 - 0x3000]            │
│  Peak usage: max(path1, path2, path3)                      │
└─────────────────────────────────────────────────────────────┘
```

### Memory Pool Creation

**Location**: [pytorch/torch/_inductor/cudagraph_trees.py:440-474](pytorch/torch/_inductor/cudagraph_trees.py#L440-L474)

```python
def cudagraphify(
    model: ModelType,
    inputs: list[InputType],
    static_input_idxs: Sequence[int] = (),
    *,
    device_index: int,
    is_backward: bool,
    is_inference: bool,
    # ... other params
) -> tuple[ModelType, OutputType]:

    # Get the tree manager for this device
    manager = get_container(device_index).get_tree_manager()

    # Add this function to the tree
    # The manager will handle memory pool creation/reuse
    return manager.add_function(
        model,
        inputs,
        static_input_idxs,
        stack_traces,
        mode,
        constants,
        placeholders,
        mutated_input_idxs,
        compile_id,
    )
```

### _use_cuda_memory_pool_manager Context

**Location**: [pytorch/torch/_inductor/cudagraph_trees.py:566-591](pytorch/torch/_inductor/cudagraph_trees.py#L566-L591)

```python
@contextlib.contextmanager
def _use_cuda_memory_pool_manager(
    device: int, mem_pool: tuple[int, int], stream: torch.cuda.Stream
) -> Generator[None, None, None]:
    """
    Context manager to use cuda graph pool for new allocations.
    All allocations inside this context go to the shared pool.
    """
    torch.cuda.synchronize()
    stream.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(stream), torch.device(device):
        # Begin allocating to the shared memory pool
        torch._C._cuda_beginAllocateCurrentThreadToPool(device, mem_pool)
        try:
            yield
        finally:
            torch._C._cuda_endAllocateToPool(device, mem_pool)
            torch._C._cuda_releasePool(device, mem_pool)

    torch.cuda.current_stream().wait_stream(stream)
```

**This is the key mechanism**: During recording, all allocations are redirected to the shared pool via `_cuda_beginAllocateCurrentThreadToPool`.

---

## Recording and Replay

### Recording Process

**Location**: [pytorch/torch/_inductor/cudagraph_trees.py:1040-1059](pytorch/torch/_inductor/cudagraph_trees.py#L1040-L1059)

The recording happens in `CUDAGraphNode.__init__`:

```python
def __init__(self, ...):
    # ... setup code ...

    # Allocate recording inputs in the shared memory pool
    recording_inputs = self._allocate_and_copy_recording_inputs(inputs)
    inputs.clear()

    # Create the CUDA graph
    self.graph = torch.cuda.CUDAGraph()

    # Register RNG states
    with torch.cuda.device(self.device):
        for rng_state in rng_states:
            self.graph.register_generator_state(rng_state)

    # Reconstruct input tensors (non-owning references)
    self.reconstructed_inputs = [
        self._reconstruct_from_tensor_metadata(self._tensor_metadata(x))
        if isinstance(x, torch.Tensor) else x
        for x in recording_inputs
    ]

    # *** RECORD THE GRAPH ***
    self.recording_outputs = self._record(wrapped_function.model, recording_inputs)

    # Save output metadata
    for out in self.recording_outputs:
        if isinstance(out, torch.Tensor):
            self.outputs_metadata.append(
                self._tensor_metadata(out, ignore_storage_offset=False)
            )
        else:
            self.outputs_metadata.append(out)

    # Replay once to finalize
    self.graph.replay()
```

### _record Method

```python
def _record(
    self, model: Callable[..., OutputType], inputs: list[InputType]
) -> OutputType:
    """
    Actually record the CUDA graph.

    Steps:
    1. Clear cublas caches
    2. Disable conv benchmark cache emptying
    3. Begin graph capture
    4. Run model
    5. End graph capture
    6. Checkpoint allocator state
    """

    with clear_cublas_manager(), disable_conv_cache_emptying():
        with get_history_recording():
            # Begin CUDA graph capture
            self.graph.capture_begin(
                pool=self.cuda_graphs_pool,
                capture_error_mode=cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal,
            )

            # *** RUN THE MODEL - THIS GETS RECORDED ***
            out = model(inputs)

            # End capture
            self.graph.capture_end()

    # *** CHECKPOINT THE ALLOCATOR STATE ***
    # This is crucial for allowing new recordings after replay
    self.checkpointed_caching_state = (
        torch._C._cuda_getCheckpointState(self.device, self.cuda_graphs_pool)
    )

    return out
```

**Key Innovation**: Checkpointing the allocator state (`_cuda_getCheckpointState`) allows us to restore the CPU-side allocator bookkeeping after replaying a graph, enabling new recordings.

### Replay Process

**Location**: [pytorch/torch/_inductor/cudagraph_trees.py:1109-1128](pytorch/torch/_inductor/cudagraph_trees.py#L1109-L1128)

```python
def run(self, new_inputs: list[InputType]) -> OutputType:
    """
    Replay the recorded graph with new inputs.
    """
    # 1. Check static inputs haven't changed addresses
    self.check_static_inputs_are_stable(new_inputs)

    # 2. Copy new input data to reconstructed inputs
    self._copy_inputs_and_remove_from_src(self.reconstructed_inputs, new_inputs)

    # 3. Replay the graph
    self.run_graph()

    # 4. Reconstruct outputs from saved metadata
    outputs = self.reconstruct_outputs()
    new_inputs.clear()

    # 5. Optionally check invariants
    if config.triton.fast_path_cudagraph_asserts:
        self.debug_check_invariants_after_invocation()

    # 6. Optionally sync (for debugging)
    if config.triton.force_cudagraph_sync:
        torch.cuda.synchronize()

    return outputs
```

### run_graph

```python
def run_graph(self) -> None:
    """Simply replay the CUDA graph."""
    assert self.graph is not None
    self.graph.replay()
```

**During replay**:
- **No** Python overhead
- **No** allocator calls
- **Only** GPU kernel execution

---

## Graph Tree Structure

### Tree Building Example

Let's trace through a function with branching:

```python
@torch.compile(mode="reduce-overhead")
def foo(x):
    y = x * x * x  # Always executes
    if y.sum() > 0:
        return y + 10  # Branch A
    else:
        return y - 10  # Branch B
```

**Execution Trace**:

```
Run 1 (x=ones):  warmup, y.sum()>0 → Branch A
Run 2 (x=ones):  RECORD Graph 1, Branch A
Run 3 (x=ones):  REPLAY Graph 1
Run 4 (x=-ones): Replay Graph 1, then y.sum()<0 → warmup Branch B
Run 5 (x=-ones): RECORD Graph 2 (child of Graph 1), Branch B
Run 6 (x=-ones): REPLAY Graph 1, REPLAY Graph 2
```

**Tree Structure**:

```
                    ┌──────────────────┐
                    │   Root: Graph 1  │
                    │   (y = x*x*x)    │
                    └──────────────────┘
                            │
                  ┌─────────┴─────────┐
                  │                   │
         ┌────────▼───────┐  ┌────────▼───────┐
         │  Child A        │  │  Child B        │
         │  (return y+10)  │  │  (return y-10)  │
         └─────────────────┘  └─────────────────┘
```

### Path Liveness

**The Innovation**: Each path in the tree is identified by which tensors from previous graphs are still **alive**.

**Location**: [pytorch/torch/_inductor/cudagraph_trees.py:600-606](pytorch/torch/_inductor/cudagraph_trees.py#L600-L606)

```python
# A path index of (depth, offset) indices into a graph that is
# `depth` number of nodes from the root at graph output offset
PathOutputIndex = tuple[int, int]

# For each node in the path, for each output, is the output alive
PathLiveness = list[list[bool]]
```

**Example**:

```python
# Suppose Graph 1 has 3 outputs
Graph1.outputs = [out1, out2, out3]

# Path to Child A: all outputs dead
path_liveness_A = [[False, False, False]]

# Path to Child B: out1 still alive
path_liveness_B = [[True, False, False]]

# These become DIFFERENT children in the tree!
```

**Why?** If `out1` is still alive, Graph B can't overwrite its memory. We need a separate recording that accounts for this.

### Child Selection

**Location**: [pytorch/torch/_inductor/cudagraph_trees.py (in add_child_graph)](pytorch/torch/_inductor/cudagraph_trees.py)

```python
def add_child_graph(
    self,
    function_id: FunctionID,
    inputs: list[InputType],
    outputs: OutputType,
) -> tuple[CUDAGraphNode, OutputType]:
    """
    Add a child graph to this node.

    Child is selected based on:
    1. Function ID (which function is being called)
    2. Path liveness (which outputs from previous graphs are alive)
    """

    # Get current liveness of outputs
    liveness_slice = self._get_liveness(self.path_weakrefs[self.depth])

    # Check if we already have a child for this path
    child = self.children.get((function_id, tuple(liveness_slice)))

    if child is not None:
        # REPLAY existing child
        return child, child.run(inputs)
    else:
        # RECORD new child
        child = self._record_child(function_id, inputs, liveness_slice)
        self.children[(function_id, tuple(liveness_slice))] = child
        return child, child.run_first_inputs(inputs)
```

---

## Checkpointing and Branching

### The Checkpointing Problem

**Problem**: After replaying a CUDA graph, the CPU-side allocator state is **stale**. It doesn't know which memory is in use.

**Example**:

```python
# Record Graph 1
with torch.cuda.graph(graph1):
    y = x * x  # Allocates at 0x1000

# Replay Graph 1
graph1.replay()
# CPU allocator: "No memory allocated"  (WRONG!)
# GPU reality: Memory at 0x1000 is in use

# Try to record Graph 2
with torch.cuda.graph(graph2):
    z = torch.rand(...)  # Might allocate at 0x1000 (COLLISION!)
```

### The Solution: Checkpointing

**Location**: [pytorch/torch/_inductor/cudagraph_trees.py (in _record)](pytorch/torch/_inductor/cudagraph_trees.py)

```python
def _record(self, model, inputs):
    # ... capture the graph ...

    # *** SAVE ALLOCATOR STATE ***
    self.checkpointed_caching_state = (
        torch._C._cuda_getCheckpointState(self.device, self.cuda_graphs_pool)
    )

    return out
```

**What it saves**:
- Which memory blocks are allocated
- Which blocks are free
- Memory pool high-water mark

### Restoring from Checkpoint

**Location**: [pytorch/torch/_inductor/cudagraph_trees.py (in _resume_from_checkpoint)](pytorch/torch/_inductor/cudagraph_trees.py)

```python
def _resume_from_checkpoint(node: CUDAGraphNode) -> None:
    """
    Restore allocator state from a checkpoint.

    This allows us to record new graphs after replaying old ones.
    """
    assert node.checkpointed_caching_state is not None

    torch._C._cuda_setCheckpointPoolState(
        node.device,
        node.checkpointed_caching_state,
    )
```

### Branching Workflow

```
┌────────────────────────────────────────────────────────────────┐
│ 1. Execute Graph 1 (root)                                      │
│    - Allocates memory for outputs                              │
│    - Checkpoint allocator state                                │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ 2. User code decides to take Branch A                          │
│    - Graph 1 outputs are still alive                           │
│    - Need to record new child                                  │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ 3. Restore from Graph 1's checkpoint                           │
│    torch._C._cuda_setCheckpointPoolState(node.checkpoint)     │
│    - Allocator now knows Graph 1 outputs are alive            │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ 4. Record Branch A graph                                       │
│    - New allocations won't overwrite Graph 1 outputs          │
│    - Checkpoint this state too                                 │
└────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────┐
│ 5. Later: User takes Branch B instead                          │
│    - Restore from Graph 1's checkpoint                         │
│    - Record Branch B graph                                     │
│    - Now we have a tree!                                       │
└────────────────────────────────────────────────────────────────┘
```

---

## Complete Example Trace

Let's trace a complete example end-to-end:

### Example Code

```python
import torch

@torch.compile(mode="reduce-overhead")
def model(x):
    y = x * x * x
    torch._dynamo.graph_break()  # Force two separate graphs
    if y.sum() > 0:
        z = y + 10
    else:
        z = y - 10
    return z

input1 = torch.ones(10, device="cuda")
input2 = torch.ones(10, device="cuda") * -1
```

### Execution Timeline

#### Run 1: `model(input1)` - Initial Trace

```
┌─────────────────────────────────────────────────────────────────┐
│ Python Level (TorchDynamo)                                      │
└─────────────────────────────────────────────────────────────────┘
1. Dynamo traces `model(input1)`
2. Encounters `y = x * x * x`
3. Encounters `graph_break()` → creates FX Graph 1
4. Encounters `if y.sum() > 0` → can't trace condition
5. Runs eagerly, condition is True
6. Traces `z = y + 10` → creates FX Graph 2
7. Returns FX graphs to Inductor

┌─────────────────────────────────────────────────────────────────┐
│ Inductor Level                                                  │
└─────────────────────────────────────────────────────────────────┘
8. Inductor compiles Graph 1 → generates Triton kernel
9. Wraps with `cudagraphify()` (BUT doesn't record yet - lazy)
10. Inductor compiles Graph 2 → generates Triton kernel
11. Wraps with `cudagraphify()`
12. Returns compiled functions

┌─────────────────────────────────────────────────────────────────┐
│ Execution (First Call)                                          │
└─────────────────────────────────────────────────────────────────┘
13. Call compiled Graph 1 → WARMUP (first time)
    - Runs Triton kernel eagerly
    - Initializes CuDNN benchmarks
14. Evaluate `y.sum() > 0` → True
15. Call compiled Graph 2 → WARMUP
    - Runs Triton kernel eagerly

Output: z = [11, 11, ..., 11]
```

#### Run 2: `model(input1)` - Recording

```
┌─────────────────────────────────────────────────────────────────┐
│ Graph 1 Recording                                               │
└─────────────────────────────────────────────────────────────────┘
1. cudagraphify_impl() called for Graph 1
2. Detect this is second run → time to record!
3. get_container(device=0).get_tree_manager()
   - Creates TreeManagerContainer if needed
   - Creates CUDAGraphTreeManager if needed
4. manager.add_function(graph1_model, inputs, ...)
5. Creates ROOT CUDAGraphNode

   CUDAGraphNode.__init__:
   a. Create torch.cuda.CUDAGraph()
   b. graph.capture_begin(pool=shared_pool)
   c. Run model → y = x * x * x
   d. graph.capture_end()
   e. Checkpoint allocator:
      checkpointed_caching_state = _cuda_getCheckpointState(...)
   f. graph.replay() (to initialize outputs)

6. Return to user code

┌─────────────────────────────────────────────────────────────────┐
│ Graph 2 Recording                                               │
└─────────────────────────────────────────────────────────────────┘
7. manager.current_node = Graph1Node
8. Evaluate `y.sum() > 0` → True
9. cudagraphify_impl() called for Graph 2
10. manager.add_function(graph2_model, inputs, ...)
11. Check: Do we have a child for this function + liveness?
    - liveness = [[False, False, ...]] (all Graph 1 outputs dead)
    - No child exists → CREATE NEW CHILD

12. Restore from checkpoint:
    _cuda_setCheckpointPoolState(Graph1Node.checkpointed_caching_state)

13. Create CHILD CUDAGraphNode
    CUDAGraphNode.__init__:
    a. parent = Graph1Node
    b. Create torch.cuda.CUDAGraph()
    c. graph.capture_begin(pool=SAME_POOL)  # ← Key!
    d. Run model → z = y + 10
    e. graph.capture_end()
    f. Checkpoint allocator
    g. graph.replay()

14. Graph1Node.children[(function_id, liveness)] = Graph2Node

Output: z = [11, 11, ..., 11]
```

**Memory Pool State After Recording**:

```
Shared Memory Pool:
├─ Graph 1 inputs:    0x1000 - 0x1100  (100 bytes)
├─ Graph 1 outputs:   0x1100 - 0x1200  (100 bytes)
├─ Graph 2 inputs:    0x1200 - 0x1300  (100 bytes)
└─ Graph 2 outputs:   0x1300 - 0x1400  (100 bytes)

Total: 400 bytes (vs. 800 bytes with separate pools!)
```

#### Run 3: `model(input1)` - Fast Replay

```
┌─────────────────────────────────────────────────────────────────┐
│ Replay Existing Path                                            │
└─────────────────────────────────────────────────────────────────┘
1. cudagraphify_impl() → fn_cache hit for Graph 1
2. Call cached function
3. Graph1Node.run(inputs):
   a. Copy input data → reconstructed_inputs
   b. graph.replay()  # ← FAST!
   c. Reconstruct outputs from metadata

4. manager.current_node = Graph1Node
5. Evaluate `y.sum() > 0` → True
6. cudagraphify_impl() → fn_cache hit for Graph 2
7. Call cached function
8. Check: Do we have child for this liveness?
   - liveness = [[False, False, ...]]
   - YES! Graph2Node exists
9. Graph2Node.run(inputs):
   a. Copy input data
   b. graph.replay()  # ← FAST!
   c. Reconstruct outputs

Output: z = [11, 11, ..., 11]
```

**Performance**:
- No Python overhead
- No kernel compilation
- No memory allocation
- Just two `graph.replay()` calls!

#### Run 4: `model(input2)` - New Branch

```
┌─────────────────────────────────────────────────────────────────┐
│ Same Start, Different Branch                                    │
└─────────────────────────────────────────────────────────────────┘
1. Replay Graph 1 (same as Run 3, steps 1-3)
2. manager.current_node = Graph1Node
3. Evaluate `y.sum() > 0` → FALSE (input2 is negative)
4. cudagraphify_impl() → fn_cache hit for Graph 2 (same function!)
5. BUT: condition took different branch

6. Check: Do we have child for this liveness?
   - liveness = [[False, False, ...]]
   - YES, but... graph break causes re-entry to manager

7. Actually, the `else` branch is a DIFFERENT graph (Graph 3)
8. cudagraphify_impl() for Graph 3 (first time)
9. manager.add_function(graph3_model, inputs, ...)
10. No child exists for Graph 3 → WARMUP RUN

Output: z = [-11, -11, ..., -11]
```

#### Run 5: `model(input2)` - Record New Branch

```
┌─────────────────────────────────────────────────────────────────┐
│ Record Branch B                                                  │
└─────────────────────────────────────────────────────────────────┘
1. Replay Graph 1 (fast)
2. Evaluate `y.sum() > 0` → FALSE
3. cudagraphify_impl() for Graph 3 (second time → RECORD)
4. Restore from Graph1 checkpoint
5. Create Graph3Node (child of Graph1Node)
   - parent = Graph1Node
   - Uses SAME memory pool
   - Records `z = y - 10`
6. Graph1Node.children[(graph3_id, liveness)] = Graph3Node

Output: z = [-11, -11, ..., -11]
```

**Final Tree Structure**:

```
                    ┌──────────────────┐
                    │   Graph 1 Node   │
                    │  (y = x*x*x)     │
                    └──────────────────┘
                            │
                  ┌─────────┴─────────┐
                  │                   │
         ┌────────▼───────┐  ┌────────▼───────┐
         │  Graph 2 Node  │  │  Graph 3 Node  │
         │  (z = y + 10)  │  │  (z = y - 10)  │
         │  if sum > 0    │  │  if sum <= 0   │
         └────────────────┘  └────────────────┘
```

#### Run 6: `model(input1)` and `model(input2)` - Fully Optimized

```
Both execution paths are now recorded and can replay instantly:

Path A (input1 > 0):
  Graph1.replay() → Graph2.replay()

Path B (input2 < 0):
  Graph1.replay() → Graph3.replay()

Memory: Only max(Path A, Path B) used
Speed: Near-zero CPU overhead
```

---

## Key Takeaways

### What CUDAGraph Trees Enable

1. **Branching**: Different execution paths can coexist in the same tree
2. **Memory efficiency**: Single shared pool, not per-graph pools
3. **Safety**: Live tensor tracking prevents memory corruption
4. **Integration**: Works seamlessly with torch.compile and graph breaks

### Performance Benefits

**Without CUDAGraphs**:
```
Per iteration:
- Python overhead: ~100μs
- Kernel launches: ~50μs per kernel × N kernels
- Total: ~100μs + N×50μs
```

**With CUDAGraph Trees**:
```
Per iteration:
- Python overhead: ~10μs
- Graph replay: ~5μs
- Total: ~15μs (regardless of N kernels!)
```

**Memory**:
```
Without trees: N graphs × memory per graph
With trees: max(memory across all paths)
Savings: Up to N× for N branches
```

### Limitations

From [pytorch/docs/source/torch.compiler_cudagraph_trees.md](pytorch/docs/source/torch.compiler_cudagraph_trees.md#L298-L332):

1. **Live outputs between iterations**: Can't preserve outputs across iterations
   ```python
   out1 = model(x)  # Iteration 1
   out2 = model(x)  # Iteration 2
   print(out1)      # ERROR: out1 overwritten!
   ```

   **Solution**: Use `torch.compiler.cudagraph_mark_step_begin()` or clone outputs

2. **Memory increase**: If both eager and graph code run, memory usage is sum of both pools

3. **Recording overhead**: First few runs are slower due to recording

4. **Static shapes**: Can't change tensor shapes (but can record multiple shapes)

---

## Code References

### Key Files

- [torch/_inductor/__init__.py](pytorch/torch/_inductor/__init__.py): Mode configuration
- [torch/_inductor/compile_fx.py](pytorch/torch/_inductor/compile_fx.py): Compilation pipeline
- [torch/_inductor/cudagraph_trees.py](pytorch/torch/_inductor/cudagraph_trees.py): CUDAGraph Trees implementation
- [torch/_inductor/cudagraph_utils.py](pytorch/torch/_inductor/cudagraph_utils.py): Utilities and validation
- [torch/_inductor/config.py](pytorch/torch/_inductor/config.py): Configuration options

### Test Examples

- [test/inductor/test_cudagraph_trees.py](pytorch/test/inductor/test_cudagraph_trees.py): Comprehensive test suite

### Configuration Options

```python
import torch._inductor.config as config

config.triton.cudagraphs = True          # Enable CUDAGraphs
config.triton.cudagraph_trees = True      # Enable trees (default)
config.triton.skip_cudagraph_warmup = True  # Skip warmup (for testing)
config.triton.fast_path_cudagraph_asserts = True  # Extra validation
config.triton.cudagraph_support_input_mutation = True  # Allow input mutation
```

---

## Summary

**CUDAGraph Trees** extend PyTorch's basic CUDA graph support with:

1. **Tree structure**: Branching execution paths in a single tree
2. **Shared memory pool**: All graphs share one pool, reducing memory overhead
3. **Allocator checkpointing**: Enables recording after replay
4. **Live tensor tracking**: Prevents memory corruption across graph boundaries
5. **torch.compile integration**: Automatic with `mode="reduce-overhead"`

The result is **near-zero Python overhead** for model execution while maintaining safety and flexibility.

**Typical speedup**: 2-10× for CPU-bound models
**Typical memory overhead**: Minimal (shared pool vs. per-graph pools)

---

*This documentation traces the complete implementation from Python API through Dynamo/Inductor to the CUDA graph recording and replay system.*
