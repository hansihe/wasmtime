initSidebarItems({"constant":[["INTERRUPTED","Sentinel value indicating that wasm has been interrupted."],["VERSION","Version number of this crate."],["WASM_MAX_PAGES","The number of pages we can have before we run out of byte index space."],["WASM_PAGE_SIZE","WebAssembly page sizes are defined to be 64KiB."]],"enum":[["CompileError","An error while compiling WebAssembly to machine code."],["EntityIndex","An index of an entity."],["MemoryStyle","Implemenation styles for WebAssembly linear memory."],["ModuleMemoryOffset","Memory definition offset in the VMContext structure."],["RelocationTarget","Destination function. Can be either user function or some special one, like `memory.grow`."],["TableStyle","Implemenation styles for WebAssembly tables."]],"fn":[["cache_create_new_config","Creates a new configuration file at specified path, or default path if None is passed. Fails if file already exists."],["translate_signature","Add environment-specific function parameters."]],"mod":[["cranelift","Support for compiling with Cranelift."],["entity",""],["ir",""],["isa",""],["settings",""],["wasm",""]],"struct":[["BuiltinFunctionIndex","An index type for builtin functions."],["CacheConfig","Global configuration for how the cache is managed"],["Compilation","The result of compiling a WebAssembly module's functions."],["CompiledFunction","Compiled function: machine code body, jump table offsets, and unwind information."],["DataInitializer","A data initializer for linear memory."],["DataInitializerLocation","A memory index and offset within that memory where a data initialization should is to be performed."],["FunctionAddressMap","Function and its instructions addresses mappings."],["FunctionBodyData","Contains function data: byte code and its offset in the module."],["InstructionAddressMap","Single source location to generated address mapping."],["MemoryPlan","A WebAssembly linear memory description along with our chosen style for implementing it."],["Module","A translated WebAssembly module, excluding the function bodies and memory initializers."],["ModuleEnvironment","Object containing the standalone environment information."],["ModuleLocal","Local information known about a wasm module, the bare minimum necessary to translate function bodies."],["ModuleTranslation","The result of translating via `ModuleEnvironment`. Function bodies are not yet translated, and data initializers have not yet been copied out of the original buffer."],["ModuleVmctxInfo","Module `vmctx` related info."],["Relocation","A record of a relocation to perform."],["StackMapInformation","The offset within a function of a GC safepoint, and its associated stack map."],["TableElements","A WebAssembly table initializer."],["TablePlan","A WebAssembly table description along with our chosen style for implementing it."],["TargetSharedSignatureIndex","Target specific type for shared signature index."],["TrapInformation","Information about trap."],["Tunables","Tunable parameters for WebAssembly compilation."],["VMOffsets","This class computes offsets to fields within `VMContext` and other related structs that JIT code accesses directly."]],"trait":[["Compiler","An implementation of a compiler from parsed WebAssembly module to native code."]],"type":[["ModuleAddressMap","Module functions addresses mappings."],["Relocations","Relocations to apply to function bodies."],["StackMaps","Information about GC safepoints and their associated stack maps within each function."],["Traps","Information about traps associated with the functions where the traps are placed."],["ValueLabelsRanges","Value ranges for functions."]]});