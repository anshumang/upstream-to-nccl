# Minimal translation unit for one gin.put, with its SASS

`efa_gda_put_probe.cu` is one kernel containing exactly one EFA GDA put in the
instantiation the NCCLOFI-1945 harness uses: THREAD sharing, `ncclCoopThread`,
`ncclGin_WeakSignalInc`, thread-scope release given, device-scope release
required, default flags, device-memory windows. Nothing is launched; the object
exists to be disassembled.

The kernel is the device-only, single-segment branch of
`ncclGin_BackendMask::put` (gin__funcs.h) reproduced line for line against a
backend mask narrowed to EFA GDA. `ncclGin` itself is
`ncclGin_BackendMask<ALL>` and dispatches on the backend at run time, so a kernel
calling `ncclGin::put` also contains the proxy backend's put path and its counts
would not be EFA GDA's; the narrowed mask does not compile with a signal because
the helper overloads are typed on `ncclGin`, hence the reproduction.

Built against this branch's headers (base `0b24e73`) with
nvcc/ptxas release 13.4, V13.4.92 (CUDA 13.4.2 redistributable components):

    nvcc -std=c++17 -O2 -gencode=arch=compute_<90|100>,code=sm_<90|100> <arm flags> \
         -I<include tree of this commit> -c efa_gda_put_probe.cu -o probe.o
    cuobjdump --dump-sass probe.o   # probe_put_kernel only

| arm | flags |
|---|---|
| `baseline` | `(none)` |
| `reg` | `-DNCCL_GIN_EFA_GDA_WQE_REGISTER_BUILD=1` |

`sass/<arm>-sm90.sass` and `sass/<arm>-sm100.sass` are the listings with the
address column, the encoding comments and absolute branch targets removed
(targets become `L<n>:` labels), so two arms diff on instructions only. The raw
dump is reproducible from the command above. `counts.csv` classifies every
instruction of the kernel, per arm and arch: fences by scope (`membar_sys/gpu/cta`)
and their companions (`errbar` = ERRBAR+CGAERRBAR, `cctl`); global/generic stores
and loads by scope (`st_mmio_sys`, `st_strong_sys`, `st_strong_gpu`,
`st_global_weak`, `ld_strong_sys`, `ld_strong_gpu`, `ld_global_weak`); local
memory (`stl`, `ldl`), `shared`, constant (`ldc`); atomics by scope (`atom_sys`,
`atom_gpu`, `red_*`, `atoms_shared`); warp cooperation (`match` =
labeled_partition, `shfl`, `vote`, `warpsync`, `bar_sync`, `bssy_bsync`
reconvergence, `redux`); spin-wait hints (`nanosleep_yield`, one per poll loop);
`bra` (branches), `exit`, `other` (ALU/moves), `instructions`. There is no
grid-level synchronization inside `gin.put` and no CTA barrier; those live in
the caller.

These are static counts over the whole put as compiled. A put with a signal is
two WQE posts (the data write and the 0-byte scratch write that carries the
signal), and the kernel also holds the >1 GiB chunk loop with its drain wait,
both doorbell ring sites and the deferred (aggregating) path. They are exact
per-arm deltas, not the number of instructions one executed 14 KiB put retires.
