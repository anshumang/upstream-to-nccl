/*************************************************************************
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * EFA GDA implementations for NCCL GIN device-side APIs.
 *
 * This file provides ncclGinApi_*<NCCL_NET_DEVICE_GIN_EFA_GDA> template
 * specializations that target EFA via efa-dp-direct.
 *
 * Implemented: Put (data + signal/counter endpoints, signal-only via
 *              scratch buffer), PutValue (inline in 128-byte RDMA-write
 *              WQEs), Get, Flush, FlushAsync, Wait,
 *              GetSignalPtr, GetCounterPtr, ResetSignal, ResetCounter.
 *
 * COMPLETION
 * ----------
 * Every write is posted with COMP_REQ = 1, so the NIC writes one CQE per write
 * and increments the posting endpoint's FI_WRITE hardware counter. Completion is
 * read three ways:
 *
 *   *ep->local_cntr_value   per-QP completion: the endpoint's FI_WRITE NIC
 *                     counter, read directly from GPU memory by the blocking
 *                     Flush. Wraps at 2^31, so compared under EFA_CNTR_MASK.
 *   *dev->completed_count_per_ctx  count of CQEs the host has drained from the
 *                     shared CQ, host-published via gdrcopy. Read by the
 *                     per-context CQ-overflow gate in postRdmaOp, which claims its
 *                     span of submitted_count_per_ctx before waiting on it.
 *   dev->ordered_completed_count_per_peer[peer]:
 *                     a contiguous prefix in that peer's own posting sequence,
 *                     derived from the shared CQ and host-published via gdrcopy.
 *                     Read by FlushAsync/Wait and put's signal-ordering wait.
 *
 * The per-context and per-peer counts come from a host thread (the plugin's CQ
 * progress pass, driven by NCCL's GIN progress thread) that drains the context's
 * shared CQ. Each WQE stamps req_id with a (peer, pseq) split (peer in the high
 * bits, pseq in the low NCCL_OFI_GDAKI_PSEQ_BITS bits), echoed by
 * efa_io_cdesc_common::req_id, so the host reads peer and pseq from the CQE.
 *
 * The per-peer count is a contiguous prefix rather than a bare count, because EFA
 * SRD completes out of order even to one peer: only "everything below N completed"
 * proves a particular earlier write finished. Only a contiguous prefix expresses
 * that, which is why the host derives per-peer from the CQ while per-QP stays the
 * NIC counter.
 *
 * The FI_REMOTE_WRITE hardware counter is used for signal delivery; the FI_WRITE
 * counter also backs the user-facing GIN counter value.
 *
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_EFA_GDA_H_
#define _NCCL_DEVICE_GIN_EFA_GDA_H_

#include <cstddef>
#include <cstdint>
#include <cuda/atomic>
#include <cooperative_groups.h>

#include "../gin_device_common.h"
#include "gin_efa_gda_dev.h"

/* efa-dp-direct device functions (inline implementations) */
#include "../../transport/net_efa_gda/efa-dp-direct/include/device/efa_cuda_dp_impl.cuh"

/* This request carries what FlushAsync hands to Wait: a peer and a snapshot of
 * that peer's submitted count.
 *
 * submitted_count_at_flush is ctx->submitted_count_per_peer[peer] read at the
 * moment of the call: how many writes this CONTEXT had admitted to that peer,
 * across every QP. FlushAsync copies the value rather than re-reading the counter
 * later, because the counter keeps growing as more puts happen; copying it both
 * excludes puts issued after the flushAsync (the contract) and guarantees the
 * wait terminates. Wait compares the copy against the host-published per-peer
 * ordered prefix, so one number suffices. */
struct ncclGinEfaGdaRequest {
  uint32_t peer;
  uint32_t submitted_count_at_flush;
  uint64_t reserved;
};
static_assert(sizeof(ncclGinEfaGdaRequest) <= sizeof(ncclGinRequest_t),
              "ncclGinEfaGdaRequest must fit in ncclGinRequest_t");

namespace nccl {
namespace gin {
namespace efa_gda {

/* The plugin returns a contiguous array of per-context dev handles
 * in GPU memory; ctx.handle points at element 0. ctx.contextId
 * selects the entry for this caller. */
NCCL_DEVICE_INLINE static nccl_ofi_gin_gdaki_dev_handle* getDevHandle(ncclGinCtx ctx) {
  return &((nccl_ofi_gin_gdaki_dev_handle*)ctx.handle)[ctx.contextId];
}

/* ── Mode mapping: NCCL → efa-dp-direct ───────────────────────────── */

template <ncclGinResourceSharingMode mode>
static constexpr cuda::thread_scope ncclGinScope =
  (mode == NCCL_GIN_RESOURCE_SHARING_CTA) ? cuda::thread_scope_block : cuda::thread_scope_device;

/* The EFA hardware completion counters (FI_WRITE / FI_REMOTE_WRITE) wrap at
 * 2^31, while the kernel-side producer cursors are uint32 (wrap at 2^32).
 * Every comparison between a producer cursor and a HW counter must therefore
 * be a modular difference reduced to 31 bits: compute (producer - consumer)
 * and mask with EFA_CNTR_MASK. The true difference (in-flight / outstanding
 * work) is always far below 2^31 (bounded by sq_size == 4096), so the masked
 * difference recovers the exact value regardless of how many times either side
 * has wrapped. Never compare absolute counter values. */
static constexpr uint32_t EFA_CNTR_MASK = 0x7fffffffu;

/* Hardware cap on a single RDMA op (write or read): efadv_device_attr.max_rdma_size,
 * 1 GiB on current EFA devices. A WQE exceeding it fails on the NIC as a CQ
 * error, which this CQ-less path never observes, so the op hangs; the u32
 * SGE length field additionally truncates sizes >= 4 GiB. Larger transfers are
 * split into chunks of at most this size in putImplMode / getImplMode. */
static constexpr uint32_t EFA_GDA_MAX_RDMA_OP_SIZE = 1u << 30;

/* Which RDMA operation the post path issues. A write and a read populate identical
 * WR fields, so the opcode is the only difference between them and it is what
 * decides which way the bytes move. */
enum efaGdaRdmaOp {
  EFA_GDA_RDMA_WRITE,
  EFA_GDA_RDMA_READ
};

/* req_id is the field the EFA completion echoes back (efa_io_cdesc_common), and it
 * carries a write's attribution: the peer in the high EFA_GDA_PEER_BITS, and the
 * write's position in that peer's posting sequence (pseq) in the low
 * EFA_GDA_PSEQ_BITS. The split must match NCCL_OFI_GDAKI_PSEQ_BITS in the plugin,
 * which sizes the host's per-peer completion bitmap from the same number, and it
 * caps the ranks this backend can address at 2^EFA_GDA_PEER_BITS. The assert below
 * pins the two widths to the field they divide, so widening one without narrowing
 * the other fails the build instead of silently aliasing peers or pseqs. */
static constexpr uint32_t EFA_GDA_PSEQ_BITS = 32;
static constexpr uint32_t EFA_GDA_PEER_BITS = 32;
static constexpr uint64_t EFA_GDA_PSEQ_MASK = (1ull << EFA_GDA_PSEQ_BITS) - 1ull;
static constexpr uint64_t EFA_GDA_PEER_MASK = (1ull << EFA_GDA_PEER_BITS) - 1ull;
static_assert(EFA_GDA_PEER_BITS + EFA_GDA_PSEQ_BITS ==
                (sizeof(decltype(efa_io_tx_meta_desc::req_id)) + sizeof(decltype(efa_io_tx_meta_desc::req_id_ex))) * 8,
              "req_id must be split entirely between its peer and pseq fields");

/* ── Atomic primitives parameterized on scope and memory order ────── */

template <cuda::thread_scope Scope, cuda::memory_order Order, typename T>
NCCL_DEVICE_INLINE static T scopedAtomicLoad(T* ptr) {
  cuda::atomic_ref<T, Scope> r(*ptr);
  return r.load(Order);
}

template <cuda::thread_scope Scope, cuda::memory_order Order, typename T>
NCCL_DEVICE_INLINE static void scopedAtomicAdd(T* ptr, T val) {
  cuda::atomic_ref<T, Scope> r(*ptr);
  r.fetch_add(val, Order);
}

template <cuda::thread_scope Scope, cuda::memory_order Order, typename T>
NCCL_DEVICE_INLINE static T scopedAtomicFetchAdd(T* ptr, T val) {
  cuda::atomic_ref<T, Scope> r(*ptr);
  return r.fetch_add(val, Order);
}

/* ── NIC-written hardware counter (FI_WRITE / FI_REMOTE_WRITE) ────── */

/* Read a NIC-written hardware counter from GPU memory. System scope makes
 * the load coherent with the NIC's PCIe writes (bypasses GPU caches).
 * Acquire is the default and matches libfabric's local-completion contract:
 * when this load observes the counter has reached a target, the NIC's prior
 * side effects (e.g. source-buffer DMA-reads complete) are ordered-before
 * whatever this thread does next (e.g. overwriting that source buffer or
 * reusing the slot). */
template <cuda::memory_order Order = cuda::memory_order_acquire>
NCCL_DEVICE_INLINE static uint64_t hwCounterLoad(uint64_t* ptr) {
  return scopedAtomicLoad<cuda::thread_scope_system, Order>(ptr);
}

/* ── Completion state (per-QP: NIC counter; per-ctx/per-peer: CQ-derived) ─── */

/* This spins until `peer`'s first `target` writes on this context have all
 * completed. It is sound on an out-of-order transport because the ordered prefix
 * advances only over a contiguous run: a missing earlier completion holds the
 * ordered prefix below the target however late that completion arrives. */
template <bool HasTimeout>
NCCL_DEVICE_INLINE static ncclResult_t waitPeerCompleted(nccl_ofi_gin_gdaki_dev_handle* dev, uint32_t peer,
                                                         uint32_t target, uint32_t* abortFlag, uint64_t startCycle,
                                                         uint64_t timeoutCycles) {
  /* The signed difference orders the per-peer uint32 counter correctly across its
   * wrap at 2^32, since the true gap is bounded by what can be in flight. */
  while ((int32_t)(scopedAtomicLoad<cuda::thread_scope_system, cuda::memory_order_acquire>(
                     &dev->ordered_completed_count_per_peer[peer]) -
                   target) < 0) {
    if NCCL_IF_CONSTEXPR (HasTimeout) {
      if (clock64() - startCycle >= timeoutCycles) return ncclTimeout;
    }
    if (abortFlag && *abortFlag) return ncclInProgress;
  }
  return ncclSuccess;
}

/* ── Single-issuer-per-QP specialization ──────────────────────────────
 *
 * NCCL_GIN_EFA_GDA_SINGLE_ISSUER_PER_QP (build-time, default OFF)
 *   Asserts, and then exploits, the precondition that exactly ONE thread ever
 *   posts to a given QP. Under that precondition the post-doorbell system
 *   fence in ringDoorbell is dead and is omitted, removing one MEMBAR.ALL.SYS
 *   per put.
 *
 *   Why it is dead. The post-doorbell fence exists only to order this thread's
 *   doorbell MMIO write before the release-stores that hand the QP off
 *   (dbrung_ref / base_ref), so that a LATER group's higher doorbell value
 *   cannot overtake this one and make the NIC observe a non-monotonic producer
 *   index. With one issuing thread there is no later group and no handoff to
 *   order against. Successive doorbell writes from one thread go to the SAME
 *   address, and PTX guarantees mmio writes are always performed, never
 *   combined, and that same-address writes follow program order in coherence
 *   order -- with the NIC in scope, since these are .sys. So monotonicity holds
 *   with no fence. Publish-before-ring is still enforced per put by the
 *   pre-doorbell WQE fence, which is NOT removed.
 *
 *   The precondition is not checked on the device: with more than one poster
 *   per QP the removed fence is NOT dead and doorbell monotonicity across
 *   groups is no longer guaranteed. The caller must guarantee one QP per
 *   issuing thread at setup time. */
#ifndef NCCL_GIN_EFA_GDA_SINGLE_ISSUER_PER_QP
#define NCCL_GIN_EFA_GDA_SINGLE_ISSUER_PER_QP 0
#endif

/* ── Single-owner fast path ────────────────────────────────────────────
 *
 * NCCL_GIN_EFA_GDA_SINGLE_OWNER_FAST_PATH (build-time, default OFF)
 *   Selected only for THREAD sharing, whose contract (ncclGinResourceSharingMode)
 *   is that one thread exclusively owns the whole context, and with it every
 *   QP, until ownership is transferred by external CUDA synchronization. Under
 *   that contract every GPU-owned producer cursor -- the admission counts
 *   (submitted_count_per_peer, submitted_count_per_ctx) and the SQ cursors (pc,
 *   wqes_completed, wqes_posted, submitted_count) -- has exactly one accessor on
 *   the GPU, so their atomics, the doorbell-order rendezvous and the handoff
 *   release-store are replaced by plain sequential updates, and the
 *   post-doorbell fence that orders the doorbell before the handoff is dropped:
 *   there is no handoff, and this owner's successive doorbell writes to the same
 *   address stay in program order at system scope. The same contract makes the
 *   warp-level grouping of posters -- by context and peer for admission, by QP
 *   for posting (coalesced_threads + labeled_partition, with their shuffles and
 *   group syncs) -- provably groups of one, so the owner path runs both passes
 *   as a solo lane and emits none of those collectives. Counters written by the
 *   NIC or the host (ordered_completed_count_per_peer, completed_count_per_ctx,
 *   the hardware counters) are still read with system-scope atomic loads. CTA
 *   and GPU sharing are unchanged. */
#ifndef NCCL_GIN_EFA_GDA_SINGLE_OWNER_FAST_PATH
#define NCCL_GIN_EFA_GDA_SINGLE_OWNER_FAST_PATH 0
#endif

template <ncclGinResourceSharingMode mode>
static constexpr bool efaGdaSingleOwner =
  NCCL_GIN_EFA_GDA_SINGLE_OWNER_FAST_PATH != 0 && mode == NCCL_GIN_RESOURCE_SHARING_THREAD;

/* Group of one: the interface the admission and posting passes need from a
 * cooperative_groups::coalesced_group, for a lane that provably has no peers in
 * its group. Every member is a compile-time constant, so nothing is emitted. */
struct EfaGdaSoloGroup {
  NCCL_DEVICE_INLINE int thread_rank() const { return 0; }
  NCCL_DEVICE_INLINE int num_threads() const { return 1; }
  template <typename T>
  NCCL_DEVICE_INLINE T shfl(T v, int) const { return v; }
  NCCL_DEVICE_INLINE void sync() const {}
};

/* Selects the groups for postRdmaOp: converged lanes partitioned by context,
 * peer and QP (shared modes), or solo groups (single owner). */
template <bool Solo>
struct EfaGdaPostGroups {
  cooperative_groups::coalesced_group active;
  cooperative_groups::coalesced_group peer;
  NCCL_DEVICE_INLINE EfaGdaPostGroups(nccl_ofi_gin_gdaki_dev_handle* dev, uint32_t peerIdx)
    : active(cooperative_groups::coalesced_threads()),
      /* The admission counters belong to a logical context, not to a peer
       * globally. Partition by the context handle first so converged lanes
       * using private contexts never reserve pseq/CQ positions on another
       * context. */
      peer(cooperative_groups::labeled_partition(
        cooperative_groups::labeled_partition(active, (unsigned long long)(uintptr_t)dev), peerIdx)) {}
  NCCL_DEVICE_INLINE cooperative_groups::coalesced_group qp(efa_cuda_qp* q) const {
    return cooperative_groups::labeled_partition(active, (unsigned long long)(uintptr_t)q);
  }
};
template <>
struct EfaGdaPostGroups<true> {
  EfaGdaSoloGroup active;
  EfaGdaSoloGroup peer;
  NCCL_DEVICE_INLINE EfaGdaPostGroups(nccl_ofi_gin_gdaki_dev_handle*, uint32_t) {}
  NCCL_DEVICE_INLINE EfaGdaSoloGroup qp(efa_cuda_qp*) const { return EfaGdaSoloGroup{}; }
};

/* ── ringDoorbell: shared doorbell-ring used by the post-path ring sites ─

 * Rings the SQ doorbell to `target`, then advances the bookkeeping cursors:
 * submitted_count grows by the newly-rung span (target - db_rung) so Flush's
 * completion wait is exact, and db_rung (wqes_posted) is published to record
 * how far the doorbell has been rung.
 *
 * Preconditions the caller must satisfy:
 *   - The caller is allowed to ring up to `target` (it holds the doorbell
 *     turn, or is draining already-handed-off slots).
 *   - Every WQE in [db_rung, target) is already visible to the NIC. Each
 *     producing group publishes its own WQEs with __threadfence_system()
 *     before handoff, so the WQE data is system-visible by the time any slot
 *     is eligible to be rung here.
 *
 * Only a post-doorbell fence is emitted (to order the doorbell MMIO write).
 * A pre-doorbell publish fence would be useless: __threadfence_system()
 * orders only the calling thread's own writes, and the WQEs being rung were
 * written by other threads.
 *
 * Under SingleIssuerPerQp the post-doorbell fence is omitted; see the
 * soundness argument at NCCL_GIN_EFA_GDA_SINGLE_ISSUER_PER_QP above. */
template <ncclGinResourceSharingMode mode, bool SingleIssuerPerQp = false>
NCCL_DEVICE_INLINE static void ringDoorbell(efa_cuda_qp* qp, uint64_t* submitted_count_ptr,
                                            cuda::atomic_ref<uint32_t, ncclGinScope<mode>>& dbrung_ref,
                                            uint32_t db_rung, uint32_t target) {
  uint64_t dbAddr = (uint64_t)__cvta_generic_to_global(qp->sq.wq.db);
  asm volatile("st.mmio.relaxed.sys.global.b32 [%0], %1;" : : "l"(dbAddr), "r"(target) : "memory");
  if NCCL_IF_CONSTEXPR (!SingleIssuerPerQp) {
    /* Order the doorbell MMIO write. Use acq_rel (MEMBAR.ALL.SYS) instead of
     * __threadfence_system (MEMBAR.SC.SYS). */
    cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_system);
  }
  scopedAtomicAdd<ncclGinScope<mode>, cuda::memory_order_relaxed>(submitted_count_ptr, (uint64_t)(target - db_rung));
  dbrung_ref.store(target, cuda::memory_order_release);
}

/* Single-owner ring: same doorbell write, plain cursor updates, no fence. No
 * other accessor can observe a handoff, and this owner's successive doorbell
 * writes to the same MMIO address stay in program order at system scope. */
NCCL_DEVICE_INLINE static void ringDoorbellSingleOwner(efa_cuda_qp* qp, uint64_t* submitted_count_ptr,
                                                        uint32_t db_rung, uint32_t target) {
  uint64_t dbAddr = (uint64_t)__cvta_generic_to_global(qp->sq.wq.db);
  asm volatile("st.mmio.relaxed.sys.global.b32 [%0], %1;" : : "l"(dbAddr), "r"(target) : "memory");
  *submitted_count_ptr += (uint64_t)(target - db_rung);
  qp->sq.wq.wqes_posted = target;
}

/* ── Register-resident WQE image ──────────────────────────────────────
 *
 * NCCL_GIN_EFA_GDA_WQE_REGISTER_BUILD (build-time, default OFF)
 *   Builds the WQE image as 64-bit words in registers and stores them to the SQ
 *   slot directly, instead of going through EfaCudaWrBuilder and a stack-resident
 *   efa_io_tx_wqe_128 staging buffer.
 *
 *   Why. The builder cannot keep the WQE in registers: it addresses the buffer
 *   through a uint8_t* at offsets loaded at runtime from efa_cuda_wr_ctx, does
 *   byte-granular read-modify-writes on the ctrl bytes, and zeroes the buffer
 *   with a runtime-trip loop. In SASS that is ~30 STL (byte/halfword/word,
 *   partially overlapping) plus 4 dependent LDG.U8 offset loads, then 12
 *   LDL.128 to read the image back before the MMIO stores: a local-memory
 *   round trip on every put.
 *
 *   The register build is ~20 integer ops with no memory traffic. Field
 *   placement is pinned at compile time with static_asserts against the
 *   efa_io_tx_wqe / efa_io_tx_wqe_128 layouts, which are exactly what the host
 *   side (efa_init_sq_wr_ctx_v0) uses to populate the runtime offsets. No device
 *   assert is emitted on the put path; a shipping version must compare the
 *   published wr_ctx offsets against EfaGdaWqeRegs::k*Offset once on the host
 *   at context creation so a layout skew fails there.
 *
 * The RDMA write/read WQE as WqeBytes/8 little-endian 64-bit words. Every index
 * below is a compile-time constant, so the array lives in registers. Word map
 * (byte offsets from efa_io_defs.h; pinned by the static_asserts):
 *
 *   w0  [ 0.. 8)  req_id:16 | ctrl1:8 | ctrl2:8 | dest_qp_num:16 | length:16
 *   w1  [ 8..16)  immediate_data:32 | ah:16 | ctrl3:8 | reserved:8
 *   w2  [16..24)  qkey:32 | reserved2[0..4)
 *   w3  [24..32)  reserved2[4..6) | req_id_ex.w[0..3)   == req_id & ~0xFFFF
 *   w4  [32..40)  remote_mem.length:32 | remote_mem.rkey:32
 *   w5  [40..48)  remote_mem.buf_addr (lo | hi<<32)
 *   w6  [48..56)  local_mem.length:32 | local_mem.lkey:32     (or inline data)
 *   w7  [56..64)  local_mem.buf_addr                           (or inline data)
 *   w8..w15       MBZ (128-byte WQE only; inline data may extend into them)
 *
 * The 64B and 128B layouts place remote_mem and local_mem at the same offsets
 * (32 and 48); RDMA-write inline data also starts at 48 and exists only in the
 * 128B form. */
#ifndef NCCL_GIN_EFA_GDA_WQE_REGISTER_BUILD
#define NCCL_GIN_EFA_GDA_WQE_REGISTER_BUILD 0
#endif

template <uint32_t WqeBytes>
struct EfaGdaWqeRegs {
  static_assert(WqeBytes == 64u || WqeBytes == 128u, "EFA WQE is 64 or 128 bytes");
  static constexpr uint32_t Words = WqeBytes / 8u;
  uint64_t w[Words];

  /* Single-bit ctrl flags from efa_io_defs.h. Spelled out here because that
   * header defines them via BIT(), which efa_cuda_dp_impl.cuh #undef's at its
   * end; the multi-bit GENMASK() fields remain usable and are taken from it. */
  static constexpr uint64_t kCtrl1MetaDesc = 1ull << 7;    /* EFA_IO_TX_META_DESC_META_DESC  */
  static constexpr uint64_t kCtrl1InlineMsg = 1ull << 5;   /* EFA_IO_TX_META_DESC_INLINE_MSG */
  static constexpr uint64_t kCtrl2Phase = 1ull << 0;       /* EFA_IO_TX_META_DESC_PHASE      */
  static constexpr uint64_t kCtrl2First = 1ull << 2;       /* EFA_IO_TX_META_DESC_FIRST      */
  static constexpr uint64_t kCtrl2Last = 1ull << 3;        /* EFA_IO_TX_META_DESC_LAST       */
  static constexpr uint64_t kCtrl2CompReq = 1ull << 4;     /* EFA_IO_TX_META_DESC_COMP_REQ   */

  /* Offsets the host writes into efa_cuda_wr_ctx (efa_init_sq_wr_ctx_v0). */
  static constexpr uint32_t kRemoteMemOffset = 32u;
  static constexpr uint32_t kLocalMemOffset = 48u;
  static constexpr uint32_t kWriteInlineOffset = (WqeBytes == 128u) ? 48u : 0u;

  static_assert(sizeof(struct efa_io_tx_wqe) == 64u, "efa_io_tx_wqe must be 64 bytes");
  static_assert(sizeof(struct efa_io_tx_wqe_128) == 128u, "efa_io_tx_wqe_128 must be 128 bytes");
  static_assert(offsetof(struct efa_io_tx_meta_desc, req_id) == 0u, "meta.req_id");
  static_assert(offsetof(struct efa_io_tx_meta_desc, ctrl1) == 2u, "meta.ctrl1");
  static_assert(offsetof(struct efa_io_tx_meta_desc, ctrl2) == 3u, "meta.ctrl2");
  static_assert(offsetof(struct efa_io_tx_meta_desc, dest_qp_num) == 4u, "meta.dest_qp_num");
  static_assert(offsetof(struct efa_io_tx_meta_desc, length) == 6u, "meta.length");
  static_assert(offsetof(struct efa_io_tx_meta_desc, immediate_data) == 8u, "meta.immediate_data");
  static_assert(offsetof(struct efa_io_tx_meta_desc, ah) == 12u, "meta.ah");
  static_assert(offsetof(struct efa_io_tx_meta_desc, ctrl3) == 14u, "meta.ctrl3");
  static_assert(offsetof(struct efa_io_tx_meta_desc, qkey) == 16u, "meta.qkey");
  static_assert(offsetof(struct efa_io_tx_meta_desc, req_id_ex) == 26u, "meta.req_id_ex");
  static_assert(offsetof(struct efa_io_tx_wqe, data.rdma_req.remote_mem) == kRemoteMemOffset, "64B remote_mem");
  static_assert(offsetof(struct efa_io_tx_wqe, data.rdma_req.local_mem) == kLocalMemOffset, "64B local_mem");
  static_assert(offsetof(struct efa_io_tx_wqe_128, data.rdma_req.remote_mem) == kRemoteMemOffset, "128B remote_mem");
  static_assert(offsetof(struct efa_io_tx_wqe_128, data.rdma_req.local_mem) == kLocalMemOffset, "128B local_mem");
  static_assert(offsetof(struct efa_io_tx_wqe_128, data.rdma_req.inline_data) == 48u, "128B inline_data");
  static_assert(offsetof(struct efa_io_remote_mem_addr, length) == 0u && offsetof(struct efa_io_remote_mem_addr, rkey) == 4u &&
                offsetof(struct efa_io_remote_mem_addr, buf_addr_lo) == 8u, "remote_mem layout");
  static_assert(offsetof(struct efa_io_tx_buf_desc, length) == 0u && offsetof(struct efa_io_tx_buf_desc, lkey) == 4u &&
                offsetof(struct efa_io_tx_buf_desc, buf_addr_lo) == 8u, "tx_buf_desc layout");

  /* Everything except the payload descriptor and the phase bit. Mirrors
   * EfaCudaWrBuilder::init_wr + set_remote_mem + set_remote +
   * set_processing_hints(BURST_PPS_SENSITIVE). meta.length and remote_mem.length
   * are left zero for the payload encoder to fill. */
  template <efaGdaRdmaOp op>
  NCCL_DEVICE_INLINE void initRdma(uint64_t reqId, uint16_t ah, uint16_t qpn, uint32_t qkey, uint64_t dstAddr,
                                   uint32_t dstRkey) {
    constexpr uint32_t opType = (op == EFA_GDA_RDMA_READ) ? (uint32_t)EFA_IO_RDMA_READ : (uint32_t)EFA_IO_RDMA_WRITE;
    constexpr uint64_t ctrl1 = kCtrl1MetaDesc | (opType & EFA_IO_TX_META_DESC_OP_TYPE_MASK);
    constexpr uint64_t ctrl2 = kCtrl2First | kCtrl2Last | kCtrl2CompReq;
    constexpr uint64_t ctrl3 = (uint64_t)EFA_IO_PROCESSING_HINT_BURST_PPS_SENSITIVE & EFA_IO_TX_META_DESC_PROCESSING_HINTS_MASK;

    w[0] = (reqId & 0xFFFFull) | (ctrl1 << 16) | (ctrl2 << 24) | ((uint64_t)qpn << 32);
    w[1] = ((uint64_t)ah << 32) | (ctrl3 << 48);
    w[2] = (uint64_t)qkey;
    w[3] = reqId & ~0xFFFFull;   /* req_id_ex.w[0..3) at bytes 26..32 */
    w[4] = (uint64_t)dstRkey << 32;
    w[5] = dstAddr;
#pragma unroll
    for (uint32_t i = 6; i < Words; i++) w[i] = 0;
  }

  /* One SGE: meta.length = 1 (SGL entry count), remote and local lengths = bytes. */
  NCCL_DEVICE_INLINE void setSge(uint32_t lkey, uint64_t addr, uint32_t bytes) {
    w[0] |= 1ull << 48;
    w[4] |= (uint64_t)bytes;
    w[6] = (uint64_t)bytes | ((uint64_t)(lkey & EFA_IO_TX_BUF_DESC_LKEY_MASK) << 32);
    w[7] = addr;
  }

  /* Inline RDMA-write payload of <= 8 bytes at byte 48 (128B WQE only; the 64B
   * WQE has no RDMA-write inline form, which EfaCudaWrBuilder reports as -EINVAL). */
  template <typename T>
  NCCL_DEVICE_INLINE void setInline(T value) {
    static_assert(sizeof(T) <= 8, "inline payload must fit one word");
    if NCCL_IF_CONSTEXPR (WqeBytes == 128u) {
      uint64_t raw = 0;
      memcpy(&raw, &value, sizeof(T));
      w[0] |= (kCtrl1InlineMsg << 16) | ((uint64_t)sizeof(T) << 48);
      w[4] |= (uint64_t)sizeof(T);
      w[6] = raw;
    } else {
      assert(false && "EFA GDA: RDMA write inline requires the 128-byte WQE");
      (void)value;
    }
  }

  NCCL_DEVICE_INLINE void setPhase(uint32_t phase) {
    w[0] |= ((uint64_t)phase & kCtrl2Phase) << 24;
  }

  /* Store the image to its SQ slot with relaxed system-scope MMIO stores in
   * ascending address order, straight from registers. No fence; caller publishes. */
  NCCL_DEVICE_INLINE void storeMmio(uint64_t dstAddr) const {
#pragma unroll
    for (uint32_t i = 0; i < Words; i++) {
      asm volatile("st.mmio.relaxed.sys.global.b64 [%0], %1;"
                   :
                   : "l"(dstAddr + i * 8u), "l"(w[i])
                   : "memory");
    }
  }
};

/* ── RDMA payload encoders ───────────────────────────────────────── */

/* Put, signal, and Get operations transfer their payload through an SGE. */
struct RdmaSgeEncoder {
  uint64_t addr;
  uint32_t lkey;
  uint32_t bytes;

  NCCL_DEVICE_INLINE void encode(EfaCudaWrBuilder& wr) {
    int ret = wr.set_sge(lkey, addr, bytes);
    assert(ret == 0 && "EFA GDA: failed to encode RDMA SGE");
    (void)ret;
  }

  template <uint32_t WqeBytes>
  NCCL_DEVICE_INLINE void encodeRegs(EfaGdaWqeRegs<WqeBytes>& regs) {
    regs.setSge(lkey, addr, bytes);
  }
};

/* PutValue carries its payload directly in the 128-byte RDMA-write WQE. */
template <typename T>
struct PutValuePayloadEncoder {
  T srcVal;

  NCCL_DEVICE_INLINE PutValuePayloadEncoder(T value) : srcVal(value) {}

  NCCL_DEVICE_INLINE void encode(EfaCudaWrBuilder& wr) {
    int ret = wr.set_inline_data(&srcVal, sizeof(T));
    assert(ret == 0 && "EFA GDA: failed to encode inline PutValue");
    (void)ret;
  }

  template <uint32_t WqeBytes>
  NCCL_DEVICE_INLINE void encodeRegs(EfaGdaWqeRegs<WqeBytes>& regs) {
    regs.setInline(srcVal);
  }
};

/* Register-build member write: build the WQE image in registers for a
 * compile-time WQE size, patch the phase bit, and store it to the slot. reqId
 * is the (peer, pseq) request id the completion echoes back. */
template <efaGdaRdmaOp op, uint32_t WqeBytes, typename PayloadEncoder>
NCCL_DEVICE_INLINE static void writeWqeRegs(uint64_t slotAddr, uint32_t wqe_phase, uint64_t reqId, uint16_t ah,
                                            uint16_t qpn, uint32_t qkey, uint64_t dstAddr, uint32_t dstRkey,
                                            PayloadEncoder& payloadEncoder) {
  EfaGdaWqeRegs<WqeBytes> regs;
  regs.template initRdma<op>(reqId, ah, qpn, qkey, dstAddr, dstRkey);
  payloadEncoder.encodeRegs(regs);
  regs.setPhase(wqe_phase);
  regs.storeMmio(slotAddr);
}

/* ── postRdmaOp: admission gate + shared post path for Put, PutValue and Get ── */

/* Posts an RDMA operation on `ep`'s local QP to the remote QP given by
 * the explicit (ah, qpn, qkey) tuple. The local poster QP and the remote
 * target QP are chosen independently by the caller: counterId selects
 * the local poster (this `ep`), the target slot selects the remote tuple
 * (via the poster's [total_slots*nranks] target table).
 *
 * PayloadEncoder runs after this lane's SQ slot is known and either attaches
 * a local SGE or writes inline data into the WQE.
 *
 *
 * `dev` and `peerIdx` identify the target peer: they drive req_id stamping and
 * the per-peer and per-context backpressure and counts. */
template <ncclGinResourceSharingMode mode, efaGdaRdmaOp op = EFA_GDA_RDMA_WRITE,
          bool SingleIssuerPerQp = (NCCL_GIN_EFA_GDA_SINGLE_ISSUER_PER_QP != 0),
          bool RegisterBuild = (NCCL_GIN_EFA_GDA_WQE_REGISTER_BUILD != 0),
          bool SingleOwner = efaGdaSingleOwner<mode>,
          typename PayloadEncoder>
NCCL_DEVICE_INLINE static void postRdmaOp(nccl_ofi_gin_gdaki_dev_handle* dev,
                                          nccl_ofi_gin_gdaki_dev_endpoint_handle* ep, uint32_t peerIdx, uint16_t ah,
                                          uint16_t qpn, uint32_t qkey, uint64_t dstAddr, uint32_t dstRkey,
                                          PayloadEncoder payloadEncoder, uint32_t optFlags = ncclGinOptFlagsDefault) {
  /* ── Admission: leader-only over contiguous per-peer blocks ──────────
   *
   * A write takes its position in the peer's posting sequence once, then joins the
   * posting pass below only when that position is inside the peer's window and the
   * shared CQ has room. The lanes converged here partition by peer, one leader per
   * peer reserves the subgroup's block with a single atomic, and that leader alone
   * waits until the block's top position has room. The members idle at
   * warp-internal syncs, so the PCIe-resident counters see one reader per peer
   * instead of one per lane.
   *
   * Waiting while holding a block is safe because admission happens before the
   * posting group is formed: a waiting lane holds no doorbell turn and no SQ slot.
   * Every position below the block's base belongs to an earlier block, groups take
   * the doorbell in strict slot order, and the max_batch gate in the posting pass
   * bounds the un-rung depth, so the completions a leader waits for always reach
   * the NIC without help from any lane waiting here. This needs
   * max_batch + 32 < peer_window, which the plugin checks at context setup. */
  EfaGdaPostGroups<SingleOwner> groups(dev, peerIdx);
  auto& active = groups.active;
  auto& peerGroup = groups.peer;

  uint32_t blockSize = (uint32_t)peerGroup.num_threads();
  uint32_t blockBase = 0;
  if (peerGroup.thread_rank() == 0) {
    if NCCL_IF_CONSTEXPR (SingleOwner) {
      /* Exclusive owner: plain read-modify-write on the producer count. */
      blockBase = dev->submitted_count_per_peer[peerIdx];
      dev->submitted_count_per_peer[peerIdx] = blockBase + blockSize;
    } else {
      blockBase = scopedAtomicFetchAdd<ncclGinScope<mode>, cuda::memory_order_relaxed>(
        &dev->submitted_count_per_peer[peerIdx], blockSize);
    }
  }
  blockBase = peerGroup.shfl(blockBase, 0);
  uint32_t pseq = blockBase + (uint32_t)peerGroup.thread_rank();

  if (peerGroup.thread_rank() == 0) {
    uint32_t peerTop = blockBase + blockSize;
    /* The block has room once the peer's ordered prefix has come within
     * peer_window of the block's claimed count, which keeps the host's per-peer
     * completion bitmap a fixed size and every live position in it distinct.
     * The prefix only advances, so once the block fits it keeps fitting. */
    while ((uint32_t)(peerTop - scopedAtomicLoad<cuda::thread_scope_system, cuda::memory_order_acquire>(
                                  &dev->ordered_completed_count_per_peer[peerIdx])) > dev->peer_window) {
      /* spin */
    }
      /* The block claims its span of the context's shared CQ with one atomic, then
     * waits until the host has read enough entries for that span to fit within
     * cq_depth, since an overflow drops completions. Claiming before waiting is
     * what bounds occupancy: concurrent leaders reserve disjoint spans instead of
     * all passing the same reading of an unclaimed counter. The host only advances
     * completed_count_per_ctx, so once the span fits it keeps fitting. */
    uint64_t cqTop;
    if NCCL_IF_CONSTEXPR (SingleOwner) {
      cqTop = *dev->submitted_count_per_ctx + (uint64_t)blockSize;
      *dev->submitted_count_per_ctx = cqTop;
    } else {
      cqTop = scopedAtomicFetchAdd<ncclGinScope<mode>, cuda::memory_order_relaxed>(dev->submitted_count_per_ctx,
                                                                                   (uint64_t)blockSize) +
              (uint64_t)blockSize;
    }
    while (cqTop -
             scopedAtomicLoad<cuda::thread_scope_system, cuda::memory_order_relaxed>(dev->completed_count_per_ctx) >
           (uint64_t)dev->cq_depth) {
        /* spin */
    }
  }
  /* The members wait for their leader's admission here, and the peer subgroups
   * then reconverge into the full active group, so the posting pass below still
   * runs the warp as one group under one doorbell. */
  peerGroup.sync();
  active.sync();

  /* ── Posting: one group per QP, one doorbell per batch ─────────────── */
  efa_cuda_qp* qp = (efa_cuda_qp*)ep->qp;
  uint64_t* submitted_count_ptr = &ep->submitted_count;

  /* WQE staging buffer (builder path only). Always the 128B form: the builder
   * zeroes and the MMIO loop copies only the QP's negotiated wqe_size, but
   * sizing the storage to the larger layout keeps one buffer valid for both 64B
   * and 128B QPs. Under RegisterBuild the image is assembled in registers
   * instead (see EfaGdaWqeRegs) and none of this is emitted. */
  efa_io_tx_wqe_128 wr_storage;
  uint16_t wqe_size = qp->sq.wr_ctx.wqe_size;

  /* req_id carries this write's (peer, pseq): peer in the high bits, pseq in the low
   * EFA_GDA_PSEQ_BITS, matching NCCL_OFI_GDAKI_PSEQ_BITS in the plugin. pseq is the
   * position this write reserved in that peer's own posting sequence, the same
   * counter FlushAsync snapshots. The EFA completion echoes req_id back
   * (efa_io_cdesc_common), so the host reads peer and pseq from the CQE and advances
   * that peer's ordered_completed_count_per_peer. The admission gate above admits a
   * write only while its position is within peer_window of that prefix, so pseq
   * uniquely identifies each live write. */
  const uint64_t wrReqId =
    (((uint64_t)peerIdx & EFA_GDA_PEER_MASK) << EFA_GDA_PSEQ_BITS) | ((uint64_t)pseq & EFA_GDA_PSEQ_MASK);
  EfaCudaWrBuilder wr(&qp->sq.wr_ctx, (uint8_t*)&wr_storage);
  if NCCL_IF_CONSTEXPR (!RegisterBuild) {
    /* The opcode is the only thing that differs between a Put and a Get here: both
     * carry the RDMA address pair (dstAddr, dstRkey) and the local buffer in the SGE,
     * and the opcode decides which way the bytes move. A read therefore arrives with
     * the REMOTE source in the RDMA pair and the LOCAL destination in the SGE. */
    if NCCL_IF_CONSTEXPR (op == EFA_GDA_RDMA_READ) {
      wr.init_rdma_read(wrReqId, dstRkey, dstAddr);
    } else {
      wr.init_rdma_write(wrReqId, dstRkey, dstAddr);
    }
    wr.set_remote(ah, (uint32_t)qpn, qkey);
    /* Tag the WQE as PPS-sensitive. GIN puts are small, high-rate writes, so
     * ask the NIC to optimize for packets-per-second (burst PPS) rather than
     * bandwidth. This sets the PROCESSING_HINTS field in the WQE meta
     * descriptor (ctrl3); it is a hint, so the device may ignore it. */
    wr.set_processing_hints(EFA_CUDA_PROCESSING_HINT_BURST_PPS_SENSITIVE);
  }

  /* Sliding-window SQ post with warp coalescing (Stage 2).
   *
   * Inlines the reserve / write / doorbell sequence directly against
   * the efa_cuda_qp ring. Two shared cursors in the QP coordinate all
   * posters (across lanes, warps and CTAs):
   *
   *   pc             : monotonic reservation index. A group's leader
   *                    claims its whole range with one atomicAdd(+g).
   *   wqes_completed : "released" cursor — the rendezvous/handoff token.
   *                    Advances when a group passes the doorbell turn,
   *                    whether or not it actually rang (so deferred
   *                    groups still hand off in strict slot order).
   *   wqes_posted    : "doorbell rung" cursor (db_rung) — the value last
   *                    written to the SQ doorbell register. With request
   *                    aggregation (ncclGinOptFlagsAggregateRequests) the
   *                    doorbell is deferred: a group writes its WQEs and
   *                    hands off without ringing; a later non-aggregated
   *                    group rings to its own chunk_next, which
   *                    is >= every deferred slot (the doorbell is
   *                    monotonic), draining the whole batch in one ring.
   *                    Un-rung WQEs are bounded by max_batch via the
   *                    window check below (gated on db_rung), so the EFA
   *                    staging limit is always respected.
   *
   * Coalescing: the active group's lanes targeting the same QP form a
   * group via labeled_partition(qp). The leader reserves g
   * = qpGroup.num_threads() contiguous slots; every member writes its own
   * WQE in parallel; the leader rings one doorbell for the batch.
   *
   * max_batch bound: a group may be larger than the EFA staging limit
   * (a warp can have up to 32 lanes on one QP), so the group is chunked
   * into windows of <= max_batch. For each chunk:
   *   - window-wait (leader): write only once the chunk fits within the
   *     released window [released, released + max_batch). This bounds
   *     un-doorbelled WQEs across ALL concurrent groups to max_batch.
   *   - members write their WQEs in parallel.
   *   - doorbell rendezvous (leader): wait until released == chunk_base
   *     (strict slot order across groups), ring the doorbell, then
   *     advance released to hand off to the next group. */
  auto qpGroup = groups.qp(qp);

  int my_idx = qpGroup.thread_rank();
  int group_size = qpGroup.num_threads();
  bool is_leader = (my_idx == 0);
  uint32_t max_batch = qp->sq.wq.max_batch;

  /* db_rung reuses the wqes_posted field, which the GIN path does not
   * otherwise use (zero-initialized by the plugin). Under SingleOwner the
   * cursors are plain fields; the shared modes wrap them in atomic_refs at
   * each use site below. */
  const bool aggregate = (optFlags & ncclGinOptFlagsAggregateRequests) != 0;

  /* Leader reserves the whole group's contiguous slot range. */
  uint32_t base = 0;
  if (is_leader) {
    if NCCL_IF_CONSTEXPR (SingleOwner) {
      base = qp->sq.wq.pc;
      qp->sq.wq.pc = base + (uint32_t)group_size;
    } else {
      cuda::atomic_ref<uint32_t, ncclGinScope<mode>> pc_ref(qp->sq.wq.pc);
      base = pc_ref.fetch_add((uint32_t)group_size, cuda::memory_order_relaxed);
    }
  }
  base = qpGroup.shfl(base, 0);

  /* Chunk the group into windows of <= max_batch. */
  for (int chunk_start = 0; chunk_start < group_size; chunk_start += (int)max_batch) {
    int chunk_size = min((int)max_batch, group_size - chunk_start);
    uint32_t chunk_base = base + (uint32_t)chunk_start;
    uint32_t chunk_next = chunk_base + (uint32_t)chunk_size;

    if (is_leader) {
      /* Backpressure: keep the number of written-but-un-rung WQEs within
       * max_batch (the hard EFA staging limit). The un-rung depth this chunk
       * would reach is chunk_next - db_rung.
       *
       * We must ring to make room rather than just wait: a deferring group
       * leaves its WQEs un-rung and hands off without ringing, so db_rung is
       * only ever advanced by a group that actively rings. If this group
       * blocked passively, no one would ring the deferred slots below it and
       * it would wait forever.
       *
       * So while the depth would exceed max_batch, make room by ringing the
       * deferred work that is already written. Only the turn-holder
       * (base_ref == chunk_base) may ring, and it rings up to chunk_base:
       * those slots were written and published by earlier groups that handed
       * off before us, and the deferred span below chunk_base is itself
       * <= max_batch, so one doorbell drains it. If we do not yet hold the
       * turn, a lower group does and will either ring (advancing db_rung) or
       * hand off (advancing base_ref); keep checking until our chunk fits. */
      if NCCL_IF_CONSTEXPR (SingleOwner) {
        /* The owner is always the turn-holder; ring the deferred batch itself. */
        uint32_t db_rung = qp->sq.wq.wqes_posted;
        if (chunk_next - db_rung > max_batch && chunk_base != db_rung) {
          ringDoorbellSingleOwner(qp, submitted_count_ptr, db_rung, chunk_base);
        }
      } else {
        cuda::atomic_ref<uint32_t, ncclGinScope<mode>> base_ref(qp->sq.wq.wqes_completed);
        cuda::atomic_ref<uint32_t, ncclGinScope<mode>> dbrung_ref(qp->sq.wq.wqes_posted);
        while (chunk_next - dbrung_ref.load(cuda::memory_order_relaxed) > max_batch) {
          if (base_ref.load(cuda::memory_order_acquire) == chunk_base) {
            uint32_t db_rung = dbrung_ref.load(cuda::memory_order_relaxed);
            if (chunk_base != db_rung) {   /* deferred, already-written batch */
              ringDoorbell<mode, SingleIssuerPerQp>(qp, submitted_count_ptr, dbrung_ref, db_rung, chunk_base);
            }
          }
        }
      }
    }
    qpGroup.sync();   /* members wait for leader's backpressure before writing */

    /* Members in this window write their own WQE into their slot. */
    if (my_idx >= chunk_start && my_idx < chunk_start + chunk_size) {
      uint32_t my_slot = chunk_base + (uint32_t)(my_idx - chunk_start);
      uint32_t sq_idx = my_slot & qp->sq.wq.queue_mask;
      uint32_t wqe_phase = (my_slot >> qp->sq.wq.queue_size_shift) & 1u;
      uint64_t* dst = (uint64_t*)(qp->sq.wq.buf + sq_idx * wqe_size);
      uint64_t slotAddr = (uint64_t)__cvta_generic_to_global(dst);

      if NCCL_IF_CONSTEXPR (RegisterBuild) {
        /* wqe_size is negotiated per QP and is 64 or 128. Branch once so the
         * image size is a compile-time constant and the words stay in registers. */
        if (wqe_size == 64u) {
          writeWqeRegs<op, 64u>(slotAddr, wqe_phase, wrReqId, ah, qpn, qkey, dstAddr, dstRkey, payloadEncoder);
        } else {
          writeWqeRegs<op, 128u>(slotAddr, wqe_phase, wrReqId, ah, qpn, qkey, dstAddr, dstRkey, payloadEncoder);
        }
      } else {
        payloadEncoder.encode(wr);

        wr_storage.meta.ctrl2 = (wr_storage.meta.ctrl2 & ~(uint8_t)1u) | ((uint8_t)wqe_phase & (uint8_t)1u);
        uint64_t* src = (uint64_t*)&wr_storage;
        /* One final system-scope fence publishes the complete WQE after these
         * relaxed MMIO stores. */
        uint32_t num_words = wqe_size / (uint32_t)sizeof(uint64_t);
        for (uint32_t i = 0; i < num_words; i++) {
          uint64_t value = src[i];
          asm volatile("st.mmio.relaxed.sys.global.b64 [%0], %1;"
                       :
                       : "l"(slotAddr + i * sizeof(uint64_t)), "l"(value)
                       : "memory");
        }
      }
      /* Publish this group's WQE writes to system scope so they are visible
       * to the NIC whenever any doorbell rings a slot in this range. */
      cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_system);
    }
    qpGroup.sync();   /* all members' WQE writes for this chunk are done */

    if (is_leader) {
      if NCCL_IF_CONSTEXPR (SingleOwner) {
        /* No rendezvous: the owner always holds the turn. Ring unless
         * aggregating, or if deferring would exceed the staging limit. The
         * deferred path needs no publish fence either: the only doorbell that
         * can ever ring these slots is this owner's, and its WQE stores are
         * already published per put by the pre-doorbell fence above. */
        uint32_t db_rung = qp->sq.wq.wqes_posted;
        bool must_ring = (!aggregate) || (chunk_next - db_rung >= max_batch);
        if (must_ring) ringDoorbellSingleOwner(qp, submitted_count_ptr, db_rung, chunk_next);
        qp->sq.wq.wqes_completed = chunk_next;
      } else {
        cuda::atomic_ref<uint32_t, ncclGinScope<mode>> base_ref(qp->sq.wq.wqes_completed);
        cuda::atomic_ref<uint32_t, ncclGinScope<mode>> dbrung_ref(qp->sq.wq.wqes_posted);
        /* Doorbell-order rendezvous: take the turn in strict slot order. */
        while (base_ref.load(cuda::memory_order_relaxed) != chunk_base) {
          /* spin */
        }

        /* Ring unless aggregating. Force a ring if deferring would leave
         * more than max_batch un-rung WQEs (db_rung is the last rung slot),
         * so the EFA staging limit is never exceeded. When we do ring, ring
         * to chunk_next: it is >= every deferred slot below us (we hold the
         * turn in slot order), so one doorbell drains the whole contiguous
         * batch. submitted_count advances by everything since db_rung. */
        uint32_t db_rung = dbrung_ref.load(cuda::memory_order_relaxed);
        bool must_ring = (!aggregate) || (chunk_next - db_rung >= max_batch);
        if (must_ring) {
          ringDoorbell<mode, SingleIssuerPerQp>(qp, submitted_count_ptr, dbrung_ref, db_rung, chunk_next);
        } else {
          /* Publish this group's WQE writes to system scope before handing off,
           * so they are visible to the NIC whenever any doorbell (this group's or
           * a later draining group's) rings a slot in this range.
           * Each group must publish its own writes: __threadfence_system()
           * orders only the calling thread's writes, and the handoff (base_ref)
           * is device/block scope, so a later thread's fence cannot publish this
           * group's writes for it. Runs only on the defer path. */
          cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_system);
        }
        base_ref.store(chunk_next, cuda::memory_order_release);   /* hand off to next group */
      }
    }
    qpGroup.sync();   /* chunk fully posted before the next chunk */
  }
}

/* ── putImplMode: mode-templated Put implementation ─────────────── */

template <ncclGinResourceSharingMode mode, typename Coop>
NCCL_DEVICE_INLINE static void putImplMode(ncclGinCtx ctx, Coop coop, int peer, bool hasWins, ncclGinWindow_t dstWin,
                                           size_t dstOff, ncclGinWindow_t srcWin, size_t srcOff, size_t bytes,
                                           ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                           uint64_t signalOpArg, bool hasCounter, ncclGinCounter_t counterId,
                                           bool hasDescriptor, ncclGinDescriptorSmem* descriptor,
                                           cuda::thread_scope required, cuda::thread_scope given, uint32_t optFlags) {
  coop.sync();
  if (coop.thread_rank() == 0) {
    nccl_ofi_gin_gdaki_dev_handle* dev = getDevHandle(ctx);

    bool hasPayload = hasWins && bytes > 0;

    /* This backend supports INDEXED signals only. EFA's FI_REMOTE_WRITE
     * counter ticks exactly once per inbound write and has no atomic-add,
     * so a signal Add-by-N is emulated as N inbound write events (see the
     * posting block below). VA-typed signals are not representable. */
    assert((signal.type == NCCL_GIN_SIGNAL_TYPE_NONE || signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED) &&
           "EFA GDA: only INDEXED signals are supported");
    assert((signal.type != NCCL_GIN_SIGNAL_TYPE_INDEXED || (int)signal.indexedSignal.signalId < dev->nSignals) &&
           "EFA GDA: signalId out of range");
    assert((!hasCounter || (int)counterId < dev->nCounters) && "EFA GDA: counterId out of range");

    /* A Put ALWAYS posts at least one WQE -- even with no payload and no
     * signal/counter, where it degenerates to a 0-byte write to the peer's
     * per-context scratch via the peer DATA endpoint (target slot 0), which
     * binds no FI_REMOTE_WRITE: remotely unobservable.
     *
     * Empty puts cannot be dropped as no-ops, because under
     * ncclGinOptFlagsAggregateRequests a put carries a LOCAL side effect: its
     * non-aggregated doorbell rendezvous in postRdmaOp is what publishes
     * earlier deferred WQEs on the QP. Dropping an "empty" put would make it
     * impossible to terminate a deferred stream whose last real put was
     * aggregated -- the tail WQEs (and their signals) would never be handed to
     * the NIC and the peer would wait forever. Callers that want to elide
     * empty puts for performance should skip at the call site, where "empty"
     * is actually known. */
    {
      /* Three WQE patterns:
       *
       * (a) Data put: posts an RDMA write of the user payload.
       *     Routed through the signal/counter endpoint when a signal or
       *     counter is attached, so the receiver's FI_REMOTE_WRITE fires
       *     on completion; otherwise routed through the data endpoint.
       *
       * (b) Signal-only: posts a 0-byte RDMA write into the peer's
       *     per-context scratch buffer. The write event bumps the
       *     receiver's FI_REMOTE_WRITE counter on the signal endpoint.
       *
       * (c) Empty (no payload, no signal/counter): same 0-byte scratch
       *     write as (b) but addressed to the peer DATA endpoint, which
       *     binds no FI_REMOTE_WRITE -- remotely unobservable. Posted for
       *     its local doorbell rendezvous (see the block comment above). */
      uint64_t absSrcAddr;
      uint64_t absDstAddr;
      uint32_t dstRkey;
      uint32_t srcLkey;
      uint32_t writeBytes;
      if (hasPayload) {
        /* Resolve the memory window to the rail this context is bound
         * to. The window is an array of per-rail mr_handle pointers;
         * dev->rail_id (= contextId % num_rails, pre-baked by the
         * plugin) selects this context's rail, keeping the path
         * rail-agnostic. */
        nccl_ofi_gin_gdaki_mr_handle* dstMh = ((nccl_ofi_gin_gdaki_mr_handle**)dstWin)[dev->rail_id];
        nccl_ofi_gin_gdaki_mr_handle* srcMh = ((nccl_ofi_gin_gdaki_mr_handle**)srcWin)[dev->rail_id];
        absSrcAddr = srcMh->local_addr + srcOff;
        absDstAddr = dstMh->peers[peer].remote_addr + dstOff;
        dstRkey = dstMh->peers[peer].rkey;
        srcLkey = srcMh->lkey;
        writeBytes = (uint32_t)bytes;
      } else {
        absSrcAddr = dev->scratch_local_addr;
        absDstAddr = dev->scratch_remote_addrs[peer];
        dstRkey = dev->scratch_remote_rkeys[peer];
        srcLkey = dev->scratch_lkey;
        writeBytes = 0;
      }

      /* A single RDMA write carries two independent QP choices:
       *
       *   - Local poster QP: the SQ we post from; its FI_WRITE counter
       *     ticks on local completion. This is a LOCAL property,
       *     selected by counterId. With a counter request we post from
       *     counter_handles[counterId] so its FI_WRITE is the counter;
       *     otherwise we post from the data endpoint (a signal-only or
       *     plain put has no local counter to track here).
       *
       *   - Remote target QP: the peer endpoint we address; the peer's
       *     FI_REMOTE_WRITE counter on THAT endpoint ticks (only when the
       *     target is a signal/sc endpoint). This is a TARGET property,
       *     selected below as a slot into the poster's target
       *     table (slot 0 = peer data EP, slot 1+signalId = peer sc EP).
       *
       * The signal (signalId) selects the remote (target) QP via the
       * slot; the counter (counterId) selects the local (poster) QP. */
      nccl_ofi_gin_gdaki_dev_endpoint_handle* main_ep =
        hasCounter ? &dev->counter_handles[counterId]->base : &dev->data;

      /* Target slot in the [total_slots*nranks] target addressing table
       * (targetSlot-major: idx = targetSlot*nranks + peer):
       *     signalling write (INDEXED) -> slot 1 + signalId (peer sc EP,
       *       whose FI_REMOTE_WRITE the receiver's waitSignal observes)
       *     plain put / counter-only    -> slot 0 (peer DATA EP, which
       *       binds no FI_REMOTE_WRITE, so the write ticks the local
       *       FI_WRITE counter without firing a signal on the receiver)
       * The local poster QP is chosen by counterId (main_ep); the slot
       * picks the remote target. */
      const bool isIndexed = (signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED);
      const uint32_t targetSlot = isIndexed ? (1u + (uint32_t)signal.indexedSignal.signalId) : 0u;
      const uint32_t targetIdx = targetSlot * (uint32_t)dev->nranks + (uint32_t)peer;

      /* Signal increment count.
       *
       * EFA's FI_REMOTE_WRITE counter advances by exactly 1 per inbound
       * write, so an Add-by-N signal is emulated as N inbound write
       * events. Inc is always +1 (signalOpArg is defined to be 1 for Inc
       * by the GIN API). A pure data put or counter request (no signal)
       * contributes a single write.
       *
       * Correctness-first: this issues the writes as separate posts (one
       * doorbell each). A future optimization can batch the doorbell over
       * a larger reservation via postRdmaOp's chunk loop; that must NOT
       * be done by suppressing doorbells across calls, which would break
       * the wqes_completed sliding-window / rendezvous invariant.
       *
       * TODO: batch the doorbells for a signal Add-by-N (and across the
       * payload + scratch writes) instead of ringing one doorbell per
       * write. Must reuse postRdmaOp's chunk loop (one doorbell per
       * max_batch reservation), NOT a doorbell-suppress flag across
       * separate calls. */
      uint32_t signalCount = 1u;
      if (isIndexed && signalOp == ncclGinSignalAdd) {
        signalCount = (uint32_t)signalOpArg;
      }

      /* Target tuples, both read from the target table at the same slot.
       *
       * The local FI_WRITE counter selected by counterId must tick
       * EXACTLY ONCE per Put, no matter how many physical writes the
       * signal Add-by-N expands into. So only the FIRST write rides
       * `main_ep` (the counterId-selected poster when hasCounter, else
       * the data EP); every remaining (signalCount - 1) increment rides
       * the DATA endpoint, whose FI_WRITE is not the caller's counter.
       * Both resolve the SAME remote target (slot), but each through its
       * own endpoint's AV — an address handle is AV-local, so the data EP
       * uses its own tuple, not main_ep's. */
      const uint16_t main_ah = main_ep->target_address_handles[targetIdx];
      const uint16_t main_qpn = main_ep->target_remote_qpns[targetIdx];
      const uint32_t main_qkey = main_ep->target_qkey[targetIdx];
      const uint16_t dataSigAh = dev->data.target_address_handles[targetIdx];
      const uint16_t dataSigQpn = dev->data.target_remote_qpns[targetIdx];
      const uint32_t dataSigQkey = dev->data.target_qkey[targetIdx];

      /* Chunk payloads that exceed the EFA per-op limit.
       *
       * A single RDMA write is capped at EFA_GDA_MAX_RDMA_OP_SIZE (1 GiB).
       * Split larger payloads into full-size leading chunks plus a tail
       * (<= cap); the tail is posted by the normal signal/counter-carrying
       * path below.
       *
       * Leading chunks target the peer's DATA EP (slot 0): no remote signal
       * fires and the caller's counter does not tick. They are posted
       * non-aggregated so their doorbells ring (the drain below only counts
       * rung WQEs). */
      if (hasPayload && bytes > (size_t)EFA_GDA_MAX_RDMA_OP_SIZE) {
        const size_t cap = (size_t)EFA_GDA_MAX_RDMA_OP_SIZE;
        const size_t nLeading = (bytes - 1) / cap; /* tail is (0, cap] */
        /* Peer's DATA EP tuple: target slot 0 -> idx = 0*nranks + peer. */
        const uint32_t dataIdx = (uint32_t)peer;
        const uint16_t dAh = dev->data.target_address_handles[dataIdx];
        const uint16_t dQpn = dev->data.target_remote_qpns[dataIdx];
        const uint32_t dQkey = dev->data.target_qkey[dataIdx];
        for (size_t i = 0; i < nLeading; i++) {
          postRdmaOp<mode>(dev, &dev->data, (uint32_t)peer, dAh, dQpn, dQkey, absDstAddr + i * cap, dstRkey,
                           RdmaSgeEncoder{absSrcAddr + i * cap, srcLkey, (uint32_t)cap}, ncclGinOptFlagsDefault);
        }
        /* EFA SRD is unordered: the tail landing does not imply the leading
         * chunks landed. So when the tail announces completion (signal or
         * counter), first wait for the leading chunks' local completions: a
         * local completion means the write has been queued towards the
         * receiver's PCIe, so any write issued after it to that same PCIe
         * destination is delivered after it by PCIe ordering rules. The tail
         * therefore lands behind the whole payload, making its signal/counter
         * mean the data is there and the source buffer safe to reuse.
         * A plain put announces nothing, so nothing can observe its chunks
         * out of order; the wait is deferred to the caller's later flush /
         * signaled put / barrier.
         *
         * Before posting the signaled tail, this code waits for this
         * peer's leading chunks to complete: it snapshots this peer's submitted count (which now
         * covers the leading chunks) and waits for the host-published per-peer
         * ordered prefix to reach it. The snapshot is per peer, so the wait covers
         * only traffic to this peer and stays fixed against concurrent posters. */
        if (isIndexed || hasCounter) {
          uint32_t target;
          if NCCL_IF_CONSTEXPR (efaGdaSingleOwner<mode>) {
            /* Exclusive owner: the plain producer count cannot move under us. */
            target = dev->submitted_count_per_peer[peer];
          } else {
            target = scopedAtomicLoad<cuda::thread_scope_system, cuda::memory_order_acquire>(
              &dev->submitted_count_per_peer[peer]);
          }
          (void)waitPeerCompleted</*HasTimeout=*/false>(dev, (uint32_t)peer, target, nullptr, 0, 0);
        }
        absSrcAddr += nLeading * cap;
        absDstAddr += nLeading * cap;
        writeBytes = (uint32_t)(bytes - nLeading * cap);
      }

      /* Final write: the payload tail (the WHOLE payload when no chunking
       * occurred above; hasPayload) or 0-byte scratch (signal-only),
       * on the counterId-selected poster so the local counter ticks once,
       * addressed to the resolved target so the receiver's FI_REMOTE_WRITE
       * fires once. absSrcAddr/absDstAddr/writeBytes already point at the
       * payload or the scratch region per the hasPayload branch above. */
      postRdmaOp<mode>(dev, main_ep, (uint32_t)peer, main_ah, main_qpn, main_qkey, absDstAddr, dstRkey,
                       RdmaSgeEncoder{absSrcAddr, srcLkey, writeBytes}, optFlags);

      /* Remaining (signalCount - 1) signal increments: 0-byte writes to
       * the peer scratch region on the DATA endpoint, so the caller's
       * counter is not over-incremented. The loop body is empty unless
       * signalCount > 1, which implies an INDEXED Add (and thus a
       * signal endpoint target). */
      for (uint32_t k = 1u; k < signalCount; k++) {
        postRdmaOp<mode>(dev, &dev->data, (uint32_t)peer, dataSigAh, dataSigQpn, dataSigQkey,
                         dev->scratch_remote_addrs[peer], dev->scratch_remote_rkeys[peer],
                         RdmaSgeEncoder{dev->scratch_local_addr, dev->scratch_lkey, 0u}, optFlags);
      }
    }
  }
  (void)hasDescriptor;
  (void)descriptor;
  (void)required;
  (void)given;
  (void)optFlags;
  coop.sync();
}

/* ── putImpl: runtime mode dispatcher ─────────────────────────────── */

template <typename Coop>
NCCL_DEVICE_INLINE static void putImpl(ncclGinCtx ctx, Coop coop, int peer, bool hasWins, ncclGinWindow_t dstWin,
                                       size_t dstOff, ncclGinWindow_t srcWin, size_t srcOff, size_t bytes,
                                       ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
                                       bool hasCounter, ncclGinCounter_t counterId, bool hasDescriptor,
                                       ncclGinDescriptorSmem* descriptor, cuda::thread_scope required,
                                       cuda::thread_scope given, uint32_t optFlags) {
  switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
  case NCCL_GIN_RESOURCE_SHARING_THREAD:
    putImplMode<NCCL_GIN_RESOURCE_SHARING_THREAD>(ctx, coop, peer, hasWins, dstWin, dstOff, srcWin, srcOff, bytes,
                                                  signal, signalOp, signalOpArg, hasCounter, counterId, hasDescriptor,
                                                  descriptor, required, given, optFlags);
    break;
  case NCCL_GIN_RESOURCE_SHARING_CTA:
    putImplMode<NCCL_GIN_RESOURCE_SHARING_CTA>(ctx, coop, peer, hasWins, dstWin, dstOff, srcWin, srcOff, bytes, signal,
                                               signalOp, signalOpArg, hasCounter, counterId, hasDescriptor, descriptor,
                                               required, given, optFlags);
    break;
  default:
    putImplMode<NCCL_GIN_RESOURCE_SHARING_GPU>(ctx, coop, peer, hasWins, dstWin, dstOff, srcWin, srcOff, bytes, signal,
                                               signalOp, signalOpArg, hasCounter, counterId, hasDescriptor, descriptor,
                                               required, given, optFlags);
    break;
  }
}

/* ── getImplMode: mode-templated Get implementation ──────────────── */

/* Fetches `bytes` from the peer's `remoteWin` into this rank's `localWin` with RDMA
 * reads on the data endpoint.
 *
 * Get is the reverse of Put and reuses the same post path: the RDMA address pair
 * carries the peer's window (the source the NIC reads) and the SGE carries the local
 * window (the destination it fills), with postRdmaOp's op selecting the
 * direction. The fetched bytes become observable to the caller once the read
 * completes, which the caller establishes with flush or flushAsync/wait. */
template <ncclGinResourceSharingMode mode, typename Coop>
NCCL_DEVICE_INLINE static void getImplMode(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t remoteWin,
                                           size_t remoteOff, ncclGinWindow_t localWin, size_t localOff, size_t bytes,
                                           uint32_t optFlags) {
  coop.sync();
  if (coop.thread_rank() == 0) {
    nccl_ofi_gin_gdaki_dev_handle* dev = getDevHandle(ctx);

    /* Both windows resolve to this context's rail, as in Put. The peer's window
     * supplies the remote address and rkey; this rank's window supplies the local
     * address and lkey. */
    nccl_ofi_gin_gdaki_mr_handle* remoteMh = ((nccl_ofi_gin_gdaki_mr_handle**)remoteWin)[dev->rail_id];
    nccl_ofi_gin_gdaki_mr_handle* localMh = ((nccl_ofi_gin_gdaki_mr_handle**)localWin)[dev->rail_id];
    uint64_t absRemoteAddr = remoteMh->peers[peer].remote_addr + remoteOff;
    uint32_t remoteRkey = remoteMh->peers[peer].rkey;
    uint64_t absLocalAddr = localMh->local_addr + localOff;
    uint32_t localLkey = localMh->lkey;

    /* Target slot 0 addresses the peer's data endpoint, which is the endpoint that
     * serves window memory. */
    const uint32_t targetIdx = 0u * (uint32_t)dev->nranks + (uint32_t)peer;
    const uint16_t ah = dev->data.target_address_handles[targetIdx];
    const uint16_t qpn = dev->data.target_remote_qpns[targetIdx];
    const uint32_t qkey = dev->data.target_qkey[targetIdx];

    /* One RDMA read carries at most EFA_GDA_MAX_RDMA_OP_SIZE, so a larger fetch splits
     * into full-size chunks plus a tail, advancing both sides together. A zero-byte
     * fetch posts one zero-length read rather than nothing, so that it still produces
     * a completion for flush and flushAsync/wait to terminate on. */
    const size_t cap = (size_t)EFA_GDA_MAX_RDMA_OP_SIZE;
    size_t remaining = bytes;
    uint64_t remoteAddr = absRemoteAddr;
    uint64_t localAddr = absLocalAddr;
    do {
      const size_t chunk = (remaining > cap) ? cap : remaining;
      postRdmaOp<mode, EFA_GDA_RDMA_READ>(dev, &dev->data, (uint32_t)peer, ah, qpn, qkey, remoteAddr, remoteRkey,
                                          RdmaSgeEncoder{localAddr, localLkey, (uint32_t)chunk}, optFlags);
      remoteAddr += chunk;
      localAddr += chunk;
      remaining -= chunk;
    } while (remaining > 0);
  }
  coop.sync();
}

/* ── getImpl: dispatch on the context's resource-sharing mode ────── */

template <typename Coop>
NCCL_DEVICE_INLINE static void getImpl(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t remoteWin, size_t remoteOff,
                                       ncclGinWindow_t localWin, size_t localOff, size_t bytes, uint32_t optFlags) {
  switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
  case NCCL_GIN_RESOURCE_SHARING_THREAD:
    getImplMode<NCCL_GIN_RESOURCE_SHARING_THREAD>(ctx, coop, peer, remoteWin, remoteOff, localWin, localOff, bytes,
                                                  optFlags);
    break;
  case NCCL_GIN_RESOURCE_SHARING_CTA:
    getImplMode<NCCL_GIN_RESOURCE_SHARING_CTA>(ctx, coop, peer, remoteWin, remoteOff, localWin, localOff, bytes,
                                               optFlags);
    break;
  default:
    getImplMode<NCCL_GIN_RESOURCE_SHARING_GPU>(ctx, coop, peer, remoteWin, remoteOff, localWin, localOff, bytes,
                                               optFlags);
    break;
  }
}

/* ── putValueImplMode: mode-templated PutValue implementation ─────── */

template <ncclGinResourceSharingMode mode, typename Coop, typename T>
NCCL_DEVICE_INLINE static void putValueImplMode(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t dstWin,
                                                size_t dstOff, T srcVal, ncclGinSignalDescriptor signal,
                                                ncclGinSignalOp_t signalOp, uint64_t signalOpArg, bool hasDescriptor,
                                                ncclGinDescriptorSmem* descriptor, cuda::thread_scope required,
                                                cuda::thread_scope given, uint32_t optFlags) {
  static_assert(sizeof(T) <= 8, "PutValue: T must fit in 8 bytes");
  coop.sync();
  if (coop.thread_rank() == 0) {
    nccl_ofi_gin_gdaki_dev_handle* dev = getDevHandle(ctx);

    /* This backend supports INDEXED signals only. */
    assert((signal.type == NCCL_GIN_SIGNAL_TYPE_NONE || signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED) &&
           "EFA GDA: only INDEXED signals are supported");

    /* Resolve the window to this context's rail. */
    nccl_ofi_gin_gdaki_mr_handle* dstMh = ((nccl_ofi_gin_gdaki_mr_handle**)dstWin)[dev->rail_id];

    /* All PutValues post from the dedicated PutValue endpoint (pvdata). */
    nccl_ofi_gin_gdaki_dev_endpoint_handle* ep = &dev->pvdata;

    /* Resolve the remote target (ah, qpn, qkey) via pvdata's target table
     * (targetSlot-major: idx = targetSlot*nranks + peer), same target slots
     * as Put:
     *     signalling write (INDEXED) -> slot 1 + signalId (peer sc EP,
     *       whose FI_REMOTE_WRITE the receiver's waitSignal observes)
     *     plain value write          -> slot 0 (peer DATA EP, no
     *       FI_REMOTE_WRITE bound, so no signal fires on the receiver)
     * The remote tuple is read through pvdata's own AV. */
    const bool isIndexed = (signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED);
    const uint32_t targetSlot = isIndexed ? (1u + (uint32_t)signal.indexedSignal.signalId) : 0u;
    const uint32_t targetIdx = targetSlot * (uint32_t)dev->nranks + (uint32_t)peer;
    uint16_t ah = ep->target_address_handles[targetIdx];
    uint16_t qpn = ep->target_remote_qpns[targetIdx];
    uint32_t qkey = ep->target_qkey[targetIdx];

    /* Signal increment count, mirroring Put: Inc (or no signal) is a single
     * write; an INDEXED Add-by-N expands into N inbound writes. signalOpArg
     * is defined to be 1 for Inc by the GIN API. */
    uint32_t signalCount = 1u;
    if (isIndexed && signalOp == ncclGinSignalAdd) {
      signalCount = (uint32_t)signalOpArg;
    }

    uint64_t absDstAddr = dstMh->peers[peer].remote_addr + dstOff;
    uint32_t dstRkey = dstMh->peers[peer].rkey;

      /* Value write: the value rides inline in the WQE and is RDMA-written to the
     * destination. The arrival ticks the target sc EP's FI_REMOTE_WRITE once
     * (signalled) or no signal (no-signal). */
    PutValuePayloadEncoder<T> payloadEncoder(srcVal);
    postRdmaOp<mode>(dev, ep, (uint32_t)peer, ah, qpn, qkey, absDstAddr, dstRkey, payloadEncoder,
                     ncclGinOptFlagsDefault);

    /* Remaining (signalCount - 1) signal increments: 0-byte writes to
     * the peer scratch region on the DATA endpoint. The loop body is empty
     * unless signalCount > 1, which implies an INDEXED Add (and thus a
     * signal endpoint target). */
    for (uint32_t k = 1u; k < signalCount; k++) {
      postRdmaOp<mode>(dev, ep, (uint32_t)peer, ah, qpn, qkey, dev->scratch_remote_addrs[peer],
                       dev->scratch_remote_rkeys[peer], RdmaSgeEncoder{dev->scratch_local_addr, dev->scratch_lkey, 0u});
    }
  }
  (void)hasDescriptor;
  (void)descriptor;
  (void)required;
  (void)given;
  (void)optFlags;
  coop.sync();
}

/* ── putValueImpl: runtime mode dispatcher ────────────────────────── */

template <typename Coop, typename T>
NCCL_DEVICE_INLINE static void putValueImpl(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t dstWin, size_t dstOff,
                                            T srcVal, ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                            uint64_t signalOpArg, bool hasDescriptor, ncclGinDescriptorSmem* descriptor,
                                            cuda::thread_scope required, cuda::thread_scope given, uint32_t optFlags) {
  switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
  case NCCL_GIN_RESOURCE_SHARING_THREAD:
    putValueImplMode<NCCL_GIN_RESOURCE_SHARING_THREAD>(ctx, coop, peer, dstWin, dstOff, srcVal, signal, signalOp,
                                                       signalOpArg, hasDescriptor, descriptor, required, given,
                                                       optFlags);
    break;
  case NCCL_GIN_RESOURCE_SHARING_CTA:
    putValueImplMode<NCCL_GIN_RESOURCE_SHARING_CTA>(ctx, coop, peer, dstWin, dstOff, srcVal, signal, signalOp,
                                                    signalOpArg, hasDescriptor, descriptor, required, given, optFlags);
    break;
  default:
    putValueImplMode<NCCL_GIN_RESOURCE_SHARING_GPU>(ctx, coop, peer, dstWin, dstOff, srcVal, signal, signalOp,
                                                    signalOpArg, hasDescriptor, descriptor, required, given, optFlags);
    break;
  }
}

/* ── flushImplMode: mode-templated Flush implementation ───────────── */

template <bool HasTimeout, ncclGinResourceSharingMode mode, typename Coop>
NCCL_DEVICE_INLINE static ncclResult_t flushImplMode(ncclGinCtx ctx, Coop coop, cuda::memory_order ord,
                                                     uint32_t* abortFlag, uint64_t timeoutCycles) {
  (void)ord;
  if NCCL_IF_CONSTEXPR (!HasTimeout) (void)timeoutCycles;

  coop.sync();
  ncclResult_t result = ncclSuccess;
  if (coop.thread_rank() == 0) {
    nccl_ofi_gin_gdaki_dev_handle* dev = getDevHandle(ctx);
    uint64_t startCycle = 0;
    if NCCL_IF_CONSTEXPR (HasTimeout) startCycle = clock64();
    else (void)startCycle; // referenced only when HasTimeout is true

    /* For each endpoint with outstanding work, spin on the NIC-written
     * FI_WRITE + FI_READ counter until it catches up with submitted_count.
     * In the shared modes submitted_count is re-read (scoped relaxed atomic
     * load matching the relaxed bumps from the post path) on every iteration
     * rather than snapshotted once: another thread may keep posting on this
     * context while we drain, and completions passing a stale snapshot would
     * leave the masked difference permanently non-zero. The exclusive THREAD
     * owner snapshots its plain cursor once. The HW counter is read with
     * system-scope acquire so the GPU bypasses caches and observes the latest
     * NIC update through PCIe-coherent memory. */
    auto wait_for_endpoint = [abortFlag, startCycle,
                              timeoutCycles](nccl_ofi_gin_gdaki_dev_endpoint_handle& ep) -> ncclResult_t {
      /* Drain-to-zero: outstanding = (submitted - completed) reduced to
       * 31 bits, since the NIC FI_WRITE + FI_READ counter wraps at 2^31. Wait until
       * no work is outstanding. Outstanding is bounded by sq_size « 2^31,
       * so the masked difference is exact and cannot be fooled by a
       * counter wrap. */
      if NCCL_IF_CONSTEXPR (efaGdaSingleOwner<mode>) {
        uint32_t target = (uint32_t)ep.submitted_count;
        while (((target - (uint32_t)hwCounterLoad(ep.local_cntr_value)) & EFA_CNTR_MASK) != 0) {
          if NCCL_IF_CONSTEXPR (HasTimeout) {
            if (clock64() - startCycle >= timeoutCycles) return ncclTimeout;
          }
          if (abortFlag && *abortFlag) return ncclInProgress;
        }
      } else {
        cuda::atomic_ref<uint64_t, ncclGinScope<mode>> target_ref(ep.submitted_count);
        while (((((uint32_t)target_ref.load(cuda::memory_order_relaxed)) - (uint32_t)hwCounterLoad(ep.local_cntr_value)) &
                EFA_CNTR_MASK) != 0) {
          if NCCL_IF_CONSTEXPR (HasTimeout) {
            if (clock64() - startCycle >= timeoutCycles) return ncclTimeout;
          }
          if (abortFlag && *abortFlag) return ncclInProgress;
        }
      }
      return ncclSuccess;
    };

    result = wait_for_endpoint(dev->data);
    if (result != ncclSuccess) goto done;

    /* The dedicated PutValue endpoint is a local poster too (all PutValue
     * writes ride it), so drain it as well. */
    result = wait_for_endpoint(dev->pvdata);
    if (result != ncclSuccess) goto done;

    /* Drain the counter endpoints only. With the decoupled model the
     * local poster QP is always the data endpoint, the PutValue endpoint,
     * or a counter endpoint (counterId selects the poster); a signal
     * endpoint is only ever a remote TARGET, never a local poster, so its
     * FI_WRITE counter never ticks from our writes and there is nothing to
     * drain. A signal QP needs no local completions at all. */
    for (int i = 0; i < dev->nCounters; i++) {
      result = wait_for_endpoint(dev->counter_handles[i]->base);
      if (result != ncclSuccess) goto done;
    }
  }
done:
  coop.sync();
  return (result == ncclInProgress) ? ncclSuccess : result;
}

/* ── flushImpl: runtime mode dispatcher ───────────────────────────── */

template <bool HasTimeout, typename Coop>
NCCL_DEVICE_INLINE static ncclResult_t flushImpl(ncclGinCtx ctx, Coop coop, cuda::memory_order ord, uint32_t* abortFlag,
                                                 uint64_t timeoutCycles) {
  switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
  case NCCL_GIN_RESOURCE_SHARING_THREAD:
    return flushImplMode<HasTimeout, NCCL_GIN_RESOURCE_SHARING_THREAD>(ctx, coop, ord, abortFlag, timeoutCycles);
  case NCCL_GIN_RESOURCE_SHARING_CTA:
    return flushImplMode<HasTimeout, NCCL_GIN_RESOURCE_SHARING_CTA>(ctx, coop, ord, abortFlag, timeoutCycles);
  default:
    return flushImplMode<HasTimeout, NCCL_GIN_RESOURCE_SHARING_GPU>(ctx, coop, ord, abortFlag, timeoutCycles);
  }
}

/* ── FlushAsync / Wait: one per-peer count against one published word ── */

/* This snapshots into the request how much this context has admitted to `peer`.
 *
 * submitted_count_per_peer[peer] is the counter the admission gate advances, so the
 * snapshot covers every write that had taken a position in this peer's sequence
 * before the call and stops there. Writes admitted after the call get positions
 * above the snapshot, so the wait terminates. */
NCCL_DEVICE_INLINE static void flushAsyncImpl(ncclGinCtx ctx, int peer, ncclGinEfaGdaRequest* req) {
  nccl_ofi_gin_gdaki_dev_handle* dev = getDevHandle(ctx);
  req->peer = (uint32_t)peer;
#if NCCL_GIN_EFA_GDA_SINGLE_OWNER_FAST_PATH
  if ((ncclGinResourceSharingMode)ctx.resourceSharingMode == NCCL_GIN_RESOURCE_SHARING_THREAD) {
    /* Exclusive owner: the plain producer count is this thread's own. */
    req->submitted_count_at_flush = dev->submitted_count_per_peer[peer];
  } else
#endif
  {
    req->submitted_count_at_flush =
      scopedAtomicLoad<cuda::thread_scope_system, cuda::memory_order_acquire>(&dev->submitted_count_per_peer[peer]);
  }
  req->reserved = 0;
}

/* This completes the flush that FlushAsync deferred: it waits for this peer's
 * first submitted_count_at_flush writes to complete via waitPeerCompleted, then
 * fences at the caller's memory order so the flushed writes' effects are visible
 * at the requested scope. */
template <bool HasTimeout>
NCCL_DEVICE_INLINE static ncclResult_t waitImpl(ncclGinCtx ctx, ncclGinRequest_t& request, cuda::memory_order ord,
                                                uint32_t* abortFlag, uint64_t timeoutCycles) {
  nccl_ofi_gin_gdaki_dev_handle* dev = getDevHandle(ctx);
  ncclGinEfaGdaRequest& req = reinterpret_cast<ncclGinEfaGdaRequest&>(request);

  uint64_t startCycle = 0;
  if NCCL_IF_CONSTEXPR (HasTimeout) startCycle = clock64();

  ncclResult_t result =
    waitPeerCompleted<HasTimeout>(dev, req.peer, req.submitted_count_at_flush, abortFlag, startCycle, timeoutCycles);
  if (result != ncclSuccess) return result;

  /* Publish the completions at the caller's chosen order, as Wait's contract
   * requires: after this returns, the flushed writes' effects are visible at the
   * requested scope. */
  cuda::atomic_thread_fence(ord, cuda::thread_scope_system);
  return ncclSuccess;
}

} // namespace efa_gda
} // namespace gin
} // namespace nccl

/* ── Put ───────────────────────────────────────────────────────────── */

template <>
struct ncclGinApi_Put<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, bool hasWins, ncclGinWindow_t dstWin,
                                      size_t dstOff, ncclGinWindow_t srcWin, size_t srcOff, size_t bytes,
                                      ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
                                      bool hasCounter, ncclGinCounter_t counterId, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, cuda::thread_scope required,
                                      cuda::thread_scope given, uint32_t optFlags = ncclGinOptFlagsDefault) {
    nccl::gin::efa_gda::putImpl(ctx, coop, peer, hasWins, dstWin, dstOff, srcWin, srcOff, bytes, signal, signalOp,
                                signalOpArg, hasCounter, counterId, hasDescriptor, descriptor, required, given,
                                optFlags);
  }
};

/* ── PutValue ─────────────────────────────────────────────────────── */

template <>
struct ncclGinApi_PutValue<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  template <typename Coop, typename T>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t dstWin, size_t dstOff,
                                      T srcVal, ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasDescriptor, ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope required, cuda::thread_scope given,
                                      uint32_t optFlags = ncclGinOptFlagsDefault) {
    nccl::gin::efa_gda::putValueImpl(ctx, coop, peer, dstWin, dstOff, srcVal, signal, signalOp, signalOpArg,
                                     hasDescriptor, descriptor, required, given, optFlags);
  }
};

/* ── Get ──────────────────────────────────────────────────────────── */

template <>
struct ncclGinApi_Get<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t remoteWin, size_t remoteOff,
                                      ncclGinWindow_t localWin, size_t localOff, size_t bytes, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, uint32_t optFlags = ncclGinOptFlagsDefault) {
    (void)hasDescriptor;
    (void)descriptor;
    nccl::gin::efa_gda::getImpl(ctx, coop, peer, remoteWin, remoteOff, localWin, localOff, bytes, optFlags);
  }
};

/* ── FlushAsync ───────────────────────────────────────────────────── */

/* Delegates to flushAsyncImpl. */
template <>
struct ncclGinApi_FlushAsync<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, int peer, ncclGinRequest_t* outRequest, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, uint32_t optFlags) {
    (void)hasDescriptor;
    (void)descriptor;
    (void)optFlags;
    ncclGinEfaGdaRequest* req = reinterpret_cast<ncclGinEfaGdaRequest*>(outRequest);
    nccl::gin::efa_gda::flushAsyncImpl(ctx, peer, req);
  }
};

/* ── Wait ─────────────────────────────────────────────────────────── */

/* Both overloads delegate to waitImpl. */
template <>
struct ncclGinApi_Wait<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinRequest_t& request, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, cuda::memory_order ord, uint32_t* abortFlag) {
    (void)hasDescriptor;
    (void)descriptor;
    /* No return channel on this overload; a completion error is surfaced through
     * queryLastError. */
    (void)nccl::gin::efa_gda::waitImpl</*HasTimeout=*/false>(ctx, request, ord, abortFlag, 0);
  }

  NCCL_DEVICE_INLINE static ncclResult_t call(ncclGinCtx ctx, ncclGinRequest_t& request, bool hasDescriptor,
                                              ncclGinDescriptorSmem* descriptor, cuda::memory_order ord,
                                              uint32_t* abortFlag, uint64_t timeoutCycles) {
    (void)hasDescriptor;
    (void)descriptor;
    ncclResult_t result =
      nccl::gin::efa_gda::waitImpl</*HasTimeout=*/true>(ctx, request, ord, abortFlag, timeoutCycles);
    /* Abort is a teardown signal, the same as Flush, so this reports success. */
    return (result == ncclInProgress) ? ncclSuccess : result;
  }
};

/* ── Flush ────────────────────────────────────────────────────────── */

template <>
struct ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, bool hasDescriptor, ncclGinDescriptorSmem* descriptor,
                                      cuda::memory_order ord, uint32_t* abortFlag) {
    (void)hasDescriptor;
    (void)descriptor;
    (void)nccl::gin::efa_gda::flushImpl<false>(ctx, coop, ord, abortFlag, 0);
  }

  template <typename Coop>
  NCCL_DEVICE_INLINE static ncclResult_t call(ncclGinCtx ctx, Coop coop, bool hasDescriptor,
                                              ncclGinDescriptorSmem* descriptor, cuda::memory_order ord,
                                              uint32_t* abortFlag, uint64_t timeoutCycles) {
    (void)hasDescriptor;
    (void)descriptor;
    return nccl::gin::efa_gda::flushImpl<true>(ctx, coop, ord, abortFlag, timeoutCycles);
  };
};

/* ── SupportsStrongSignal ────────────────────────────────────────────────────────── */
template <>
struct ncclGinApi_SupportsStrongSignal<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static bool call(ncclGinCtx) {
    return false;
  }
};

/* ── FlushesAllPutsOnAnySignal ───────────────────────────────────────────────────── */
template <>
struct ncclGinApi_FlushesAllPutsOnAnySignal<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  // EFA is not MLX5 and does not flush previously-received puts on an unrelated signal, so the
  // capability is unconditionally absent.
  NCCL_DEVICE_INLINE static bool call(ncclGinCtx) {
    return false;
  }
};

/* ── GetSignalPtr ─────────────────────────────────────────────────── */

template <>
struct ncclGinApi_GetSignalPtr<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static ncclGinOffsetPtr call(ncclGinCtx ctx, ncclGinSignal_t signalId) {
    nccl_ofi_gin_gdaki_dev_handle* dev = nccl::gin::efa_gda::getDevHandle(ctx);
    nccl_ofi_gin_gdaki_dev_counter_handle* h = dev->signal_handles[signalId];
    return {(uint64_t*)h->cntr_value, h->cntr_offset};
  }
};

/* ── GetCounterPtr ────────────────────────────────────────────────── */

template <>
struct ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static ncclGinOffsetPtr call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    nccl_ofi_gin_gdaki_dev_handle* dev = nccl::gin::efa_gda::getDevHandle(ctx);
    nccl_ofi_gin_gdaki_dev_counter_handle* h = dev->counter_handles[counterId];
    return {(uint64_t*)h->cntr_value, h->cntr_offset};
  }
};

/* ── ResetSignal ──────────────────────────────────────────────────── */

template <>
struct ncclGinApi_ResetSignal<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinSignalDescriptor signal) {
    nccl_ofi_gin_gdaki_dev_handle* dev = nccl::gin::efa_gda::getDevHandle(ctx);
    assert(signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED && "EFA GDA ResetSignal: only INDEXED signals are supported");
    assert((int)signal.indexedSignal.signalId < dev->nSignals && "EFA GDA ResetSignal: signalId out of range");
    /* Offset-based reset: the NIC counter cannot be written, so snapshot
     * its current value into cntr_offset. Subsequent reads/waits subtract
     * the offset, making the signal appear reset. */
    nccl_ofi_gin_gdaki_dev_counter_handle* h = dev->signal_handles[signal.indexedSignal.signalId];
    h->cntr_offset = nccl::gin::efa_gda::hwCounterLoad((uint64_t*)h->cntr_value);
  }
};

/* ── ResetCounter ─────────────────────────────────────────────────── */

template <>
struct ncclGinApi_ResetCounter<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    nccl_ofi_gin_gdaki_dev_handle* dev = nccl::gin::efa_gda::getDevHandle(ctx);
    assert((int)counterId < dev->nCounters && "EFA GDA ResetCounter: counterId out of range");
    /* Offset-based reset: snapshot the NIC counter into cntr_offset
     * instead of writing the (NIC-owned) counter. */
    nccl_ofi_gin_gdaki_dev_counter_handle* h = dev->counter_handles[counterId];
    h->cntr_offset = nccl::gin::efa_gda::hwCounterLoad((uint64_t*)h->cntr_value);
  }
};

#endif /* _NCCL_DEVICE_GIN_EFA_GDA_H_ */
