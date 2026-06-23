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
 *              scratch buffer), Flush, GetSignalPtr, GetCounterPtr,
 *              ResetSignal, ResetCounter.
 * Stub: PutValue, Get, FlushAsync, Wait.
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_EFA_GDA_H_
#define _NCCL_DEVICE_GIN_EFA_GDA_H_

#include <cstdint>
#include <cuda/atomic>
#include <cooperative_groups.h>

#include "../gin_device_common.h"
#include "gin_efa_gda_dev.h"

/* efa-dp-direct device functions (inline implementations) */
#include "../../transport/net_efa_gda/efa-dp-direct/include/device/efa_cuda_dp_impl.cuh"

namespace nccl {
namespace gin {
namespace efa_gda {

/* The plugin returns a contiguous array of per-context dev handles
 * in GPU memory; ctx.handle points at element 0. ctx.contextId
 * selects the entry for this caller. */
NCCL_DEVICE_INLINE static nccl_ofi_gin_gdaki_dev_handle*
getDevHandle(ncclGinCtx ctx) {
  return &((nccl_ofi_gin_gdaki_dev_handle*)ctx.handle)[ctx.contextId];
}

/* ── Mode mapping: NCCL → efa-dp-direct ───────────────────────────── */

template <ncclGinResourceSharingMode mode>
static constexpr cuda::thread_scope ncclGinScope =
    (mode == NCCL_GIN_RESOURCE_SHARING_CTA)
        ? cuda::thread_scope_block : cuda::thread_scope_device;

/* The EFA hardware completion counters (FI_WRITE / FI_REMOTE_WRITE) wrap at
 * 2^31, while the kernel-side producer cursors are uint32 (wrap at 2^32).
 * Every comparison between a producer cursor and a HW counter must therefore
 * be a modular difference reduced to 31 bits: compute (producer - consumer)
 * and mask with EFA_CNTR_MASK. The true difference (in-flight / outstanding
 * work) is always far below 2^31 (bounded by sq_size == 4096), so the masked
 * difference recovers the exact value regardless of how many times either side
 * has wrapped. Never compare absolute counter values. */
static constexpr uint32_t EFA_CNTR_MASK = 0x7fffffffu;

/* ── Atomic primitives parameterized on scope and memory order ────── */

template <cuda::thread_scope Scope, cuda::memory_order Order>
NCCL_DEVICE_INLINE static uint64_t scopedAtomicLoad(uint64_t *ptr) {
  cuda::atomic_ref<uint64_t, Scope> r(*ptr);
  return r.load(Order);
}

template <cuda::thread_scope Scope, cuda::memory_order Order>
NCCL_DEVICE_INLINE static void scopedAtomicAdd(uint64_t *ptr, uint64_t val) {
  cuda::atomic_ref<uint64_t, Scope> r(*ptr);
  r.fetch_add(val, Order);
}

/* ── NIC-written hardware counter (FI_WRITE / FI_REMOTE_WRITE) ────── */

/* Read a NIC-written hardware counter from GPU memory. Uses system-scope
 * acquire so the load is coherent with the NIC's PCIe writes (bypasses
 * GPU caches) and subsequent operations on this thread cannot be
 * reordered to before the load. The acquire matches libfabric's
 * local-completion contract: when this load observes the counter has
 * reached a target, the NIC's prior side effects (e.g. source-buffer
 * DMA-reads complete) are ordered-before whatever this thread does
 * next (e.g. overwriting that source buffer or reusing the slot). */
NCCL_DEVICE_INLINE static uint64_t hwCounterLoad(uint64_t *ptr) {
  return scopedAtomicLoad<cuda::thread_scope_system, cuda::memory_order_acquire>(ptr);
}

/* Store to a NIC-written hardware counter from GPU memory. Used by
 * ResetSignal / ResetCounter to zero the counter from the GPU side;
 * system-scope release so any prior GPU writes are visible to the NIC
 * before it observes the new counter value. */
NCCL_DEVICE_INLINE static void hwCounterStore(uint64_t *ptr, uint64_t val) {
  scopedAtomicStore<cuda::thread_scope_system, cuda::memory_order_release>(ptr, val);
}

/* ── postRdmaWrite: shared post path for Put and PutValue ─────────────
 *
 * Posts an RDMA write on `ep`'s local QP (its FI_WRITE counter tracks
 * local completion) to the remote QP given by the explicit
 * (ah, qpn, qkey) tuple. The local poster QP and the remote target QP
 * are chosen independently by the caller: counterId selects the local
 * poster (this `ep`), signalId selects the remote tuple (via the
 * poster's sig_* table); a counter-only write targets the peer's data EP
 * via cnt_*. For non-signalling/non-counter writes the caller passes
 * `ep`'s own per-peer addressing tuple.
 *
 * Inlined sliding-window SQ post with warp coalescing: drives the
 * efa_cuda_qp ring directly via two shared cursors rather than a per-QP
 * spinlock + the efa-dp-direct start_sq_batch / flush_sq_wrs helpers.
 *
 *   qp->sq.wq.pc             : monotonic reservation index. A group's
 *                              leader claims its whole range with one
 *                              atomicAdd(+group_size).
 *   qp->sq.wq.wqes_completed : "released" cursor — the doorbell has been
 *                              rung up to here. Doubles as the
 *                              sliding-window base and the doorbell-order
 *                              rendezvous token across groups, serializing
 *                              concurrent posters in slot order WITHOUT a
 *                              held lock.
 *
 * Threads concurrently posting to the same QP form a group
 * (coalesced_threads + labeled_partition by qp), chunked into windows of
 * <= max_batch (the EFA staging limit). Per chunk the leader applies
 * sliding-window + SQ ring-overflow backpressure; members write their own
 * WQEs in parallel; the leader rings one doorbell for the chunk in slot
 * order and hands off to the next group.
 *
 * PutValue staging (pvSrcVal != nullptr): EFA RDMA_WRITE cannot carry
 * inline data, so for PutValue each lane first stages its value into its
 * own slot of the endpoint's putvalue pool (slice base ep->putvalue_slice_base,
 * slot index = reserved SQ slot % sq_size) and the WR's SGE points at that
 * slot. Put (pvSrcVal == nullptr) uses the caller's fixed srcAddr/srcLkey.
 * pvSrcVal/pvValBytes are read per lane; the same value goes to every
 * lane's slot (each call posts one WR per calling thread). */
template <ncclGinResourceSharingMode mode>
NCCL_DEVICE_INLINE static void postRdmaWrite(
    nccl_ofi_gin_gdaki_dev_endpoint_handle *ep, uint16_t ah, uint16_t qpn,
    uint32_t qkey, uint64_t srcAddr, uint32_t srcLkey, uint32_t writeBytes,
    uint64_t dstAddr, uint32_t dstRkey,
    const void *pvSrcVal = nullptr, uint32_t pvValBytes = 0,
    uint32_t pvLkey = 0, uint64_t pvSliceBase = 0, uint32_t pvSlotSize = 0) {

  efa_cuda_qp       *qp                  = (efa_cuda_qp *)ep->qp;
  uint64_t          *submitted_count_ptr  = &ep->submitted_count;
  uint64_t          *local_cntr_ptr       = ep->local_cntr_value;
  uint32_t           sq_size_val          = ep->sq_size;

  cooperative_groups::coalesced_group active = cooperative_groups::coalesced_threads();
  auto group = cooperative_groups::labeled_partition(active, (unsigned long long)(uintptr_t)qp);

  int  my_idx     = group.thread_rank();
  int  group_size = group.num_threads();
  bool is_leader  = (my_idx == 0);
  uint32_t max_batch = qp->sq.wq.max_batch;

  cuda::atomic_ref<uint32_t, ncclGinScope<mode>> pc_ref(qp->sq.wq.pc);
  cuda::atomic_ref<uint32_t, ncclGinScope<mode>> base_ref(qp->sq.wq.wqes_completed);

  /* Leader reserves the whole group's contiguous slot range with one
   * atomicAdd; this is the linearization point across all groups. */
  uint32_t base = 0;
  if (is_leader) {
    base = pc_ref.fetch_add((uint32_t)group_size, cuda::memory_order_relaxed);
  }
  base = group.shfl(base, 0);

  /* Chunk the group into windows of <= max_batch. */
  for (int chunk_start = 0; chunk_start < group_size; chunk_start += (int)max_batch) {
    int      chunk_size = min((int)max_batch, group_size - chunk_start);
    uint32_t chunk_base = base + (uint32_t)chunk_start;
    uint32_t chunk_next = chunk_base + (uint32_t)chunk_size;

    if (is_leader) {
      /* Sliding-window backpressure: keep cumulative un-doorbelled WQEs
       * (across all groups) within max_batch. */
      while (chunk_next > base_ref.load(cuda::memory_order_acquire) + max_batch) {
        /* spin */
      }
      /* SQ ring-overflow backpressure on the chunk's high-water slot.
       * System-scope acquire so we see the latest NIC FI_WRITE update
       * and the WQE stores below can't hoist above this load.
       *
       * In-flight count is computed as a 31-bit modular difference
       * (producer chunk_next minus the NIC FI_WRITE counter): the HW
       * counter wraps at 2^31 and chunk_next is uint32, so a plain
       * widened subtraction would underflow once either side wraps.
       * The true in-flight depth is bounded by sq_size (4096) « 2^31,
       * so the masked difference is exact. */
      while (((chunk_next - (uint32_t)hwCounterLoad(local_cntr_ptr)) & EFA_CNTR_MASK) > sq_size_val) {
        /* spin */
      }
    }
    group.sync();   /* members wait for leader's backpressure before writing */

    /* Members in this window write their own WQE into their slot. */
    bool in_window = (my_idx >= chunk_start && my_idx < chunk_start + chunk_size);
    if (in_window) {
      uint32_t my_slot = chunk_base + (uint32_t)(my_idx - chunk_start);

      /* WQE wr_id -> wqe.meta.req_id (completion-descriptor tag). Use the
       * reserved absolute SQ slot index, not threadIdx.x: the slot is
       * unique per outstanding WQE on this QP, whereas threadIdx.x is
       * neither unique across a coalesced group nor stable and would
       * alias/truncate into the uint16_t req_id. */
      uint64_t wrSrcAddr = srcAddr;
      uint32_t wrSrcLkey = srcLkey;
      if (pvSrcVal != nullptr) {
        /* PutValue: stage the value into this slot's slot of the pool, then
         * point the SGE at it. slot folds 1:1 onto a pool slot (slice holds
         * sq_size slots; sq_size is a power of two so % is a mask). */
        uint64_t slot_idx   = (uint64_t)my_slot % (uint64_t)sq_size_val;
        uint64_t local_addr = pvSliceBase + slot_idx * (uint64_t)pvSlotSize;
        for (uint32_t b = 0; b < pvValBytes; b++)
          ((uint8_t *)local_addr)[b] = ((const uint8_t *)pvSrcVal)[b];
        wrSrcAddr  = local_addr;
        wrSrcLkey  = pvLkey;
      }

      efa_io_tx_wqe wr;
      efa_cuda_init_rdma_write_wr(&wr, (uint16_t)my_slot, dstRkey, dstAddr);
      efa_cuda_wr_set_sge(&wr, wrSrcLkey, wrSrcAddr, writeBytes);
      efa_cuda_wr_set_remote(&wr, ah, (uint32_t)qpn, qkey);

      uint32_t sq_idx    = my_slot & qp->sq.wq.queue_mask;
      int      wqe_phase = (int)((my_slot >> qp->sq.wq.queue_size_shift) & 1u);
      EFA_SET(&wr.meta.ctrl2, EFA_IO_TX_META_DESC_PHASE, wqe_phase);
      uint64_t *src = (uint64_t *)&wr;
      uint64_t *dst = (uint64_t *)(qp->sq.wq.buf + sq_idx * sizeof(efa_io_tx_wqe));
      for (int i = 0; i < 8; i++)
        dst[i] = src[i];
    }
    group.sync();   /* all members' WQE (+ PutValue staging) stores done */

    /* Each writing lane publishes ITS OWN WQE + staging stores to system
     * scope. A single leader-only __threadfence_system() does NOT order
     * other lanes' stores ahead of the doorbell — threadfence only orders
     * the calling thread's accesses — so every writer must fence its own
     * writes here. */
    if (in_window) {
      __threadfence_system();
    }
    group.sync();   /* all per-lane publishes retired before the doorbell */

    if (is_leader) {
      /* Doorbell-order rendezvous: ring in slot order across groups. */
      while (base_ref.load(cuda::memory_order_acquire) != chunk_base) {
        /* spin */
      }
      *qp->sq.wq.db = chunk_next;
      __threadfence_system();   /* order the doorbell MMIO write itself */
      scopedAtomicAdd<ncclGinScope<mode>, cuda::memory_order_relaxed>(submitted_count_ptr, (uint64_t)chunk_size);
      base_ref.store(chunk_next, cuda::memory_order_release);   /* hand off to next group */
    }
    group.sync();   /* chunk fully posted before the next chunk */
  }
}

/* ── putImplMode: mode-templated Put implementation ─────────────── */

template <ncclGinResourceSharingMode mode, typename Coop>
NCCL_DEVICE_INLINE static void putImplMode(ncclGinCtx ctx, Coop coop, int peer, bool hasWins,
                                    ncclGinWindow_t dstWin, size_t dstOff, ncclGinWindow_t srcWin,
                                    size_t srcOff, size_t bytes,
                                    ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                    uint64_t signalOpArg, bool hasCounter,
                                    ncclGinCounter_t counterId, bool hasDescriptor,
                                    ncclGinDescriptorSmem* descriptor,
                                    cuda::thread_scope required, cuda::thread_scope given,
                                    uint32_t optFlags) {
  coop.sync();
  if (coop.thread_rank() == 0) {
    nccl_ofi_gin_gdaki_dev_handle *dev = getDevHandle(ctx);

    bool hasPayload    = hasWins && bytes > 0;
    bool needsSignalEp = (signal.type != NCCL_GIN_SIGNAL_TYPE_NONE) || hasCounter;

    /* This backend supports INDEXED signals only. EFA's FI_REMOTE_WRITE
     * counter ticks exactly once per inbound write and has no atomic-add,
     * so a signal Add-by-N is emulated as N inbound write events (see the
     * posting block below). VA-typed signals are not representable. */
    assert((signal.type == NCCL_GIN_SIGNAL_TYPE_NONE
            || signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED)
           && "EFA GDA: only INDEXED signals are supported");
    assert((signal.type != NCCL_GIN_SIGNAL_TYPE_INDEXED
            || (int)signal.indexedSignal.signalId < dev->nSignals)
           && "EFA GDA: signalId out of range");
    assert((!hasCounter || (int)counterId < dev->nCounters)
           && "EFA GDA: counterId out of range");

    if (hasPayload || needsSignalEp) {
      /* Two WQE patterns:
       *
       * (a) Data put: posts an RDMA write of the user payload.
       *     Routed through signal/counter endpoint when needsSignalEp
       *     so the receiver's FI_REMOTE_WRITE fires on completion;
       *     otherwise routed through the data endpoint.
       *
       * (b) Signal-only: posts a 0-byte RDMA write into the peer's
       *     per-context scratch buffer. The write event bumps the
       *     receiver's FI_REMOTE_WRITE counter on the signal endpoint. */
      uint64_t absSrcAddr;
      uint64_t absDstAddr;
      uint32_t dstRkey;
      uint32_t srcLkey;
      uint32_t writeBytes;
      if (hasPayload) {
        nccl_ofi_gin_gdaki_mr_handle *dstMh = (nccl_ofi_gin_gdaki_mr_handle *)dstWin;
        nccl_ofi_gin_gdaki_mr_handle *srcMh = (nccl_ofi_gin_gdaki_mr_handle *)srcWin;
        absSrcAddr = srcMh->local_addr + srcOff;
        absDstAddr = dstMh->peers[peer].remote_addr + dstOff;
        dstRkey    = dstMh->peers[peer].rkey;
        srcLkey    = srcMh->lkey;
        writeBytes = (uint32_t)bytes;
      } else {
        absSrcAddr = dev->scratch_local_addr;
        absDstAddr = dev->scratch_remote_addrs[peer];
        dstRkey    = dev->scratch_remote_rkeys[peer];
        srcLkey    = dev->scratch_lkey;
        writeBytes = 0;
      }

      /* Decoupled local-poster / remote-target selection.
       *
       * A single RDMA write carries two independent QP choices:
       *
       *   - Local poster QP: the SQ we post from; its FI_WRITE counter
       *     ticks on local completion. This is a LOCAL property,
       *     selected by counterId. With a counter request we post from
       *     counter_handles[counterId] so its FI_WRITE is the counter;
       *     otherwise we post from the data endpoint (a signal-only or
       *     plain put has no local counter to track here).
       *
       *   - Remote target QP: the peer endpoint we address; the peer's
       *     FI_REMOTE_WRITE counter on THAT endpoint ticks. This is a
       *     TARGET property, selected by signalId. We resolve it through
       *     the poster's own sig_* table at [signalId * nranks + peer],
       *     which addresses peer P's sc-endpoint[signalId] regardless of
       *     which local endpoint posts. Non-signalling writes use the
       *     poster's plain per-peer addressing.
       *
       * This replaces the old logic that picked the local poster by
       * signalId (which conflated the two and made the signal a local
       * property — see Amit/Arun: "the signal needs to pick the peer
       * QP, the counter needs to pick the local QP"). */
      nccl_ofi_gin_gdaki_dev_endpoint_handle *ep =
          hasCounter ? &dev->counter_handles[counterId]->base : &dev->data;

      uint16_t ah;
      uint16_t qpn;
      uint32_t qkey;
      if (signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED) {
        /* signalId selects the remote target QP via the poster's
         * signalId-major sig_* table: idx = signalId * nranks + peer. */
        uint32_t sigIdx =
            (uint32_t)signal.indexedSignal.signalId * (uint32_t)dev->nranks
            + (uint32_t)peer;
        ah   = ep->sig_address_handles[sigIdx];
        qpn  = ep->sig_remote_qpns[sigIdx];
        qkey = ep->sig_qkey[sigIdx];
      } else if (hasCounter) {
        /* Reached only when there is NO signal (the INDEXED branch
         * above did not match) AND a counter was requested — i.e. the
         * counter-only case. (A signal+counter Put is handled by the
         * sig_* branch above; the counter is still satisfied there
         * because the local poster QP was selected by counterId.)
         *
         * The poster here IS a counter sc endpoint, whose plain
         * per-peer addressing resolves to peer's SAME-INDEX sc
         * endpoint — and that endpoint's FI_REMOTE_WRITE counter is GIN
         * signal C on the receiver, so a plain write would spuriously
         * tick signal[counterId]. A counter is a sender-local concept
         * (the local FI_WRITE) and must produce no receiver-observable
         * signal, so address peer's DATA endpoint (which binds no
         * FI_REMOTE_WRITE counter) via this endpoint's cnt_* table,
         * indexed by peer. The write still posts from this endpoint's
         * QP, so the correct local counter ticks. */
        ah   = ep->cnt_address_handles[peer];
        qpn  = ep->cnt_remote_qpns[peer];
        qkey = ep->cnt_qkey[peer];
      } else {
        /* No signal, no counter: address the poster's own same-index
         * peer endpoint (the data endpoint's plain table resolves to
         * peer's data endpoint). */
        ah   = ep->address_handles[peer];
        qpn  = ep->remote_qpns[peer];
        qkey = ep->qkey[peer];
      }

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
       * a larger reservation via postRdmaWrite's chunk loop; that must NOT
       * be done by suppressing doorbells across calls, which would break
       * the wqes_completed sliding-window / rendezvous invariant.
       *
       * TODO: batch the doorbells for a signal Add-by-N (and across the
       * payload + scratch writes) instead of ringing one doorbell per
       * write. Must reuse postRdmaWrite's chunk loop (one doorbell per
       * max_batch reservation), NOT a doorbell-suppress flag across
       * separate calls. */
      uint32_t signalCount = 1u;
      if (signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED
          && signalOp == ncclGinSignalAdd) {
        signalCount = (uint32_t)signalOpArg;
      }

      if (hasPayload) {
        /* The payload write itself bumps the receiver's FI_REMOTE_WRITE
         * counter by 1 (it is routed through the resolved signal-target
         * tuple when needsSignalEp). Any remaining (signalCount - 1)
         * increments are 0-byte writes to the peer scratch region,
         * addressed to the same target QP. */
        postRdmaWrite<mode>(ep, ah, qpn, qkey, absSrcAddr, srcLkey,
                            writeBytes, absDstAddr, dstRkey);
        if (needsSignalEp) {
          for (uint32_t k = 1u; k < signalCount; k++) {
            postRdmaWrite<mode>(ep, ah, qpn, qkey, dev->scratch_local_addr,
                                dev->scratch_lkey, 0u,
                                dev->scratch_remote_addrs[peer],
                                dev->scratch_remote_rkeys[peer]);
          }
        }
      } else {
        /* Signal-only: all signalCount increments are 0-byte scratch
         * writes (absSrcAddr/absDstAddr already point at the scratch
         * region from the branch above), addressed to the resolved
         * target QP. */
        for (uint32_t k = 0u; k < signalCount; k++) {
          postRdmaWrite<mode>(ep, ah, qpn, qkey, absSrcAddr, srcLkey,
                              writeBytes, absDstAddr, dstRkey);
        }
      }
    }
  }
  (void)hasDescriptor; (void)descriptor;
  (void)required; (void)given; (void)optFlags;
  coop.sync();
}

/* ── putImpl: runtime mode dispatcher ─────────────────────────────── */

template <typename Coop>
NCCL_DEVICE_INLINE static void putImpl(ncclGinCtx ctx, Coop coop, int peer, bool hasWins,
                                    ncclGinWindow_t dstWin, size_t dstOff, ncclGinWindow_t srcWin,
                                    size_t srcOff, size_t bytes,
                                    ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                    uint64_t signalOpArg, bool hasCounter,
                                    ncclGinCounter_t counterId, bool hasDescriptor,
                                    ncclGinDescriptorSmem* descriptor,
                                    cuda::thread_scope required, cuda::thread_scope given,
                                    uint32_t optFlags) {
  switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
    case NCCL_GIN_RESOURCE_SHARING_CTA:
      putImplMode<NCCL_GIN_RESOURCE_SHARING_CTA>(
        ctx, coop, peer, hasWins, dstWin, dstOff, srcWin, srcOff, bytes,
        signal, signalOp, signalOpArg, hasCounter, counterId,
        hasDescriptor, descriptor, required, given, optFlags);
      break;
    default:
      putImplMode<NCCL_GIN_RESOURCE_SHARING_GPU>(
        ctx, coop, peer, hasWins, dstWin, dstOff, srcWin, srcOff, bytes,
        signal, signalOp, signalOpArg, hasCounter, counterId,
        hasDescriptor, descriptor, required, given, optFlags);
      break;
  }
}

/* ── putValueImplMode: mode-templated PutValue implementation ─────── */

template <ncclGinResourceSharingMode mode, typename T>
NCCL_DEVICE_INLINE static void putValueImplMode(
    ncclGinCtx ctx, int peer, ncclGinWindow_t dstWin, size_t dstOff, T srcVal,
    ncclGinSignalDescriptor signal,
    cuda::thread_scope required, cuda::thread_scope given) {

  nccl_ofi_gin_gdaki_dev_handle *dev = getDevHandle(ctx);
  nccl_ofi_gin_gdaki_mr_handle *dstMh = (nccl_ofi_gin_gdaki_mr_handle *)dstWin;

  /* Pick the local poster endpoint, mirroring Put: a signal request
   * routes to signal_handles[] (its FI_WRITE counter tracks local
   * completion); otherwise the data endpoint. */
  nccl_ofi_gin_gdaki_dev_endpoint_handle *ep;
  if (signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED) {
    ep = &dev->signal_handles[signal.indexedSignal.signalId]->base;
  } else {
    ep = &dev->data;
  }

  /* Resolve the remote target (ah, qpn, qkey), mirroring Put. A signal
   * request selects the target QP via the poster's signalId-major sig_*
   * table (idx = signalId * nranks + peer) so the write's arrival ticks
   * the receiver's FI_REMOTE_WRITE counter for that signalId; otherwise
   * the poster's own per-peer table addresses peer's same-index endpoint
   * (the data endpoint resolves to peer's data endpoint). */
  uint16_t ah;
  uint16_t qpn;
  uint32_t qkey;
  if (signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED) {
    uint32_t sigIdx =
        (uint32_t)signal.indexedSignal.signalId * (uint32_t)dev->nranks
        + (uint32_t)peer;
    ah   = ep->sig_address_handles[sigIdx];
    qpn  = ep->sig_remote_qpns[sigIdx];
    qkey = ep->sig_qkey[sigIdx];
  } else {
    ah   = ep->address_handles[peer];
    qpn  = ep->remote_qpns[peer];
    qkey = ep->qkey[peer];
  }

  uint64_t absDstAddr = dstMh->peers[peer].remote_addr + dstOff;
  uint32_t dstRkey    = dstMh->peers[peer].rkey;

  /* PutValue reuses postRdmaWrite. EFA RDMA_WRITE can't carry inline data,
   * so postRdmaWrite's PutValue-staging path (pvSrcVal != nullptr) stages
   * srcVal into each lane's slot of the endpoint's putvalue pool and points
   * the WR's SGE there; the reserved SQ slot folds 1:1 onto a pool slot, so
   * concurrent posters never collide and a slot is only reused once its
   * prior WQE has retired (gated by postRdmaWrite's SQ-overflow check). The
   * pool's source length is the value size; the destination write length is
   * also sizeof(T). */
  (void)required; (void)given;   /* NIC reads the staged slot at system
                                  * scope; postRdmaWrite fences every writer
                                  * before the doorbell, so the caller's
                                  * release-scope hint is not needed here. */
  postRdmaWrite<mode>(ep, ah, qpn, qkey, /*srcAddr=*/0, /*srcLkey=*/0,
                      /*writeBytes=*/(uint32_t)sizeof(T), absDstAddr, dstRkey,
                      /*pvSrcVal=*/&srcVal, /*pvValBytes=*/(uint32_t)sizeof(T),
                      /*pvLkey=*/dev->putvalue_lkey,
                      /*pvSliceBase=*/ep->putvalue_slice_base,
                      /*pvSlotSize=*/dev->putvalue_slot_size);
}

/* ── flushImplMode: mode-templated Flush implementation ───────────── */

template <ncclGinResourceSharingMode mode, typename Coop>
NCCL_DEVICE_INLINE static void flushImplMode(ncclGinCtx ctx, Coop coop, cuda::memory_order ord, uint32_t* abortFlag) {
  (void)ord;
  coop.sync();
  if (coop.thread_rank() == 0) {
    nccl_ofi_gin_gdaki_dev_handle *dev = getDevHandle(ctx);

    /* For each endpoint with outstanding work, snapshot submitted_count
     * (scoped atomic load matching the relaxed bumps from the post
     * path), then spin on the NIC-written FI_WRITE counter until it
     * reaches the snapshot. The HW counter is read with system-scope
     * acquire so the GPU bypasses caches and observes the latest NIC
     * update through PCIe-coherent memory. */
    auto wait_for_endpoint = [abortFlag](nccl_ofi_gin_gdaki_dev_endpoint_handle &ep) -> bool {
      uint64_t target = scopedAtomicLoad<ncclGinScope<mode>, cuda::memory_order_relaxed>(&ep.submitted_count);

      /* Drain-to-zero: outstanding = (submitted - completed) reduced to
       * 31 bits, since the NIC FI_WRITE counter wraps at 2^31. Wait until
       * no work is outstanding. Outstanding is bounded by sq_size « 2^31,
       * so the masked difference is exact and cannot be fooled by a wrap
       * (unlike the original absolute-value `completed < target`). */
      while (((((uint32_t)target) - (uint32_t)hwCounterLoad(ep.local_cntr_value)) & EFA_CNTR_MASK) != 0) {
        if (abortFlag && *abortFlag) return false;
      }
      return true;
    };

    if (!wait_for_endpoint(dev->data)) return;

    /* Drain the counter endpoints only. With the decoupled model the
     * local poster QP is always either the data endpoint or a counter
     * endpoint (counterId selects the poster); a signal endpoint is
     * only ever a remote TARGET, never a local poster, so its FI_WRITE
     * counter never ticks from our writes and there is nothing to drain
     * (see Amit/Arun: "we only need to do this on the counters, not
     * signals... for a signal QP we don't need local completions at
     * all"). */
    for (int i = 0; i < dev->nCounters; i++) {
      if (!wait_for_endpoint(dev->counter_handles[i]->base)) return;
    }
  }
  coop.sync();
}

/* ── flushImpl: runtime mode dispatcher ───────────────────────────── */

template <typename Coop>
NCCL_DEVICE_INLINE static void flushImpl(ncclGinCtx ctx, Coop coop, cuda::memory_order ord, uint32_t* abortFlag) {
  switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
    case NCCL_GIN_RESOURCE_SHARING_CTA:
      flushImplMode<NCCL_GIN_RESOURCE_SHARING_CTA>(ctx, coop, ord, abortFlag);
      break;
    default:
      flushImplMode<NCCL_GIN_RESOURCE_SHARING_GPU>(ctx, coop, ord, abortFlag);
      break;
  }
}

} // namespace efa_gda
} // namespace gin
} // namespace nccl

/* ── Put ───────────────────────────────────────────────────────────── */

template <>
struct ncclGinApi_Put<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, bool hasWins,
                                      ncclGinWindow_t dstWin, size_t dstOff, ncclGinWindow_t srcWin,
                                      size_t srcOff, size_t bytes,
                                      ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasCounter,
                                      ncclGinCounter_t counterId, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope required, cuda::thread_scope given,
                                      uint32_t optFlags = ncclGinOptFlagsDefault) {
    nccl::gin::efa_gda::putImpl(
      ctx, coop, peer, hasWins, dstWin, dstOff, srcWin, srcOff, bytes,
      signal, signalOp, signalOpArg, hasCounter, counterId,
      hasDescriptor, descriptor, required, given, optFlags);
  }
};

/* ── PutValue ─────────────────────────────────────────────────────── */

template <>
struct ncclGinApi_PutValue<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  template <typename Coop, typename T>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t dstWin,
                                      size_t dstOff, T srcVal,
                                      ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope required, cuda::thread_scope given,
                                      uint32_t optFlags = ncclGinOptFlagsDefault) {
    /* EFA RDMA_WRITE doesn't support inline data (efa-dp-direct's
     * wr_set_inline_data only supports SEND opcode). Stage srcVal into
     * a registered local source slot, then post an RDMA_WRITE from the
     * slot to the user's destination.
     *
     * Routing matches Put: when the caller asks for a signal, the WQE
     * goes on signal_handles[signalId]'s sc_endpoint so the arrival of
     * the write at the receiver bumps that endpoint's FI_REMOTE_WRITE
     * counter — i.e. value-and-signal in one WQE. When there is no
     * signal the WQE goes on the data endpoint.
     *
     * Slot pool is shared across the data endpoint and every sc
     * endpoint. Each endpoint owns its slice; the slice base lives
     * on the endpoint handle itself (ep.putvalue_slice_base). Slice
     * size is implied by ep.sq_size. Slot reuse safety: each in-flight
     * WQE on endpoint E owns slot (E.submitted_count % E.sq_size)
     * inside E's slice; the per-endpoint SQ-overflow backpressure
     * check on E's FI_WRITE counter gates new posts until the NIC
     * has drained the old WR before the slot is reused.
     *
     * EFA backend signal contract (matches Put):
     *   - INDEXED signals only.
     *   - Inc-by-1 only (FI_REMOTE_WRITE ticks once per inbound write);
     *     signalOpArg is ignored. */
    coop.sync();
    static_assert(sizeof(T) <= 8, "PutValue: T must fit in 8 bytes");

    /* Only INDEXED + Inc-by-1 are supported on this backend. */
    assert(signal.type == NCCL_GIN_SIGNAL_TYPE_NONE
        || signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED);
    assert(signal.type == NCCL_GIN_SIGNAL_TYPE_NONE
        || signalOp == ncclGinSignalInc
        || signalOpArg == 1);

    /* One WQE per calling thread, posted lock-free with the same
     * coalesced-group doorbell coalescing as Put. The mode-templated
     * body uses block- vs device-scope atomics per the context's
     * resource-sharing mode. */
    if (coop.thread_rank() == 0) {
      switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
        case NCCL_GIN_RESOURCE_SHARING_CTA:
          nccl::gin::efa_gda::putValueImplMode<NCCL_GIN_RESOURCE_SHARING_CTA>(
            ctx, peer, dstWin, dstOff, srcVal, signal, required, given);
          break;
        default:
          nccl::gin::efa_gda::putValueImplMode<NCCL_GIN_RESOURCE_SHARING_GPU>(
            ctx, peer, dstWin, dstOff, srcVal, signal, required, given);
          break;
      }
    }
    (void)signalOp; (void)signalOpArg;
    (void)hasDescriptor; (void)descriptor;
    (void)optFlags;
    coop.sync();
  }
};

/* ── Get ──────────────────────────────────────────────────────────── */

template <>
struct ncclGinApi_Get<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t remoteWin, size_t remoteOff,
                                      ncclGinWindow_t localWin, size_t localOff, size_t bytes,
                                      bool hasDescriptor, ncclGinDescriptorSmem* descriptor,
                                      uint32_t optFlags = ncclGinOptFlagsDefault) {
    coop.sync();
    /* TODO: implement with efa_cuda_init_rdma_read_wr */
    (void)ctx; (void)peer; (void)remoteWin; (void)remoteOff;
    (void)localWin; (void)localOff; (void)bytes;
    (void)hasDescriptor; (void)descriptor; (void)optFlags;
    coop.sync();
  }
};

/* ── FlushAsync ───────────────────────────────────────────────────── */

template <>
struct ncclGinApi_FlushAsync<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, int peer, ncclGinRequest_t* outRequest, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, uint32_t optFlags) {
    (void)ctx; (void)peer; (void)outRequest; (void)hasDescriptor; (void)descriptor; (void)optFlags;
  }
};

/* ── Wait ─────────────────────────────────────────────────────────── */

template <>
struct ncclGinApi_Wait<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinRequest_t& request, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, cuda::memory_order ord, uint32_t* abortFlag) {
    (void)ctx; (void)request; (void)hasDescriptor;
    (void)descriptor; (void)ord; (void)abortFlag;
  }
};

/* ── Flush ────────────────────────────────────────────────────────── */

template <>
struct ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, bool hasDescriptor, ncclGinDescriptorSmem* descriptor,
                                      cuda::memory_order ord, uint32_t* abortFlag) {
    (void)hasDescriptor; (void)descriptor;
    nccl::gin::efa_gda::flushImpl(ctx, coop, ord, abortFlag);
  }
};

/* ── GetSignalPtr ─────────────────────────────────────────────────── */

template <>
struct ncclGinApi_GetSignalPtr<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static ncclGinOffsetPtr call(ncclGinCtx ctx, ncclGinSignal_t signalId) {
    nccl_ofi_gin_gdaki_dev_handle *dev = nccl::gin::efa_gda::getDevHandle(ctx);
    nccl_ofi_gin_gdaki_dev_counter_handle *h = dev->signal_handles[signalId];
    return { (uint64_t *)h->cntr_value, h->cntr_offset };
  }
};

/* ── GetCounterPtr ────────────────────────────────────────────────── */

template <>
struct ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static ncclGinOffsetPtr call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    nccl_ofi_gin_gdaki_dev_handle *dev = nccl::gin::efa_gda::getDevHandle(ctx);
    nccl_ofi_gin_gdaki_dev_counter_handle *h = dev->counter_handles[counterId];
    return { (uint64_t *)h->cntr_value, h->cntr_offset };
  }
};

/* ── ResetSignal ──────────────────────────────────────────────────── */

template <>
struct ncclGinApi_ResetSignal<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinSignalDescriptor signal) {
    nccl_ofi_gin_gdaki_dev_handle *dev = nccl::gin::efa_gda::getDevHandle(ctx);
    assert(signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED
           && "EFA GDA ResetSignal: only INDEXED signals are supported");
    assert((int)signal.indexedSignal.signalId < dev->nSignals
           && "EFA GDA ResetSignal: signalId out of range");
    /* Offset-based reset: the NIC counter cannot be written, so snapshot
     * its current value into cntr_offset. Subsequent reads/waits subtract
     * the offset, making the signal appear reset. */
    nccl_ofi_gin_gdaki_dev_counter_handle *h =
        dev->signal_handles[signal.indexedSignal.signalId];
    h->cntr_offset = nccl::gin::efa_gda::hwCounterLoad((uint64_t *)h->cntr_value);
  }
};

/* ── ResetCounter ─────────────────────────────────────────────────── */

template <>
struct ncclGinApi_ResetCounter<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    nccl_ofi_gin_gdaki_dev_handle *dev = nccl::gin::efa_gda::getDevHandle(ctx);
    assert((int)counterId < dev->nCounters
           && "EFA GDA ResetCounter: counterId out of range");
    /* Offset-based reset: snapshot the NIC counter into cntr_offset
     * instead of writing the (NIC-owned) counter. */
    nccl_ofi_gin_gdaki_dev_counter_handle *h = dev->counter_handles[counterId];
    h->cntr_offset = nccl::gin::efa_gda::hwCounterLoad((uint64_t *)h->cntr_value);
  }
};

#endif /* _NCCL_DEVICE_GIN_EFA_GDA_H_ */
