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
static constexpr efa_qp_sharing_mode qpSharingMode =
    (mode == NCCL_GIN_RESOURCE_SHARING_CTA)
        ? EFA_QP_SHARING_MODE_CTA : EFA_QP_SHARING_MODE_GPU;

template <ncclGinResourceSharingMode mode>
static constexpr cuda::thread_scope ncclGinScope =
    (mode == NCCL_GIN_RESOURCE_SHARING_CTA)
        ? cuda::thread_scope_block : cuda::thread_scope_device;

/* ── Mode-scoped submitted_count helpers ──────────────────────────── */

template <ncclGinResourceSharingMode mode>
NCCL_DEVICE_INLINE static void submittedCountAdd(uint64_t *ptr, uint64_t val) {
  cuda::atomic_ref<uint64_t, ncclGinScope<mode>> sc(*ptr);
  sc.fetch_add(val, cuda::memory_order_relaxed);
}

template <ncclGinResourceSharingMode mode>
NCCL_DEVICE_INLINE static uint64_t submittedCountLoad(uint64_t *ptr) {
  cuda::atomic_ref<uint64_t, ncclGinScope<mode>> sc(*ptr);
  return sc.load(cuda::memory_order_relaxed);
}

/* ── postRdmaWrite: shared post path for Put and PutValue ─────────── */

template <ncclGinResourceSharingMode mode>
NCCL_DEVICE_INLINE static void postRdmaWrite(
    nccl_ofi_gin_gdaki_dev_endpoint_handle *ep, int peer,
    uint64_t srcAddr, uint32_t srcLkey, uint32_t writeBytes,
    uint64_t dstAddr, uint32_t dstRkey) {

  efa_cuda_qp       *qp                  = (efa_cuda_qp *)ep->qp;
  uint16_t           ah                   = ep->address_handles[peer];
  uint16_t           qpn                  = ep->remote_qpns[peer];
  uint32_t           qkey                 = ep->qkey[peer];
  volatile uint64_t *submitted_count_ptr  = &ep->submitted_count;
  volatile uint64_t *local_cntr_ptr       = ep->local_cntr_value;
  uint32_t           sq_size_val          = ep->sq_size;

  efa_io_tx_wqe wr;
  efa_cuda_init_rdma_write_wr(&wr, (uint16_t)threadIdx.x, dstRkey, dstAddr);
  efa_cuda_wr_set_sge(&wr, srcLkey, srcAddr, writeBytes);
  efa_cuda_wr_set_remote(&wr, ah, (uint32_t)qpn, qkey);

  /* Lock-free post with coalesced-group doorbell coalescing.
   *
   * Threads concurrently posting to the same QP form a group:
   * coalesced_threads() captures the converged threads and
   * labeled_partition splits them by target QP. The group leader
   * reserves a contiguous slot range for the whole group, every
   * member writes its WR in parallel, and the leader rings a single
   * doorbell for the batch. The group handle stays coherent across
   * the shfl and sync calls under independent thread scheduling. */
  cooperative_groups::coalesced_group active = cooperative_groups::coalesced_threads();
  auto group = cooperative_groups::labeled_partition(active, (unsigned long long)(uintptr_t)qp);

  int  my_idx     = group.thread_rank();
  int  batch_size = group.num_threads();
  bool is_leader  = (my_idx == 0);

  uint32_t base = 0;
  if (is_leader) {
    while (true) {
      uint64_t in_flight = *submitted_count_ptr - *local_cntr_ptr;
      if (in_flight + (uint64_t)batch_size <= (uint64_t)sq_size_val) break;
    }
    base = efa_cuda_start_sq_batch<qpSharingMode<mode>>(qp, (uint32_t)batch_size);
  }
  base = group.shfl(base, 0);

  efa_cuda_sq_batch_place_wr(qp, base + (uint32_t)my_idx, &wr);
  group.sync();

  if (is_leader) {
    efa_cuda_flush_sq_wrs<qpSharingMode<mode>>(qp, base, (uint32_t)batch_size);
    submittedCountAdd<mode>(const_cast<uint64_t*>(submitted_count_ptr), (uint64_t)batch_size);
  }
  group.sync();
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

    /* This backend supports INDEXED signals only, incremented by 1.
     * EFA's FI_REMOTE_WRITE counter ticks exactly once per inbound
     * write, so VA-typed signals and signalOpArg > 1 are not
     * representable. */
    assert((signal.type == NCCL_GIN_SIGNAL_TYPE_NONE
            || signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED)
           && "EFA GDA: only INDEXED signals are supported");
    assert((signal.type == NCCL_GIN_SIGNAL_TYPE_NONE
            || signalOp == ncclGinSignalInc
            || signalOpArg == 1)
           && "EFA GDA: only inc-by-1 signals are supported");

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

      /* Pick endpoint. A signal request routes to signal_handles[];
       * otherwise a counter request routes to counter_handles[].
       * Both arrays index the same underlying sc_endpoint, so the post
       * lands on the same QP; we dereference whichever the caller's
       * request uses, tolerating the other array being NULL. Signal
       * takes priority when both are set. */
      nccl_ofi_gin_gdaki_dev_endpoint_handle *ep;
      if (needsSignalEp) {
        nccl_ofi_gin_gdaki_dev_counter_handle *sch;
        if (signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED) {
          sch = dev->signal_handles[signal.indexedSignal.signalId];
        } else {
          /* hasCounter is true here (needsSignalEp with no signal). */
          sch = dev->counter_handles[counterId];
        }
        ep = &sch->base;
      } else {
        ep = &dev->data;
      }

      postRdmaWrite<mode>(ep, peer, absSrcAddr, srcLkey, writeBytes, absDstAddr, dstRkey);
    }
  }
  (void)signalOp; (void)signalOpArg;
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

  /* Pick endpoint, mirroring Put: a signal request routes to
   * signal_handles[] so the write's arrival ticks the receiver's
   * FI_REMOTE_WRITE counter; otherwise the data endpoint. */
  nccl_ofi_gin_gdaki_dev_endpoint_handle *ep;
  if (signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED) {
    ep = &dev->signal_handles[signal.indexedSignal.signalId]->base;
  } else {
    ep = &dev->data;
  }

  efa_cuda_qp       *qp                  = (efa_cuda_qp *)ep->qp;
  uint16_t           ah                   = ep->address_handles[peer];
  uint16_t           qpn                  = ep->remote_qpns[peer];
  uint32_t           qkey                 = ep->qkey[peer];
  volatile uint64_t *submitted_count_ptr  = &ep->submitted_count;
  volatile uint64_t *local_cntr_ptr       = ep->local_cntr_value;
  uint32_t           sq_size_val          = ep->sq_size;
  uint64_t           slice_base           = ep->putvalue_slice_base;

  uint64_t absDstAddr = dstMh->peers[peer].remote_addr + dstOff;
  uint32_t dstRkey    = dstMh->peers[peer].rkey;

  /* Lock-free post with coalesced-group doorbell coalescing, matching
   * postRdmaWrite. The reserved SQ slot index (base + my_idx) also
   * selects this WQE's source slot in the endpoint's slice of the
   * shared PutValue pool, so concurrent posters never collide on a
   * staging slot. The slice holds sq_size slots, exactly covering the
   * SQ ring depth, so a slot can only be reused once its prior WQE has
   * been retired — gated by the SQ-overflow backpressure below. */
  cooperative_groups::coalesced_group active = cooperative_groups::coalesced_threads();
  auto group = cooperative_groups::labeled_partition(active, (unsigned long long)(uintptr_t)qp);

  int  my_idx     = group.thread_rank();
  int  batch_size = group.num_threads();
  bool is_leader  = (my_idx == 0);

  uint32_t base = 0;
  if (is_leader) {
    while (true) {
      uint64_t in_flight = *submitted_count_ptr - *local_cntr_ptr;
      if (in_flight + (uint64_t)batch_size <= (uint64_t)sq_size_val) break;
    }
    base = efa_cuda_start_sq_batch<qpSharingMode<mode>>(qp, (uint32_t)batch_size);
  }
  base = group.shfl(base, 0);
  uint32_t abs_idx = base + (uint32_t)my_idx;
  /* Stage srcVal into this WQE's slot, then build the RDMA_WRITE WQE
   * from the slot. EFA RDMA_WRITE can't carry inline data, so the NIC
   * DMAs the value from the registered pool. The slice holds sq_size
   * slots (== SQ ring depth), so the reserved absolute SQ index folds
   * 1:1 onto a slot; sq_size is a power of two so % is a mask. */
  uint64_t slot_idx   = (uint64_t)abs_idx % (uint64_t)sq_size_val;
  uint64_t local_addr = slice_base + slot_idx * (uint64_t)dev->putvalue_slot_size;
  *(T *)local_addr = srcVal;
  if (required == cuda::thread_scope_system && given < required) {
    __threadfence_system();
  }

  efa_io_tx_wqe wr;
  efa_cuda_init_rdma_write_wr(&wr, (uint16_t)threadIdx.x, dstRkey, absDstAddr);
  efa_cuda_wr_set_sge(&wr, dev->putvalue_lkey, local_addr, (uint32_t)sizeof(T));
  efa_cuda_wr_set_remote(&wr, ah, (uint32_t)qpn, qkey);

  efa_cuda_sq_batch_place_wr(qp, abs_idx, &wr);
  group.sync();

  if (is_leader) {
    efa_cuda_flush_sq_wrs<qpSharingMode<mode>>(qp, base, (uint32_t)batch_size);
    submittedCountAdd<mode>(const_cast<uint64_t*>(submitted_count_ptr), (uint64_t)batch_size);
  }
  group.sync();
}

/* ── flushImplMode: mode-templated Flush implementation ───────────── */

template <ncclGinResourceSharingMode mode, typename Coop>
NCCL_DEVICE_INLINE static void flushImplMode(ncclGinCtx ctx, Coop coop, cuda::memory_order ord, uint32_t* abortFlag) {
  (void)ord;
  coop.sync();
  if (coop.thread_rank() == 0) {
    nccl_ofi_gin_gdaki_dev_handle *dev = getDevHandle(ctx);

    /* For each endpoint with outstanding work, snapshot submitted_count
     * atomically, then spin on *local_cntr_value until the firmware
     * has caught up. */
    auto wait_for_endpoint = [abortFlag](nccl_ofi_gin_gdaki_dev_endpoint_handle &ep) -> bool {
      uint64_t target = submittedCountLoad<mode>(&ep.submitted_count);

      while (*ep.local_cntr_value < target) {
        if (abortFlag && *abortFlag) return false;
      }
      return true;
    };

    if (!wait_for_endpoint(dev->data)) return;

    for (int i = 0; i < dev->nSignals; i++) {
      if (!wait_for_endpoint(dev->signal_handles[i]->base)) return;
    }

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
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, int peer, ncclGinRequest_t* outRequest, uint32_t optFlags) {
    (void)ctx; (void)peer; (void)outRequest; (void)optFlags;
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
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, cuda::memory_order ord, uint32_t* abortFlag) {
    nccl::gin::efa_gda::flushImpl(ctx, coop, ord, abortFlag);
  }
};

/* ── GetSignalPtr ─────────────────────────────────────────────────── */

template <>
struct ncclGinApi_GetSignalPtr<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static uint64_t* call(ncclGinCtx ctx, ncclGinSignal_t signalId) {
    nccl_ofi_gin_gdaki_dev_handle *dev = nccl::gin::efa_gda::getDevHandle(ctx);
    return (uint64_t *)dev->signal_handles[signalId]->cntr_value;
  }
};

/* ── GetCounterPtr ────────────────────────────────────────────────── */

template <>
struct ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static uint64_t* call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    nccl_ofi_gin_gdaki_dev_handle *dev = nccl::gin::efa_gda::getDevHandle(ctx);
    return (uint64_t *)dev->counter_handles[counterId]->cntr_value;
  }
};

/* ── ResetSignal ──────────────────────────────────────────────────── */

template <>
struct ncclGinApi_ResetSignal<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinSignalDescriptor signal) {
    if (signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED) {
      nccl_ofi_gin_gdaki_dev_handle *dev = nccl::gin::efa_gda::getDevHandle(ctx);
      *((volatile uint64_t *)dev->signal_handles[signal.indexedSignal.signalId]->cntr_value) = 0;
    }
  }
};

/* ── ResetCounter ─────────────────────────────────────────────────── */

template <>
struct ncclGinApi_ResetCounter<NCCL_NET_DEVICE_GIN_EFA_GDA> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    nccl_ofi_gin_gdaki_dev_handle *dev = nccl::gin::efa_gda::getDevHandle(ctx);
    *((volatile uint64_t *)dev->counter_handles[counterId]->cntr_value) = 0;
  }
};

#endif /* _NCCL_DEVICE_GIN_EFA_GDA_H_ */
