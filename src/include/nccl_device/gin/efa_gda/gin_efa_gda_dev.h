/*************************************************************************
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * EFA GDA device handle struct, mirroring the layout defined in
 * aws-ofi-nccl (nccl_ofi_gin_gdaki_dev.h). The plugin's createContext()
 * populates this struct in GPU memory; the kernel code reads it.
 *
 * IMPORTANT: Must stay in sync with the plugin-side definition.
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_EFA_GDA_DEV_H_
#define _NCCL_DEVICE_GIN_EFA_GDA_DEV_H_

#include <stdint.h>

/*
 * Common per-endpoint state shared by every endpoint flavor. Holds
 * the GPU-resident QP/CQ, per-peer addressing, the per-QP spinlock
 * that serializes the device-side WQE-post sequence, and the
 * counter-based completion tracking fields.
 */
struct nccl_ofi_gin_gdaki_dev_endpoint_handle {
  void *qp;                        /* GPU-resident QP (efa_cuda_qp layout) */
  void *cq;                        /* GPU-resident CQ (efa_cuda_cq layout) */
  uint16_t *address_handles;       /* [nranks] per-peer addressing */
  uint16_t *remote_qpns;
  uint32_t *qkey;

  /* Per-QP spinlock for the device-side WQE post path. efa-dp-direct's
   * start_sq_batch / sq_batch_place_wr / flush_sq_wrs sequence is
   * single-threaded per QP (per the efa-dp-direct CUDA README). One
   * lock per endpoint lets multiple CTAs targeting different endpoints
   * proceed in parallel; only CTAs targeting the same endpoint contend. */
  uint32_t sq_lock;

  /* Counter-based completion tracking.
   *
   * `local_cntr_value` points at the FI_WRITE hardware counter for this
   * endpoint's QP. The NIC increments it on every locally-completed
   * outgoing WR. The kernel reads it directly from GPU memory.
   *
   * `submitted_count` is incremented by the device under sq_lock after
   * a successful flush_sq_wrs. (submitted_count - *local_cntr_value)
   * gives the number of WRs still in flight on this QP — used by the
   * SQ-overflow backpressure check and by Flush to wait for local
   * completion. */
  volatile uint64_t *local_cntr_value;
  uint64_t submitted_count;

  /* SQ ring size for this endpoint's QP. Used by Put to gate new
   * batches against in-flight WRs (efa-dp-direct's start_sq_batch
   * does not validate ring overflow on its own). The kernel spins
   * until (submitted_count - *local_cntr_value + batch_size)
   * <= sq_size before reserving slots. */
  uint32_t sq_size;

  uint32_t putvalue_pad;

  /* Base address of this endpoint's slice of the shared PutValue
   * source slot pool. The pool is registered as a single MR on the
   * plugin side; the same lkey (dev_handle->putvalue_lkey) covers
   * every slice. Slice size is implied by sq_size; per-call slot
   * byte offset is
   *   (submitted_count % sq_size) * dev_handle->putvalue_slot_size.
   * Set by the plugin at createContext time; read-only on the device. */
  uint64_t putvalue_slice_base;
};

/*
 * Per-signal/counter endpoint handle, returned to device code through
 * dev_handle->signal_handles[] and dev_handle->counter_handles[].
 *
 * Composes nccl_ofi_gin_gdaki_dev_endpoint_handle (qp / cq / addressing /
 * sq_lock / counter completion tracking) and adds the cntr_value
 * pointer that the kernel reads to observe signal arrivals
 * (FI_REMOTE_WRITE) or counter increments (FI_WRITE).
 */
struct nccl_ofi_gin_gdaki_dev_counter_handle {
  /* Endpoint-common fields (qp, cq, addressing, sq_lock,
   * counter completion tracking). */
  struct nccl_ofi_gin_gdaki_dev_endpoint_handle base;
  /* NIC writes the hardware counter value here (GPU memory).
   * For signals: FI_REMOTE_WRITE count. For counters: FI_WRITE count. */
  volatile uint64_t *cntr_value;
};

/*
 * Device-visible handle returned from createContext, populated in GPU
 * memory by the plugin and dereferenced by the kernel.
 */
struct nccl_ofi_gin_gdaki_dev_handle {
  /* Data endpoint. The plugin binds a FI_WRITE counter to this
   * endpoint and populates data.local_cntr_value at createContext
   * time so the kernel can poll for local completion. */
  struct nccl_ofi_gin_gdaki_dev_endpoint_handle data;

  /* Per-counter / per-signal endpoint handles, populated when the
   * caller asked createContext for nCounters / nSignals > 0. Both
   * arrays index into the same underlying signal/counter endpoint
   * (sc_endpoint) on the plugin side; they expose two views of that
   * endpoint with cntr_value pointing at the FI_WRITE counter
   * (counter_handles) or the FI_REMOTE_WRITE counter (signal_handles).
   * The array pointer is NULL when the corresponding count is zero. */
  struct nccl_ofi_gin_gdaki_dev_counter_handle **counter_handles; /* [nCounters] or NULL */
  struct nccl_ofi_gin_gdaki_dev_counter_handle **signal_handles;  /* [nSignals]  or NULL */
  int32_t nCounters;
  int32_t nSignals;

  int32_t nranks;
  int32_t rank;

  /* Multi-rail: the rail (EFA NIC) this logical context is bound to.
   * The plugin opens this context's endpoints on rail rail_id's domain
   * and bakes that rail's scratch / putvalue MR keys into this handle.
   * The kernel uses rail_id only to index the per-rail memory handle
   * array that regMrSym returns as the window (see Put/PutValue); all
   * endpoint / counter / scratch / putvalue fields here are already
   * rail-resolved, so the rest of the device path is rail-agnostic.
   *
   * The plugin sets rail_id = contextId % num_rails, so on a
   * 2-NIC-per-GPU node distinct contextIds spread across both NICs. */
  uint32_t rail_id;

  /* Per-context signal-only scratch buffer, used by Put when the
   * caller has no payload (hasWins=false || bytes=0) but has
   * requested a signal/counter. EFA's RDMA write needs a real
   * remote address to bump the receiver's FI_REMOTE_WRITE counter,
   * so the plugin allocates a small buffer per createContext,
   * registers it on the proxy domain, and allgathers the
   * (local_addr, rkey) per rank. The kernel posts a 0-byte RDMA
   * write to the peer's scratch on the signal endpoint; the buffer
   * content is never read.
   *
   * scratch_remote_addrs / scratch_remote_rkeys are [nranks] arrays
   * in GPU memory.
   */
  uint32_t scratch_lkey;
  uint32_t scratch_pad;
  uint64_t scratch_local_addr;
  uint64_t *scratch_remote_addrs;
  uint32_t *scratch_remote_rkeys;

  /* PutValue source slot pool, shared across the data endpoint and
   * every signal/counter (sc) endpoint.
   *
   * Mirrors the plumbing in
   * aws-ofi-nccl/include/rdma/gin/nccl_ofi_gin_gdaki_dev.h. EFA's
   * RDMA_WRITE WQE cannot use inline data, so ncclGinApi_PutValue
   * stages the user value through a registered source slot, then
   * RDMA-writes the slot to the user destination. The arrival of that
   * write at the receiver bumps the FI_REMOTE_WRITE counter on the
   * chosen sc_endpoint, providing the signal as a side effect of the
   * value transfer.
   *
   * Routing (matches Put):
   *   signal != NONE  -> sc_endpoints[signalId]
   *   signal == NONE  -> data endpoint
   *
   * Each participating endpoint owns a slice; the slice base lives
   * on its dev_endpoint_handle (see putvalue_slice_base above). One
   * MR covers the whole pool, so a single lkey + a single uniform
   * slot stride live here. */
  uint32_t putvalue_lkey;
  uint32_t putvalue_slot_size;
};

/*
 * Per-peer MR metadata. EFA's domain advertises FI_MR_VIRT_ADDR, so
 * RDMA write WQEs take absolute virtual addresses for both local and
 * remote buffers. The kernel passes an offset (srcOff / dstOff) and
 * the absolute address is computed as base_va + offset.
 */
struct nccl_ofi_gin_gdaki_mr_peer {
  uint64_t remote_addr;  /* remote rank's base VA for this MR */
  uint32_t rkey;         /* remote rank's rkey */
  uint32_t pad;
};

struct nccl_ofi_gin_gdaki_mr_handle {
  uint32_t lkey;
  int32_t nranks;
  uint64_t local_addr;                       /* local base VA for this MR */
  struct nccl_ofi_gin_gdaki_mr_peer peers[]; /* [nranks] flex array */
};

/* regMrSym registers a window's memory once per rail (each rail has its
 * own libfabric domain, hence its own lkey and the peer's per-rail rkey)
 * and returns, as the window handle, an array of per-rail mr_handle
 * pointers: (nccl_ofi_gin_gdaki_mr_handle *)[num_rails]. The kernel
 * selects the handle for the rail its logical context is bound to by
 * indexing that array with dev->rail_id — see Put/PutValue. The caller
 * therefore selects the rail purely by which contextId (hence which
 * dev_handle) it issues through; the rest of the path is rail-agnostic. */

#endif /* _NCCL_DEVICE_GIN_EFA_GDA_DEV_H_ */
