/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * GPU-side test for ncclGinApi_PutValue<NCCL_NET_DEVICE_GIN_EFA_GDA>.
 *
 * Two phases:
 *   Phase 1 (no-signal):
 *     rank 0 -> rank 1 via the data endpoint. signal arg = ncclGin_None{}.
 *     rank 1 polls the destination memory directly.
 *
 *   Phase 2 (signal):
 *     rank 0 -> rank 1 with ncclGin_SignalInc{signalId=0}. The PutValue
 *     RDMA_WRITE rides on signal_handles[0] so its arrival ticks the
 *     receiver's FI_REMOTE_WRITE counter on signalId=0. rank 1 calls
 *     waitSignal(0, least=1) to confirm and then verifies the
 *     destination memory also carries the new value.
 *
 *   Phase 3 (aggregate-requests):
 *     rank 0 -> rank 1 issues BURST=8 PutValues at offsets 0..7 with
 *     ncclGinOptFlagsAggregateRequests on the first 7 and no flag on
 *     the 8th. Only the 8th post should ring the SQ doorbell; the
 *     earlier 7 stage WQEs and skip the MMIO write. rank 1 polls for
 *     the 8th sentinel value at offset 7 (which forces all 7 deferred
 *     WQEs to land first), then verifies offsets 0..6 match their
 *     respective sentinels.
 *
 * Run: mpirun -np 2 ./gin_putvalue_gpu
 * Env: NCCL_NET_PLUGIN=<path to libnccl-net-ofi.so>
 *      OFI_NCCL_GIN_GDAKI=1
 *      FI_PROVIDER=efa
 *      FI_EFA_USE_DEVICE_RDMA=1
 *      FI_EFA_USE_HW_CNTR=1
 *      NCCL_GIN_TYPE=4
 *      NCCL_SYM_GIN_KERNELS_ENABLE=0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include <cuda_runtime.h>

#include "nccl.h"
#include "nccl_device.h"
#include "nccl_device/gin.h"

#define BUF_SIZE 64
#define VAL_NO_SIGNAL 0x42424242
#define VAL_WITH_SIGNAL 0x43434343
/* Phase 3: BURST PutValues, sentinels stride from VAL_BURST_BASE. */
#define BURST 8
#define VAL_BURST_BASE 0x44440000
#define WAIT_TIMEOUT_CYCLES 7500000000UL  /* ~5s on a 1.5GHz GPU */

#define CUDACHECK(cmd) do {                                \
  cudaError_t e = cmd;                                     \
  if (e != cudaSuccess) {                                  \
    printf("CUDA error %s:%d '%s'\n",                      \
           __FILE__, __LINE__, cudaGetErrorString(e));     \
    MPI_Abort(MPI_COMM_WORLD, 1);                          \
  }                                                        \
} while(0)

#define NCCLCHECK(cmd) do {                                \
  ncclResult_t r = cmd;                                    \
  if (r != ncclSuccess) {                                  \
    printf("NCCL error %s:%d '%s'\n",                      \
           __FILE__, __LINE__, ncclGetErrorString(r));     \
    MPI_Abort(MPI_COMM_WORLD, 1);                          \
  }                                                        \
} while(0)

/* Phase 1: no-signal. rank 0 sender. */
__global__ void putvalue_no_signal_kernel(
    ncclDevComm devComm, int ctxId,
    ncclWindow_t dstWin, size_t dstOff,
    int peer, int value)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("SENDER[no-signal]: kernel entered, peer=%d value=0x%x\n", peer, value);

  ncclGin gin{devComm, ctxId};
  gin.putValue<int>(ncclTeamWorld(devComm),
                    peer,
                    dstWin,
                    dstOff,
                    value,
                    ncclGin_None{},
                    ncclCoopThread());
  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("SENDER[no-signal]: putValue posted\n");
  __threadfence_system();
}

/* Phase 1 receiver: poll the destination memory directly. */
__global__ void wait_no_signal_kernel(volatile int *dst, int expected, uint64_t timeout_cycles)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("RECEIVER[no-signal]: polling dst=%p for value 0x%x (timeout %lu cycles)\n",
           (void*)dst, (unsigned)expected, (unsigned long)timeout_cycles);
  uint64_t start = clock64();
  int seen;
  uint64_t iters = 0;
  do {
    seen = *dst;
    iters++;
    if (clock64() - start > timeout_cycles) break;
  } while (seen != expected);
  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("RECEIVER[no-signal]: dst=0x%x after %lu iters %s\n",
           (unsigned)seen, (unsigned long)iters,
           seen == expected ? "PASS" : "TIMEOUT");
}

/* Phase 2 sender: PutValue with ncclGin_SignalInc{signalId=0}. */
__global__ void putvalue_with_signal_kernel(
    ncclDevComm devComm, int ctxId,
    ncclWindow_t dstWin, size_t dstOff,
    int peer, int value, ncclGinSignal_t signalId)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("SENDER[signal]: kernel entered, peer=%d value=0x%x signalId=%d\n",
           peer, value, (int)signalId);

  ncclGin gin{devComm, ctxId};
  gin.putValue<int>(ncclTeamWorld(devComm),
                    peer,
                    dstWin,
                    dstOff,
                    value,
                    ncclGin_SignalInc{signalId},
                    ncclCoopThread());
  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("SENDER[signal]: putValue posted\n");
  __threadfence_system();
}

/* Phase 2 receiver: wait for the FI_REMOTE_WRITE signal counter to tick,
 * then read dst memory and report. */
__global__ void wait_signal_kernel(
    ncclDevComm devComm, int ctxId,
    volatile int *dst, int expected, ncclGinSignal_t signalId)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("RECEIVER[signal]: waiting on signalId=%d (least=1)\n", (int)signalId);

  ncclGin gin{devComm, ctxId};
  /* Wait for signal to reach 1: the put's arrival bumps the
   * FI_REMOTE_WRITE counter on signal_handles[signalId]. */
  gin.waitSignal(ncclCoopThread{}, signalId, /*least=*/1);

  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("RECEIVER[signal]: waitSignal returned, reading dst\n");
  __threadfence_system();
  int seen = *dst;
  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("RECEIVER[signal]: dst=0x%x %s\n",
           (unsigned)seen, seen == expected ? "PASS" : "FAIL");
}

/* Phase 3 sender: BURST PutValues, deferring the doorbell on the first
 * BURST-1 calls via ncclGinOptFlagsAggregateRequests. Only the last
 * call rings the doorbell, so the firmware sees one MMIO write per
 * burst (instead of BURST). */
__global__ void putvalue_burst_aggregate_kernel(
    ncclDevComm devComm, int ctxId,
    ncclWindow_t dstWin, size_t dstOffBase,
    int peer)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("SENDER[burst]: kernel entered, peer=%d burst=%d base_dstOff=%zu\n",
           peer, BURST, (size_t)dstOffBase);

  ncclGin gin{devComm, ctxId};
  for (int i = 0; i < BURST; i++) {
    int v = (int)(VAL_BURST_BASE + i);
    /* Defer the SQ doorbell on every post except the last; the last
     * post (no flag) rings everything that has been staged. */
    uint32_t flags = (i < BURST - 1) ? (uint32_t)ncclGinOptFlagsAggregateRequests
                                     : (uint32_t)ncclGinOptFlagsDefault;
    gin.putValue<int>(ncclTeamWorld(devComm),
                      peer,
                      dstWin,
                      dstOffBase + (size_t)i * sizeof(int),
                      v,
                      ncclGin_None{},
                      ncclCoopThread{},
                      ncclGin_None{},
                      cuda::thread_scope_thread,
                      cuda::thread_scope_device,
                      flags);
  }
  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("SENDER[burst]: %d putValues posted (last with no flag)\n", BURST);
  __threadfence_system();
}

/* Phase 3 receiver: poll until all BURST sentinels have landed, then
 * verify each is in place at its respective offset.
 *
 * Deferring the doorbell coalesces the MMIO writes but does not impose
 * delivery order: EFA SRD gives no ordering between work requests, so
 * the final non-aggregate post (which rings the doorbell and releases
 * every deferred WQE) can itself be delivered before some of the
 * earlier deferred posts. The aggregate-requests contract is only that
 * all posts land once the doorbell is rung — not that they arrive in
 * order — so the receiver polls for every offset rather than keying off
 * the last one. */
__global__ void wait_burst_kernel(volatile int *dst_base, uint64_t timeout_cycles, int *result_out)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("RECEIVER[burst]: polling dst[0..%d] for all %d sentinels\n",
           BURST - 1, BURST);

  /* Wait until every offset carries its sentinel (or timeout). */
  uint64_t start = clock64();
  uint64_t iters = 0;
  int remaining;
  do {
    __threadfence_system();
    remaining = 0;
    for (int i = 0; i < BURST; i++) {
      if (dst_base[i] != (int)(VAL_BURST_BASE + i)) remaining++;
    }
    iters++;
    if (clock64() - start > timeout_cycles) break;
  } while (remaining != 0);

  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("RECEIVER[burst]: %d/%d sentinels landed after %lu iters %s\n",
           BURST - remaining, BURST, (unsigned long)iters,
           remaining == 0 ? "(all seen)" : "(TIMEOUT)");

  /* Verify all BURST sentinels are in place. */
  __threadfence_system();
  int errors = 0;
  for (int i = 0; i < BURST; i++) {
    int got = dst_base[i];
    int want = (int)(VAL_BURST_BASE + i);
    if (got != want) {
      if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("RECEIVER[burst]: dst[%d]=0x%x expected=0x%x MISMATCH\n",
               i, (unsigned)got, (unsigned)want);
      errors++;
    }
  }
  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("RECEIVER[burst]: %d/%d sentinels match %s\n",
           BURST - errors, BURST, errors == 0 ? "PASS" : "FAIL");

  if (threadIdx.x == 0 && blockIdx.x == 0)
    *result_out = errors;
}

int main(int argc, char **argv)
{
  int rank, nranks;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  if (nranks != 2) {
    if (rank == 0) printf("This test requires exactly 2 ranks (one per node).\n");
    MPI_Finalize();
    return 1;
  }

  CUDACHECK(cudaSetDevice(0));

  ncclUniqueId id;
  if (rank == 0) NCCLCHECK(ncclGetUniqueId(&id));
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, nranks, id, rank));
  printf("Rank %d: ncclCommInitRank done\n", rank);

  /* Allocate dst buffer */
  int *dst_gpu = nullptr;
  NCCLCHECK(ncclMemAlloc((void**)&dst_gpu, BUF_SIZE));
  CUDACHECK(cudaMemset(dst_gpu, 0, BUF_SIZE));

  /* Register window symmetric */
  ncclWindow_t dstWin = nullptr;
  NCCLCHECK(ncclGroupStart());
  NCCLCHECK(ncclCommWindowRegister(comm, dst_gpu, BUF_SIZE, &dstWin, NCCL_WIN_COLL_SYMMETRIC));
  NCCLCHECK(ncclGroupEnd());
  printf("Rank %d: window registered\n", rank);

  /* Create devComm:
   *   ginContextCount = 1
   *   ginSignalCount  = 1   (so signal_handles[0] / sc_endpoints[0] exist)
   */
  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.barrierCount = 1;
  reqs.ginContextCount = 1;
  reqs.ginSignalCount = 1;
  reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  reqs.ginForceEnable = true;

  ncclDevComm *devCommPtr = nullptr;
  CUDACHECK(cudaMallocManaged(&devCommPtr, sizeof(ncclDevComm)));

  NCCLCHECK(ncclGroupStart());
  NCCLCHECK(ncclDevCommCreate(comm, &reqs, devCommPtr));
  NCCLCHECK(ncclGroupEnd());
  printf("Rank %d: ncclDevCommCreate done, nRanks=%d\n", rank, devCommPtr->nRanks);

  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  /* ============== Phase 1: no-signal ============== */
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    putvalue_no_signal_kernel<<<1, 1, 0, stream>>>(
        *devCommPtr, /*ctxId=*/0,
        dstWin, /*dstOff=*/0,
        /*peer=*/1, /*value=*/VAL_NO_SIGNAL);
    CUDACHECK(cudaStreamSynchronize(stream));
    printf("R0: phase1 (no-signal) sender done\n");
  } else {
    wait_no_signal_kernel<<<1, 1, 0, stream>>>(dst_gpu, VAL_NO_SIGNAL, WAIT_TIMEOUT_CYCLES);
    CUDACHECK(cudaStreamSynchronize(stream));
    int seen;
    CUDACHECK(cudaMemcpy(&seen, dst_gpu, sizeof(int), cudaMemcpyDeviceToHost));
    printf("R1: phase1 (no-signal) final dst=0x%x %s\n",
           (unsigned)seen, seen == VAL_NO_SIGNAL ? "PASS" : "FAIL");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  /* ============== Phase 2: signal ============== */
  /* Reset dst on rank 1 so the receiver test cleanly distinguishes
   * phase-1 from phase-2 arrivals. */
  if (rank == 1) {
    CUDACHECK(cudaMemset(dst_gpu, 0, sizeof(int)));
  }
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    putvalue_with_signal_kernel<<<1, 1, 0, stream>>>(
        *devCommPtr, /*ctxId=*/0,
        dstWin, /*dstOff=*/0,
        /*peer=*/1, /*value=*/VAL_WITH_SIGNAL,
        /*signalId=*/0);
    CUDACHECK(cudaStreamSynchronize(stream));
    printf("R0: phase2 (signal) sender done\n");
  } else {
    wait_signal_kernel<<<1, 1, 0, stream>>>(
        *devCommPtr, /*ctxId=*/0,
        dst_gpu, VAL_WITH_SIGNAL, /*signalId=*/0);
    CUDACHECK(cudaStreamSynchronize(stream));
    int seen;
    CUDACHECK(cudaMemcpy(&seen, dst_gpu, sizeof(int), cudaMemcpyDeviceToHost));
    printf("R1: phase2 (signal) final dst=0x%x %s\n",
           (unsigned)seen, seen == VAL_WITH_SIGNAL ? "PASS" : "FAIL");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  /* ============== Phase 3: aggregate-requests ============== */
  /* Reset the receiver's first BURST*sizeof(int) bytes of dst so we can
   * unambiguously detect arrival of each sentinel. */
  if (rank == 1) {
    CUDACHECK(cudaMemset(dst_gpu, 0, BURST * sizeof(int)));
  }
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    putvalue_burst_aggregate_kernel<<<1, 1, 0, stream>>>(
        *devCommPtr, /*ctxId=*/0,
        dstWin, /*dstOffBase=*/0,
        /*peer=*/1);
    CUDACHECK(cudaStreamSynchronize(stream));
    printf("R0: phase3 (aggregate-requests) sender done\n");
  } else {
    int *errors_dev = nullptr;
    CUDACHECK(cudaMalloc(&errors_dev, sizeof(int)));
    CUDACHECK(cudaMemset(errors_dev, -1, sizeof(int)));
    wait_burst_kernel<<<1, 1, 0, stream>>>(dst_gpu, WAIT_TIMEOUT_CYCLES, errors_dev);
    CUDACHECK(cudaStreamSynchronize(stream));
    int errors_host = -1;
    CUDACHECK(cudaMemcpy(&errors_host, errors_dev, sizeof(int), cudaMemcpyDeviceToHost));
    int seen[BURST] = {0};
    CUDACHECK(cudaMemcpy(seen, dst_gpu, BURST * sizeof(int), cudaMemcpyDeviceToHost));
    printf("R1: phase3 (aggregate-requests) errors=%d  values=[", errors_host);
    for (int i = 0; i < BURST; i++) {
      printf("0x%x%s", (unsigned)seen[i], i == BURST - 1 ? "" : ", ");
    }
    printf("] %s\n", errors_host == 0 ? "PASS" : "FAIL");
    CUDACHECK(cudaFree(errors_dev));
  }
  MPI_Barrier(MPI_COMM_WORLD);

  /* Cleanup */
  CUDACHECK(cudaStreamDestroy(stream));
  NCCLCHECK(ncclCommWindowDeregister(comm, dstWin));
  NCCLCHECK(ncclMemFree(dst_gpu));
  NCCLCHECK(ncclDevCommDestroy(comm, devCommPtr));
  NCCLCHECK(ncclCommFinalize(comm));
  NCCLCHECK(ncclCommDestroy(comm));
  CUDACHECK(cudaFree(devCommPtr));

  printf("Rank %d: done\n", rank);
  MPI_Finalize();
  return 0;
}
