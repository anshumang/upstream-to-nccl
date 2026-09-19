/*
 * efa_gda_put_probe.cu — per-put SASS probe for the EFA GDA gin.put device path.
 *
 * One kernel, one put, no other GIN calls, so the SASS of probe_put_kernel is the
 * instruction stream of exactly one EFA GDA put in the instantiation the
 * NCCLOFI-1945 harness uses: THREAD sharing, ncclCoopThread, WeakSignalInc,
 * thread-scope release given, device-scope release required, default flags,
 * device-memory windows.
 *
 * Why not simply `ncclGin::put`: `ncclGin` is ncclGin_BackendMask<ALL>, and its
 * put dispatches on the context's backend at run time, so the kernel would also
 * contain the proxy (and any other enabled) backend's put path and the static
 * counts would not be EFA GDA's. A narrowed ncclGin_BackendMask<EFA_GDA> folds the
 * dispatch, but the signal/counter helper overloads in gin__funcs.h are typed on
 * `ncclGin`, so its put() does not compile with a signal. The kernel below is
 * therefore the device-only, single-segment branch of ncclGin_BackendMask::put
 * (gin__funcs.h) reproduced line for line against the narrowed mask, with the
 * WeakSignalInc helpers expanded to what they return. Nothing is launched; the
 * object exists only to be disassembled:
 *
 *   nvcc -std=c++17 -O2 -gencode arch=compute_90,code=sm_90 [-D<arm flags>] \
 *        -I<nccl include> -c efa_gda_put_probe.cu -o probe.o
 *   cuobjdump --dump-sass probe.o
 */
#include <cstdint>
#include "nccl.h"
#include "nccl_device.h"
#include "nccl_device/gin.h"

constexpr unsigned kEfaGdaMask = 1u << (unsigned)NCCL_NET_DEVICE_GIN_EFA_GDA;
using ncclGinEfaGda = ncclGin_BackendMask<kEfaGdaMask>;

struct ProbeArgs {
  ncclDevComm dev_comm;
  ncclWindow_t send_window;
  ncclWindow_t recv_window;
  size_t offset;
  size_t bytes;
  int peer;
  int context;
  uint32_t signal;
};

__global__ void probe_put_kernel(ProbeArgs a) {
  using nccl::utility::loadConst;
  using nccl::gin::internal::getGinWindow;
  using nccl::gin::internal::teamRankToGinRank;

  ncclGinEfaGda gin{a.dev_comm, a.context, NCCL_GIN_RESOURCE_SHARING_THREAD};
  const ncclTeam team = ncclTeamWorld(a.dev_comm);
  ncclCoopThread coop;

  /* == ncclGin_BackendMask<beMask>::put, device-only branch, for
   *    RemoteAction = ncclGin_WeakSignalInc{a.signal}, LocalAction = ncclGin_None,
   *    DescriptorSmem = ncclGin_None, givenRelease = thread, requiredRelease = device. */
  ncclGinSignalDescriptor sig{};
  sig.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  sig.indexedSignal.signalId = a.signal;
  sig.isStrong = false;

  ncclGinCtx_M<kEfaGdaMask> ctx = gin._makeCtx();
  coop.sync();
  if (coop.thread_rank() == 0) {
    ncclGinCall<ncclGinApi_Put>(
      ctx, ncclCoopThread(), teamRankToGinRank(gin.comm, team, a.peer), /*hasWins=*/true,
      getGinWindow(a.recv_window, gin.comm.backendIndex, gin.connectionId),
      4096 * size_t(loadConst(&a.recv_window->ginOffset4K)) + a.offset,
      getGinWindow(a.send_window, gin.comm.backendIndex, gin.connectionId),
      4096 * size_t(loadConst(&a.send_window->ginOffset4K)) + a.offset, a.bytes,
      sig, ncclGinSignalInc, /*signalOpArg=*/uint64_t(1), /*hasCounter=*/false, ncclGinCounter_t(0),
      /*hasDescriptor=*/false, (ncclGinDescriptorSmem*)nullptr,
      /*requiredRelease=*/cuda::thread_scope_device, /*givenRelease=*/cuda::thread_scope_thread,
      ncclGinOptFlagsDefault);
  }
  coop.sync();
}
