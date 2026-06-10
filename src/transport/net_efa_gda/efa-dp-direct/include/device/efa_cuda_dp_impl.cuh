// SPDX-License-Identifier: Apache-2.0
// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

#ifndef EFA_CUDA_DP_IMPL_CUH
#define EFA_CUDA_DP_IMPL_CUH

#include <stdio.h>
#include <cstring>
#include <errno.h>
#include <stdint.h>
#include <cuda/atomic>

#include "../host/efa_cuda_dp.h"
#include "../common/efa_io_defs.h"

#define BIT(nr)		(1UL << (nr))

#define __bf_shf(x)	(__builtin_ffsll(x) - 1)

#define EFA_FIELD_GET(_mask, _reg)						\
	({									\
		(typeof(_mask))(((_reg) & (_mask)) >> __bf_shf(_mask));		\
	})

#define EFA_FIELD_PREP(_mask, _val)						\
	({									\
		((typeof(_mask))(_val) << __bf_shf(_mask)) & (_mask);		\
	})

#define BITS_PER_LONG	(8 * sizeof(long))

#define GENMASK(h, l)								\
	(((~0UL) - (1UL << (l)) + 1) & (~0UL >> (BITS_PER_LONG - 1 - (h))))

#define EFA_GET(ptr, mask)							\
	EFA_FIELD_GET(mask##_MASK, *(typeof(*ptr) volatile *)(ptr))

#define EFA_SET(ptr, mask, value)						\
	({									\
		typeof(ptr) _ptr = ptr;				        	\
		*_ptr = (*_ptr & ~(mask##_MASK)) |				\
			EFA_FIELD_PREP(mask##_MASK, value);		        \
	})

#define efa_container_of(ptr, type, field)					\
	((type *) ((char *)ptr - offsetof(type, field)))


__device__ static inline int efa_cuda_cqe_is_pending(const efa_io_cdesc_common *cqe_common, int phase)
{
	return EFA_GET(&cqe_common->flags, EFA_IO_CDESC_COMMON_PHASE) == phase;
}

__device__ static inline efa_io_cdesc_common *efa_cuda_get_cqe(efa_cuda_cq *cq, uint32_t position)
{
	uint32_t index = (cq->cc + position) & cq->queue_mask;
	return (efa_io_cdesc_common *)(cq->buf + (index * cq->entry_size));
}

__device__ static inline int efa_cuda_get_cqe_phase(efa_cuda_cq *cq, uint32_t position)
{
	return cq->phase ^ (((cq->cc & cq->queue_mask) + position) >> cq->queue_size_shift);
}

__device__ static inline void *efa_cuda_cq_poll(efa_cuda_cq *cq, uint32_t position)
{
	efa_io_cdesc_common *cqe = efa_cuda_get_cqe(cq, position);
	int cqe_phase = efa_cuda_get_cqe_phase(cq, position);

	if (efa_cuda_cqe_is_pending(cqe, cqe_phase)) {
		__threadfence_block();

		return cqe;
	}
	return nullptr;
}
__device__ static inline int efa_cuda_cq_pop(efa_cuda_cq *cq, uint32_t amount)
{
	cq->phase = efa_cuda_get_cqe_phase(cq, amount);
	cq->cc += amount;

	return 0;
}

__device__ static inline enum efa_cuda_wc_opcode efa_cuda_wc_read_opcode(void *wc_buf)
{
	enum efa_io_send_op_type op_type;
	struct efa_io_cdesc_common *cqe = (struct efa_io_cdesc_common *)wc_buf;

	op_type = (enum efa_io_send_op_type)EFA_GET(&cqe->flags, EFA_IO_CDESC_COMMON_OP_TYPE);

	if (EFA_GET(&cqe->flags, EFA_IO_CDESC_COMMON_Q_TYPE) == EFA_IO_SEND_QUEUE) {
		if (op_type == EFA_IO_RDMA_WRITE)
			return EFA_CUDA_WC_RDMA_WRITE;

		if (op_type == EFA_IO_RDMA_READ)
			return EFA_CUDA_WC_RDMA_READ;

		return EFA_CUDA_WC_SEND;
	}

	if (op_type == EFA_IO_RDMA_WRITE)
		return EFA_CUDA_WC_RECV_RDMA_WITH_IMM;

	return EFA_CUDA_WC_RECV;
}

__device__ static inline bool efa_cuda_wc_is_unsolicited(void *wc_buf)
{
	struct efa_io_cdesc_common *cqe = (struct efa_io_cdesc_common *)wc_buf;

	return EFA_GET(&cqe->flags, EFA_IO_CDESC_COMMON_UNSOLICITED);
}

__device__ static inline uint16_t efa_cuda_wc_read_req_id(void *wc_buf)
{
	struct efa_io_cdesc_common *cqe = (struct efa_io_cdesc_common *)wc_buf;

	return cqe->req_id;
}

__device__ static inline uint32_t efa_cuda_wc_read_vendor_err(void *wc_buf)
{
	struct efa_io_cdesc_common *cqe = (struct efa_io_cdesc_common *)wc_buf;

	return cqe->status;
}

__device__ static inline bool efa_cuda_wc_has_imm(void *wc_buf)
{
	struct efa_io_cdesc_common *cqe = (struct efa_io_cdesc_common *)wc_buf;

	return EFA_GET(&cqe->flags, EFA_IO_CDESC_COMMON_HAS_IMM);
}

__device__ static inline uint32_t efa_cuda_wc_read_imm_data(void *wc_buf)
{
	struct efa_io_rx_cdesc *rcqe;

	rcqe = efa_container_of(wc_buf, struct efa_io_rx_cdesc, common);

	return rcqe->imm;
}

__device__ static inline uint32_t efa_cuda_wc_read_byte_len(void *wc_buf)
{
	struct efa_io_cdesc_common *cqe = (struct efa_io_cdesc_common *)wc_buf;
	struct efa_io_rx_cdesc_ex *rcqe;
	uint32_t length;

	if (EFA_GET(&cqe->flags, EFA_IO_CDESC_COMMON_Q_TYPE) != EFA_IO_RECV_QUEUE)
		return 0;

	rcqe = efa_container_of(cqe, struct efa_io_rx_cdesc_ex, base.common);

	length = rcqe->base.length;
	if (EFA_GET(&cqe->flags, EFA_IO_CDESC_COMMON_OP_TYPE) == EFA_IO_RDMA_WRITE)
		length |= ((uint32_t)rcqe->u.rdma_write.length_hi << 16);

	return length;
}

__device__ static inline uint32_t efa_cuda_wc_read_qp_num(void *wc_buf)
{
	struct efa_io_cdesc_common *cqe = (struct efa_io_cdesc_common *)wc_buf;

	return cqe->qp_num;
}

__device__ static inline uint32_t efa_cuda_wc_read_src_qp(void *wc_buf)
{
	struct efa_io_rx_cdesc *rcqe;

	rcqe = efa_container_of(wc_buf, struct efa_io_rx_cdesc, common);

	return rcqe->src_qp_num;
}

__device__ static inline uint32_t efa_cuda_wc_read_slid(void *wc_buf)
{
	struct efa_io_rx_cdesc *rcqe;

	rcqe = efa_container_of(wc_buf, struct efa_io_rx_cdesc, common);

	return rcqe->ah;
}

__device__ static inline int efa_cuda_sq_init_wr(void *wr_buf, enum efa_io_send_op_type op_type, uint16_t wr_id)
{
	struct efa_io_tx_wqe *wqe = (struct efa_io_tx_wqe *)wr_buf;

	memset(wqe, 0, sizeof(*wqe));
	EFA_SET(&wqe->meta.ctrl1, EFA_IO_TX_META_DESC_META_DESC, 1);
	EFA_SET(&wqe->meta.ctrl1, EFA_IO_TX_META_DESC_OP_TYPE, op_type);
	EFA_SET(&wqe->meta.ctrl2, EFA_IO_TX_META_DESC_FIRST, 1);
	EFA_SET(&wqe->meta.ctrl2, EFA_IO_TX_META_DESC_LAST, 1);
	EFA_SET(&wqe->meta.ctrl2, EFA_IO_TX_META_DESC_COMP_REQ, 1);

	wqe->meta.req_id = wr_id;

	return 0;
}

__device__ static inline void efa_cuda_set_wqe_imm_data(void *wr_buf, uint32_t imm_data)
{
	struct efa_io_tx_wqe *wqe = (struct efa_io_tx_wqe *)wr_buf;

	wqe->meta.immediate_data = imm_data;
	EFA_SET(&wqe->meta.ctrl1, EFA_IO_TX_META_DESC_HAS_IMM, 1);
}

__device__ static inline void efa_cuda_set_remote_mem(struct efa_io_remote_mem_addr *remote_mem, uint32_t rkey, uint64_t remote_addr)
{
	remote_mem->rkey = rkey;
	remote_mem->buf_addr_lo = remote_addr & 0xFFFFFFFF;
	remote_mem->buf_addr_hi = remote_addr >> 32;
}

__device__ static inline void efa_cuda_set_tx_buf(struct efa_io_tx_buf_desc *tx_buf, uint64_t addr, uint32_t lkey, uint32_t length)
{
	tx_buf->length = length;
	EFA_SET(&tx_buf->lkey, EFA_IO_TX_BUF_DESC_LKEY, lkey);
	tx_buf->buf_addr_lo = addr & 0xffffffff;
	tx_buf->buf_addr_hi = addr >> 32;
}

__device__ static inline int efa_cuda_init_send_wr(void *wr_buf, uint16_t wr_id)
{
	return efa_cuda_sq_init_wr(wr_buf, EFA_IO_SEND, wr_id);
}

__device__ static inline int efa_cuda_init_send_imm_wr(void *wr_buf, uint16_t wr_id, uint32_t imm_data)
{
	int ret;

	ret = efa_cuda_sq_init_wr(wr_buf, EFA_IO_SEND, wr_id);
	if (ret)
		return ret;

	efa_cuda_set_wqe_imm_data(wr_buf, imm_data);

	return 0;
}

__device__ static inline int efa_cuda_init_rdma_read_wr(void *wr_buf, uint16_t wr_id, uint32_t rkey, uint64_t remote_addr)
{
	struct efa_io_tx_wqe *wqe = (struct efa_io_tx_wqe *)wr_buf;
	int ret;

	ret = efa_cuda_sq_init_wr(wr_buf, EFA_IO_RDMA_READ, wr_id);
	if (ret)
		return ret;

	efa_cuda_set_remote_mem(&wqe->data.rdma_req.remote_mem, rkey, remote_addr);

	return 0;
}

__device__ static inline int efa_cuda_init_rdma_write_wr(void *wr_buf, uint16_t wr_id, uint32_t rkey, uint64_t remote_addr)
{
	struct efa_io_tx_wqe *wqe = (struct efa_io_tx_wqe *)wr_buf;
	int ret;

	ret = efa_cuda_sq_init_wr(wr_buf, EFA_IO_RDMA_WRITE, wr_id);
	if (ret)
		return ret;

	efa_cuda_set_remote_mem(&wqe->data.rdma_req.remote_mem, rkey, remote_addr);

	return 0;
}

__device__ static inline int efa_cuda_init_rdma_write_imm_wr(void *wr_buf, uint16_t wr_id, uint32_t rkey, uint64_t remote_addr, uint32_t imm_data)
{
	struct efa_io_tx_wqe *wqe = (struct efa_io_tx_wqe *)wr_buf;
	int ret;

	ret = efa_cuda_sq_init_wr(wr_buf, EFA_IO_RDMA_WRITE, wr_id);
	if (ret)
		return ret;

	efa_cuda_set_remote_mem(&wqe->data.rdma_req.remote_mem, rkey, remote_addr);
	efa_cuda_set_wqe_imm_data(wr_buf, imm_data);

	return 0;
}

__device__ static inline void efa_cuda_wr_set_remote(void *wr_buf, uint16_t ah, uint32_t remote_qpn, uint32_t remote_qkey)
{
	struct efa_io_tx_wqe *wqe = (struct efa_io_tx_wqe *)wr_buf;

	wqe->meta.ah = ah;
	wqe->meta.dest_qp_num = remote_qpn;
	wqe->meta.qkey = remote_qkey;
}

__device__ static inline int efa_cuda_wr_set_inline_data(void *wr_buf, void *addr, size_t length)
{
	struct efa_io_tx_wqe *wqe = (struct efa_io_tx_wqe *)wr_buf;
	uint8_t op_type;

	if (length > sizeof(wqe->data.inline_data))
		return -EINVAL;

	op_type = EFA_GET(&wqe->meta.ctrl1, EFA_IO_TX_META_DESC_OP_TYPE);
	if (op_type != EFA_IO_SEND)
		return -EINVAL;

	EFA_SET(&wqe->meta.ctrl1, EFA_IO_TX_META_DESC_INLINE_MSG, 1);
	memcpy(wqe->data.inline_data, addr, length);
	wqe->meta.length = length;

	return 0;
}

__device__ static inline int efa_cuda_wr_set_sge(void *wr_buf, uint32_t lkey, uint64_t addr, uint32_t length)
{
	struct efa_io_tx_buf_desc *buf;
	struct efa_io_tx_wqe *wqe;
	uint8_t op_type;

	wqe = (struct efa_io_tx_wqe *)wr_buf;
	wqe->meta.length = 1;

	op_type = EFA_GET(&wqe->meta.ctrl1, EFA_IO_TX_META_DESC_OP_TYPE);
	switch (op_type) {
	case EFA_IO_SEND:
		buf = &wqe->data.sgl[0];
		break;
	case EFA_IO_RDMA_READ:
	case EFA_IO_RDMA_WRITE:
		wqe->data.rdma_req.remote_mem.length = length;
		buf = &wqe->data.rdma_req.local_mem[0];
		break;
	default:
		return -EINVAL;
	}

	efa_cuda_set_tx_buf(buf, addr, lkey, length);
	return 0;
}

__device__ static inline int efa_cuda_get_wqe_phase(efa_cuda_wq *wq, uint32_t abs_idx)
{
	return (int)((abs_idx >> wq->queue_size_shift) & 1u);
}

/* ==================================================================
 * Lock-free SQ post
 * ------------------------------------------------------------------
 * Concurrent CTAs / threads can post to the same QP without a spin
 * lock. Scoped on efa_qp_sharing_mode (CTA -> block-scope atomics,
 * GPU -> device-scope atomics).
 *
 * Uses the wqes_posted field as the reservation index and the
 * wqes_completed field as the ready/commit cursor.
 * ================================================================== */

enum efa_qp_sharing_mode {
	EFA_QP_SHARING_MODE_CTA = 0,
	EFA_QP_SHARING_MODE_GPU = 1,
};

template <enum efa_qp_sharing_mode mode>
static constexpr cuda::thread_scope efa_qp_scope =
	(mode == EFA_QP_SHARING_MODE_CTA)
		? cuda::thread_scope_block : cuda::thread_scope_device;

template <enum efa_qp_sharing_mode mode>
__device__ static inline uint32_t efa_cuda_atomic_add_u32(uint32_t *ptr, uint32_t val)
{
	cuda::atomic_ref<uint32_t, efa_qp_scope<mode>> r(*ptr);
	return r.fetch_add(val, cuda::memory_order_relaxed);
}

template <enum efa_qp_sharing_mode mode>
__device__ static inline uint32_t efa_cuda_atomic_load_u32(uint32_t *ptr)
{
	cuda::atomic_ref<uint32_t, efa_qp_scope<mode>> r(*ptr);
	return r.load(cuda::memory_order_acquire);
}

template <enum efa_qp_sharing_mode mode>
__device__ static inline void efa_cuda_atomic_store_u32(uint32_t *ptr, uint32_t val)
{
	cuda::atomic_ref<uint32_t, efa_qp_scope<mode>> r(*ptr);
	r.store(val, cuda::memory_order_release);
}
/* Commit slots [base, base+count) and ring the doorbell once the
 * producer cursor reaches our range. Serializes doorbell ordering
 * in slot order without a held lock. Chunks into max_batch-sized
 * doorbells to respect the hardware limit. */
template <enum efa_qp_sharing_mode mode>
__device__ static inline void efa_cuda_flush_sq_wrs(efa_cuda_qp *qp, uint32_t base, uint32_t count)
{
	uint32_t offset = 0;
	while (offset < count) {
		uint32_t chunk = min(count - offset, (uint32_t)qp->sq.wq.max_batch);
		uint32_t chunk_base = base + offset;
		uint32_t chunk_next = chunk_base + chunk;

		__threadfence_system();

		while (efa_cuda_atomic_load_u32<mode>(&qp->sq.wq.wqes_completed) != chunk_base) { /* spin */ }
		efa_cuda_atomic_store_u32<mode>(&qp->sq.wq.wqes_completed, chunk_next);

		qp->sq.wq.pc = chunk_next;
		__threadfence_system();
		*qp->sq.wq.db = qp->sq.wq.pc;
		__threadfence_system();

		offset += chunk;
	}
}

/* Reserve `count` contiguous SQ slots; returns the base absolute index.
 * Lock-free: each caller atomically advances wqes_posted. */
template <enum efa_qp_sharing_mode mode>
__device__ static inline uint32_t efa_cuda_start_sq_batch(efa_cuda_qp *qp, uint32_t batch_size)
{
	return efa_cuda_atomic_add_u32<mode>(&qp->sq.wq.wqes_posted, batch_size);
}

/* Write one WR into the ring at an absolute slot index. Phase is
 * computed from the absolute index, so concurrent writers to
 * different slots don't interfere. */
__device__ static inline int efa_cuda_sq_batch_place_wr(efa_cuda_qp *qp, uint32_t abs_idx, void *wr_buf)
{
	int wqe_phase = efa_cuda_get_wqe_phase(&qp->sq.wq, abs_idx);
	struct efa_io_tx_wqe *wqe = (struct efa_io_tx_wqe *)wr_buf;
	uint32_t sq_desc_offset;
	uint64_t *src;
	uint64_t *dst;

	EFA_SET(&wqe->meta.ctrl2, EFA_IO_TX_META_DESC_PHASE, wqe_phase);

	src = (uint64_t *)wqe;
	sq_desc_offset = (abs_idx & qp->sq.wq.queue_mask) * sizeof(struct efa_io_tx_wqe);
	dst = (uint64_t *)(qp->sq.wq.buf + sq_desc_offset);
	for (int i = 0 ; i < 8 ; i++)
		dst[i] = src[i];

	return 0;
}

__device__ static inline int efa_cuda_post_recv_wr(efa_cuda_qp *qp, uint64_t addr, uint32_t length, uint32_t lkey)
{
	struct efa_io_rx_desc wqe = {0};
	uint32_t rq_desc_offset;

	EFA_SET(&wqe.lkey_ctrl, EFA_IO_RX_DESC_FIRST, 1);
	EFA_SET(&wqe.lkey_ctrl, EFA_IO_RX_DESC_LAST, 1);

	EFA_SET(&wqe.lkey_ctrl, EFA_IO_RX_DESC_LKEY, lkey);
	wqe.buf_addr_lo = addr;
	wqe.buf_addr_hi = addr >> 32;
	wqe.length = length;

	/* Copy descriptor to RX ring */
	rq_desc_offset = (qp->rq.wq.pc & qp->rq.wq.queue_mask) * sizeof(wqe);
	memcpy(qp->rq.wq.buf + rq_desc_offset, &wqe, sizeof(wqe));

	qp->rq.wq.pc++;
	if (!(qp->rq.wq.pc & qp->rq.wq.queue_mask))
		qp->rq.wq.phase++;

	qp->rq.wq.wqes_pending++;
	if (qp->rq.wq.wqes_pending == qp->rq.wq.max_batch) {
		__threadfence_system();
		*qp->rq.wq.db = qp->rq.wq.pc;

		qp->rq.wq.wqes_pending = 0;
	}

	return 0;
}

__device__ static inline void efa_cuda_flush_rq_wrs(efa_cuda_qp *qp)
{
	if (!qp->rq.wq.wqes_pending)
		return;

	__threadfence_system();
	*qp->rq.wq.db = qp->rq.wq.pc;
	qp->rq.wq.wqes_pending = 0;
}

__device__ static inline bool efa_cuda_is_cq_compatible(efa_cuda_cq *cq)
{
	return cq->comp_mask == 0;
}

__device__ static inline bool efa_cuda_is_qp_compatible(efa_cuda_qp *qp)
{
	return qp->comp_mask == 0;
}

#endif
