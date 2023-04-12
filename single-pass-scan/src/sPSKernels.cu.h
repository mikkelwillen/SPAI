#ifndef SPS_KERS
#define SPS_KERS

#include <cuda_runtime.h>
#include "helperKernels.cu.h"


// Kernel, der executer hele lortet
template<class OP, int CHUNK>
__global__ void sPSKernel( typename OP::RedElTp* d_out 
                         , typename OP::RedElTp* d_in
                         , uint32_t N
                         , int* block_num
                         , volatile uint32_t* d_fs
                         , volatile typename OP::RedElTp* d_as
                         , volatile typename OP::RedElTp* d_ps
                         , int version) {
    // Dynamically determine block id
    volatile __shared__ int block_id;
    if (threadIdx.x == 0) {
        block_id = atomicAdd(block_num, 1);
    } __syncthreads();

    extern __shared__ char sh_mem[];
    // shared memory for the input elements (types)
    volatile typename OP::RedElTp* shmem = (typename OP::RedElTp*)sh_mem;

    // arrays for flags, aggregates and prefixes
    volatile uint32_t* sh_fs = (uint32_t*) sh_mem;
    volatile typename OP::RedElTp* sh_vs =
        (typename OP::RedElTp*) (sh_mem + WARP * sizeof(uint32_t));

    // number of elements to be processed by each block
    uint32_t num_elems_per_block = CHUNK * blockDim.x;

    // the current block start processing input elements from this offset:
    uint32_t block_offs = num_elems_per_block * block_id;
    
    // per block exlusive prefix
    typename OP::RedElTp block_exl_prefix;

    // register memory for storing the scanned elements
    typename OP::RedElTp chunk[CHUNK];


    // 1. copy `CHUNK` input elements per thread from global to shared
    // memory in coalesced fashion (for global memory)
    copyFromGlb2ShrMem<typename OP::RedElTp, CHUNK>
        (block_offs, N, OP::identity(), d_in, shmem);


    // 2. each thread scans sequentially its CHUNK elements.
    typename OP::RedElTp acc = OP::identity();
    uint32_t shmem_offset = threadIdx.x * CHUNK;
    #pragma unroll
    for (uint32_t i = 0; i < CHUNK; i++) {
        typename OP::RedElTp elm = shmem[shmem_offset + i];
        acc = OP::apply(acc, elm);
        chunk[i] = acc;
    }
    __syncthreads();


    // 3.
    // 3.1 last elem of each thread is placed in shmem_buf.
    shmem[threadIdx.x] = acc;
    __syncthreads();

    // 3.2 block-wise scan over the shmem_buff
    acc = scanIncBlock<OP>(shmem, threadIdx.x);
    __syncthreads();

    shmem[threadIdx.x] = acc;
    __syncthreads();

    // chunk_exl_prefix is set to be the previous chunks inclusive prefix
    typename OP::RedElTp chunk_exl_prefix =
        (threadIdx.x == 0) ? OP::remVolatile(shmem[blockDim.x - 1])
                           : OP::remVolatile(shmem[threadIdx.x - 1]);
    __syncthreads();

    block_exl_prefix = OP::identity();
    // 4 and 5.
    if (version == 0) { // chained scan
        if (threadIdx.x == 0) {
            if (block_id == 0) {
                d_ps[block_id] = chunk_exl_prefix;
                __threadfence();
                d_fs[block_id] = 2;
                chunk_exl_prefix = OP::identity();
                sh_vs[0] = OP::identity();
            } else {
                uint32_t prev = block_id - 1;
                while(d_fs[prev] != 2) {
                    // wait
                }
                block_exl_prefix = OP::apply(block_exl_prefix, d_ps[prev]);
            }
            sh_vs[0] = block_exl_prefix;
        } __syncthreads();
    }

    if (version == 1 || version == 2) {
        if (threadIdx.x == 0) {
            if (block_id == 0) {
                d_ps[block_id] = chunk_exl_prefix;
                __threadfence();
                d_fs[block_id] = 2;
                chunk_exl_prefix = OP::identity();
                sh_vs[0] = OP::identity();
            } else {
                d_as[block_id] = chunk_exl_prefix;
                __threadfence();
                d_fs[block_id] = 1;
                sh_fs[0] = d_fs[block_id - 1];
            }
        } __syncthreads();
    }

    if (version == 1) { // naive decoupled lookback
        if (threadIdx.x == 0) {
            if (block_id != 0){
                uint32_t prev = block_id - 1;
                while(d_fs[prev] != 2) {
                    if (d_fs[prev] == 1) {
                        block_exl_prefix = OP::apply(block_exl_prefix, d_as[prev]);
                        prev -= 1;
                    }
                }
                block_exl_prefix = OP::apply(block_exl_prefix, d_ps[prev]);
            }
            sh_vs[0] = block_exl_prefix;
        } __syncthreads();
    }


    if (version == 2) { // WARP thread copy - single thread lookback:
        int32_t window_offset = block_id - WARP;
        int32_t loop_stop = - WARP;
        if (block_id != 0 && threadIdx.x < WARP) {
            while(window_offset > loop_stop) {
                int lookup_block_id = window_offset + threadIdx.x;
                if (lookup_block_id >= 0) {
                    while (d_fs[lookup_block_id] == 0) {}  // wait for some result
                    uint32_t flag = d_fs[lookup_block_id];
                    sh_fs[threadIdx.x] = flag;
                    if (flag == 1) {
                        sh_vs[threadIdx.x] = d_as[lookup_block_id];
                    } else if (flag == 2) {
                        sh_vs[threadIdx.x] = d_ps[lookup_block_id];
                    }
                }

                if (threadIdx.x == 0) {
                    // do lookback
                    int i = 0;
                    while (i < WARP) {
                        int index = (WARP - 1) - i;
                        uint32_t flag = sh_fs[index];
                        if (flag == 1) {
                            block_exl_prefix = OP::apply(block_exl_prefix,
                                                            sh_vs[index]);
                            i++;
                        } else if (flag == 2) {
                            block_exl_prefix = OP::apply(block_exl_prefix,
                                                            sh_vs[index]);
                            window_offset = loop_stop;
                            i = WARP;
                        }
                    }
                }
                window_offset -= WARP;
            }

            if (threadIdx.x == 0) {
                sh_vs[0] = block_exl_prefix;
            }
        }
        __syncthreads();
    }

    if (version == 3) { // WARP thread copy - WARP thread lookback:

        if (block_id == 0 && threadIdx.x == 0) {
            d_ps[block_id] = chunk_exl_prefix;
            __threadfence();
            d_fs[block_id] = 2;
            chunk_exl_prefix = OP::identity();
            sh_vs[0] = OP::identity();
        } __syncthreads();

        int32_t window_offset = block_id - WARP;
        int32_t loop_stop = - WARP;
        if (block_id != 0 && threadIdx.x < WARP) {
            if (threadIdx.x == 0) {
                d_as[block_id] = chunk_exl_prefix;
                __threadfence();
                d_fs[block_id] = 1;
                sh_fs[0] = d_fs[block_id - 1];
            }

            if(sh_fs[0] == 2) {
                if (threadIdx.x == 0) {
                    block_exl_prefix = d_ps[block_id - 1];
                }
            } else {
                while(window_offset > loop_stop) {
                    int lookup_block_id = window_offset + threadIdx.x;
                    uint32_t flag = 0;
                    typename OP::RedElTp value = OP::identity();
                    if (lookup_block_id >= 0) {
                        while (d_fs[lookup_block_id] == 0) {}  // wait for some result
                        flag = d_fs[lookup_block_id];
                        if (flag == 1) {
                            value = d_as[lookup_block_id];
                        } else if (flag == 2) {
                            value = d_ps[lookup_block_id];
                        }
                    }
                    sh_vs[threadIdx.x] = value;
                    sh_fs[threadIdx.x] = flag;

                    // WARP-Level inclusive scan
                    if (sh_fs[WARP - 1] != 2) {
                        const uint32_t lane = threadIdx.x & (WARP - 1);
                        #pragma unroll
                        for(int d = 0; d < lgWARP; d++) {
                            uint32_t h = 1 << d;
                            if (lane >= h) {
                                uint32_t flag1 = sh_fs[threadIdx.x];
                                uint32_t flag2 = sh_fs[threadIdx.x - h];
                                typename OP::RedElTp value1 = sh_vs[threadIdx.x];
                                typename OP::RedElTp value2 = sh_vs[threadIdx.x - h];
                                if (flag1 == 1) {
                                    sh_fs[threadIdx.x] = flag2;
                                    sh_vs[threadIdx.x] = OP::apply(value1, value2);
                                }
                            }
                        }
                    }

                    flag = sh_fs[WARP - 1];
                    if (flag == 2) {
                        window_offset = loop_stop;
                    } else if (flag == 1) {
                        window_offset -= WARP;
                    }

                    if (flag != 0) {
                        block_exl_prefix = OP::apply(block_exl_prefix, sh_vs[WARP - 1]);
                    }
                    __threadfence();
                }
            }
            if (threadIdx.x == 0) {
                d_ps[block_id] = OP::apply(block_exl_prefix, chunk_exl_prefix);
                __threadfence();
                d_fs[block_id] = 2;
                sh_vs[0] = block_exl_prefix;
                chunk_exl_prefix = OP::identity();
            }
        }
        __syncthreads();
    }

    // publish prefix results
    if (version == 0 || version == 1 || version == 2) {
        // update prefix
        if (threadIdx.x == 0 && block_id != 0) {
            d_ps[block_id] = OP::apply(block_exl_prefix, chunk_exl_prefix);
            __threadfence();
            d_fs[block_id] = 2;
            chunk_exl_prefix = OP::identity();
        }
    }

    block_exl_prefix = sh_vs[0];
    // if (threadIdx.x == 0) {chunk_exl_prefix = OP::identity(); } // tester
    chunk_exl_prefix = OP::apply(block_exl_prefix, chunk_exl_prefix);

    // 3.3 update the per thread-scan with the corrosponding prefix
    // (after 4 and 5, when they are implemented)
    #pragma unroll
    for (uint32_t i = 0; i < CHUNK; i++) {
        shmem[shmem_offset + i] = OP::apply(chunk_exl_prefix, chunk[i]);
    } __syncthreads();


    // ?. write back from shared to global memory in coalesced fashion.
    copyFromShr2GlbMem<typename OP::RedElTp, CHUNK>
        (block_offs, N, d_out, shmem);

}

#endif
