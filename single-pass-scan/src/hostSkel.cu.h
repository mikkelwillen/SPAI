#ifndef SCAN_HOST
#define SCAN_HOST

#include "constants.cu.h"
#include "rTSKernels.cu.h"
#include "sPSKernels.cu.h"

int gpuAssert(cudaError_t code) {
  if(code != cudaSuccess) {
    printf("GPU Error: %s\n", cudaGetErrorString(code));
    return -1;
  }
  return 0;
}

uint32_t closestMul32(uint32_t x) {
    return ((x + 31) / 32) * 32;
}

/**
 * `N` is the input-array length
 * `B` is the CUDA block size
 * This function attempts to virtualize the computation so
 *   that it spawns at most 1024 CUDA blocks; otherwise an
 *   error is thrown. It should not throw an error for any
 *   B >= 64.
 * The return is the number of blocks, and `CHUNK * (*num_chunks)`
 *   is the number of elements to be processed sequentially by
 *   each thread so that the number of blocks is <= 1024. 
 */
template<int CHUNK>
uint32_t getNumBlocks(const uint32_t N, const uint32_t B, uint32_t* num_chunks) {
    const uint32_t max_inp_thds = (N + CHUNK - 1) / CHUNK;
    const uint32_t num_thds0    = min(max_inp_thds, MAX_HWDTH);

    const uint32_t min_elms_all_thds = num_thds0 * CHUNK;
    *num_chunks = max(1, N / min_elms_all_thds);

    const uint32_t seq_chunk = (*num_chunks) * CHUNK;
    const uint32_t num_thds = (N + seq_chunk - 1) / seq_chunk;
    const uint32_t num_blocks = (num_thds + B - 1) / B;

    if(num_blocks > MAX_BLOCK) {
        printf("Broken Assumption: number of blocks %d exceeds maximal block size: %d. Exiting!"
              , num_blocks, MAX_BLOCK);
        exit(1);
    }

    return num_blocks;
}

template<class OP>
void CPU_scan (typename OP::InpElTp *d_in, int N, typename OP::RedElTp *d_out) {
    typename OP::RedElTp acc = OP::identity();
    for (int i = 0; i < N; i++) {
        typename OP::InpElTp elm = d_in[i];
        typename OP::RedElTp red = OP::mapFun(elm);
        acc = OP::apply(acc, red);
        d_out[i] = acc;
    }
}

/**
 * Host Wrapper orchestraiting the execution of scan:
 * d_in  is the input array
 * d_out is the result array (result of scan)
 * t_tmp is a temporary array (used to scan in-place across the per-block results)
 * Implementation consist of three phases:
 *   1. elements are partitioned across CUDA blocks such that the number of
 *      spawned CUDA blocks is <= 1024. Each CUDA block reduces its elements
 *      and publishes the per-block partial result in `d_tmp`. This is 
 *      implemented in the `redAssocKernel` kernel.
 *   2. The per-block reduced results are scanned within one CUDA block;
 *      this is implemented in `scan1Block` kernel.
 *   3. Then with the same partitioning as in step 1, the whole scan is
 *      performed again at the level of each CUDA block, and then the
 *      prefix of the previous block---available in `d_tmp`---is also
 *      accumulated to each-element result. This is implemented in
 *      `scan3rdKernel` kernel and concludes the whole scan.
 */
template<class OP>                     // element-type and associative operator properties
void scanInc( const uint32_t     B     // desired CUDA block size ( <= 1024, multiple of 32)
            , const size_t       N     // length of the input array
            , typename OP::RedElTp* d_out // device array of length: N
            , typename OP::InpElTp* d_in  // device array of length: N
            , typename OP::RedElTp* d_tmp // device array of max length: MAX_BLOCK
) {
    const uint32_t inp_sz = sizeof(typename OP::InpElTp);
    const uint32_t red_sz = sizeof(typename OP::RedElTp);
    const uint32_t max_tp_size = (inp_sz > red_sz) ? inp_sz : red_sz;
    const uint32_t CHUNK = ELEMS_PER_THREAD*4 / max_tp_size;
    uint32_t num_seq_chunks;
    const uint32_t num_blocks = getNumBlocks<CHUNK>(N, B, &num_seq_chunks);
    const size_t   shmem_size = B * max_tp_size * CHUNK;

    //
    redAssocKernel<OP, CHUNK><<< num_blocks, B, shmem_size >>>
        (d_tmp, d_in, N, num_seq_chunks);

    {
        const uint32_t block_size = closestMul32(num_blocks);
        const size_t shmem_size = block_size * sizeof(typename OP::RedElTp);
        scan1Block<OP><<< 1, block_size, shmem_size>>>(d_tmp, num_blocks);
    }

    scan3rdKernel<OP, CHUNK><<< num_blocks, B, shmem_size >>>
        (d_out, d_in, d_tmp, N, num_seq_chunks);
}

template<class OP>                     // element-type and associative operator properties
void singlePassScan( const uint32_t     B     // desired CUDA block size ( <= 1024, multiple of 32)
                   , const size_t       N     // length of the input array
                   , typename OP::RedElTp* d_out // device array of length: N
                   , typename OP::RedElTp* d_in  // device array of length: N
                   // , uint32_t* d_fs
                   // , typename OP::RedElTp* d_as
                   // , typename OP::RedElTp* d_ps
                   , const int version
) {
    const uint32_t scan_sz = sizeof(typename OP::RedElTp);
    const uint32_t CHUNK = ELEMS_PER_THREAD * 4 / scan_sz;
    const uint32_t num_blocks = (N + CHUNK - 1) / (B * CHUNK) + 1;
    // const uint32_t num_blocks = (N + CHUNK - 1) / (B * CHUNK);
    // const uint32_t num_blocks = getNumBlocks<CHUNK>(N, B, &num_seq_chunks);
    const size_t   shmem_size = B * scan_sz * CHUNK;

    uint32_t* d_fs; // flag array
    typename OP::RedElTp* d_as; // aggregate array
    typename OP::RedElTp* d_ps; // prefix array

    gpuAssert( cudaMalloc((void**)&d_fs, num_blocks * sizeof(uint32_t)) );
    // initialize to 0s
    gpuAssert( cudaMemset(d_fs, 0, num_blocks * sizeof(uint32_t)) );
    gpuAssert( cudaMalloc((void**)&d_as, num_blocks * scan_sz) );
    gpuAssert( cudaMalloc((void**)&d_ps, num_blocks * scan_sz) );

    int* block_num;
    gpuAssert( cudaMalloc((void**)&block_num, sizeof(int)) );
    // gpuAssert( cudaMallocManaged((void**)&block_num, sizeof(int)) );
    //cudaMallocManaged(&block_num, 4);
    // *block_num = 0;
    int init = 0;
    gpuAssert( cudaMemcpy(block_num, &init, sizeof(int), cudaMemcpyHostToDevice) );

    // printf("# blocks: %d\n", num_blocks);

    sPSKernel<OP, CHUNK><<< num_blocks, B, shmem_size >>>
        (d_out, d_in, N, block_num, d_fs, d_as, d_ps, version);

    gpuAssert( cudaFree(block_num) );
    gpuAssert( cudaFree(d_fs) );
    gpuAssert( cudaFree(d_as) );
    gpuAssert( cudaFree(d_ps) );
}

#endif //SCAN_HOST
