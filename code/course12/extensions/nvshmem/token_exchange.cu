// Double-buffered Token exchange: one PE/GPU, FP32, two-projection GELU MLP
// Experts.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cutlass/gemm/thread/mma.h>
#include <cutlass/layout/matrix.h>
#include <filesystem>
#include <fstream>
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <vector>

#define CUDA_OK(call)                                                          \
  do {                                                                         \
    auto err = (call);                                                         \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA %s\n", cudaGetErrorString(err));                   \
      MPI_Abort(MPI_COMM_WORLD, 2);                                            \
    }                                                                          \
  } while (0)

struct TokenExchangeBuffers {
  // [2,P,C,H], capacity C records per source/destination block.
  float *send, *recv, *computed, *returned;
  // [2,P]: source-ready, returned-ready and receive-consumed epochs.
  uint64_t *ready, *done, *consumed;
  int *send_rows, *recv_rows; // [P], device int32, each value <= C.
  int pe, npes, capacity, hidden;
  uint64_t
      *ticks; // optional [6,4], local SM clock ticks, never compared across PEs
  float *up_weights, *down_weights; // private [4,4], PyTorch [out,in] layout
};

__device__ void wait_for_epoch(uint64_t *signal, uint64_t epoch, int pe,
                               int peer, int phase) {
  unsigned long long start = clock64();
  while (nvshmem_signal_fetch(signal) < epoch) {
    if (clock64() - start > 10000000000ULL) {
      printf("timeout pe=%d peer=%d epoch=%llu phase=%d\n", pe, peer,
             (unsigned long long)epoch, phase);
      asm volatile("trap;");
    }
  }
  // Establish NVSHMEM target-side visibility after polling the signal.
  nvshmem_signal_wait_until(signal, NVSHMEM_CMP_GE, epoch);
}

__device__ void expert_mlp_row(const float *x, float *y,
                               const float *up_weights,
                               const float *down_weights) {
  using Mma = cutlass::gemm::thread::Mma<
      cutlass::gemm::GemmShape<1, 4, 4>, float, cutlass::layout::RowMajor,
      float, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor>;
  Mma::FragmentA a;
  Mma::FragmentB w;
  Mma::FragmentC z, hidden, out;
  z.clear();
  for (int k = 0; k < 4; ++k)
    a[k] = x[k];
  // PyTorch stores weights [out,in]; CUTLASS B is [in,out].
  for (int k = 0; k < 4; ++k)
    for (int n = 0; n < 4; ++n)
      w[k * 4 + n] = up_weights[n * 4 + k];
  Mma{}(hidden, a, w, z);
  for (int k = 0; k < 4; ++k)
    a[k] = 0.5f * hidden[k] * (1.0f + erff(hidden[k] * 0.7071067811865475f));
  for (int k = 0; k < 4; ++k)
    for (int n = 0; n < 4; ++n)
      w[k * 4 + n] = down_weights[n * 4 + k];
  Mma{}(out, a, w, z);
  for (int n = 0; n < 4; ++n)
    y[n] = out[n];
}

// send/recv/computed/returned: FP32 symmetric [2,P,C,H]; signals: uint64 [2,P].
// send is destination-major; recv/computed are source-major; returned is
// Expert-major.
__global__ void exchange_compute_return(TokenExchangeBuffers b, int round) {
  int slot = round % 2, peer_block_elements = b.capacity * b.hidden;
  uint64_t epoch = (uint64_t)round + 1;
  int signal_base = slot * b.npes;
  if (threadIdx.x == 0 && b.ticks)
    b.ticks[round * 4] = clock64();
  // One resident CUDA Block: publish every destination before waiting.
  if (threadIdx.x == 0) {
    for (int peer = 0; peer < b.npes; ++peer) {
      if (round >= 2)
        wait_for_epoch(b.consumed + signal_base + peer, epoch - 2, b.pe, peer,
                       0);
      nvshmem_putmem_signal(b.recv + (signal_base + b.pe) * peer_block_elements,
                            b.send + (signal_base + peer) * peer_block_elements,
                            b.send_rows[peer] * b.hidden * sizeof(float),
                            b.ready + signal_base + b.pe, epoch,
                            NVSHMEM_SIGNAL_SET, peer);
    }
    for (int source = 0; source < b.npes; ++source)
      wait_for_epoch(b.ready + signal_base + source, epoch, b.pe, source, 1);
    if (b.ticks)
      b.ticks[round * 4 + 1] = clock64();
  }
  __syncthreads();
  // Apply Up Projection, GELU and Down Projection to each received row.
  // Expert ID equals PE; rows keep their source-PE order.
  for (int source = 0; source < b.npes; ++source) {
    for (int row = threadIdx.x; row < b.recv_rows[source]; row += blockDim.x) {
      int row_offset =
          (signal_base + source) * peer_block_elements + row * b.hidden;
      expert_mlp_row(b.recv + row_offset, b.computed + row_offset, b.up_weights,
                     b.down_weights);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (b.ticks)
      b.ticks[round * 4 + 2] = clock64();
    for (int source = 0; source < b.npes; ++source) {
      nvshmem_putmem_signal(
          b.returned + (signal_base + b.pe) * peer_block_elements,
          b.computed + (signal_base + source) * peer_block_elements,
          b.recv_rows[source] * b.hidden * sizeof(float),
          b.done + signal_base + b.pe, epoch, NVSHMEM_SIGNAL_SET, source);
      // Receiver has consumed its input; original sender may reuse this slot.
      nvshmemx_signal_op(b.consumed + signal_base + b.pe, epoch,
                         NVSHMEM_SIGNAL_SET, source);
    }
    for (int peer = 0; peer < b.npes; ++peer)
      wait_for_epoch(b.done + signal_base + peer, epoch, b.pe, peer, 2);
    if (b.ticks)
      b.ticks[round * 4 + 3] = clock64();
    nvshmem_quiet(); // Complete source accesses before this launch ends.
  }
}

__global__ void combine_returned(TokenExchangeBuffers b, const int *origin,
                                 float *output, int topk, int round) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 7 * 4)
    return;
  int token = i / 4, hidden = i % 4, slot = round % 2, first = 0;
  float sum = 0;
  for (int dest = 0; dest < b.npes; ++dest) {
    for (int row = 0; row < b.send_rows[dest]; ++row)
      if (origin[first + row] / topk == token)
        sum += b.returned[(slot * b.npes + dest) * b.capacity * 4 + row * 4 +
                          hidden] /
               topk;
    first += b.send_rows[dest];
  }
  output[i] = sum;
}

std::vector<float> host_reference(int pe, int npes, int topk, int scenario,
                                  float scale) {
  std::vector<float> result(28);
  for (int t = 0; t < 7; ++t)
    for (int k = 0; k < topk; ++k) {
      int e =
          scenario ? (k == 0 ? 0 : npes - 1) : ((pe * 7 + t) * topk + k) % npes;
      float x[4], hidden[4]{};
      for (int h = 0; h < 4; ++h)
        x[h] = scale * .01f * ((pe * 7 + t) * 4 + h + 1);
      for (int f = 0; f < 4; ++f) {
        for (int h = 0; h < 4; ++h)
          hidden[f] += x[h] * .01f * (e + 1) * (f * 4 + h + 1);
        hidden[f] =
            .5f * hidden[f] * (1.f + std::erf(hidden[f] * .7071067811865475f));
      }
      for (int h = 0; h < 4; ++h) {
        float value = 0;
        for (int f = 0; f < 4; ++f)
          value += hidden[f] * .01f * (e + 1) * (h * 4 + f + 1);
        result[t * 4 + h] += value / topk;
      }
    }
  return result;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int pe, npes;
  MPI_Comm_rank(MPI_COMM_WORLD, &pe);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);
  int devices;
  CUDA_OK(cudaGetDeviceCount(&devices));
  if (npes > devices || (npes != 2 && npes != 4))
    MPI_Abort(MPI_COMM_WORLD, 3);
  CUDA_OK(cudaSetDevice(pe));
  nvshmemx_init_attr_t attr{};
  MPI_Comm comm = MPI_COMM_WORLD;
  attr.mpi_comm = &comm;
  if (nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr) != 0)
    MPI_Abort(comm, 7);
  constexpr int N = 7, H = 4, C = 14, ROUNDS = 6;
  size_t values = 2 * npes * C * H, signals = 2 * npes;
  TokenExchangeBuffers b{};
  b.pe = pe;
  b.npes = npes;
  b.capacity = C;
  b.hidden = H;
  b.send = (float *)nvshmem_malloc(values * sizeof(float));
  b.recv = (float *)nvshmem_malloc(values * sizeof(float));
  b.computed = (float *)nvshmem_malloc(values * sizeof(float));
  b.returned = (float *)nvshmem_malloc(values * sizeof(float));
  b.ready = (uint64_t *)nvshmem_malloc(signals * sizeof(uint64_t));
  b.done = (uint64_t *)nvshmem_malloc(signals * sizeof(uint64_t));
  b.consumed = (uint64_t *)nvshmem_malloc(signals * sizeof(uint64_t));
  CUDA_OK(cudaMalloc(&b.send_rows, npes * sizeof(int)));
  CUDA_OK(cudaMalloc(&b.recv_rows, npes * sizeof(int)));
  CUDA_OK(cudaMalloc(&b.up_weights, 16 * sizeof(float)));
  CUDA_OK(cudaMalloc(&b.down_weights, 16 * sizeof(float)));
  std::vector<float> weights(16);
  for (int i = 0; i < 16; ++i)
    weights[i] = 0.01f * (pe + 1) * (i + 1);
  CUDA_OK(cudaMemcpy(b.up_weights, weights.data(), 16 * sizeof(float),
                     cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(b.down_weights, weights.data(), 16 * sizeof(float),
                     cudaMemcpyHostToDevice));
  std::filesystem::path prefix(argc > 1 ? argv[1] : "/tmp/nvshmem");
  if (!prefix.parent_path().empty())
    std::filesystem::create_directories(prefix.parent_path());
  std::ofstream results(std::string(argc > 1 ? argv[1] : "/tmp/nvshmem") +
                        "-rank" + std::to_string(pe) + ".json");
  results << "[";
  bool firstcase = true;
  if (std::getenv("MINIEP_DEVICE_TIMESTAMPS"))
    CUDA_OK(cudaMalloc(&b.ticks, ROUNDS * 4 * sizeof(uint64_t)));
  int *device_origin;
  float *device_output;
  CUDA_OK(cudaMalloc(&device_origin, N * 2 * sizeof(int)));
  CUDA_OK(cudaMalloc(&device_output, N * H * sizeof(float)));
  cudaEvent_t started, ended;
  CUDA_OK(cudaEventCreate(&started));
  CUDA_OK(cudaEventCreate(&ended));
  cudaStream_t stream;
  CUDA_OK(cudaStreamCreate(&stream));
  for (int topk = 1; topk <= 2; ++topk)
    for (int scenario = 0; scenario < 2; ++scenario) {
      std::vector<int> counts(npes), received(npes), origins;
      std::vector<float> packed(values), original(N * H), expected(N * H),
          got(N * H);
      for (int t = 0; t < N; ++t)
        for (int h = 0; h < H; ++h)
          original[t * H + h] = 0.01f * ((pe * N + t) * H + h + 1);
      // Destination-major packing; origin is the original q=t*K+slot.
      for (int dest = 0; dest < npes; ++dest)
        for (int t = 0; t < N; ++t)
          for (int k = 0; k < topk; ++k) {
            int expert = scenario ? (k == 0 ? 0 : npes - 1)
                                  : ((pe * N + t) * topk + k) % npes;
            if (expert == dest) {
              int row = counts[dest]++;
              for (int h = 0; h < H; ++h)
                for (int s = 0; s < 2; ++s)
                  packed[(s * npes + dest) * C * H + row * H + h] =
                      original[t * H + h];
              origins.push_back(t * topk + k);
            }
          }
      CUDA_OK(cudaMemcpy(device_origin, origins.data(),
                         origins.size() * sizeof(int), cudaMemcpyHostToDevice));
      MPI_Alltoall(counts.data(), 1, MPI_INT, received.data(), 1, MPI_INT,
                   MPI_COMM_WORLD);
      CUDA_OK(cudaMemcpy(b.send_rows, counts.data(), npes * sizeof(int),
                         cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemcpy(b.recv_rows, received.data(), npes * sizeof(int),
                         cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemcpy(b.send, packed.data(), values * sizeof(float),
                         cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemset(b.ready, 0, signals * sizeof(uint64_t)));
      CUDA_OK(cudaMemset(b.done, 0, signals * sizeof(uint64_t)));
      CUDA_OK(cudaMemset(b.consumed, 0, signals * sizeof(uint64_t)));
      nvshmem_barrier_all();
      std::vector<float> samples;
      for (int round = 0; round < ROUNDS; ++round) {
        float scale =
            std::getenv("MINIEP_VARY_ROUNDS") ? 1.f + .1f * round : 1.f;
        if (std::getenv("MINIEP_VARY_ROUNDS")) {
          std::vector<float> varied = packed;
          for (float &v : varied)
            v *= scale;
          CUDA_OK(cudaMemcpy(b.send, varied.data(), values * sizeof(float),
                             cudaMemcpyHostToDevice));
        }
        expected = host_reference(pe, npes, topk, scenario, scale);
        CUDA_OK(cudaEventRecord(started, stream));
        void *args[] = {&b, &round};
        int maxgrid = 0;
        if (nvshmemx_collective_launch_query_gridsize(
                (void *)exchange_compute_return, dim3(128), args, 0,
                &maxgrid) != 0 ||
            maxgrid < 1)
          MPI_Abort(comm, 4);
        if (nvshmemx_collective_launch((void *)exchange_compute_return, dim3(1),
                                       dim3(128), args, 0, stream) != 0)
          MPI_Abort(comm, 5);
        combine_returned<<<1, 32, 0, stream>>>(b, device_origin, device_output,
                                               topk, round);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaEventRecord(ended, stream));
        CUDA_OK(cudaStreamSynchronize(stream));
        float elapsed;
        CUDA_OK(cudaEventElapsedTime(&elapsed, started, ended));
        samples.push_back(elapsed);
        CUDA_OK(cudaMemcpy(got.data(), device_output, N * H * sizeof(float),
                           cudaMemcpyDeviceToHost));
        for (int i = 0; i < N * H; ++i)
          if (std::abs(got[i] - expected[i]) > 1e-4) {
            fprintf(stderr, "mismatch PE=%d K=%d round=%d i=%d\n", pe, topk,
                    round, i);
            MPI_Abort(comm, 6);
          }
      }
      std::vector<float> max_samples(ROUNDS);
      MPI_Allreduce(samples.data(), max_samples.data(), ROUNDS, MPI_FLOAT,
                    MPI_MAX, comm);
      samples = max_samples;
      if (!firstcase)
        results << ",";
      firstcase = false;
      results << "{\"input_scale\":"
              << (std::getenv("MINIEP_VARY_ROUNDS") ? 1.f + .1f * (ROUNDS - 1)
                                                    : 1.f)
              << ",\"expert\":\"mlp\",\"top_k\":" << topk
              << ",\"scenario\":" << scenario << ",\"values\":[";
      for (int i = 0; i < N * H; ++i) {
        if (i)
          results << ",";
        results << got[i];
      }
      results << "],\"prepared_samples_ms\":[";
      for (int i = 0; i < ROUNDS; ++i) {
        if (i)
          results << ",";
        results << samples[i];
      }
      results << "]";
      auto write_metadata = [&](const char *name, const std::vector<int> &v) {
        results << ",\"" << name << "\":[";
        for (size_t i = 0; i < v.size(); ++i) {
          if (i)
            results << ",";
          results << v[i];
        }
        results << "]";
      };
      write_metadata("send_rows", counts);
      write_metadata("recv_rows", received);
      write_metadata("origin", origins);
      if (b.ticks) {
        std::vector<uint64_t> ticks(ROUNDS * 4);
        CUDA_OK(cudaMemcpy(ticks.data(), b.ticks, ROUNDS * 4 * sizeof(uint64_t),
                           cudaMemcpyDeviceToHost));
        results << ",\"phase_ticks\":[";
        for (int i = 0; i < ROUNDS * 4; ++i) {
          if (i)
            results << ",";
          results << ticks[i];
        }
        results << "]";
      }
      results << "}";
      printf("NVSHMEM PE=%d P=%d K=%d skew=%d rounds=%d expert=mlp PASS\n", pe,
             npes, topk, scenario, ROUNDS);
    }
  results << "]";
  results.close();
  nvshmem_barrier_all();
  if (b.ticks)
    CUDA_OK(cudaFree(b.ticks));
  CUDA_OK(cudaEventDestroy(started));
  CUDA_OK(cudaEventDestroy(ended));
  CUDA_OK(cudaFree(device_origin));
  CUDA_OK(cudaFree(device_output));
  CUDA_OK(cudaFree(b.up_weights));
  CUDA_OK(cudaFree(b.down_weights));
  CUDA_OK(cudaStreamDestroy(stream));
  CUDA_OK(cudaFree(b.send_rows));
  CUDA_OK(cudaFree(b.recv_rows));
  nvshmem_free(b.consumed);
  nvshmem_free(b.done);
  nvshmem_free(b.ready);
  nvshmem_free(b.returned);
  nvshmem_free(b.computed);
  nvshmem_free(b.recv);
  nvshmem_free(b.send);
  nvshmem_finalize();
  MPI_Finalize();
  return 0;
}
