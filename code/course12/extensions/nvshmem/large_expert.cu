// FP16 Tensor Core MLP: matched NVSHMEM fused and split execution.
// Build with nvcc -rdc=true -arch=sm_90 -ccbin mpicxx and NVSHMEM headers/libs.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/epilogue/thread/linear_combination_generic.h>
#include <cooperative_groups.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <climits>
#include <type_traits>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#define CK(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,2);} } while(0)
#define NK(x) do {if((x)!=0) MPI_Abort(MPI_COMM_WORLD,3);}while(0)
struct Buffers {
  // Symmetric FP16 [P,Q,C,H], Q=E/P experts per rank.
  // send [destination,local_expert,row,H]; recv/computed [source,local_expert,row,H].
  // returned [expert_owner,local_expert,row,H] restores global expert ordering.
  half *send,*recv,*computed,*returned;
  uint64_t *ready,*done; // symmetric uint64[P,Q], monotonically increasing epoch
  // Private contiguous buffers, int32 counts measured in token rows.
  half *activation,*up,*down; // [P,Q,C,F], [Q,F,H], [Q,H,F], weights [out,in]
  int *send_rows,*recv_rows,*restore; // [P,Q], [P,Q], [N,K,2] => (expert,row)
  float *output; // [N,H], routing-weighted FP32 sum in original token order
  int pe,p,q,c,n,h,f,k;
};
__device__ void await_epoch(uint64_t *signal,uint64_t epoch) {
  auto start=clock64();
  while(nvshmem_signal_fetch(signal)<epoch)
    if(clock64()-start>20000000000ULL){printf("epoch timeout\n");asm("trap;");}
  nvshmem_signal_wait_until(signal,NVSHMEM_CMP_GE,epoch);
}
__device__ void forward(Buffers b,int epoch) {
  for(int peer=0;peer<b.p;++peer)for(int e=0;e<b.q;++e){
    int target=b.pe*b.q+e,local=peer*b.q+e;
    nvshmemx_putmem_signal_block(b.recv+target*b.c*b.h,b.send+local*b.c*b.h,
        size_t(b.send_rows[local])*b.h*sizeof(half),b.ready+target,epoch,NVSHMEM_SIGNAL_SET,peer);
  }
  if(threadIdx.x==0)for(int source=0;source<b.p*b.q;++source)await_epoch(b.ready+source,epoch);
  __syncthreads();
}
__device__ void backward(Buffers b,int epoch) {
  for(int source=0;source<b.p;++source)for(int e=0;e<b.q;++e){
    int target=b.pe*b.q+e,local=source*b.q+e;
    nvshmemx_putmem_signal_block(b.returned+target*b.c*b.h,b.computed+local*b.c*b.h,
        size_t(b.recv_rows[local])*b.h*sizeof(half),b.done+target,epoch,NVSHMEM_SIGNAL_SET,source);
  }
  if(threadIdx.x==0){
    for(int expert=0;expert<b.p*b.q;++expert)await_epoch(b.done+expert,epoch);
    nvshmem_quiet();
  }
  __syncthreads();
}
// Explicit tile coordinates let a resident CTA reuse CUTLASS's complete GEMM
// kernel mainloop and epilogue for successive tiles; no custom MMA is used.
struct ExplicitTileSwizzle {
  CUTLASS_HOST_DEVICE int get_log_tile(cutlass::gemm::GemmCoord) const { return 0; }
  CUTLASS_DEVICE cutlass::gemm::GemmCoord get_tile_offset(int encoded) const {
    return {encoded >> 16, encoded & 65535, 0};
  }
};
template<class T> struct RoundedGelu;
template<int Count>struct RoundedGelu<cutlass::Array<float,Count>> {
  CUTLASS_HOST_DEVICE cutlass::Array<float,Count> operator()(cutlass::Array<float,Count> const& x) const {
    cutlass::Array<float,Count> result;
    for(int i=0;i<Count;++i){
      float v=float(cutlass::half_t(x[i])); // torch FP16 Up output boundary
      result[i]=.5f*v*(1.f+erff(v*.7071067811865475f));
    }
    return result;
  }
};
using UpOutput=cutlass::epilogue::thread::LinearCombinationGeneric<RoundedGelu,cutlass::half_t,8,float,float>;
using DownOutput=cutlass::epilogue::thread::LinearCombination<cutlass::half_t,8,float,float>;
template<class Output>using Gemm=typename cutlass::gemm::kernel::DefaultGemm<
    cutlass::half_t,cutlass::layout::RowMajor,8,
    cutlass::half_t,cutlass::layout::ColumnMajor,8,
    cutlass::half_t,cutlass::layout::RowMajor,float,
    cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<32,32,32>,
    cutlass::gemm::GemmShape<16,8,16>,Output,ExplicitTileSwizzle,2,false,
    cutlass::arch::OpMultiplyAdd>::GemmKernel;
using UpGemm=Gemm<UpOutput>;using DownGemm=Gemm<DownOutput>;
union GemmStorage {UpGemm::SharedStorage up;DownGemm::SharedStorage down;};
// Both modes use these exact CUTLASS kernels, tile assignments and CTA counts.
// Half-precision contiguous operands need H and F divisible by eight.
template<bool IsUp>__device__ void project(Buffers b,int worker,int workers,GemmStorage &storage) {
  using Kernel=typename std::conditional<IsUp,UpGemm,DownGemm>::type;
  int in=IsUp?b.h:b.f,out=IsUp?b.f:b.h;
  int mt=(b.c+63)/64,nt=(out+63)/64;
  for(int task=worker;task<b.p*b.q*mt*nt;task+=workers){
    int source=task/(mt*nt),m=task/nt%mt,n=task%nt;
    int expert=source%b.q;
    auto weights=reinterpret_cast<cutlass::half_t*>((IsUp?b.up:b.down)+size_t(expert)*b.h*b.f);
    if(m*64>=b.recv_rows[source])continue;
    auto x=reinterpret_cast<cutlass::half_t*>(IsUp?b.recv+source*b.c*b.h:b.activation+source*b.c*b.f);
    auto y=reinterpret_cast<cutlass::half_t*>(IsUp?b.activation+source*b.c*b.f:b.computed+source*b.c*b.h);
    typename Kernel::Params params({b.recv_rows[source],out,in},{mt,nt,1},
        {x,in},{weights,in},{y,out},{y,out},typename Kernel::OutputOp::Params(1.f,0.f));
    params.swizzle_log_tile=(m<<16)|n;
    if constexpr(IsUp)Kernel{}(params,storage.up);else Kernel{}(params,storage.down);
    __syncthreads(); // Shared storage is reused by this CTA's next tile.
  }
}
__global__ void fused_mlp(Buffers b,int epoch) {
  __shared__ GemmStorage storage;
  auto grid=cooperative_groups::this_grid();
  if(blockIdx.x==0)forward(b,epoch);
  grid.sync(); // NVSHMEM target writes become visible before Up projection.
  if(blockIdx.x)project<true>(b,blockIdx.x-1,gridDim.x-1,storage);
  grid.sync(); // All hidden tiles must be ready for Down projection.
  if(blockIdx.x)project<false>(b,blockIdx.x-1,gridDim.x-1,storage);
  grid.sync();
  if(blockIdx.x==0)backward(b,epoch);
}
__global__ void exchange(Buffers b,int epoch,bool is_return) {
  if(is_return)backward(b,epoch);else forward(b,epoch);
}
__global__ void projection(Buffers b,bool up) {
  __shared__ GemmStorage storage;
  if(up)project<true>(b,blockIdx.x,gridDim.x,storage);else project<false>(b,blockIdx.x,gridDim.x,storage);
}
__global__ void combine(Buffers b) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=b.n*b.h)return;
  int t=i/b.h,h=i%b.h;float value=0;
  for(int k=0;k<b.k;++k){
    int expert=b.restore[(t*b.k+k)*2],row=b.restore[(t*b.k+k)*2+1];
    value+=__half2float(b.returned[(expert*b.c+row)*b.h+h])/b.k;
  }
  b.output[i]=value;
}
template<class T>void dump(std::string name,const std::vector<T>&v){
  std::ofstream file(name,std::ios::binary);file.write((const char*)v.data(),v.size()*sizeof(T));
}
template<class T>void alloc(T **p,size_t n){CK(cudaMalloc(p,n*sizeof(T)));}
template<class T>void upload(T *p,const std::vector<T>&v){CK(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
int main(int argc,char **argv){
  MPI_Init(&argc,&argv);int pe,p;MPI_Comm_rank(MPI_COMM_WORLD,&pe);MPI_Comm_size(MPI_COMM_WORLD,&p);
  if(argc!=10 && argc!=11){if(!pe)fprintf(stderr,"usage: program prefix N H F K skew warmup repeats workers [num_experts]\n");MPI_Abort(MPI_COMM_WORLD,4);}
  std::string prefix=argv[1];int n=atoi(argv[2]),h=atoi(argv[3]),f=atoi(argv[4]),k=atoi(argv[5]),skew=atoi(argv[6]),warm=atoi(argv[7]),reps=atoi(argv[8]),workers=atoi(argv[9]);
  int experts=argc==11?atoi(argv[10]):p;
  int devices;CK(cudaGetDeviceCount(&devices));
  if(n<=0||h<=0||f<=0||h%8||f%8||k<1||experts<=0||experts%p||k>experts||k>2||p>devices||warm<1||reps<1||workers<1||
      int64_t(experts)*n*k*std::max(h,f)>INT_MAX){
    fprintf(stderr,"invalid N=%d H=%d F=%d K=%d P=%d devices=%d warmup=%d repeats=%d workers=%d; require positive dimensions, H/F multiples of 8, K=1/2, P<=devices, int32 layout offsets\n",n,h,f,k,p,devices,warm,reps,workers);
    fprintf(stderr,"num_experts=%d must be positive and divisible by P=%d; K<=num_experts\n",experts,p);
    MPI_Abort(MPI_COMM_WORLD,5);
  }
  CK(cudaSetDevice(pe));MPI_Comm comm=MPI_COMM_WORLD;nvshmemx_init_attr_t attr{};attr.mpi_comm=&comm;NK(nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM,&attr));
  Buffers b{};b.pe=pe;b.p=p;b.q=experts/p;b.n=n;b.h=h;b.f=f;b.k=k;b.c=n*k;
  size_t count=size_t(experts)*b.c*h;
  b.send=(half*)nvshmem_malloc(count*2);b.recv=(half*)nvshmem_malloc(count*2);
  b.computed=(half*)nvshmem_malloc(count*2);b.returned=(half*)nvshmem_malloc(count*2);
  b.ready=(uint64_t*)nvshmem_calloc(experts,sizeof(uint64_t));b.done=(uint64_t*)nvshmem_calloc(experts,sizeof(uint64_t));
  if(!b.send||!b.recv||!b.computed||!b.returned||!b.ready||!b.done)MPI_Abort(comm,6);
  alloc(&b.activation,size_t(experts)*b.c*f);alloc(&b.up,size_t(b.q)*f*h);alloc(&b.down,size_t(b.q)*h*f);
  alloc(&b.send_rows,experts);alloc(&b.recv_rows,experts);alloc(&b.restore,n*k*2);alloc(&b.output,n*h);
  std::mt19937 rng(20261007+pe);std::uniform_real_distribution<float> uniform(-1,1);
  std::vector<half>x(n*h),up(size_t(b.q)*f*h),down(size_t(b.q)*h*f),packed(count);
  for(auto &v:x)v=__float2half(uniform(rng));
  for(auto &v:up)v=__float2half(uniform(rng)/sqrtf(h));
  for(auto &v:down)v=__float2half(uniform(rng)/sqrtf(f));
  std::vector<int>send(experts),recv(experts),routes(n*k),restore(n*k*2);
  for(int t=0;t<n;++t)for(int j=0;j<k;++j){
    int dest=skew?(j==0?0:experts-1):((pe*n+t)*k+j)%experts;
    routes[t*k+j]=dest;int row=send[dest]++;
    restore[(t*k+j)*2]=dest;restore[(t*k+j)*2+1]=row;
    std::copy_n(x.begin()+t*h,h,packed.begin()+(dest*b.c+row)*h);
  }
  MPI_Alltoall(send.data(),b.q,MPI_INT,recv.data(),b.q,MPI_INT,comm);
  upload(b.send,packed);upload(b.up,up);upload(b.down,down);upload(b.send_rows,send);upload(b.recv_rows,recv);upload(b.restore,restore);
  if(!std::filesystem::path(prefix).parent_path().empty())std::filesystem::create_directories(std::filesystem::path(prefix).parent_path());
  std::string rankprefix=prefix+"-rank"+std::to_string(pe);
  dump(rankprefix+"-x.bin",x);dump(rankprefix+"-up.bin",up);dump(rankprefix+"-down.bin",down);dump(rankprefix+"-routes.bin",routes);
  cudaStream_t stream;CK(cudaStreamCreate(&stream));cudaEvent_t begin,end;CK(cudaEventCreate(&begin));CK(cudaEventCreate(&end));
  int epoch=0,maxgrid=0;void *args[]={&b,&epoch};
  NK(nvshmemx_collective_launch_query_gridsize((void*)fused_mlp,dim3(128),args,0,&maxgrid));
  int available=maxgrid-1;MPI_Allreduce(MPI_IN_PLACE,&available,1,MPI_INT,MPI_MIN,comm);workers=std::min(workers,available);if(workers<1)MPI_Abort(comm,7);
  // Alternate mode order across four blocks. Every sample is reduced by MAX rank.
  std::vector<float>samples[2];std::vector<float>outputs[2];
  for(int block=0;block<4;++block){int mode=(block==0||block==3)?0:1;
    for(int it=0;it<warm+reps;++it){++epoch;
      // All prior readers and writers finish before the next epoch reuses buffers.
      // MPI barrier and host-side correctness copies are outside the timed range.
      MPI_Barrier(comm);CK(cudaEventRecord(begin,stream));
      if(mode==0)NK(nvshmemx_collective_launch((void*)fused_mlp,dim3(workers+1),dim3(128),args,0,stream));
      else{
        bool ret=false;void *exargs[]={&b,&epoch,&ret};
        NK(nvshmemx_collective_launch((void*)exchange,dim3(1),dim3(128),exargs,0,stream));
        projection<<<workers,128,0,stream>>>(b,true);projection<<<workers,128,0,stream>>>(b,false);
        ret=true;NK(nvshmemx_collective_launch((void*)exchange,dim3(1),dim3(128),exargs,0,stream));
      }
      combine<<<(n*h+255)/256,256,0,stream>>>(b);CK(cudaGetLastError());CK(cudaEventRecord(end,stream));CK(cudaEventSynchronize(end));
      float ms;CK(cudaEventElapsedTime(&ms,begin,end));MPI_Allreduce(MPI_IN_PLACE,&ms,1,MPI_FLOAT,MPI_MAX,comm);
      if(it>=warm)samples[mode].push_back(ms);
    }
    outputs[mode].resize(n*h);CK(cudaMemcpy(outputs[mode].data(),b.output,n*h*sizeof(float),cudaMemcpyDeviceToHost));
    dump(rankprefix+(mode==0?"-fused.bin":"-split.bin"),outputs[mode]);
  }
  float error=0;for(int i=0;i<n*h;++i)error=std::max(error,std::abs(outputs[0][i]-outputs[1][i]));
  if(error!=0){fprintf(stderr,"matched output mismatch %g\n",error);MPI_Abort(comm,8);}
  cudaDeviceProp prop;CK(cudaGetDeviceProperties(&prop,pe));cudaFuncAttributes fun;CK(cudaFuncGetAttributes(&fun,fused_mlp));
  std::ofstream json(rankprefix+".json");
  json<<"{\"n\":"<<n<<",\"h\":"<<h<<",\"f\":"<<f<<",\"k\":"<<k<<",\"p\":"<<p<<",\"skew\":"<<skew<<",\"workers\":"<<workers<<",\"warmup_per_block\":"<<warm<<",\"repeats_per_block\":"<<reps<<",\"registers\":"<<fun.numRegs<<",\"static_shared_bytes\":"<<fun.sharedSizeBytes<<",\"sm_count\":"<<prop.multiProcessorCount<<",\"matched_max_abs_error\":"<<error;
  auto write=[&](const char*key,auto v){json<<",\""<<key<<"\":[";for(size_t i=0;i<v.size();++i){if(i)json<<",";json<<v[i];}json<<"]";};
  std::vector<int>send_rank(p),recv_rank(p);
  for(int rank=0;rank<p;++rank)for(int e=0;e<b.q;++e){send_rank[rank]+=send[rank*b.q+e];recv_rank[rank]+=recv[rank*b.q+e];}
  json<<",\"e\":"<<experts<<",\"experts_per_rank\":"<<b.q;
  write("send_expert_rows",send);write("recv_expert_rows",recv);
  write("send_rows",send_rank);write("recv_rows",recv_rank);write("fused_ms",samples[0]);write("split_ms",samples[1]);json<<"}";json.close();
  printf("PASS PE=%d N=%d H=%d F=%d P=%d K=%d skew=%d workers=%d matched_error=%g\n",pe,n,h,f,p,k,skew,workers,error);
  nvshmem_barrier_all();
  cudaEventDestroy(begin);cudaEventDestroy(end);cudaStreamDestroy(stream);
  cudaFree(b.activation);cudaFree(b.up);cudaFree(b.down);cudaFree(b.send_rows);cudaFree(b.recv_rows);cudaFree(b.restore);cudaFree(b.output);
  nvshmem_free(b.done);nvshmem_free(b.ready);nvshmem_free(b.returned);nvshmem_free(b.computed);nvshmem_free(b.recv);nvshmem_free(b.send);
  nvshmem_finalize();MPI_Finalize();return 0;
}
