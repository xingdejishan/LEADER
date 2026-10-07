#include <cuda_runtime.h>
#include <stdint.h>

__global__ void build_terms(const float* points, const float* tables, const float* gradient, const float* spacings, float* output, int n, int levels, int size, int features) {
    int64_t total=(int64_t)n*levels*8*features*3;
    for(int64_t flat=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;flat<total;flat+=(int64_t)blockDim.x*gridDim.x){
        int64_t index=flat;
        int axis=index%3;index/=3;
        int feature=index%features;index/=features;
        int corner=index%8;index/=8;
        int level=index%levels;index/=levels;
        int point=index;
        float spacing=spacings[level];
        float axes[3];int64_t grid[3];int bits[3]={(corner>>2)&1,(corner>>1)&1,corner&1};
        for(int k=0;k<3;k++){
            float scaled=__fdiv_rn(points[point*3+k],spacing);
            float integral=floorf(scaled);
            float fraction=__fsub_rn(scaled,integral);
            grid[k]=(int64_t)integral+bits[k];
            axes[k]=bits[k]?fraction:__fsub_rn(1.f,fraction);
        }
        int64_t hash=(grid[0]^(grid[1]*(int64_t)2654435761ULL)^(grid[2]*(int64_t)805459861ULL))&(size-1);
        float derivative=axis==0?__fmul_rn(axes[1],axes[2]):axis==1?__fmul_rn(axes[0],axes[2]):__fmul_rn(axes[0],axes[1]);
        derivative=__fdiv_rn(__fmul_rn(derivative,(float)(2*bits[axis]-1)),spacing);
        float value=tables[((int64_t)level*size+hash)*features+feature];
        float g=gradient[((int64_t)point*levels+level)*features+feature];
        output[flat]=__fmul_rn(__fmul_rn(g,value),derivative);
    }
}

extern "C" int mixed_gradient_terms(const float* points,const float* tables,const float* gradient,const float* spacings,float* output,int n,int levels,int size,int features,cudaStream_t stream){
    int64_t total=(int64_t)n*levels*8*features*3;
    int blocks=(int)((total+255)/256);
    if(blocks>4096)blocks=4096;
    if(blocks>0)build_terms<<<blocks,256,0,stream>>>(points,tables,gradient,spacings,output,n,levels,size,features);
    return (int)cudaGetLastError();
}
