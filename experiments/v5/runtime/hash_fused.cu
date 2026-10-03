#include <cuda_runtime.h>
#include <stdint.h>

__device__ unsigned hash_index(int x,int y,int z,unsigned mask) {
    return (unsigned(x) ^ (unsigned(y)*2654435761u) ^ (unsigned(z)*805459861u)) & mask;
}

__global__ void encode_kernel(const float* points,const float* table,const float* spacing,float* out,int count,int levels,int size) {
    int n=blockIdx.x*blockDim.x+threadIdx.x;
    int l=blockIdx.y;
    if(n>=count)return;
    float s=spacing[l];
    float x=points[3*n]/s,y=points[3*n+1]/s,z=points[3*n+2]/s;
    int bx=int(floorf(x)),by=int(floorf(y)),bz=int(floorf(z));
    float fx=x-float(bx),fy=y-float(by),fz=z-float(bz);
    float a=0.f,b=0.f;
    #pragma unroll
    for(int k=0;k<8;k++) {
        int cx=(k>>2)&1,cy=(k>>1)&1,cz=k&1;
        float w=(cx?fx:1.f-fx)*(cy?fy:1.f-fy)*(cz?fz:1.f-fz);
        unsigned id=hash_index(bx+cx,by+cy,bz+cz,unsigned(size-1));
        int offset=(l*size+id)*2;
        a+=table[offset]*w;
        b+=table[offset+1]*w;
    }
    out[n*levels*2+l*2]=a;
    out[n*levels*2+l*2+1]=b;
}

__device__ void grouped_add(float* target,unsigned id,float value) {
    unsigned active=__activemask();
    unsigned group=__match_any_sync(active,id);
    int leader=__ffs(group)-1;
    float total=0.f;
    unsigned todo=group;
    while(todo) {
        int lane=__ffs(todo)-1;
        total+=__shfl_sync(group,value,lane);
        todo&=todo-1;
    }
    if((threadIdx.x&31)==leader)atomicAdd(target+id,total);
}

__global__ void backward_kernel(const float* points,const float* table,const float* spacing,const float* grad,float* grad_table,float* partial,int count,int levels,int size) {
    int n=blockIdx.x*blockDim.x+threadIdx.x;
    int l=blockIdx.y;
    if(n>=count)return;
    float s=spacing[l];
    float x=points[3*n]/s,y=points[3*n+1]/s,z=points[3*n+2]/s;
    int bx=int(floorf(x)),by=int(floorf(y)),bz=int(floorf(z));
    float fx=x-float(bx),fy=y-float(by),fz=z-float(bz);
    float ga=grad[n*levels*2+l*2],gb=grad[n*levels*2+l*2+1];
    float dx=0.f,dy=0.f,dz=0.f;
    #pragma unroll
    for(int k=0;k<8;k++) {
        int cx=(k>>2)&1,cy=(k>>1)&1,cz=k&1;
        float wx=cx?fx:1.f-fx,wy=cy?fy:1.f-fy,wz=cz?fz:1.f-fz;
        float w=wx*wy*wz;
        unsigned id=hash_index(bx+cx,by+cy,bz+cz,unsigned(size-1));
        unsigned offset=(l*size+id)*2;
        grouped_add(grad_table,offset,ga*w);
        grouped_add(grad_table,offset+1,gb*w);
        float g=ga*table[offset]+gb*table[offset+1];
        dx+=g*(cx?1.f:-1.f)*wy*wz;
        dy+=g*wx*(cy?1.f:-1.f)*wz;
        dz+=g*wx*wy*(cz?1.f:-1.f);
    }
    int offset=(n*levels+l)*3;
    partial[offset]=dx/s;
    partial[offset+1]=dy/s;
    partial[offset+2]=dz/s;
}

extern "C" int hash_forward(const float* points,const float* table,const float* spacing,float* out,int count,int levels,int size,void* stream) {
    if(count==0)return 0;
    encode_kernel<<<dim3((count+127)/128,levels),128,0,(cudaStream_t)stream>>>(points,table,spacing,out,count,levels,size);
    return int(cudaGetLastError());
}

extern "C" int hash_backward(const float* points,const float* table,const float* spacing,const float* grad,float* grad_table,float* partial,int count,int levels,int size,void* stream) {
    if(count==0)return 0;
    backward_kernel<<<dim3((count+127)/128,levels),128,0,(cudaStream_t)stream>>>(points,table,spacing,grad,grad_table,partial,count,levels,size);
    return int(cudaGetLastError());
}
