#include <cuda_runtime.h>
#include <stdint.h>

__global__ void mixed_backward_kernel(const float* points,const float* table,const float* gradient,const float* spacings,const float* upstream,float* point_partial,float* table_gradient,float* feature_gradient,int count,int levels,int size,int need_point,int need_table,int need_feature){
    int n=blockIdx.x*blockDim.x+threadIdx.x;
    int l=blockIdx.y;
    if(n>=count)return;
    float s=spacings[l],s2=s*s;
    float fraction[3];int64_t integral[3];
    for(int axis=0;axis<3;axis++){
        float scaled=__fdiv_rn(points[n*3+axis],s);
        float base=floorf(scaled);
        integral[axis]=(int64_t)base;
        fraction[axis]=scaled-base;
    }
    float ux=upstream[n*3],uy=upstream[n*3+1],uz=upstream[n*3+2];
    float g0=gradient[(n*levels+l)*2],g1=gradient[(n*levels+l)*2+1];
    float px=0.f,py=0.f,pz=0.f,fg0=0.f,fg1=0.f;
    #pragma unroll
    for(int corner=0;corner<8;corner++){
        int bx=(corner>>2)&1,by=(corner>>1)&1,bz=corner&1;
        float sx=2*bx-1,sy=2*by-1,sz=2*bz-1;
        float ax=bx?fraction[0]:1.f-fraction[0];
        float ay=by?fraction[1]:1.f-fraction[1];
        float az=bz?fraction[2]:1.f-fraction[2];
        int64_t index=((integral[0]+bx)^((integral[1]+by)*(int64_t)2654435761ULL)^((integral[2]+bz)*(int64_t)805459861ULL))&(size-1);
        int64_t offset=((int64_t)l*size+index)*2;
        float v0=table[offset],v1=table[offset+1];
        float dx=__fdiv_rn((ay*az)*sx,s),dy=__fdiv_rn((ax*az)*sy,s),dz=__fdiv_rn((ax*ay)*sz,s);
        float directional=(dx*ux+dy*uy)+dz*uz;
        if(need_table){
            atomicAdd(table_gradient+offset,g0*directional);
            atomicAdd(table_gradient+offset+1,g1*directional);
        }
        if(need_feature){fg0+=v0*directional;fg1+=v1*directional;}
        if(need_point){
            float hxy=__fdiv_rn(az*(sx*sy),s2),hxz=__fdiv_rn(ay*(sx*sz),s2),hyz=__fdiv_rn(ax*(sy*sz),s2);
            float coefficient=g0*v0+g1*v1;
            px+=coefficient*(uy*hxy+uz*hxz);
            py+=coefficient*(ux*hxy+uz*hyz);
            pz+=coefficient*(ux*hxz+uy*hyz);
        }
    }
    if(need_feature){feature_gradient[(n*levels+l)*2]=fg0;feature_gradient[(n*levels+l)*2+1]=fg1;}
    if(need_point){int offset=(n*levels+l)*3;point_partial[offset]=px;point_partial[offset+1]=py;point_partial[offset+2]=pz;}
}

extern "C" int mixed_backward_complete(const float* points,const float* table,const float* gradient,const float* spacings,const float* upstream,float* point_partial,float* table_gradient,float* feature_gradient,int count,int levels,int size,int need_point,int need_table,int need_feature,cudaStream_t stream){
    if(count) mixed_backward_kernel<<<dim3((count+127)/128,levels),128,0,stream>>>(points,table,gradient,spacings,upstream,point_partial,table_gradient,feature_gradient,count,levels,size,need_point,need_table,need_feature);
    return (int)cudaGetLastError();
}
