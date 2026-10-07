#include <cuda_runtime.h>
#include <math.h>

__global__ void target_backward(const double* q,const double* scale,const double* up_value,const double* up_gradient,double* output){
    int j=threadIdx.x;
    if(j>=9)return;
    double s[3]={.5*(q[7]-q[5]),.5*(q[2]-q[6]),.5*(q[3]-q[1])};
    double square=s[0]*s[0]+s[1]*s[1]+s[2]*s[2];
    double n=sqrt(fmax(square,1e-20));
    double trace=q[0]+q[4]+q[8],c=.5*(trace-1),h=c*c+n*n;
    double angle=atan2(n,c),tau=scale[0];
    bool small=square<1e-8&&c>0,active=square>=1e-20;
    double ds[3]={0,0,0};
    if(j==7)ds[0]=.5;if(j==5)ds[0]=-.5;
    if(j==2)ds[1]=.5;if(j==6)ds[1]=-.5;
    if(j==3)ds[2]=.5;if(j==1)ds[2]=-.5;
    int row=j/3,col=j%3;
    double dt=row==col?1:0,dc=.5*dt,dn=active?(s[0]*ds[0]+s[1]*ds[1]+s[2]*ds[2])/n:0;
    double dangle=(c*dn-n*dc)/h;
    double dv_value=small?(s[0]*ds[0]+s[1]*ds[1]+s[2]*ds[2])/(tau*tau):angle*dangle/(tau*tau);
    double dh=2*c*dc+2*n*dn,k=angle/(tau*h),dk=(dangle*h-angle*dh)/(tau*h*h);
    double result=up_value[0]*dv_value;
    for(int i=0;i<3;i++){
        double v=0,dv=0;
        for(int a=0;a<3;a++){
            double b=.5*((a==i?trace:0)-q[a*3+i]);
            v+=b*s[a];
            dv+=b*ds[a];
        }
        dv+=.5*(dt*s[i]-(i==col?s[row]:0));
        double derivative;
        if(small)derivative=dv/tau;
        else{
            double normal=active?v/n:0;
            double dnormal=active?(dv/n-v*dn/(n*n)):0;
            double b=c*normal+n*s[i];
            double db=dc*normal+c*dnormal+dn*s[i]+n*ds[i];
            derivative=dk*b+k*db;
        }
        result+=up_gradient[i]*derivative;
    }
    output[j]=result;
}

extern "C" int target_mixed_backward(const double* q,const double* scale,const double* up_value,const double* up_gradient,double* output,void* stream){
    target_backward<<<1,32,0,(cudaStream_t)stream>>>(q,scale,up_value,up_gradient,output);
    return (int)cudaGetLastError();
}
