import numpy as np
from subspace import solve


def main():
    x=np.eye(512)[0]
    a=np.zeros((1,512)); a[0,0]=-.01; a[0,1]=1
    for q,rho,exact in [(np.eye(512)[:,1:2],.05,.04),(np.eye(512)[:,2:3],.05,-.01),(np.eye(512)[:,1:2],0.,-.01)]:
        c,w,lo,hi,_=solve(x,a,q,rho)
        assert lo<=exact<=hi and hi-lo<1e-8
        assert np.linalg.norm(q@c)<=rho+1e-15
        assert w.min()>=0 and abs(w.sum()-1)<1e-14
    q=np.eye(512)[:,1:2]
    c,w,lo,hi,_=solve(x,np.vstack([a,np.zeros(512)]),q,.05)
    assert lo<=0<=hi and hi-lo<1e-8
    print('Analytic reachable, orthogonal-blocked, locked and tie checks passed')


if __name__=='__main__':
    main()
