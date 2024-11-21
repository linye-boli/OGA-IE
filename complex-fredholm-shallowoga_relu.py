import argparse
import numpy as np
import torch 
torch.set_default_dtype(torch.float64)

from utils import relative_l2, init_records
from network import ReLUk
torch.manual_seed(0)

class ShallowOGAFredholm:
    def __init__(
            self, 
            data,
            lops,
            nNeuron,
            act,
            device):

        # dataset 
        self.data = data 
        self.device = device 
        self.inpDim = data['xTrain'].shape[1]
        self.nPts = data['xTrain'].shape[0]

        X = data['xTrain']
        Xtest = data['xTest']

        self.X = torch.hstack((X, torch.ones((X.shape[0],1))))
        self.Xtest = torch.hstack((Xtest, torch.ones((Xtest.shape[0],1))))
        self.fTrain = data['fTrain']
        self.uTrain = data['uTrain']
        self.uTest = data['uTest']
        self.uk = torch.zeros_like(self.fTrain)

        # nn config 
        self.nNeuron = nNeuron
        self.act = act
        self.Alpha = None
        self.WB = torch.zeros((self.nNeuron, self.inpDim+1))
        self.lops = lops

        # put to device 
        self.device = device 
        self.to_device()

    def to_device(self):
        self.fTrain = self.fTrain.to(self.device)
        self.uTrain = self.uTrain.to(self.device)
        self.uTest = self.uTest.to(self.device)
        self.X = self.X.to(self.device)
        self.Xtest = self.Xtest.to(self.device)
        self.WB = self.WB.to(self.device)
        self.uk = self.uk.to(self.device)
        self.lops.to(self.device)

    def random_guess(self, nr):
        if (self.inpDim == 1):
            phi = torch.rand(nr).to(self.device) * 2*torch.pi
            b = (torch.rand(nr)*2 - 1).to(self.device) * 2**1.5
            Wx = torch.cos(phi).reshape(-1,1)
            B = b.reshape(-1,1)
            self.WBs = torch.concatenate([Wx, B], axis=1)  
            self.nParam = self.WBs.shape[0]

        if self.inpDim == 2:
            phi = torch.rand(nr).to(self.device) * 2*torch.pi
            b = (torch.rand(nr)*2 - 1).to(self.device) * 2**1.5
            w1 = torch.cos(phi).reshape(-1,1)
            w2 = torch.sin(phi).reshape(-1,1)
            b = b.reshape(-1,1)
            self.WBs = torch.concatenate([w1, w2, b], axis=1)  
            self.nParam = self.WBs.shape[0]

        if self.inpDim == 3:
            phi1 = torch.rand(nr).to(self.device) * torch.pi 
            phi2 = torch.rand(nr).to(self.device) *  2 * torch.pi 
            b = (torch.rand(nr)*2 - 1.0).to(self.device) * 4 # for poisson 2D            
            w1 = torch.cos(phi1)
            w2 = torch.sin(phi1) * torch.cos(phi2)
            w3 = torch.sin(phi1) * torch.sin(phi2)
            w1 = w1.reshape(-1,1)
            w2 = w2.reshape(-1,1)
            w3 = w3.reshape(-1,1)
            b = b.reshape(-1,1)
            self.WBs = torch.concatenate([w1, w2, w3, b], axis=1)   
            self.nParam = self.WBs.shape[0]

        if self.inpDim == 4:
            phi1 = torch.rand(nr).to(self.device) * torch.pi 
            phi2 = torch.rand(nr).to(self.device) * torch.pi 
            phi3 = torch.rand(nr).to(self.device) * 2 * torch.pi 
            b = (torch.rand(nr)*2 - 1.0).to(self.device) * 4 # for poisson 2D
            w1 = torch.cos(phi1)
            w2 = torch.sin(phi1) * torch.cos(phi2)
            w3 = torch.sin(phi1) * torch.sin(phi2) * torch.cos(phi3)
            w4 = torch.sin(phi1) * torch.sin(phi2) * torch.sin(phi3)
            w1 = w1.reshape(-1,1)
            w2 = w2.reshape(-1,1)
            w3 = w3.reshape(-1,1)
            w4 = w4.reshape(-1,1)
            b = b.reshape(-1,1)
            self.WBs = torch.concatenate([w1, w2, w3, w4, b], axis=1)   
            self.nParam = self.WBs.shape[0]
        self.gs = self.act(self.WBs @ self.X.T)
 
    def brute_search(self):
        # init search
        rG = self.gs @ (self.fTrain - (self.uk - self.lops.map(self.uk)))
        E = -0.5 * rG ** 2 
        
        idx = torch.argmin(E)
        wbk = self.WBs[idx]
        return wbk
    
    def projection(self, k):    
        gsub = self.act(self.WB[:k+1] @ self.X.T)
        fsub = (gsub.T - self.lops.map(gsub.T)).T

        A = torch.einsum('kn,pn->kp', gsub, fsub)
        b = torch.einsum('kn,ns->ks', gsub, self.fTrain)

        alpha_k = torch.linalg.solve(A, b)
        return alpha_k
   
    def optimize_random(self, nr=1024):

        for k in range(self.nNeuron):
            self.random_guess(nr)
            wbk = self.brute_search()
            self.WB[k] = wbk

            try:
                alpha_k = self.projection(k)
                self.Alpha = alpha_k
            except:
                print("LinAlgWarning, we should stop adding neurons")
                break 

            # update yk
            self.uk = (self.Alpha.T @ self.act(self.WB[:k+1] @ self.X.T)).T
            self.upred = (self.Alpha.T @ self.act(self.WB[:k+1] @ self.Xtest.T)).T

            url2_train = relative_l2(self.uk, self.uTrain)
            url2_test = relative_l2(self.upred, self.uTest)

            if k % 10 == 0:
                print('{:}th (y) : train rl2 {:.4e} - val rl2 : {:.4e}'.format(
                    k, url2_train, url2_test))

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OGA for function fitting')
    parser.add_argument('--task', type=str, default='fredholm_3d_ex1',
                        help='task name.')
    parser.add_argument('--nNeuron', type=int, default=1024,
                        help='maximum number of neurons')
    parser.add_argument('--k', type=int, default=1,
                        help='order of relu')
    parser.add_argument('--device', type=int, default=0,
                        help='device id.')
    
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.device}')
    
    if args.task == 'fredholm_3d_ex1':
        from dataset import fredholm_3d_ex1
        from linearops import MCIntegralOperator
        data = fredholm_3d_ex1(dtype=torch.float64)
        lops = MCIntegralOperator(K=data['KTrain'], beta=data['beta'])
    elif args.task == 'fredholm_4d_ex1':
        from dataset import fredholm_4d_ex1
        from linearops import MCIntegralOperator
        data = fredholm_4d_ex1(dtype=torch.float64)
        lops = MCIntegralOperator(K=data['KTrain'], beta=data['beta'])
    elif args.task == 'inhomo_helmholtz_2d':
        from dataset import inhomo_helmholtz_2d
        from linearops import CBSOperator
        data = inhomo_helmholtz_2d(dtype=torch.float64)
        lops = CBSOperator(gamma=data['gamma'], V=data['V'], g0=data['g0'])
        complex = True
    
    act = ReLUk(k=args.k)
    model = ShallowOGAFredholm(
        data = data,
        lops=lops,
        nNeuron=args.nNeuron, 
        act=act, 
        device=device)
    model.optimize_random(nr=512)

    # exp_nm = 'oga-complex-{:}-{:}-{:}-{:}-{:}-relu{:}'.format(args.nNeuron, args.nr, args.param, args.nTrain, args.nTest, args.k)
    # print(exp_nm)
    # log_outpath, upred_outpath, model_outpath = init_records('./results', args.task, exp_nm)
    # np.save(log_outpath, model.log)
    # np.save(upred_outpath, model.ypred)
    # np.save(model_outpath, model.model_weights)
