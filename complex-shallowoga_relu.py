import argparse
import numpy as np
import torch 
torch.set_default_dtype(torch.float64)

from utils import relative_l2, init_records
from network import ReLUk
torch.manual_seed(0)

class ComplexShallowOGAFitter:
    def __init__(
            self, 
            data,
            nNeuron,
            act,
            device):

        # dataset 
        self.data = data 
        self.device = device 
        self.inpDim = data['Xtrain'].shape[1]
        self.nPts = data['Xtrain'].shape[0]

        X = data['Xtrain']
        Xtest = data['Xtest']
        Xgrid = data['Xgrid']

        self.X = torch.hstack((X, torch.ones((X.shape[0],1))))
        self.y_real = data['ytrain'].real
        self.y_img = data['ytrain'].imag

        self.Xtest = torch.hstack((Xtest, torch.ones((Xtest.shape[0],1))))
        self.ytest_real = data['ytest'].real
        self.ytest_img = data['ytest'].imag

        self.Xgrid = torch.hstack((Xgrid, torch.ones((Xgrid.shape[0],1))))
        self.ygrid = data['ygrid']
        
        self.h = 1/self.nPts
        
        self.yk_real = torch.zeros_like(self.y_real)
        self.yk_img = torch.zeros_like(self.y_img)
        self.ylog = []

        # nn config 
        self.nNeuron = nNeuron
        self.act = act
        self.Alpha = None
        self.WB_real = torch.zeros((self.nNeuron, self.inpDim+1))
        self.WB_imag = torch.zeros((self.nNeuron, self.inpDim+1))

        # put to device 
        self.device = device 
        self.to_device()

    def to_device(self):
        self.X = self.X.to(self.device)
        
        self.y_real = self.y_real.to(self.device)
        self.y_img = self.y_img.to(self.device)
        
        self.Xtest = self.Xtest.to(self.device)

        self.ytest_real = self.ytest_real.to(self.device)
        self.ytest_img = self.ytest_img.to(self.device)

        self.Xgrid = self.Xgrid.to(self.device)

        self.yk_real = self.yk_real.to(self.device)
        self.yk_img = self.yk_img.to(self.device)

        self.WB_real = self.WB_real.to(self.device)
        self.WB_imag = self.WB_imag.to(self.device)

    def random_guess(self, nr):
        if (self.inpDim == 1):
            phi = torch.rand(nr).to(self.device) * 2*torch.pi
            b = (torch.rand(nr)*2 - 1).to(self.device) * 2**1.5
            Wx = torch.cos(phi).reshape(-1,1)
            B = b.reshape(-1,1)
            self.WBs = torch.concatenate([Wx, B], axis=1)  
            self.nParam = self.WBs.shape[0]

        if self.inpDim == 2:
            def rand_weights(nr):
                phi = torch.rand(nr).to(self.device) * 2*torch.pi
                b = (torch.rand(nr)*2 - 1).to(self.device) * 2**1.5
                w1 = torch.cos(phi).reshape(-1,1)
                w2 = torch.sin(phi).reshape(-1,1)
                b = b.reshape(-1,1)
                return w1, w2, b

            w1, w2, b = rand_weights(nr)
            self.WBs_real = torch.concatenate([w1, w2, b], axis=1)  
            
            w1, w2, b = rand_weights(nr)
            self.WBs_imag = torch.concatenate([w1, w2, b], axis=1)  

            self.nParam = self.WBs_real.shape[0]

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
        
        self.gs_real = self.act(self.WBs_real @ self.X.T)
        self.gs_img = self.act(self.WBs_imag @ self.X.T)

    def brute_search(self):
        # init search
        rG = self.h * ( self.gs_real @ (self.y_real - self.yk_real) +\
                        self.gs_img @ (self.y_img - self.yk_img))
        E = -0.5 * rG ** 2 
        
        idx = torch.argmin(E)
        wbk_real = self.WBs_real[idx]
        wbk_img = self.WBs_imag[idx]
        return wbk_real, wbk_img
    
    def projection(self, k):    
        gsub_real = self.act(self.WB_real[:k+1] @ self.X.T)
        gsub_img = self.act(self.WB_imag[:k+1] @ self.X.T)

        A = torch.einsum('kn,pn->kp', gsub_real, gsub_real) + torch.einsum('kn,pn->kp', gsub_img, gsub_img)
        b = torch.einsum('kn,ns->ks', gsub_real, self.y_real) + torch.einsum('kn,ns->ks', gsub_img, self.y_img)

        alpha_k = torch.linalg.solve(A, b)
        return alpha_k
   
    def optimize_random(self, nr=1024):

        for k in range(self.nNeuron):
            self.random_guess(nr)
            wbk_real, wbk_img = self.brute_search()
            self.WB_real[k] = wbk_real 
            self.WB_imag[k] = wbk_img

            try:
                alpha_k = self.projection(k)
                self.Alpha = alpha_k
            except:
                print("LinAlgWarning, we should stop adding neurons")
                break 

            # update yk
            self.yk_real = (self.Alpha.T @ self.act(self.WB_real[:k+1] @ self.X.T)).T
            self.yk_img = (self.Alpha.T @ self.act(self.WB_imag[:k+1] @ self.X.T)).T
            
            self.ypred_real = (self.Alpha.T @ self.act(self.WB_real[:k+1] @ self.Xtest.T)).T
            self.ypred_img = (self.Alpha.T @ self.act(self.WB_imag[:k+1] @ self.Xtest.T)).T

            yreal_rl2_test = relative_l2(self.ypred_real, self.ytest_real)
            yimg_rl2_test = relative_l2(self.ypred_img, self.ytest_img)
            
            self.ylog.append([yreal_rl2_test.cpu().numpy(), yimg_rl2_test.cpu().numpy()])

            if k % 10 == 0:
                print('{:}th (y) : real rl2 {:.4e} - img rl2 : {:.4e}'.format(
                    k, yreal_rl2_test, yimg_rl2_test))
        
        ypred_real = (self.Alpha.T @ self.act(self.WB_real @ self.Xgrid.T)).T
        ypred_img = (self.Alpha.T @ self.act(self.WB_imag @ self.Xgrid.T)).T
        ypred = ypred_real + 1j * ypred_img
        self.ypred = {"pred":ypred.cpu().numpy(), "ref":self.ygrid.numpy()}
        self.model_weights = {
            "WB_real":self.WB_real.detach().cpu().numpy(), 
            "WB_imag":self.WB_imag.detach().cpu().numpy(),
            "Alpha":self.Alpha.detach().cpu().numpy()}
        self.log = {"y_rl2" : np.array(self.ylog)}
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OGA for function fitting')
    parser.add_argument('--task', type=str, default='fredholm_3d_ex1',
                        help='task name.')
    parser.add_argument('--nNeuron', type=int, default=1024,
                        help='maximum number of neurons')
    parser.add_argument('--k', type=int, default=1,
                        help='order of relu')
    parser.add_argument('--param', type=int, default=4,
                        help='wavenumber')
    parser.add_argument('--device', type=int, default=0,
                        help='device id.')
    parser.add_argument('--nTrain', type=int, default=2500, 
                        help='number of training samples')
    parser.add_argument('--nTest', type=int, default=5000, 
                        help='number of training samples')
    parser.add_argument('--nr', type=int, default=512, 
                        help='mesh density')
    
    args = parser.parse_args()
    device = torch.device(f'cuda:{args.device}')
    
    if args.task == 'helmholtz-green_1d':
        from dataset import helmholtz_green_1d
        data = helmholtz_green_1d(
            dtype=torch.float64, k=args.param, nTrain=args.nTrain, nTest=args.nTest)
    
    act = ReLUk(k=args.k)
    model = ComplexShallowOGAFitter(
        data = data,
        nNeuron=args.nNeuron, 
        act=act,
        device=device)
    model.optimize_random(nr=args.nr)

    exp_nm = 'oga-complex-{:}-{:}-{:}-{:}-{:}-relu{:}'.format(args.nNeuron, args.nr, args.param, args.nTrain, args.nTest, args.k)
    print(exp_nm)
    log_outpath, upred_outpath, model_outpath = init_records('./results', args.task, exp_nm)
    np.save(log_outpath, model.log)
    np.save(upred_outpath, model.ypred)
    np.save(model_outpath, model.model_weights)


