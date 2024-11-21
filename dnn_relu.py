import torch 
import torch.nn as nn
import argparse 
from utils import relative_l2
from network import DeepNN
# torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

class DNNSolver:
    def __init__(self, data, lops, complex, device):
        
        self.data = data
        self.lops = lops
        self.complex = complex
        self.device = device

        self.inpDim = data['xTrain'].shape[-1]
        if complex:
            self.model = DeepNN([self.inpDim,50,50,50,2], nn.ReLU())
        else:
            self.model = DeepNN([self.inpDim,50,50,50,1], nn.ReLU())

        self.to_device()


    def to_device(self):
        for k, v in self.data.items():
            if k != 'beta':
                self.data[k] = v.to(self.device)
        self.model = self.model.to(self.device)
        self.lops.to(self.device)

    def optimize_adam(self, niter, lr=1e-3, dispstep=100):
        self.opt_adam = torch.optim.Adam(self.model.parameters(), lr)
        
        print()
        print("Adam optimization start")
        print()

        for n in range(niter):
            self.model.train()
            self.opt_adam.zero_grad()
            uPred = self.model(self.data['xTrain'])
            if self.complex:
                uPred = torch.complex(uPred[...,0], uPred[...,1])
                uPred = uPred.reshape(256, 256)
            fPred = uPred - self.lops.map(uPred)
            loss = torch.linalg.norm(fPred - self.data['fTrain'])
            loss.backward()
            self.opt_adam.step()

            if n % dispstep == 0:
                self.model.eval()
                with torch.no_grad():
                    uPred = self.model(self.data['xTrain'])
                    if self.complex:
                        uPred = torch.complex(uPred[...,0], uPred[...,1])
                        uPred = uPred.reshape(256, 256)
                    train_rl2 = relative_l2(uPred, self.data['uTrain'])
                    
                    uPred = self.model(self.data['xTest'])
                    if self.complex:
                        uPred = torch.complex(uPred[...,0], uPred[...,1])
                        uPred = uPred.reshape(256, 256)
                    test_rl2 = relative_l2(uPred, self.data['uTest'])
                
                print("{:}th - train rl2 : {:.4e} - val rl2 : {:.4e}".format(
                    n, train_rl2, test_rl2))           

    def optimize_lbfgs(self, niter, lr=1e-3, dispstep=10):
        self.opt_lbfgs = torch.optim.LBFGS(self.model.parameters(), lr)
        
        print()
        print("L-BFGS optimization start")
        print()

        for n in range(niter):
            self.model.train()
            def closure():
                self.opt_lbfgs.zero_grad()
                uPred = self.model(self.data['xTrain'])
                if self.complex:
                    uPred = torch.complex(uPred[...,0], uPred[...,1])
                fPred = uPred - self.lops.map(uPred)
                loss = torch.linalg.norm(fPred - self.data['fTrain'])
                loss.backward()
                return loss 

            loss_prev = closure()
            self.opt_lbfgs.step(closure)
            loss_cur = closure()
            
            if n % dispstep == 0:
                self.model.eval()
                with torch.no_grad():
                    uPred = self.model(self.data['xTrain'])
                    if self.complex:
                        uPred = torch.complex(uPred[...,0], uPred[...,1])
                    train_rl2 = relative_l2(uPred, self.data['uTrain'])
                    
                    uPred = self.model(self.data['xTest'])
                    if self.complex:
                        uPred = torch.complex(uPred[...,0], uPred[...,1])
                    test_rl2 = relative_l2(uPred, self.data['uTest'])
                
                print("{:}th - train rl2 : {:.4e} - val rl2 : {:.4e}".format(
                    n, train_rl2, test_rl2))    

            if (loss_prev - loss_cur).abs() < 1e-16:
                print("lbfgs converge")
                break         
        pass 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dnn function fitting')
    parser.add_argument('--task', type=str, default='fredholm_3d_ex1',
                        help='task name.')
    parser.add_argument('--opt', type=str, default='adam',
                        help='adam or lbfgs')
    parser.add_argument('--device', type=int, default=0,
                        help='device id.')
    
    args = parser.parse_args()
    if args.device == -1:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.device}')

    if args.task == 'fredholm_3d_ex1':
        from dataset import fredholm_3d_ex1
        from linearops import MCIntegralOperator
        data = fredholm_3d_ex1()
        lops = MCIntegralOperator(K=data['KTrain'], beta=data['beta'])
        complex = False
    elif args.task == 'fredholm_4d_ex1':
        from dataset import fredholm_4d_ex1
        from linearops import MCIntegralOperator
        data = fredholm_4d_ex1()
        lops = MCIntegralOperator(K=data['KTrain'], beta=data['beta'])
        complex = False
    elif args.task == 'inhomo_helmholtz_2d':
        from dataset import inhomo_helmholtz_2d
        from linearops import CBSOperator
        data = inhomo_helmholtz_2d()
        lops = CBSOperator(gamma=data['gamma'], V=data['V'], g0=data['g0'])
        complex = True
    
    model = DNNSolver(data, lops, complex, device)

    if args.opt == 'adam':
        adam_niter = 30000
        model.optimize_adam(adam_niter, lr=1e-3, dispstep=1000)
    elif args.opt == 'lbfgs':
        lbfgs_niter = 1000
        model.optimize_lbfgs(lbfgs_niter, lr=1e-3, dispstep=10)