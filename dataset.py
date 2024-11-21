import torch 
# torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

def fredholm_3d_ex1(nTrain=1000, nTest=1000, dtype=torch.float32):
    # u + Ku = f
    # domain: [1,2] x [1,2] x [1,2]
    # given:     K(x,y,z, s,t,v) = exp(-xs-yt-zv)
    #            f(x,y,z) = 1 - (exp(-2x) - exp(-x))(exp(-2y) - exp(-y))(exp(-2z) - exp(-z))/(xyz)
    # solution : u(x,y,z) = 1

    def kernel_func(xy):
        x = xy[:,:3]
        y = xy[:,3:]
        return -torch.exp(-(x * y).sum(axis=1))

    def f_func(x):
        def exp(x):
            return (torch.exp(-2*x) - torch.exp(-x))/x
        return 1 - (exp(x[:,[0]])*exp(x[:,[1]])*exp(x[:,[2]]))

    def u_func(x):
        return torch.ones((x.shape[0],1), dtype=dtype)

    xTrain = torch.rand(nTrain,3,dtype=dtype) + 1
    fTrain = f_func(xTrain)
    uTrain = u_func(xTrain)
    XYIdx = torch.cartesian_prod(torch.arange(nTrain), torch.arange(nTrain))
    XYTrain = torch.concatenate([xTrain[XYIdx[:,0]], xTrain[XYIdx[:,1]]], axis=1)
    KTrain = kernel_func(XYTrain).reshape(nTrain, nTrain)

    xTest = torch.rand(nTest, 3, dtype=dtype) + 1
    fTest = f_func(xTest)
    uTest = u_func(xTest)
    XYIdx = torch.cartesian_prod(torch.arange(nTest), torch.arange(nTest))
    XYTest = torch.concatenate([xTest[XYIdx[:,0]], xTest[XYIdx[:,1]]], axis=1)
    KTest = kernel_func(XYTest).reshape(nTest, nTest)

    return {
        "xTrain":xTrain, 
        "fTrain":fTrain, 
        "uTrain":uTrain, 
        "KTrain":KTrain,
        "xTest":xTest, 
        "fTest":fTest, 
        "uTest":uTest, 
        "KTest":KTest,
        "beta":1}

def fredholm_4d_ex1(nTrain=5000, nTest=5000, dtype=torch.float32):
    # u + Ku = f
    # domain: [1,2] x [1,2] x [1,2]
    # given:     K(x,y,z, s,t,v) = exp(-xs-yt-zv)
    #            f(x,y,z) = 1 - (exp(-2x) - exp(-x))(exp(-2y) - exp(-y))(exp(-2z) - exp(-z))/(xyz)
    # solution : u(x,y,z) = 1

    def kernel_func(xy):
        x = xy[:,:4]
        y = xy[:,4:]
        return x[:,[0]]*x[:,[1]]*x[:,[2]]*x[:,[3]]

    def f_func(x):
        return 15/16 * (x[:,[0]]*x[:,[1]]*x[:,[2]]*x[:,[3]])

    def u_func(x):
        return x[:,[0]]*x[:,[1]]*x[:,[2]]*x[:,[3]]

    xTrain = torch.rand(nTrain,4,dtype=dtype)
    fTrain = f_func(xTrain)
    uTrain = u_func(xTrain)
    XYIdx = torch.cartesian_prod(torch.arange(nTrain), torch.arange(nTrain))
    XYTrain = torch.concatenate([xTrain[XYIdx[:,0]], xTrain[XYIdx[:,1]]], axis=1)
    KTrain = kernel_func(XYTrain).reshape(nTrain, nTrain)

    xTest = torch.rand(nTest, 4, dtype=dtype)
    fTest = f_func(xTest)
    uTest = u_func(xTest)
    XYIdx = torch.cartesian_prod(torch.arange(nTest), torch.arange(nTest))
    XYTest = torch.concatenate([xTest[XYIdx[:,0]], xTest[XYIdx[:,1]]], axis=1)
    KTest = kernel_func(XYTest).reshape(nTest, nTest)

    return {
        "xTrain":xTrain, 
        "fTrain":fTrain, 
        "uTrain":uTrain, 
        "KTrain":KTrain,
        "xTest":xTest, 
        "fTest":fTest, 
        "uTest":uTest, 
        "KTest":KTest,
        "beta":1}


def inhomo_helmholtz_2d(dtype=torch.float32, device="cpu"):
    from cbs.cbsmodel import ConvergentBornSeries 
    model = ConvergentBornSeries(lamb=1,
                             sos=None,# default homogenous, sound of speed
                             dx=None,
                             src_loc=[100,40],
                             boundary_width=[8,8],
                             boundary_strength=1,
                             boundary_type='PML3',
                             dtype=dtype).to(device)
    if dtype == torch.float32:
        model.solve(maxiter=10000, tol=1e-7)
    elif dtype == torch.float64:
        model.solve(maxiter=10000, tol=1e-14)

    u = model.u
    f = model.f 
    gamma = model.gamma
    g0 = model.g0 
    V = model.V
    m,n = u.shape
    x = torch.stack(torch.meshgrid(
        [torch.linspace(0,1,m), torch.linspace(0,1,n)])).permute(1,2,0)
    x = x.reshape(-1,2)

    return {
        "xTrain":x,
        "uTrain":u, 
        "fTrain":f,
        "xTest":x,
        "uTest":u,
        "gamma":gamma, 
        "g0":g0, 
        "V":V,}


def helmholtz_green_1d(nTrain=2500, nTest=5000, n=101, k=2, dtype=torch.float32, device='cpu'):
    def func(X):
        x = X[:,[0]]
        ksi = X[:,[1]]
        r = (x - ksi).abs()
        return 1j / (2*k) * torch.exp(1j * k * r)
        
    X = torch.rand(nTrain,2,dtype=dtype)*2 - 1
    y = func(X)

    Xtest = torch.rand(nTest,2,dtype=dtype)*2 - 1
    ytest = func(Xtest)

    Xgrid = torch.concat([torch.zeros(n,1), 
                          torch.linspace(-1,1,n).reshape(-1,1)], axis=1)
    ygrid = func(Xgrid)

    return {
        "Xtrain":X,
        "ytrain":y, 
        "Xtest":Xtest,
        "ytest":ytest, 
        "Xgrid":Xgrid,
        "ygrid":ygrid,}  
