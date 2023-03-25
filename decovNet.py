import torch
import torch.fft

from function import thresh_l1, thresh_l2_array

class DecovNetIndiviual(torch.nn.Module):
    def __init__(self, convergence=False, convergence_type=0, random_initial=True, filter_number=2, is_isotropic=False,
                 max_iter=10, beta1=2e5, beta2=2e5, mu=5e4):
        super().__init__()

        self.DecovNetIterList = torch.nn.ModuleList()
        self.convergence = convergence
        for i in range(max_iter):
            self.DecovNetIterList.append(
                DecovNetIter(filter_number, is_isotropic, beta1, beta2, mu, convergence, random_initial))


    def forward(self, y, k, g=None, u=None):
        # g = None
        # u = None
        e = torch.tensor(1)
        for item in self.DecovNetIterList:
            x, g, u = item(y, k, g, u)
        return x, g, u


class DecovNetShare(torch.nn.Module):
    def __init__(self, convergence=False, convergence_type=0, random_initial=True, filter_number=2, is_isotropic=False,
                 max_iter=10, beta1=2e5, beta2=2e5, mu=5e4):
        super().__init__()
        self.max_iter = max_iter
        self.convergence = convergence
        self.DecovNetIterList = torch.nn.ModuleList()
        self.DecovNetIterList.append(
            DecovNetIter(filter_number, is_isotropic, beta1, beta2, mu, convergence, random_initial))

    def forward(self, y, k, g=None, u=None):
        # g = None
        # u = None
        e = torch.tensor(1)
        for item in range(self.max_iter):
            x, g, u = self.DecovNetIterList[0](y, k, g, u)
        return x, g, u


class DecovNetConvergence(torch.nn.Module):
    def __init__(self, convergence=True, convergence_type=0, random_initial=True, filter_number=2, is_isotropic=False,
                 max_iter=10, beta1=2e5, beta2=2e5, mu=5e4):
        super().__init__()

        self.convergence_type = convergence_type
        self.DecovNetIterList = torch.nn.ModuleList()
        self.filters = torch.nn.ParameterList()
        if filter_number == 2 and not random_initial:
            self.filters.append(torch.nn.Parameter(torch.tensor([[[[-1.0, 1.0], [0, 0]]]]).double()))
            self.filters.append(torch.nn.Parameter(torch.tensor([[[[-1.0, 0], [1.0, 0]]]]).double()))
        else:
            for item in range(filter_number):
                self.filters.append(torch.nn.Parameter(torch.rand(1, 1, 4, 4, dtype=torch.double)))

        self.convergence = convergence
        for i in range(max_iter):
            self.DecovNetIterList.append(
                DecovNetIter(filter_number, is_isotropic, beta1, beta2, mu, convergence, random_initial,
                             share_filters=True))

    def forward(self, y, k, g=None, u=None):
        # g = None
        # u = None
        e = torch.tensor(1)
        for index, item in enumerate(self.DecovNetIterList):
            if self.convergence_type == 0:
                e = e * torch.tensor(0.5)
            elif self.convergence_type == 1:
                e = 1 / (index + 1) / (index + 1)
            x, g, u = item(y, k, g, u, e, filters=self.filters)
        return x, g, u


class DecovNetIter(torch.nn.Module):
    def __init__(self, filter_number=2, is_isotropic=False, beta1=2e5, beta2=2e5, mu=5e4, convergence=False,
                 random_initial=True, share_filters=False):
        super().__init__()

        if not share_filters:
            self.filters = torch.nn.ParameterList()
            if filter_number == 2 and not random_initial:
                self.filters.append(torch.nn.Parameter(torch.tensor([[[[-1.0, 1.0], [0, 0]]]]).double()))
                self.filters.append(torch.nn.Parameter(torch.tensor([[[[-1.0, 0], [1.0, 0]]]]).double()))
            else:
                for item in range(filter_number):
                    self.filters.append(torch.nn.Parameter(torch.rand(1, 1, 4, 4, dtype=torch.double)))

        if convergence:
            self.filters_epsilon = torch.nn.ParameterList()
            if filter_number == 2 and not random_initial:
                self.filters_epsilon.append(torch.nn.Parameter(torch.tensor([[[[0, 0], [0, 0]]]]).double()))
                self.filters_epsilon.append(torch.nn.Parameter(torch.tensor([[[[0, 0], [0, 0]]]]).double()))
            else:
                for item in range(filter_number):
                    self.filters_epsilon.append(torch.nn.Parameter(torch.rand(1, 1, 4, 4, dtype=torch.double)))

        # self.filters = torch.nn.ParameterList()
        # for i in range(2):
        #     self.filters.append(torch.nn.Parameter(torch.randn((1, 1, 2, 2)).double()))

        self.is_isotropic = is_isotropic
        self.beta1 = beta1
        self.beta2 = beta2
        self.mu = mu

        # y_shape = (1,1,437,277)
        y_shape = (1,1,424,424)
        # y_shape = (1,1,168,168)
        k_shape = (44,44,44,44)
        M = torch.nn.functional.pad(torch.ones(y_shape), k_shape).to("cuda")
        self.M = M + self.beta2 / self.mu * torch.ones(M.shape).to("cuda")

        self.PALayerList = torch.nn.ModuleList()
        for item in range(filter_number):
            self.PALayerList.append(PALayer())

    def forward(self, y, k, g=None, u=None, e=0, filters=None):
        [bantch_size, channel_size, mb, nb] = y.shape
        [bantch_size, channel_size, mk, nk] = k.shape

        # Size of the latent image
        mi = mb + mk - 1
        ni = nb + nk - 1
        if filters is None:
            filters = self.filters
        if e == 0:
            filters_iter = filters
        else:
            filters_iter = [filters[index] + self.filters_epsilon[index] * e for index in range(len(filters))]
        mg = torch.tensor(mi + max([filterItem.shape[2] for filterItem in filters_iter]) - 1)
        ng = torch.tensor(ni + max([filterItem.shape[3] for filterItem in filters_iter]) - 1)
        # Optimal size for FFT
        mf = int(torch.tensor(2).pow(torch.log2(torch.maximum(mg, torch.tensor(mi + mk - 1, dtype=torch.float))).ceil()))
        nf = int(torch.tensor(2).pow(torch.log2(torch.maximum(ng, torch.tensor(ni + nk - 1, dtype=torch.float))).ceil()))

        Ff = [torch.fft.fft2(filterItem, (mf, nf)) for filterItem in filters_iter]
        Fk = torch.fft.fft2(k, (mf, nf))

        yPadding = torch.nn.functional.pad(y, (k.shape[2] - 1, k.shape[2] - 1,
                                           k.shape[3] - 1, k.shape[3] - 1))
        if u is None:
            u = torch.nn.functional.pad(y, (k.shape[2] - 1, k.shape[2] - 1,
                                        k.shape[3] - 1, k.shape[3] - 1), mode='replicate')

        Fu = torch.fft.fft2(u, (mf, nf))

        if g is None:
            g = [torch.zeros((bantch_size, channel_size, int(mg), int(ng))).to("cuda") for filterItem in filters_iter]

        if self.is_isotropic:
            z = thresh_l2_array(g, 1 / self.beta1)
        else:
            z = [thresh_l1(gItem, 1 / self.beta1) for gItem in g]

        za = []
        for item in range(len(self.PALayerList)):
            za.append(self.PALayerList[item](z[item]))

        Fz = [torch.fft.fft2(zItem, (mf, nf)) for zItem in za]

        den = (self.beta2 / self.beta1 * torch.abs(Fk).pow(2))
        for item in Ff:
            den += torch.abs(item).pow(2)

        num = self.beta2 / self.beta1 * torch.conj(Fk).mul(Fu)
        for index in range(len(filters_iter)):
            num += torch.conj(Ff[index]).mul(Fz[index])

        x = torch.real(torch.fft.ifft2(num.div(den), (mf, nf))).double()
        x = x[:, :, :mi, :ni]
        g = [torch.real(torch.fft.ifft2(torch.fft.fft2(x, (mg, ng)).mul(torch.fft.fft2(filterItem, (mg, ng))))).double()
            for filterItem in filters_iter]
        gk = torch.real(torch.fft.ifft2(torch.fft.fft2(x, (mi+ mk - 1, ni+ nk - 1)).mul(torch.fft.fft2(k, (mi+ mk - 1, ni+ nk - 1))))).double()

        # g = [torch.conv2d(x, filterItem.rot90(2, (-2, -1)),
        #                   padding=(filterItem.shape[2] - 1, filterItem.shape[3] - 1)) for filterItem in
        #      filters_iter]
        #
        # gk = torch.stack([torch.conv2d(x[index].unsqueeze(0), k[index].unsqueeze(0).rot90(2, (-2, -1)),
        #                                padding=(k.shape[2] - 1, k.shape[3] - 1))
        #                   for index in range(bantch_size)]).squeeze(dim=2)
        u = (yPadding + self.beta2 / self.mu * gk).div(self.M)

        # cost = 0.5 * self.mu * torch.norm(y.sub(gk[:,:,mk-1:-mk+1, nk-1:-nk+1]), dim=(-2, -1), p='fro').pow(2)
        #
        # # for index in range(len(filters_iter)):
        # #     cost += torch.norm(z[index].sub(g[index]), dim=(-2, -1), p='fro').pow(2)
        #
        # if self.is_isotropic:
        #     cost = cost + torch.sum(torch.sum(torch.stack([gitem.pow(2) for gitem in g]), dim=0).sqrt())
        # else:
        #     cost = cost + torch.sum(torch.stack([torch.max(torch.sum(gitem.abs())) for gitem in g]), dim=0)
        cost = 0
        # print('Iteration n, cost \n' , cost.tolist())
        return x, g, u


class PALayer(torch.nn.Module):
    def __init__(self):
        super(PALayer, self).__init__()
        self.pa = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, 7, padding="same", bias=True),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(1, 1, 7, padding="same", bias=True),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x.float()).double()
        return y*x

