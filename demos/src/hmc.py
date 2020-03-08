import numpy as np
import torch as th
from gytools import *
from gyplots import *
from torch.autograd import Variable
from datetime import datetime
import time
from scipy.stats import multivariate_normal
epsilon = 1e-8

class Optim:
    def __init__(self, forwardObj, sumfuncObj, args, device, usePrior=True):
        print("Initial Class hmc::Optim()")
        self.lr = args.lr
        self.xs = []
        self.lpdfs = []
        self.dir = args.out_dir
        self.usePrior= usePrior
        self.device = device
        self.to_save = args.to_save

        self.forwardObj = forwardObj
        self.sumfuncObj = sumfuncObj

    def optim(self, N):
        para = th.tensor(self.forwardObj.para, dtype=th.float32, device=self.device)
        x = self.forwardObj.orig_to_norm(para)

        x = Variable(x, requires_grad=True)
        optimizer = th.optim.Adam([x], lr=self.lr)

        xs = []
        losses = []
        for i in range(N):
            img = self.forwardObj.eval_render(x)
            loss = self.sumfuncObj.logpdf(img)
            if self.usePrior:
                lpdf_prior = self.forwardObj.eval_prior_lpdf(x)
                loss += lpdf_prior

            xs.append(self.forwardObj.norm_to_orig(x).detach().cpu().numpy())
            losses.append(loss.item())

            if i%10==0:
                now = datetime.now(); print(now)
                print('%d/%d: loss:%f' % (i,N,losses[-1]))

                xs_arr = np.vstack(xs)
                losses_arr = np.vstack(losses)
                id = np.vstack(np.arange(len(losses)))
                np.savetxt(os.path.join(self.dir, 'optim.csv'),
                    np.concatenate((id, losses_arr, xs_arr), 1),
                    delimiter=",", fmt='%.3f')

                if self.to_save == 'fig':
                    plotOptimFrames(self.dir,
                        np.vstack(losses),
                        img.detach().cpu().numpy(),
                        self.sumfuncObj.target.cpu().numpy(),
                        i, N)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if not self.to_save == 'fig':
            plotFigureOptim(self.dir,
                np.vstack(losses),
                img.detach().cpu().numpy(),
                self.sumfuncObj.target.cpu().numpy(),
                i)

        return xs, losses


class HMC:
    def __init__(self, forwardObj, sumfuncObj, args, usePrior=True):
        print("Initial Class hmc::HMC()")
        self.delta = args.lf_lens / args.lf_steps
        self.L = args.lf_steps
        print('leapfrog delta:', self.delta)
        print('leapfrog L:', self.L)
        self.xs = []
        self.lpdfs = []
        self.dir = args.out_dir
        self.num_reject = 0
        self.num_outBound = 0
        self.usePrior = usePrior
        self.to_save = args.to_save

        self.forwardObj = forwardObj
        self.sumfuncObj = sumfuncObj

    def leapfrog(self, x, p):
        d = self.delta
        p = p - d/2 * self.dU(x)
        x = x + d   * self.dK(p)
        for i in range(self.L-1):
            p = p - d * self.dU(x)
            x = x + d * self.dK(p)
        p = p - d/2 * self.dU(x)
        return x, p

    def U(self, x):
        img = self.forwardObj.eval_render(x)
        lpdf = self.sumfuncObj.logpdf(img)
        if self.usePrior:
            lpdf_prior = self.forwardObj.eval_prior_lpdf(x)
            lpdf += lpdf_prior

        return lpdf, img

    def dU(self, x):
        x = Variable(x, requires_grad=True)
        if x.grad is not None:
            x.grad.zero_()
        lpdf, img = self.U(x)
        lpdf.backward()
        return x.grad

    def K(self, p): return 0.5 * (p * p).sum()

    def dK(self, p): return p

    def sample(self, N):
        para = th.tensor(self.forwardObj.para, dtype=th.float32, device=self.forwardObj.device)
        x0 = self.forwardObj.orig_to_norm(para)

        # print(x0)
        lpdf, img = self.U(x0)
        now = datetime.now(); print(now)
        if self.usePrior:
            print('0/%d, lpdf: %f, %f' % (N, lpdf.item(), self.forwardObj.eval_prior_lpdf(x0).item()))
        else:
            print('0/%d, lpdf: %f' % (N, lpdf.item()))

        xs = [self.forwardObj.norm_to_orig(x0).cpu().numpy()]
        lpdfs = [lpdf.item()]
        if self.to_save == 'fig':
            plotFigure(self.dir,
                np.vstack(lpdfs),
                xs,
                img.detach().cpu().numpy(),
                self.sumfuncObj.target.cpu().numpy(),
                1,
                0,
                0)
        time_start = time.time()
        while len(xs) < N+1:
            p0 = th.randn_like(x0)
            x, p = self.leapfrog(x0, p0)
            # print(x)
            if th.isfinite(x).all():
                lpdf0, _ = self.U(x0)
                H0 = lpdf0 + self.K(p0)
                lpdf, img = self.U(x)
                H  = lpdf + self.K(p)
                alpha = min(1, np.exp(H0.item() - H.item()))

                if np.random.rand() < alpha:
                    x0 = x
                    xs.append(self.forwardObj.norm_to_orig(x0).cpu().numpy())
                    lpdfs.append(lpdf.item())

                    if len(xs)%100==0:
                        now = datetime.now(); print(now)
                        if self.usePrior:
                            print('%d/%d, lpdf: %f, %f' % (len(xs), N, lpdf.item(), self.forwardObj.eval_prior_lpdf(x0).item()))
                        else:
                            print('%d/%d, lpdf: %f' % (len(xs), N, lpdf.item()))
                        print('rejection rate: %f'
                            % (self.num_reject/(self.num_reject + len(xs) - 1)))

                        xs_arr = np.vstack(xs)
                        lpdfs_arr = np.vstack(lpdfs)
                        id = np.vstack(np.arange(len(lpdfs))+1)
                        np.savetxt(os.path.join(self.dir, 'sample.csv'),
                            np.concatenate((id, lpdfs_arr, xs_arr), 1),
                            delimiter=",", fmt='%.3f')

                        if self.to_save == 'fig':
                            plotFigure(self.dir,
                                np.vstack(lpdfs),
                                xs,
                                img.detach().cpu().numpy(),
                                self.sumfuncObj.target.cpu().numpy(),
                                len(xs),
                                self.num_reject,
                                time.time()-time_start)
                else:
                    self.num_reject += 1
                    # print('rejection rate: %f'
                    #     % (self.num_reject/(self.num_reject + len(xs) - 1)))

            else:
                self.num_outBound += 1
                print('hmc::sample():NaN')

        if self.to_save == 'fig':
            png2gif(self.dir)
        else:
            plotFigure(self.dir,
                np.vstack(lpdfs),
                xs,
                img.detach().cpu().numpy(),
                self.sumfuncObj.target.cpu().numpy(),
                len(xs),
                self.num_reject,
                time.time()-time_start)

        return xs, lpdfs

class MALA:
    def __init__(self, forwardObj, sumfuncObj, args, usePrior=True):
        print("Initial Class hmc::MALA()")
        self.delta = args.lr
        self.xs = []
        self.lpdfs = []
        self.dir = args.out_dir
        self.num_reject = 0
        self.num_outBound = 0
        self.usePrior = usePrior
        self.to_save = args.to_save

        self.forwardObj = forwardObj
        self.sumfuncObj = sumfuncObj

        self.alpha = 0.9
        self.beta = 0.999
        self.c1 = args.diminish
        self.c2 = args.diminish


    def U(self, x):
        img = self.forwardObj.eval_render(x)
        lpdf = self.sumfuncObj.logpdf(img)
        if self.usePrior:
            lpdf_prior = self.forwardObj.eval_prior_lpdf(x)
            lpdf += lpdf_prior

        return lpdf, img

    def dU(self, x):
        x = Variable(x, requires_grad=True)
        if x.grad is not None:
            x.grad.zero_()
        lpdf, img = self.U(x)
        if lpdf != lpdf:
            print('lpdf:', lpdf)
            exit()
        lpdf.backward()
        return x.grad

    def sample(self, N):
        para = th.tensor(self.forwardObj.para, dtype=th.float32, device=self.forwardObj.device)
        x0 = self.forwardObj.orig_to_norm(para)
        G0 = 0
        d0 = 0
        t = 1

        # print(x0)
        lpdf, img = self.U(x0)
        now = datetime.now(); print(now)
        if self.usePrior:
            print('0/%d, lpdf: %f, %f' % (N, lpdf.item(), self.forwardObj.eval_prior_lpdf(x0).item()))
        else:
            print('0/%d, lpdf: %f' % (N, lpdf.item()))

        xs = [self.forwardObj.norm_to_orig(x0).cpu().numpy()]
        lpdfs = [lpdf.item()]
        if self.to_save == 'fig':
            plotFigure(self.dir,
                np.vstack(lpdfs),
                xs,
                img.detach().cpu().numpy(),
                self.sumfuncObj.target.cpu().numpy(),
                1,
                0,
                0)
        time_start = time.time()
        while len(xs) < N+1:
            ##### tmp
            g0 = self.dU(x0)

            G = self.beta * G0 + (1-self.beta) * g0 * g0
            d = self.alpha * d0 + (1-self.alpha) * g0

            M = 1/(epsilon + t**-self.c1 * G.clamp(min=epsilon).sqrt())
            m = t**-self.c2 * d + g0

            if len(xs)%100==0:
                print('G:', G)
                print('d:', d)
                print('M:', M)
                print('m:', m)

            mu = x0 - 0.5 * self.delta * M * m
            sigma2 = self.delta * M
            sigma = sigma2.clamp(min=epsilon).sqrt()
            x = mu - sigma * th.randn_like(mu)

            q0 = multivariate_normal.pdf(x.cpu().numpy(), mu.cpu().numpy(), sigma2.cpu().numpy(), allow_singular = True)

            ##### this
            g = self.dU(x)

            G_tmp = self.beta * G0 + (1-self.beta) * g * g
            d_tmp = self.alpha * d0 + (1-self.alpha) * g

            M = 1/(epsilon+ t**-self.c1 * G_tmp.clamp(min=epsilon).sqrt())
            m = t**-self.c2 * d_tmp + g

            mu = x - 0.5 * self.delta * M * m
            sigma2 = self.delta * M

            q = multivariate_normal.pdf(x0.cpu().numpy(), mu.cpu().numpy(), sigma2.cpu().numpy(), allow_singular = True)

            ##### #######
            lpdf0, _ = self.U(x0)
            lpdf, img = self.U(x)
            lpdf_diff = lpdf0.cpu().numpy()-lpdf.cpu().numpy()
            alpha = min(1, np.exp(np.clip(lpdf_diff, a_min=None, a_max=88)))# * q/(q0+epsilon))
            # print(lpdf0.cpu().numpy(), lpdf.cpu().numpy())
            # print(q0, q)
            # print(alpha)

            t += 1
            if np.random.rand() < alpha:
                G0 = G
                d0 = d
                x0 = x
                xs.append(self.forwardObj.norm_to_orig(x0).cpu().numpy())
                lpdfs.append(lpdf.item())

                if len(xs)%100==0:
                    now = datetime.now(); print(now)
                    if self.usePrior:
                        print('%d/%d, lpdf: %f, %f' % (len(xs), N, lpdf.item(), self.forwardObj.eval_prior_lpdf(x0).item()))
                    else:
                        print('%d/%d, lpdf: %f' % (len(xs), N, lpdf.item()))
                    print('rejection rate: %f'
                        % (self.num_reject/(self.num_reject + len(xs) - 1)))
                    # print(t**-self.c1)
                    # print('Sigma2:', sigma2.cpu().numpy())

                    xs_arr = np.vstack(xs)
                    lpdfs_arr = np.vstack(lpdfs)
                    id = np.vstack(np.arange(len(lpdfs))+1)
                    np.savetxt(os.path.join(self.dir, 'sample.csv'),
                        np.concatenate((id, lpdfs_arr, xs_arr), 1),
                        delimiter=",", fmt='%.3f')

                    if self.to_save == 'fig':
                        plotFigure(self.dir,
                            np.vstack(lpdfs),
                            xs,
                            img.detach().cpu().numpy(),
                            self.sumfuncObj.target.cpu().numpy(),
                            len(xs),
                            self.num_reject,
                            time.time()-time_start)
            else:
                self.num_reject += 1
                print('rejection rate: %f'
                    % (self.num_reject/(self.num_reject + len(xs) - 1)))

        if self.to_save == 'fig':
            png2gif(self.dir)
        else:
            plotFigure(self.dir,
                np.vstack(lpdfs),
                xs,
                img.detach().cpu().numpy(),
                self.sumfuncObj.target.cpu().numpy(),
                len(xs),
                self.num_reject,
                time.time()-time_start)

        return xs, lpdfs

def saveImage(x, n):
    Image.fromarray((x.cpu().numpy() * 255).astype(np.uint8)).save('tmp%03d.png' % n)

class HMC_Texture(HMC):
    def __init__(self, forwardObj, sumfuncObj, step_size, num_steps, usePrior=False):
        print("Initial Class hmc::HMC_Texture()")
        super().__init__(forwardObj, sumfuncObj, step_size, num_steps, usePrior)

    def sample(self, x0, N):
        saveImage(x0, 0)

        lpdf, _ = self.U(x0)
        print('\n0/%d, lpdf: %f' % (N, lpdf.item()))
        print('image mean and var:', self.sumfuncObj.img_mean.cpu().numpy(), self.sumfuncObj.img_var.cpu().numpy())

        num_accept = 0
        while num_accept < N:
            p0 = th.randn_like(x0)
            x, p = self.leapfrog(x0, p0)
            if th.isfinite(x).all():
                lpdf0, _ = self.U(x0)
                H0 = lpdf0 + self.K(p0)
                lpdf, _ = self.U(x)
                H = lpdf + self.K(p)
                alpha = min(1, np.exp(H0.item() - H.item()))
                if np.random.rand() < alpha:
                    x0 = x
                    num_accept += 1
                    if num_accept%100==0:
                        print('\n%d/%d, lpdf: %f' % (num_accept, N, lpdf.item()))
                        print('image mean and var:', self.sumfuncObj.img_mean.cpu().numpy(), self.sumfuncObj.img_var.cpu().numpy())
                        saveImage(x0, 1)
                else:
                    self.num_reject += 1
                    print('rejection rate: %f'
                        % (self.num_reject/(self.num_reject + num_accept)))
            else:
                self.num_outBound += 1
                print('hmc::sample():NaN')

class Optim_Texture(Optim):
    def __init__(self, forwardObj, sumfuncObj, lr, usePrior=True):
        print("Initial Class hmc::HMC_Texture()")
        super().__init__(forwardObj, sumfuncObj, lr, usePrior)

    def optim(self, x, N):
        x = th.tensor(x, dtype=th.float32, device=self.forwardObj.device)
        x = Variable(x, requires_grad=True)
        optimizer = th.optim.Adam([x], lr=self.lr)

        xs = []
        losses = []
        imTarget = self.sumfuncObj.target.cpu().numpy()
        for i in range(N):
            img = self.forwardObj.eval_render(x)
            loss = self.sumfuncObj.errloss(img)
            if self.usePrior:
                loss_laplacian = self.forwardObj.eval_img_laplacian(x)
                loss += loss_laplacian

            losses.append(loss.item())

            if i%10==0:
                print('%d/%d' % (i,N))
                plotFigureOptim(np.vstack(losses),
                    img.detach().cpu().numpy(),
                    imTarget)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return losses