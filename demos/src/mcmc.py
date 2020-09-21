from util import *
from gyplots import *
from scipy.stats import multivariate_normal

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

class MCMCsample:
    def __init__(self, forwardObj, sumfuncObj, args, usePrior):
        self.xs = []
        self.lpdfs = []
        self.num_accept = 0
        self.num_reject = 0
        self.dir = args.out_dir
        self.usePrior = usePrior
        self.to_save = args.to_save

        self.forwardObj = forwardObj
        self.sumfuncObj = sumfuncObj

    # def plot(self):
    #     if len(xs)%1000==0:
    #         now = datetime.now(); print(now)
    #         if self.usePrior:
    #             print('%d/%d, lpdf: %f, %f' % (len(xs), N, lpdf.item(), self.forwardObj.eval_prior_lpdf(x0).item()))
    #         else:
    #             print('%d/%d, lpdf: %f' % (len(xs), N, lpdf.item()))
    #         print('rejection rate: %f'
    #             % (self.num_reject/(self.num_reject + len(xs) - 1)))
    #         # print(t**-self.c1)
    #         # print('Sigma2:', sigma2.cpu().numpy())

    #         xs_arr = np.vstack(xs)
    #         lpdfs_arr = np.vstack(lpdfs)
    #         id = np.vstack(np.arange(len(lpdfs))+1)
    #         np.savetxt(os.path.join(self.dir, 'sample.csv'),
    #             np.concatenate((id, lpdfs_arr, xs_arr), 1),
    #             delimiter=",", fmt='%.3f')

    #         if self.to_save == 'fig':
    #             plotFigure(self.dir,
    #                 np.vstack(lpdfs),
    #                 xs,
    #                 img.detach().cpu().numpy(),
    #                 self.sumfuncObj.target.cpu().numpy(),
    #                 len(xs),
    #                 self.num_reject,
    #                 time.time()-time_start)


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

    def mcmc(self, x):
        return 0

    def sample(self, T):
        para = th.tensor(self.forwardObj.para, dtype=th.float32, device=self.forwardObj.device)
        x0 = self.forwardObj.orig_to_norm(para)
        xs = [self.forwardObj.norm_to_orig(x0).cpu().numpy()]

        lpdf, img = self.U(x0)
        lpdfs = [lpdf.item()]

        best_lpdf = lpdfs[-1]
        best_id = 0
        best_img = img
        best_para = xs[-1]

        initTime = time.time()
        while (time.time() - initTime) < T:
            self.N = self.num_accept + self.num_reject
            if self.N%100==0:
                print('accept:%d, reject:%d, accept rate:%.2f%%' % \
                    (self.num_accept, self.num_reject, self.num_accept*100/max(1,self.N)))
                print('Best parameters so far: ', best_para)

                xs_arr = np.vstack(xs)
                lpdfs_arr = np.vstack(lpdfs)
                id = np.vstack(np.arange(len(lpdfs)))
                np.savetxt(os.path.join(self.dir, 'sample.csv'),
                    np.concatenate((id, lpdfs_arr, xs_arr), 1),
                    delimiter=",", fmt='%.3f')
                gyArray2PIL(gyTensor2Array(best_img)).save(os.path.join(self.dir, 'bestfit_%d_%.3f.png' % (best_id, best_lpdf)))
                best_lpdf = 99999

            arej, x_tmp, lpdf, img = self.mcmc(x0)

            if np.random.rand() < min(1, arej):
                self.num_accept += 1
                x0 = x_tmp
                xs.append(self.forwardObj.norm_to_orig(x0).cpu().numpy())
                lpdfs.append(lpdf.item())
                if lpdfs[-1] < best_lpdf:
                    best_lpdf = lpdfs[-1]
                    best_id = len(lpdfs)-1
                    best_img = img
                    best_para = xs[-1]
            else:
                self.num_reject += 1

        print('accept:%d, reject:%d, accept rate:%.2f%%' % \
            (self.num_accept, self.num_reject, self.num_accept*100/max(1,self.N)))
        print('Best parameters so far: ', best_para)

        plotFigure(self.dir,
            np.vstack(lpdfs),
            xs,
            img.detach().cpu().numpy(),
            self.sumfuncObj.target.cpu().numpy(),
            len(xs),
            self.num_reject,
            time.time()-initTime)

        return xs, lpdfs

class MALA(MCMCsample):
    def __init__(self, forwardObj, sumfuncObj, args, usePrior=True):
        print("Initial Class hmc::MALA()")
        super().__init__(forwardObj, sumfuncObj, args, usePrior)
        self.alpha = 0.9
        self.beta = 0.999
        self.c1 = 0.25
        self.c2 = 0.25
        self.delta = args.lr
        self.V1 = 0
        self.V2 = 0

    def mcmc(self, x0):
        ##### tmp
        U_this, _ = self.U(x0)
        dU_this   = self.dU(x0)

        V1_this = self.alpha * self.V1 + (1 - self.alpha) * dU_this
        V2_this = self.beta  * self.V2 + (1 - self.beta)  * dU_this * dU_this

        M1 = max(eps, self.num_accept)**-self.c1 * V1_this + dU_this
        M2 = 1/(eps + max(eps, self.num_accept)**-self.c2 * V2_this.clamp(min=eps).sqrt())

        mu = x0 - 0.5 * self.delta * M1 * M2
        sigma2 = self.delta * M2
        sigma = sigma2.clamp(min=eps).sqrt()
        x_tmp = mu + sigma * th.randn_like(mu)
        # print('\nmu:')
        # print(mu)
        # print('\nsigma:')
        # print(sigma)
        # print('\nx_tmp:')
        # print(x_tmp)

        q_this = multivariate_normal.pdf(x_tmp.cpu().numpy(), mu.cpu().numpy(), sigma2.cpu().numpy(), allow_singular = True)
        # print('\nq_this:', q_this.item())

        ##### this
        U_tmp, img = self.U(x_tmp)
        dU_tmp     = self.dU(x_tmp)

        V1_tmp = self.alpha * self.V1 + (1 - self.alpha) * dU_tmp
        V2_tmp = self.beta  * self.V2 + (1 - self.beta)  * dU_tmp * dU_tmp

        M1 = max(eps, self.num_accept)**-self.c1 * V1_tmp + dU_tmp
        M2 = 1/(eps + max(eps, self.num_accept)**-self.c2 * V2_tmp.clamp(min=eps).sqrt())

        mu = x_tmp - 0.5 * self.delta * M1 * M2
        sigma2 = self.delta * M2
        q_tmp = multivariate_normal.pdf(x0.cpu().numpy(), mu.cpu().numpy(), sigma2.cpu().numpy(), allow_singular = True)

        # print('\nmu:')
        # print(mu)
        # print('\nsigma2:')
        # print(sigma2)
        # print('\nx0:')
        # print(x0)

        # print('\nq_tmp:', q_tmp.item())

        ##### #######
        arej = np.exp(np.clip((U_this-U_tmp).cpu().numpy(), a_min=None, a_max=88)) * q_tmp/(q_this+eps)
        print('\naref:', arej)

        self.V1 = V1_this
        self.V2 = V2_this

        return arej, x_tmp, U_tmp, img

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