import numpy as np
from concurrent.futures import ThreadPoolExecutor


def _proj_simplex_vec(v,z=1.0):
    v = np.asarray(v,dtype=float)
    n = v.size

    if n == 0:
        return v.copy()

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)-z
    ind = np.arange(1,n+1)
    cond = u-cssv/ind>0

    if not np.any(cond):
        return np.full(n,z/n)

    rho = ind[cond][-1]
    theta = cssv[cond][-1]/rho

    return np.maximum(v-theta,0.0)


def _proj_matrix_to_column_simplex(W):
    W = np.asarray(W,dtype=float).copy()
    N = W.shape[0]

    for j in range(N):
        idx = np.arange(N)!=j
        col = _proj_simplex_vec(W[idx,j],z=1.0)
        W[:,j] = 0.0
        W[idx,j] = col
        W[j,j] = 0.0

    return W


def _proj_simplex_on_support(v,support,z=1.0):
    v = np.asarray(v,dtype=float)
    support = np.asarray(support,dtype=bool)
    out = np.zeros_like(v)

    if support.sum() == 0:
        return out

    out[support] = _proj_simplex_vec(v[support],z=z)

    return out


def _threshold_matrix_end(W,tau,keep_at_least_one=True):
    W = np.asarray(W,dtype=float)
    N = W.shape[0]
    W_thr = np.zeros_like(W)

    for j in range(N):
        idx = np.arange(N)!=j
        col = W[idx,j].copy()
        support = col>=tau

        if keep_at_least_one and not np.any(support):
            support[np.argmax(col)] = True

        col_thr = _proj_simplex_on_support(col,support,z=1.0)
        W_thr[idx,j] = col_thr
        W_thr[j,j] = 0.0

    return W_thr


def random_split_indices(n,fold_n,seed=None):
    rng = np.random.default_rng(seed)
    inds = np.arange(n)
    rng.shuffle(inds)

    return np.array_split(inds,fold_n)


class GNARMM:
    def __init__(self,p=1,alpha=0.0,intercept=False,global_intercept=True):
        self.p = p
        self.alpha = alpha
        self.intercept = intercept
        self.global_intercept = global_intercept

        self.W = None
        self.beta = None
        self.losses = None
        self.intercept_ = None

        self._fit_cache = None
        self._fit_cache_key = None

    def _make_lags(self,vts):
        vts = np.asarray(vts,dtype=float)
        T,N = vts.shape

        if T <= self.p:
            raise ValueError("Need T > p.")

        Y = vts[self.p:]
        X_lags = [vts[self.p-(l+1):T-(l+1)] for l in range(self.p)]

        return X_lags,Y

    def _unpack_beta(self,beta,N):
        b = np.asarray(beta,dtype=float).reshape(-1)

        if b.size < 2*self.p:
            raise ValueError("beta has wrong length: need at least 2p entries.")

        a = b[:self.p]
        bb = b[self.p:2*self.p]
        c = None

        if self.intercept:
            if self.global_intercept:
                if b.size < 2*self.p+1:
                    raise ValueError("beta missing global intercept term.")
                c = float(b[2*self.p])
            else:
                if b.size < 2*self.p+N:
                    raise ValueError("beta missing node-specific intercept terms.")
                c = b[2*self.p:2*self.p+N]

        return a,bb,c

    def _predict_from_lags(self,X_lags,W,beta):
        T_eff,N = X_lags[0].shape
        a,bb,c = self._unpack_beta(beta,N)
        pred = np.zeros((T_eff,N),dtype=float)

        for l in range(self.p):
            pred += a[l]*X_lags[l]+bb[l]*(X_lags[l]@W)

        if self.intercept:
            if self.global_intercept:
                pred += c
            else:
                pred += c.reshape(1,-1)

        return pred

    def _data_key(self,vts):
        ptr = int(vts.__array_interface__["data"][0])

        return (vts.shape,vts.strides,ptr,self.p,self.intercept,self.global_intercept)

    def _prepare_fast_cache(self,X,Y):
        X = np.asarray(X,dtype=float)
        Y = np.asarray(Y,dtype=float)

        Tm1,N = X.shape

        XtX = (X.T@X)/Tm1
        XtY = (X.T@Y)/Tm1
        Y2 = float(np.sum(Y**2)/Tm1)

        sxx = float(np.trace(XtX))
        sxy = float(np.trace(XtY))

        idx_list = [np.arange(N)!=j for j in range(N)]
        G_list = [XtX[np.ix_(idx_list[j],idx_list[j])].copy() for j in range(N)]
        lammax_base = np.array([float(np.linalg.eigvalsh(G_list[j]).max()) for j in range(N)],dtype=float)

        return {"XtX":XtX,"XtY":XtY,"Y2":Y2,"sxx":sxx,"sxy":sxy,"idx_list":idx_list,"G_list":G_list,"lammax_base":lammax_base,"Tm1":Tm1,"N":N}

    def _fast_theta_ols(self,W,cache):
        XtX = cache["XtX"]
        XtY = cache["XtY"]

        XtXW = XtX@W

        sxz = float(np.sum(XtX*W.T))
        szz = float(np.sum(W*XtXW))
        szy = float(np.sum(W*XtY))

        G = np.array([[cache["sxx"],sxz],[sxz,szz]],dtype=float)
        h = np.array([cache["sxy"],szy],dtype=float)

        try:
            theta = np.linalg.solve(G,h)
        except np.linalg.LinAlgError:
            theta = np.linalg.lstsq(G,h,rcond=None)[0]

        return theta.reshape(-1,1)

    def _fast_loss(self,W,b,cache):
        a = float(b[0,0])
        beta = float(b[1,0])

        XtX = cache["XtX"]
        XtY = cache["XtY"]

        XtXW = XtX@W

        sxz = float(np.sum(XtX*W.T))
        szz = float(np.sum(W*XtXW))
        szy = float(np.sum(W*XtY))

        Q = cache["Y2"]+a*a*cache["sxx"]+2.0*a*beta*sxz+beta*beta*szz-2.0*a*cache["sxy"]-2.0*beta*szy
        pen = float(self.alpha)*float(np.sum(np.linalg.norm(W,axis=0)))

        return float(Q-pen)

    def _initialise_W(self,X,Y):
        Tm1,N = X.shape

        Ahat,*_ = np.linalg.lstsq(X,Y,rcond=None)

        alpha0 = float(np.trace(Ahat)/N)
        beta0 = float((np.sum(Ahat)-np.trace(Ahat))/N)

        if abs(beta0) < 1e-14:
            raise ValueError("VAR init produced beta^(0) ~ 0; cannot form Wtilde^(0).")

        Wtilde = (Ahat-alpha0*np.eye(N))/beta0
        np.fill_diagonal(Wtilde,0.0)

        return _proj_matrix_to_column_simplex(Wtilde)

    def _fit_XY_fast(self,X,Y,max_iter=500,rtol=1e-6,rtol_inner=1e-6,W=None,inner_max_iter=30,eps_norm=1e-10,epsilon=1e-8,max_workers=None,step_size_mult=1.0,show_progress=False,cache=None):
        X = np.asarray(X,dtype=float)
        Y = np.asarray(Y,dtype=float)

        Tm1,N = X.shape

        if Y.shape != (Tm1,N):
            raise ValueError("Y must have shape (T_eff, N) matching X.")

        if cache is None:
            cache = self._prepare_fast_cache(X,Y)

        if W is None:
            W = self._initialise_W(X,Y)
        else:
            W = np.asarray(W,dtype=float).copy()

            if W.shape != (N,N):
                raise ValueError("W must have shape (N, N).")

            np.fill_diagonal(W,0.0)
            W = _proj_matrix_to_column_simplex(W)

        idx_list = cache["idx_list"]
        G_list = cache["G_list"]
        lammax_base = cache["lammax_base"]
        XtX = cache["XtX"]
        XtY = cache["XtY"]

        if callable(step_size_mult):
            def mult_for(j,beta):
                mult = float(step_size_mult(j,np.array([beta]),lammax_base[j],epsilon))

                if mult <= 0:
                    raise ValueError("step_size_mult must be positive.")

                return mult
        else:
            mult_arr = np.asarray(step_size_mult,dtype=float)

            if mult_arr.ndim == 0:
                mult_const = float(mult_arr)

                if mult_const <= 0:
                    raise ValueError("step_size_mult must be positive.")

                def mult_for(j,beta):
                    return mult_const
            else:
                if mult_arr.size != N:
                    raise ValueError("step_size_mult array must have length N.")

                if np.any(mult_arr <= 0):
                    raise ValueError("All entries of step_size_mult must be positive.")

                def mult_for(j,beta):
                    return float(mult_arr[j])

        L_inner = int(inner_max_iter)

        if L_inner <= 0:
            raise ValueError("inner_max_iter must be a positive integer.")

        b0 = self._fast_theta_ols(W,cache)
        losses = [self._fast_loss(W,b0,cache)]

        def update_one_column(j,a,beta,U_col,W_col_init,eta):
            idx = idx_list[j]

            w = W_col_init[idx].copy()
            u = U_col[idx].copy()

            if abs(beta) < 1e-14:
                e = np.zeros_like(w)
                e[np.argmax(u)] = 1.0
                col = np.zeros(N)
                col[idx] = e
                col[j] = 0.0

                return j,col

            G = G_list[j]
            Atr = beta*(XtY[idx,j]-a*XtX[idx,j])
            beta2 = beta*beta

            for _ in range(L_inner):
                w_old = w.copy()
                g = 2.0*beta2*(G@w)-2.0*Atr-float(self.alpha)*u
                w = _proj_simplex_vec(w-eta*g,z=1.0)

                rel_change = np.linalg.norm(w-w_old,ord=2)/max(np.linalg.norm(w_old,ord=2),1e-12)

                if rel_change <= rtol_inner:
                    break

            col = np.zeros(N)
            col[idx] = w
            col[j] = 0.0

            return j,col

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(max_iter))
            except Exception:
                iterator = range(max_iter)
        else:
            iterator = range(max_iter)

        b_prev = None

        use_threads = max_workers != 1 and N > 1
        executor = ThreadPoolExecutor(max_workers=max_workers) if use_threads else None

        try:
            for it in iterator:
                W_old = W.copy()

                b = self._fast_theta_ols(W,cache)
                a = float(b[0,0])
                beta = float(b[1,0])

                if b_prev is None:
                    rel_theta = np.inf
                else:
                    rel_theta = np.linalg.norm(b-b_prev)/max(np.linalg.norm(b_prev),1e-12)

                eta_list = np.array([mult_for(j,beta)/(2.0*beta*beta*lammax_base[j]+epsilon) for j in range(N)],dtype=float)

                norms = np.linalg.norm(W,axis=0)
                U = W/np.maximum(norms,eps_norm)
                np.fill_diagonal(U,0.0)

                if executor is None:
                    cols = [update_one_column(j,a,beta,U[:,j],W[:,j],eta_list[j])[1] for j in range(N)]
                else:
                    futures = [executor.submit(update_one_column,j,a,beta,U[:,j],W[:,j],eta_list[j]) for j in range(N)]
                    cols = [None]*N

                    for fut in futures:
                        j,col = fut.result()
                        cols[j] = col

                W = np.column_stack(cols)
                np.fill_diagonal(W,0.0)

                losses.append(self._fast_loss(W,b,cache))

                rel_W = np.linalg.norm(W-W_old,ord="fro")/max(np.linalg.norm(W_old,ord="fro"),1e-12)
                rel_change = max(rel_W,rel_theta)

                if show_progress and hasattr(iterator,"set_postfix"):
                    iterator.set_postfix(rel_W=f"{rel_W:.2e}",rel_theta=f"{rel_theta:.2e}",LT=f"{losses[-1]:.3e}")

                b_prev = b.copy()

                if rel_change <= rtol:
                    break
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        b = self._fast_theta_ols(W,cache)

        return W,b,losses

    def _fit_XY_generic(self,X,Y,max_iter=500,rtol=1e-6,rtol_inner=1e-6,W=None,inner_max_iter=30,eps_norm=1e-10,epsilon=1e-8,max_workers=None,step_size_mult=1.0,show_progress=False):
        Y = np.asarray(Y,dtype=float)

        if self.p == 1:
            if isinstance(X,(list,tuple)):
                if len(X) != 1:
                    raise ValueError("For p=1, X must be an array (T-1,N) or a list/tuple of length 1.")
                X_lags = [np.asarray(X[0],float)]
            else:
                X_lags = [np.asarray(X,float)]
        else:
            if not isinstance(X,(list,tuple)) or len(X) != self.p:
                raise ValueError("For p>1, X must be a list/tuple of length p containing lag matrices.")
            X_lags = [np.asarray(Xl,float) for Xl in X]

        Tm1,N = X_lags[0].shape

        if Y.shape != (Tm1,N):
            raise ValueError("Y must have shape (T_eff, N) matching X lags.")

        for l in range(self.p):
            if X_lags[l].shape != (Tm1,N):
                raise ValueError("All X lag matrices must have the same shape (T_eff, N).")

        if Tm1 < 1:
            raise ValueError("Need at least one transition.")

        if W is None:
            Ahat,*_ = np.linalg.lstsq(X_lags[0],Y,rcond=None)
            alpha0 = float(np.trace(Ahat)/N)
            beta0 = float((np.sum(Ahat)-np.trace(Ahat))/N)

            if abs(beta0) < 1e-14:
                raise ValueError("VAR init produced beta^(0) ~ 0; cannot form Wtilde^(0).")

            Wtilde = (Ahat-alpha0*np.eye(N))/beta0
            np.fill_diagonal(Wtilde,0.0)
            W = _proj_matrix_to_column_simplex(Wtilde)
        else:
            W = np.asarray(W,dtype=float).copy()

            if W.shape != (N,N):
                raise ValueError("W must have shape (N, N).")

            np.fill_diagonal(W,0.0)
            W = _proj_matrix_to_column_simplex(W)

        Y_vec = Y.reshape(-1,1)
        X_vecs = [Xl.reshape(-1,1) for Xl in X_lags]

        def design_matrix(Wmat):
            XW_vecs = [(Xl@Wmat).reshape(-1,1) for Xl in X_lags]
            V = np.hstack(X_vecs+XW_vecs)

            if self.intercept:
                if self.global_intercept:
                    V = np.hstack([V,np.ones((Tm1*N,1))])
                else:
                    Z = np.kron(np.eye(N),np.ones((Tm1,1)))
                    V = np.hstack([V,Z])

            return V

        def theta_ols(Wmat):
            V = design_matrix(Wmat)
            b,*_ = np.linalg.lstsq(V,Y_vec,rcond=None)

            return b

        def unpack_theta(b):
            a = np.asarray(b[:self.p,0],float)
            bb = np.asarray(b[self.p:2*self.p,0],float)

            if not self.intercept:
                c = None
            elif self.global_intercept:
                c = float(b[2*self.p,0])
            else:
                c = np.asarray(b[2*self.p:2*self.p+N,0],float)

            return a,bb,c

        def residual_matrix(Wmat,b):
            a,bb,c = unpack_theta(b)
            pred = np.zeros_like(Y)

            for l in range(self.p):
                pred += a[l]*X_lags[l]+bb[l]*(X_lags[l]@Wmat)

            if self.intercept:
                if self.global_intercept:
                    pred += c
                else:
                    pred += c.reshape(1,-1)

            return Y-pred

        def loss(Wmat,b=None):
            if b is None:
                b = theta_ols(Wmat)

            R = residual_matrix(Wmat,b)
            Q = float(np.sum(R**2))/Tm1
            pen = float(self.alpha)*float(np.sum(np.linalg.norm(Wmat,axis=0)))

            return Q-pen

        losses = [loss(W)]

        idx_list = [np.arange(N)!=j for j in range(N)]
        A_list = [[X_lags[l][:,idx_list[j]] for l in range(self.p)] for j in range(N)]

        G_list = []
        lammax_base = np.zeros(N,dtype=float)

        for j in range(N):
            Gj = [[None]*self.p for _ in range(self.p)]

            for l in range(self.p):
                Al = A_list[j][l]

                for m in range(self.p):
                    Am = A_list[j][m]
                    Gj[l][m] = (Al.T@Am)/Tm1

            G_list.append(Gj)

            if self.p == 1:
                lammax_base[j] = float(np.linalg.eigvalsh(Gj[0][0]).max())
            else:
                Gblock = np.block([[Gj[l][m] for m in range(self.p)] for l in range(self.p)])
                lammax_base[j] = float(np.linalg.eigvalsh(Gblock).max())

        if callable(step_size_mult):
            def mult_for(j,beta_vec):
                mult = float(step_size_mult(j,beta_vec,lammax_base[j],epsilon))

                if mult <= 0:
                    raise ValueError("step_size_mult must be positive.")

                return mult
        else:
            mult_arr = np.asarray(step_size_mult,dtype=float)

            if mult_arr.ndim == 0:
                mult_const = float(mult_arr)

                if mult_const <= 0:
                    raise ValueError("step_size_mult must be positive.")

                def mult_for(j,beta_vec):
                    return mult_const
            else:
                if mult_arr.size != N:
                    raise ValueError("step_size_mult array must have length N.")

                if np.any(mult_arr <= 0):
                    raise ValueError("All entries of step_size_mult must be positive.")

                def mult_for(j,beta_vec):
                    return float(mult_arr[j])

        if self.p == 1:
            def eta_for(j,beta_vec):
                return mult_for(j,beta_vec)/(2.0*beta_vec[0]**2*lammax_base[j]+epsilon)
        else:
            def eta_for(j,beta_vec):
                beta_sq = float(np.sum(np.asarray(beta_vec,float)**2))

                return mult_for(j,beta_vec)/(2.0*beta_sq*lammax_base[j]+epsilon)

        L_inner = int(inner_max_iter)

        if L_inner <= 0:
            raise ValueError("inner_max_iter must be a positive integer.")

        def update_one_column(j,a_vec,b_vec,c,U_col,W_col_init,eta):
            idx = idx_list[j]
            w = W_col_init[idx].copy()
            u = U_col[idx].copy()

            r = Y[:,j].copy()

            for l in range(self.p):
                r -= a_vec[l]*X_lags[l][:,j]

            if self.intercept:
                if self.global_intercept:
                    r -= c
                else:
                    r -= c[j]

            Gj = G_list[j]
            AtA = np.zeros((N-1,N-1),dtype=float)

            for l in range(self.p):
                for m in range(self.p):
                    AtA += b_vec[l]*b_vec[m]*Gj[l][m]

            Atr = np.zeros(N-1,dtype=float)

            for l in range(self.p):
                Atr += b_vec[l]*((A_list[j][l].T@r)/Tm1)

            if np.max(np.abs(b_vec)) < 1e-14:
                e = np.zeros_like(w)
                e[np.argmax(u)] = 1.0
                col = np.zeros(N)
                col[idx] = e
                col[j] = 0.0

                return j,col

            for _ in range(L_inner):
                w_old = w.copy()
                g = 2.0*(AtA@w)-2.0*Atr-float(self.alpha)*u
                w = _proj_simplex_vec(w-eta*g,z=1.0)
                rel_change = np.linalg.norm(w-w_old,ord=2)/max(np.linalg.norm(w_old,ord=2),1e-12)

                if rel_change <= rtol_inner:
                    break

            col = np.zeros(N)
            col[idx] = w
            col[j] = 0.0

            return j,col

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(max_iter))
            except Exception:
                iterator = range(max_iter)
        else:
            iterator = range(max_iter)

        b_prev = None

        use_threads = max_workers != 1 and N > 1
        executor = ThreadPoolExecutor(max_workers=max_workers) if use_threads else None

        try:
            for it in iterator:
                W_old = W.copy()

                b = theta_ols(W)
                a_vec,b_vec,c = unpack_theta(b)

                if b_prev is None:
                    rel_theta = np.inf
                else:
                    rel_theta = np.linalg.norm(b-b_prev)/max(np.linalg.norm(b_prev),1e-12)

                eta_list = np.array([eta_for(j,b_vec) for j in range(N)],dtype=float)

                norms = np.linalg.norm(W,axis=0)
                U = W/np.maximum(norms,eps_norm)
                np.fill_diagonal(U,0.0)

                if executor is None:
                    cols = [update_one_column(j,a_vec,b_vec,c,U[:,j],W[:,j],eta_list[j])[1] for j in range(N)]
                else:
                    futures = [executor.submit(update_one_column,j,a_vec,b_vec,c,U[:,j],W[:,j],eta_list[j]) for j in range(N)]
                    cols = [None]*N

                    for fut in futures:
                        j,col = fut.result()
                        cols[j] = col

                W = np.column_stack(cols)
                np.fill_diagonal(W,0.0)

                losses.append(loss(W,b=b))

                rel_W = np.linalg.norm(W-W_old,ord="fro")/max(np.linalg.norm(W_old,ord="fro"),1e-12)
                rel_change = max(rel_W,rel_theta)

                if show_progress and hasattr(iterator,"set_postfix"):
                    iterator.set_postfix(rel_W=f"{rel_W:.2e}",rel_theta=f"{rel_theta:.2e}",LT=f"{losses[-1]:.3e}")

                b_prev = b.copy()

                if rel_change <= rtol:
                    break
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

        b = theta_ols(W)

        return W,b,losses

    def _fit_XY(self,X,Y,max_iter=500,rtol=1e-6,rtol_inner=1e-6,W=None,inner_max_iter=30,eps_norm=1e-10,epsilon=1e-8,max_workers=None,step_size_mult=1.0,show_progress=False,cache=None):
        if self.p == 1 and not self.intercept:
            return self._fit_XY_fast(X,Y,max_iter=max_iter,rtol=rtol,rtol_inner=rtol_inner,W=W,inner_max_iter=inner_max_iter,eps_norm=eps_norm,epsilon=epsilon,
                                     max_workers=max_workers,step_size_mult=step_size_mult,show_progress=show_progress,cache=cache)

        return self._fit_XY_generic(X,Y,max_iter=max_iter,rtol=rtol,rtol_inner=rtol_inner,W=W,inner_max_iter=inner_max_iter,eps_norm=eps_norm,
                                    epsilon=epsilon,max_workers=max_workers,step_size_mult=step_size_mult,show_progress=show_progress)

    def fit(self,vts,max_iter=500,rtol=1e-6,rtol_inner=1e-6,W=None,inner_max_iter=30,eps_norm=1e-10,epsilon=1e-8,max_workers=None,threshold=None,
            refit_beta_after_threshold=True,step_size_mult=1.0,show_progress=False,warm_start=False,append_losses=True):
        vts = np.asarray(vts,dtype=float)

        X_lags,Y = self._make_lags(vts)
        X_core = X_lags if self.p > 1 else X_lags[0]

        if warm_start:
            if W is not None:
                raise ValueError("Use either warm_start=True or provide W, not both.")

            if self.W is None:
                raise RuntimeError("warm_start=True requires a previously fitted self.W.")

            W_init = self.W.copy()
            old_losses = self.losses if append_losses and self.losses is not None else None
        else:
            W_init = W
            old_losses = None

        cache = None

        if self.p == 1 and not self.intercept:
            key = self._data_key(vts)

            if warm_start and self._fit_cache is not None and self._fit_cache_key == key:
                cache = self._fit_cache
            else:
                cache = self._prepare_fast_cache(X_core,Y)
                self._fit_cache = cache
                self._fit_cache_key = key

        W_hat,beta_hat,losses = self._fit_XY(X_core,Y,max_iter=max_iter,rtol=rtol,rtol_inner=rtol_inner,W=W_init,inner_max_iter=inner_max_iter,eps_norm=eps_norm,
                                             epsilon=epsilon,max_workers=max_workers,step_size_mult=step_size_mult,show_progress=show_progress,cache=cache)

        if threshold is not None:
            W_hat = _threshold_matrix_end(W_hat,tau=float(threshold),keep_at_least_one=True)

            if refit_beta_after_threshold:
                W_hat,beta_hat,_ = self._fit_XY(X_core,Y,max_iter=0,rtol=rtol,rtol_inner=rtol_inner,W=W_hat,inner_max_iter=inner_max_iter,eps_norm=eps_norm,
                                                epsilon=epsilon,max_workers=max_workers,step_size_mult=step_size_mult,show_progress=False,cache=cache)

        self.W = W_hat
        self.beta = beta_hat

        if old_losses is not None:
            self.losses = list(old_losses)+list(losses[1:])
        else:
            self.losses = losses

        b = np.asarray(beta_hat,dtype=float).reshape(-1)
        N = vts.shape[1]

        if self.intercept:
            if self.global_intercept:
                self.intercept_ = float(b[2*self.p])
            else:
                self.intercept_ = b[2*self.p:2*self.p+N]
        else:
            self.intercept_ = 0.0

        return self

    def predict(self,vts):
        if self.W is None or self.beta is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        X_lags,_ = self._make_lags(vts)

        return self._predict_from_lags(X_lags,self.W,self.beta)

    def test(self,vts):
        if self.W is None or self.beta is None:
            raise RuntimeError("Model must be fitted before calling test().")

        X_lags,Y = self._make_lags(vts)
        Y_pred = self._predict_from_lags(X_lags,self.W,self.beta)

        return float(np.sum((Y-Y_pred)**2))

    def CV(self,vts,fold_n,max_iter=500,rtol=1e-6,rtol_inner=1e-6,W=None,inner_max_iter=30,eps_norm=1e-10,epsilon=1e-8,max_workers=None,threshold=None,
           refit_beta_after_threshold=True,step_size_mult=1.0,random=True,seed=None,show_progress=False):
        vts = np.asarray(vts,dtype=float)

        X_lags_all,Y_all = self._make_lags(vts)
        T_eff,N = Y_all.shape

        if fold_n < 2 or fold_n > T_eff:
            raise ValueError("fold_n must satisfy 2 <= fold_n <= T-p.")

        if random:
            fold_inds = random_split_indices(T_eff,fold_n,seed)
        else:
            fold_inds = np.array_split(np.arange(T_eff),fold_n)

        sse = 0.0

        for i in range(fold_n):
            fold_ind = fold_inds[i]

            mask = np.ones(T_eff,dtype=bool)
            mask[fold_ind] = False

            Y_train = Y_all[mask]
            Y_val = Y_all[fold_ind]

            X_train = [Xl[mask] for Xl in X_lags_all]
            X_val = [Xl[fold_ind] for Xl in X_lags_all]

            X_core_train = X_train if self.p > 1 else X_train[0]

            if self.p == 1 and not self.intercept:
                cache = self._prepare_fast_cache(X_core_train,Y_train)
            else:
                cache = None

            W_hat,beta_hat,_ = self._fit_XY(X_core_train,Y_train,max_iter=max_iter,rtol=rtol,rtol_inner=rtol_inner,W=W,inner_max_iter=inner_max_iter,
                                            eps_norm=eps_norm,epsilon=epsilon,max_workers=max_workers,step_size_mult=step_size_mult,show_progress=show_progress,cache=cache)

            if threshold is not None:
                W_hat = _threshold_matrix_end(W_hat,tau=float(threshold),keep_at_least_one=True)

                if refit_beta_after_threshold:
                    W_hat,beta_hat,_ = self._fit_XY(X_core_train,Y_train,max_iter=0,rtol=rtol,rtol_inner=rtol_inner,W=W_hat,inner_max_iter=inner_max_iter,
                                                    eps_norm=eps_norm,epsilon=epsilon,max_workers=max_workers,step_size_mult=step_size_mult,show_progress=False,cache=cache)

            Y_pred = self._predict_from_lags(X_val,W_hat,beta_hat)
            sse += np.mean((Y_val-Y_pred)**2)/fold_n

        return float(sse)