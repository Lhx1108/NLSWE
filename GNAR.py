import numpy as np
from scipy import sparse

from sklearn.linear_model import Lasso
from concurrent.futures import ThreadPoolExecutor

def random_split_indices(N, m, seed=None):
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.permutation(N)
    return np.array_split(indices, m)

def zero_expand_matrix(A):
    T, N = A.shape
    
    if not isinstance(A, sparse.csr_matrix):
        A = sparse.csr_matrix(A)  # Convert to CSR if not already sparse
    
    row_indices, col_indices = A.nonzero()  # Get the indices of nonzero elements
    new_row_indices = row_indices + col_indices * T  # Adjust row indices for expansion
    data = A.data  # Keep the nonzero values

    # Create the sparse output matrix
    B = sparse.coo_matrix((data, (new_row_indices, col_indices)), shape=(T * N, N))

    return B.tocsr()  # Convert to CSR for efficient row operations


class NetworkModel:
    def __init__(self, alpha_order, beta_order, intercept=True, global_intercept=False, global_alpha=True, global_beta=True,nbhds=None):
        self.beta_order = beta_order
        self.intercept = intercept
        self.global_intercept = global_intercept
        self.global_alpha = global_alpha
        self.global_beta = global_beta
        self.nbhds = None if nbhds is None else np.asarray(nbhds)
        if self.nbhds is not None:
            if not self.global_alpha or not self.global_beta:
                raise ValueError("nbhds requires global_alpha=True and global_beta=True")
            self.nbhd_labels = np.unique(self.nbhds)
            self.n_nbhd = len(self.nbhd_labels)
        if isinstance(alpha_order,int):
            self.alpha_order = alpha_order
            self.alpha_orders = [1 for _ in range(alpha_order)]
        else:
            self.alpha_order = len(alpha_order)
            self.alpha_orders = alpha_order

    def transformVTS(self,vts):
        if self.nbhds is not None:
            if len(self.nbhds) != vts.shape[1]:
                raise ValueError("nbhds must have one entry for each node")
            n_time = vts.shape[0]-self.alpha_order
            nbhd_masks = np.column_stack([np.repeat(self.nbhds == label,n_time) for label in self.nbhd_labels])
        X = []
        for t in range(self.alpha_order):
            if self.beta_order[t] != 0:
                X_t_beta = np.transpose(vts[np.newaxis,self.alpha_order-t-1:-t-1,:]@self.network.w_mats[1:1+self.beta_order[t]],axes=(1,2,0))
                if self.global_beta:
                    if self.nbhds is None:
                        X_t_beta = X_t_beta.reshape((-1,self.beta_order[t]),order="F")
                    else:
                        X_t_beta = np.hstack([X_t_beta[:,:,i].reshape((-1,1),order="F")*nbhd_masks for i in range(self.beta_order[t])])
                    if not self.global_alpha:
                        X_t_beta = sparse.csr_matrix(X_t_beta)
                else:
                    X_t_beta = sparse.hstack([zero_expand_matrix(X_t_beta[:,:,i]) for i in range(self.beta_order[t])])
            else:
                X_t_beta = np.empty(((vts.shape[0]-self.alpha_order)*vts.shape[1],0))
                if (not self.global_alpha) or (not self.global_beta):
                    X_t_beta = sparse.csr_matrix(X_t_beta)
            if self.alpha_orders[t] == 1:
                X_t_alpha = vts[self.alpha_order-t-1:-t-1,:]
                if self.global_alpha:
                    X_t_alpha = X_t_alpha.reshape((-1,1),order="F")
                    if self.nbhds is not None:
                        X_t_alpha = X_t_alpha*nbhd_masks
                    if not self.global_beta:
                        X_t_alpha = sparse.csr_matrix(X_t_alpha)
                else:
                    X_t_alpha = zero_expand_matrix(X_t_alpha)
            else:
                X_t_alpha = np.empty(((vts.shape[0]-self.alpha_order)*vts.shape[1],0))
                if (not self.global_alpha) or (not self.global_beta):
                    X_t_alpha = sparse.csr_matrix(X_t_alpha)
            X.append(self.hstack((X_t_alpha,X_t_beta)))
        if self.intercept:
            if self.global_intercept:
                X_intercept = np.ones((self.network.size*(len(vts)-self.alpha_order),1))
            else:
                X_intercept = np.repeat(np.identity(self.network.size),len(vts)-self.alpha_order,axis=0)
            if not (self.global_alpha and self.global_beta):
                X_intercept = sparse.csr_matrix(X_intercept)
            X.append(X_intercept)
        X = self.hstack(X)
        return X

    def hstack(self,x):
        if self.global_alpha and self.global_beta:
            return np.hstack(x)
        else:
            return sparse.hstack(x)
    
    def return_coef(self,t,type="all"):
        if self.nbhds is not None:
            start = 0 if t == 1 else self.coef_index[2*t-3]
            alpha_end = self.coef_index[2*t-2]
            beta_end = self.coef_index[2*t-1]
            if type=="alpha":
                if alpha_end == start:
                    return 0
                return self.coefs[start:alpha_end]
            elif type=="beta":
                return self.coefs[alpha_end:beta_end]
            elif type=="all":
                return self.coefs[start:beta_end]
        else:
            if type=="alpha":
                if self.coef_index[2*t-2] != 0 and self.coef_index[2*t-2] != self.coef_index[2*t-3]:
                    return self.coefs[self.coef_index[2*t-2]-1]
                else:
                    return 0
            elif type=="beta":
                return self.coefs[self.coef_index[2*t-2]:self.coef_index[2*t-1]]
            elif type=="all":
                return self.coefs[self.coef_index[2*t-2]-1:self.coef_index[2*t-1]]

    def AIC(self):
        return 2*len(self.coefs)-2*self.loglik()
    def BIC(self):
        return len(self.coefs)*np.log(len(self.y))-2*self.loglik()
    def group_NTS(self, network, vts, size):
        self.network = network
        X = self.transformVTS(vts)
        T,n = len(vts),self.network.size
        self.k = int(np.ceil((T-self.alpha_order)/size))
        G = np.zeros(((T-self.alpha_order)*n,self.k*X.shape[1]))
        for i in range(self.k-1):
            for j in range(i+1):
                G[i*size*n:(i+1)*size*n,j*X.shape[1]:(j+1)*X.shape[1]] = X[i*size*n:(i+1)*size*n,:] 
        for j in range(self.k):
            G[(self.k-1)*size*n:,j*X.shape[1]:(j+1)*X.shape[1]] = X[(self.k-1)*size*n:,:]
        return G

from scipy.optimize import lsq_linear
from scipy.stats import multivariate_normal, norm
from itertools import chain, combinations


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def min_part(xs, r):
    xs = np.sort(xs)
    part = []
    ys = []
    for x in xs:
        for y in ys:
            if abs(x-y) >= r:
                part.append(ys)
                ys = []
                break
        ys.append(x)
    part.append(ys)
    return part
            

class GNAR(NetworkModel):
    def __init__(self, alpha_order, beta_order, intercept=True, global_intercept=False, global_alpha=True, global_beta=True,nbhds=None):
        super().__init__(alpha_order,beta_order,intercept,global_intercept,global_alpha,global_beta,nbhds=nbhds)
        mult = 1 if self.nbhds is None else self.n_nbhd
        self.coef_index = np.cumsum([[self.alpha_orders[i]*mult]+[beta_order[i]*mult] for i in range(self.alpha_order)])
        self.coef_order = self.coef_index[-1]
        
    def simulate(self, network, initial_vts, length, coefs, error_cov_mat):
        self.network = network
        l = len(initial_vts)
        vts_sim = np.zeros((length+l,self.network.size))
        vts_sim[:l,:] = initial_vts
        for i in range(length):
            vts_sim[l+i,:] = self.transformVTS(vts_sim[l+i-self.alpha_order:l+i+1,:])@coefs + multivariate_normal(
                cov=error_cov_mat).rvs(1)
        return vts_sim[len(initial_vts):]
        
    def fit(self, network, vts, use_ls=True, l2_penal=0):
        # the order of cols of vts must match the order of network nodes
        # i.e. the first vts col is the ts for the first node in the network, etc
        self.network = network
        self.vts = vts
        self.vts_end = vts[-self.alpha_order:,:]
        self.y = vts[self.alpha_order:,:].flatten("F")
        self.X = self.transformVTS(vts)
        if self.global_alpha:
            if l2_penal != 0:
                self.coefs = np.linalg.solve(self.X.T@self.X + l2_penal*np.identity(self.X.shape[1]), self.X.T@self.y)
            else:
                if use_ls:
                    self.coefs,self.res = np.linalg.lstsq(self.X, self.y, rcond=None)[:2]
                else:
                    self.coefs = np.linalg.solve(self.X.T@self.X, self.X.T@self.y)
        else:
            if use_ls:
                self.coefs = sparse.linalg.lsqr(self.X,self.y)[0]
            else:
                self.coefs = sparse.linalg.spsolve(self.X.T@self.X, self.X.T@self.y)
        self.vts_fitted = np.reshape((self.X@self.coefs),(-1,network.size),"F")
        #self.sigma2 = np.sum((self.y-self.X@self.coefs)**2)/(len(self.y)-len(self.coefs)) #assuming equal error var
        
    def predict(self, length, nodes=None, vts_end =None):
        if vts_end is None:
            vts_end = self.vts_end
        vts_pred = np.zeros((length+self.alpha_order,self.network.size))
        vts_pred[:self.alpha_order,:] = vts_end
        for i in range(length):
            vts_pred[self.alpha_order+i,:] = self.transformVTS(vts_pred[i:self.alpha_order+i+1,:])@self.coefs
        if nodes is None:
            return vts_pred[self.alpha_order:,:]
        else:
            return vts_pred[self.alpha_order:,nodes]

    def CV(self, network, vts, fold_n, seed, use_ls=True, random=True, full_random=False, output_errors=False):
        sse = 0
        n_val = 0
        self.fit(network,vts,use_ls=use_ls)
    
        T_eff = int(len(self.y)/self.network.size)
    
        if output_errors:
            errors_vec = np.zeros(len(self.y))
    
        if random:
            if full_random:
                fold_inds = random_split_indices(len(self.y),fold_n,seed)
            else:
                fold_inds = random_split_indices(T_eff,fold_n,seed)
        else:
            fold_inds = np.array_split(np.arange(T_eff),fold_n)
    
        if not full_random or not random:
            for i in range(fold_n):
                fold_inds[i] = np.concatenate([fold_inds[i] + int(n*T_eff) for n in range(self.network.size)])
    
        for i in range(fold_n):
            fold_ind = fold_inds[i]
            X_val = self.X[fold_ind]
    
            mask = np.ones(len(self.y),dtype=bool)
            mask[fold_ind] = False
    
            X_train = self.X[mask]
            y_train = self.y[mask]
    
            if self.global_alpha:
                if use_ls:
                    coefs = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
                else:
                    coefs = np.linalg.solve(X_train.T@X_train, X_train.T@y_train)
            else:
                if use_ls:
                    coefs = sparse.linalg.lsqr(X_train,y_train)[0]
                else:
                    coefs = sparse.linalg.spsolve(X_train.T@X_train,X_train.T@y_train)
    
            error = self.y[fold_ind] - X_val@coefs
    
            if output_errors:
                errors_vec[fold_ind] = error
            else:
                sse += np.sum(error**2)
                n_val += len(error)
    
        if output_errors:
            return errors_vec.reshape((T_eff,self.network.size),order="F")
        else:
            return sse/n_val

    def validate(self, vts):
        X = self.transformVTS(vts)
        y = vts[self.alpha_order:,:].flatten("F")
        return np.sum((y-X@self.coefs)**2)
            

def GNAR_sim_piecewise(networks, alpha_order, beta_order, coefs_list, intercept,global_intercept,error_cov_mat_list, length_list, burn_in = 100):
    num_piece = len(coefs_list)
    vts = np.zeros((sum(length_list)+burn_in+alpha_order,networks[0].size))
    vts[:alpha_order] = norm(loc=0).rvs(size=(alpha_order,networks[0].size))
    current_index = alpha_order
    length_list[0] += burn_in
    for l in range(num_piece):
        model = GNAR(alpha_order,beta_order,intercept, global_intercept)
        vts[current_index:current_index+length_list[l]]=model.simulate(networks[l],vts[:current_index],length_list[l],coefs_list[l],error_cov_mat_list[l])
        current_index = current_index+length_list[l]
    return vts[burn_in+alpha_order:]

def standardize(arr):
    """
    Standardize a 2D numpy array column-wise.
    Each column will have mean 0 and standard deviation 1.
    
    Parameters:
        arr (np.ndarray): 2D array to standardize.
        
    Returns:
        np.ndarray: Standardized 2D array.
    """
    arr = np.array(arr, dtype=float)  # ensure float for division
    means = np.mean(arr, axis=0)      # column means
    stds = np.std(arr, axis=0, ddof=0)  # column std deviations
    
    # Avoid division by zero
    stds[stds == 0] = 1
    
    standardized = (arr - means) / stds
    return standardized

import torch

class VARLasso:
    def __init__(
        self,
        p=1,
        alpha=1,
        fit_intercept=False,
        n_jobs=None,
        gpu=False,
        device=None,
        tol=1e-6,
    ):
        self.p = p
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.n_jobs = n_jobs
        self.gpu = gpu
        self.device = device
        self.tol = tol

    def _fit_one_cpu(self, X, y, max_iter):
        model = Lasso(
            self.alpha,
            max_iter=max_iter,
            fit_intercept=self.fit_intercept,
        ).fit(X, y)
        return model.coef_, model.intercept_

    @staticmethod
    def _soft_threshold_torch(Z, lam):
        return torch.sign(Z) * torch.clamp(torch.abs(Z) - lam, min=0.0)

    def _fit_gpu_lasso(self, X_np, Y_np, max_iter):
        if not torch.cuda.is_available():
            raise RuntimeError("gpu=True but CUDA is not available.")

        device = self.device if self.device is not None else "cuda"
        dtype = torch.float64 if X_np.dtype == np.float64 else torch.float32

        X = torch.as_tensor(X_np, dtype=dtype, device=device)
        Y = torch.as_tensor(Y_np, dtype=dtype, device=device)

        T_eff, P = X.shape
        N = Y.shape[1]

        if self.fit_intercept:
            X_mean = X.mean(0, keepdim=True)
            Y_mean = Y.mean(0, keepdim=True)
            Xc = X - X_mean
            Yc = Y - Y_mean
        else:
            X_mean = torch.zeros((1, P), dtype=dtype, device=device)
            Y_mean = torch.zeros((1, N), dtype=dtype, device=device)
            Xc = X
            Yc = Y

        Xt = Xc.T
        # objective matches sklearn scaling:
        # (1/(2*T_eff))||Y - XW||_F^2 + alpha * ||W||_1
        L = (torch.linalg.norm(Xc, ord=2) ** 2) / T_eff
        L = torch.clamp(L, min=torch.tensor(1e-12, dtype=dtype, device=device))

        W = torch.zeros((P, N), dtype=dtype, device=device)
        Z = W.clone()
        t = torch.tensor(1.0, dtype=dtype, device=device)

        prev_obj = None
        for it in range(max_iter):
            grad = (Xt @ (Xc @ Z - Yc)) / T_eff
            W_new = self._soft_threshold_torch(Z - grad / L, self.alpha / L)

            t_new = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * t * t))
            Z = W_new + ((t - 1.0) / t_new) * (W_new - W)

            if it % 50 == 0 or it == max_iter - 1:
                R = Yc - Xc @ W_new
                obj = 0.5 * torch.sum(R * R) / T_eff + self.alpha * torch.sum(torch.abs(W_new))
                obj_val = obj.item()
                if prev_obj is not None:
                    rel = abs(prev_obj - obj_val) / max(1.0, abs(prev_obj))
                    if rel < self.tol:
                        W = W_new
                        break
                prev_obj = obj_val

            W = W_new
            t = t_new

        if self.fit_intercept:
            b = (Y_mean - X_mean @ W).reshape(-1)
        else:
            b = torch.zeros(N, dtype=dtype, device=device)

        return W.detach().cpu().numpy(), b.detach().cpu().numpy()

    def fit(self, vts, max_iter=100000):
        N = vts.shape[1]
        X = np.hstack([vts[self.p - i:-i] for i in range(1, 1 + self.p)])
        Y = vts[self.p:]

        if self.alpha == 0:
            if self.fit_intercept:
                X_aug = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
                coef = np.linalg.lstsq(X_aug, Y, rcond=None)[0]
                self.W = coef[:-1]
                self.intercept_ = coef[-1]
            else:
                self.W = np.linalg.lstsq(X, Y, rcond=None)[0]
                self.intercept_ = np.zeros(N, dtype=X.dtype)
            return self

        if self.gpu:
            self.W, self.intercept_ = self._fit_gpu_lasso(X, Y, max_iter=max_iter)
            return self

        self.W = np.zeros((N * self.p, N), dtype=X.dtype)
        self.intercept_ = np.zeros(N, dtype=X.dtype)

        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            results = list(
                executor.map(
                    lambda j: self._fit_one_cpu(X, Y[:, j], max_iter),
                    range(N),
                )
            )

        for j, (coef_j, intercept_j) in enumerate(results):
            self.W[:, j] = coef_j
            self.intercept_[j] = intercept_j

        return self

    def CV(self, vts, fold_n,max_iter=100000, random=True,seed=None, output_errors=False):
        sse = 0
        N = vts.shape[1]
        X = np.hstack([vts[self.p - i:-i] for i in range(1, 1 + self.p)])
        y = vts[self.p:]

        if output_errors:
            errors = np.zeros_like(y)

        if random:
            fold_inds = random_split_indices(len(y),fold_n,seed)
        else:
            fold_inds = np.array_split(np.arange(len(y)),fold_n)

        for i in range(fold_n):
            fold_ind = fold_inds[i]
            X_val = X[fold_ind]
            mask = np.ones(len(y),dtype=bool)
            mask[fold_ind] = False
            X_train = X[mask]
            y_train = y[mask]

            if self.alpha == 0:
                if self.fit_intercept:
                    X_aug = np.hstack([X_train, np.ones((X_train.shape[0], 1), dtype=X.dtype)])
                    coef = np.linalg.lstsq(X_aug, y_train, rcond=None)[0]
                    W = coef[:-1]
                    intercept_ = coef[-1]
                else:
                    W = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
                    intercept_ = np.zeros(N, dtype=X.dtype)
            else:
                if not self.gpu:
                    W = np.zeros((N * self.p, N), dtype=X.dtype)
                    intercept_ = np.zeros(N, dtype=X.dtype)

                    with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                        results = list(
                            executor.map(
                                lambda j: self._fit_one_cpu(X_train, y_train[:, j], max_iter),
                                range(N),
                            )
                        )
                    for j, (coef_j, intercept_j) in enumerate(results):
                        W[:, j] = coef_j
                        intercept_[j] = intercept_j
                else:
                    W, intercept_ = self._fit_gpu_lasso(X_train, y_train, max_iter=max_iter)

            error = y[fold_ind] - X_val@W-intercept_

            if output_errors:
                errors[fold_ind] = error
            else:
                sse += np.mean(error**2)/fold_n

        if output_errors:
            return errors
        else:
            return sse

    def predict(self, vts):
        X = np.hstack([vts[self.p - 1:]] + [vts[self.p - i:-i + 1] for i in range(2, 1 + self.p)])
        return X @ self.W + self.intercept_

    def validate(self, vts):
        X = np.hstack([vts[self.p - i:-i] for i in range(1, 1 + self.p)])
        Y = vts[self.p:]
        Y_pred = X @ self.W + self.intercept_
        return np.sum((Y_pred - Y) ** 2)
            