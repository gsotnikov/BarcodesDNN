import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F

from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
import sys
import os
from time import time
import numpy as np
import pandas as pd

from functools import reduce
from itertools import combinations
import operator
from scipy.optimize import minimize

from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Custom imports
from resnet import ResNet, MultipleShallowNets
from train_functions import make_loaders

class RegularizationL2:
    def __init__(self, alpha, device):
        self.alpha = alpha
        self.device = device
        
    def __call__(self, model):
        
        alpha = torch.tensor(self.alpha, requires_grad=False).to(self.device)
        l2_reg = torch.tensor(0., requires_grad=False).to(self.device)
        
        for param in model.parameters():
            l2_reg += alpha * torch.norm(param)
            
        return l2_reg


class ComputeLoss:
    def __init__(self, model, model_params, dataset, bs=256, part='train', device='cuda:0'):
        '''
        Main contribution of this paper: Class to find optimal path between minimas of a loss function.
        '''
        
        self.ds = dataset
        self.bs = bs
        self.part = part
        self.device = device
        self.basic_net = model
        self.mp = model_params
        
        self.regularization = self.mp['regularization']
        self.regularizer = RegularizationL2(self.regularization, self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        
    def make_wide_net(self, thetas):
        shallow_nets = []
        for index, theta in enumerate(thetas):            
            # self.basic_net
            net = self.basic_net()
            
            if index==0:
                self.layer_dict = net.layer_dict

            net.net = self.set_weights(net.net, theta)
            shallow_nets.append(net)
            
        wide_net = MultipleShallowNets(shallow_nets)
        wide_net.num_of_basic = len(thetas)
        
        return wide_net.to(self.device)
        
    def init_loaders(self, seed):
        """
        Returns train/test loader
        """
        if self.part == 'train':
            train, _ = make_loaders(seed, dataset=self.ds, batch_size=self.bs)
            return train
        elif self.part == 'test':
            _, test = make_loaders(seed, dataset=self.ds, batch_size=self.bs)
            return test
               
    
    def evaluate_on_batch(self, batch, label, model):
        """
        input:
        batch, label - from dataloader
        model - wide net
        N - number of basic nets in wide net
        
        returns losses and gradients
        """
        loss_thetas, gradients_thetas = [], []


        pred_student = model(batch.to(self.device))
        Pred_student = torch.cat(pred_student.split(10, dim=1))
        
        Labels = torch.cat([label for _ in range(model.num_of_basic)]).to(self.device)
        
        Loss = self.criterion(Pred_student, Labels) * model.num_of_basic
        
        if self.regularization > 0:
            Loss += self.regularizer(model)
        
        Loss.backward()
        
        for index in range(model.num_of_basic):
            model_gradient_batch = self.unfold_gradients(model, index)
            gradients_thetas.append(model_gradient_batch)

        model.zero_grad()

        # list of losses, list of N - (|Thetas| x 1) gradients
        return np.array(loss_thetas).reshape(-1, 1), np.array(gradients_thetas)[:, :, 0]
    

    def evaluate(self, model, gradients=True, accuracy=False, mode=1, dataset_ratio=1):
        """
        input:
        model - desired wide network
        N - number of basic models in a wide network
        
        output:
        gradients & losses w.r.t. Thetas of the model
        """
        loss_thetas, gradients_thetas, acc_thetas, acc_thetas_relative = [], [], [], []

        loader = self.init_loaders(26)
        for idx, (batch, label) in enumerate(loader):
            if idx/len(loader) < dataset_ratio:
            
                # Lists to store losses and gradients on each batch
                loss_thetas_batch, gradients_thetas_batch = [], []

                pred_student = model(batch.to(self.device))
                Pred_student = torch.cat(pred_student.split(10, dim=1))

                Labels = torch.cat([label for _ in range(model.num_of_basic)]).to(self.device)

                Loss = self.criterion(Pred_student, Labels) * model.num_of_basic   

                for point in range(model.num_of_basic):  

                    loss = self.criterion(Pred_student[self.bs * point: self.bs * (point+1)],
                                          Labels[self.bs * point: self.bs * (point+1)]).item()


                    if self.regularization > 0:
                        loss += self.regularizer(model[point]).item()

                    loss_thetas_batch.append(loss)


                if gradients:
                    Loss = self.criterion(Pred_student, Labels) * model.num_of_basic

                    acc = (Pred_student.cpu().detach().argmax(axis=1) == Labels.cpu()).float().split(Labels.shape[0]//model.num_of_basic, dim=0)

                    acc_thetas.append([torch.mean(point) for point in acc])

                    Loss.backward()

                    for index in range(model.num_of_basic):
                        model_gradient_batch = self.unfold_gradients(model, index)
                        gradients_thetas_batch.append(model_gradient_batch)

                    model.zero_grad()

                if idx/len(loader) < mode:
                    # Append lists of losses and gradient after a batch to a global pool
                    loss_thetas.append(loss_thetas_batch)
                    gradients_thetas.append(gradients_thetas_batch)
                    
        
        loss_thetas = np.array(loss_thetas).mean(axis=0).reshape(-1, 1) # (Number of models x 1)
        
        if gradients:
            print(np.array(gradients_thetas).shape)
            gradients_thetas = np.array(gradients_thetas).mean(axis=0)[:, :, 0] # (Number of models x Number of parameters)
            if accuracy:
                return loss_thetas, gradients_thetas, np.vstack(acc_thetas).mean(axis=0)

            return loss_thetas, gradients_thetas # Loss, Gradient
        
        return loss_thetas
        
    def unfold_gradients(self, network, network_number):
        """
        network - probably a wide_network
        
        output: vector (|Thetas x 1| - of gradients w.r.t. every parameter)
        """
        N = network.num_of_basic
        network = network.net
        
        parameter_folding = []
        
        for index, (name, par) in enumerate(network.named_parameters()):
            grad_width = par.shape[0]//N
            group_range = [network_number * grad_width, (network_number + 1) * grad_width]
            grad = par.grad[group_range[0] : group_range[1]]                
            flatten_par = grad.reshape(-1)

            parameter_folding.append(flatten_par)
        
        return torch.cat(parameter_folding).reshape(-1, 1).cpu().numpy()
    
    def set_weights(self, net, theta):
        pos = 0
        for _, p in net.named_parameters():
            dim = reduce(operator.mul, p.size(), 1)
            p.data = torch.tensor(
                theta[
                    pos:pos+dim
                ], dtype=torch.float32
            ).reshape(p.size())
            pos += dim
        return net    
    
    def parametrize_vector(self, theta1, theta2, num_points):
        """
        input:
        theta1, theta2 (string) - path to file with parameters
        num_points - number of points to be sampled on a vector
        
        output:
        list of points which lay on a high-dimensional vector (theta1, theta2)
        """
        dot1 = self.basic_net()
        dot1.load_state_dict(torch.load(theta1, map_location=self.device))

        dot2 = self.basic_net()
        dot2.load_state_dict(torch.load(theta2, map_location=self.device))

        theta1, theta2 = dot1.flatten_parameters(), dot2.flatten_parameters()

        delta = theta2 - theta1
        space = [theta1 + alpha * delta for alpha in np.linspace(0,1, num_points)]

        outer_borders = True
        if outer_borders:
            self.left_space = [theta1 + alpha * delta for alpha in np.linspace(-0.1, 0, num_points//10)]
            self.right_space = [theta1 + alpha * delta for alpha in np.linspace(1, 1.1, num_points//10)]
            self.loss_on_borders = 0

        return space
    
    def straight_path(self, theta1, theta2, num_points, N):
        """
        Calculates value of a loss function which belong to a vector (theta1, theta2)
        
        theta1, theta2 (string) - start and end of the path
        num_points - number of points along the path which would be uniformly distributed
        N - number of points to be computed together in a WideShallowNet
        """
        s = time()

        points = self.parametrize_vector(theta1, theta2, num_points)
        bins = num_points//N 
        
        # Points are splited into parts: 1.(p1, p2,..., pN), ..., bins.(p1, p2,..., pN)
        # A wide net is applied on top of each group
        loss_on_vector, grad_on_vector = [], []
        
        for b in range(bins):            
            current_group_of_points = points[b * N: (b + 1) * N]            
            wide_net = self.make_wide_net(current_group_of_points)
            
            loss_on_points, gradient_on_points = self.evaluate(wide_net, mode=0.005)
            
            # Store loss across all Thetas
            loss_on_vector.append(loss_on_points.tolist())
            
            # Store gradient across all Thetas
            grad_on_vector.append(gradient_on_points)
        
        # post processing grad_on_vector (|Theta|, Num_points)
        grad_on_vector_ = []
        for batch in grad_on_vector:
            for theta in batch:
                grad_on_vector_.append(theta)
        
        print(f'Path found in {np.round((time() - s)/60, 2)} minutes') 

        #(Array - Num of pts x 1), (Array - Num of pts x Num of params), Thetas (Array - Num of pts x Num of params)
        return np.array(loss_on_vector[0]).reshape(-1,1), np.vstack(grad_on_vector_), np.hstack(points).T 
    
    def projection(self, start, end, alpha):
        stds = np.hstack([start.reshape(-1, 1), end.reshape(-1, 1)]).std(axis=1)
        stds = stds/stds.sum()

        def piecewise_linear(x):
            return 0.5 * x + 0.5

        diff = end - start
        center = start + diff/2
        
        length = np.linalg.norm(diff)
        normed_diff = diff/length

        multiplier = length * alpha / piecewise_linear(stds)
        v = np.random.randn(start.shape[0]) * multiplier 

        projection = normed_diff * (normed_diff @ v)
        orthogonal = v - projection
        
        return orthogonal + center


    def stochastic_path(self, start, end, alpha, total_iters):
        alpha, decay = alpha['alpha'], alpha['alpha_decay']

        dot1 = self.basic_net()
        dot2 = self.basic_net()
        
        dot1.load_state_dict(torch.load(start, map_location=self.device))
        dot2.load_state_dict(torch.load(end, map_location=self.device))

        start, end = np.squeeze(dot1.flatten_parameters()), np.squeeze(dot2.flatten_parameters())

        points = [start, end]
        for epoch in range(total_iters):  
            centers, cache = [], []
            pairs = [[points[i], points[i + 1]] for i in range(len(points) - 1)]

            for p in pairs: 
                center = self.projection(p[0], p[1], alpha)
                centers.append(center)

            for index, p in enumerate(points[:-1]):
                cache.extend([p, centers[index]])
            cache.append(points[-1])
            points = cache.copy()
            
            alpha = alpha * decay
            
        stochastic_thetas = np.vstack(points)

        wide_net = self.make_wide_net(stochastic_thetas)
        loss_on_points, gradient_on_points = self.evaluate(wide_net)

        return loss_on_points, gradient_on_points, stochastic_thetas


    def chain_distance(self, thetas):
        l2 = np.sqrt(np.sum(np.square(np.diff(thetas, axis=0)), axis=1))
        return l2

    def flatten_theta(self, theta):
        """
        input:
        theta (string) - path to file with parameters
        
        output:
        vector (-1, 1) - flattened parameters of a particular net
        """
        
        flatten_params = []
        params = torch.load(theta, map_location=self.device)
        
        for values in params:
            if 'net' in values:
                flatten_params.append(params[values].cpu().reshape(-1, 1))
        
        return np.vstack(flatten_params)
    
    def normalize_vectors(self, v1, v2):
        diff = v2 - v1
        return diff/np.linalg.norm(diff)

    def scalar_product(self, gradient, normalized_vector):
        return np.dot(gradient.reshape(1,-1), normalized_vector.reshape(-1,1))

    def correction(self, gradient, v1, vector, v2):
        normalized_left = self.normalize_vectors(v1, vector)
        normalized_right = self.normalize_vectors(vector, v2)

        term1 = self.scalar_product(gradient, normalized_left) * normalized_left
        term2 = self.scalar_product(gradient, normalized_right) * normalized_right

        orthogonal_gradient = gradient - 0.5 * (term1 + term2)
        return orthogonal_gradient.reshape(1, -1), (term1 + term2).reshape(1, -1)

    def corrected_gradients(self, gradient, thetas, orthogonal=False, memory_block=None, lr=0):
        updated_grads = []
        orthogonal_grads = []
        
        beta = 0.5
        
        def stepwise(x, q=0.55):        
            u, l = np.quantile(x, q), np.quantile(x, 1-q)
            return np.minimum(np.maximum(x, l), u)
                
        for T in range(1, thetas.shape[0] - 1):
                        
            # Since that moment, memory_block - index
            accumulated_gradient = beta * self.gradient_cache[memory_block][T].reshape(-1) + lr * gradient[T, :].reshape(-1)
            
            if True:
                orthogonal_gradient, projections = self.correction(lr * gradient[T, :].reshape(-1),
                                                                   thetas[T-1, :], thetas[T, :], thetas[T+1, :])
            else:
                orthogonal_gradient, projections = self.correction(self.gradient_cache[memory_block][T],
                                                   thetas[T-1, :], thetas[T, :], thetas[T+1, :])

            updated_grads.append(orthogonal_gradient)
            orthogonal_grads.append(projections)            
        
        updated_grads = np.vstack(updated_grads)

        if orthogonal:
            orthogonal_grads = np.vstack(orthogonal_grads)
            return orthogonal_grads

        return updated_grads
    
    def make_step(self, thetas, corrected):
        thetas[1:-1, :] = thetas[1:-1, :].astype(np.float32) - corrected[:, :].astype(np.float32)
        
        return thetas
    
    def plot_pca(self, path):
        fig = plt.figure(figsize=(18, 8))
        ax2d = fig.add_subplot(121)
        ax3d = fig.add_subplot(122, projection='3d')
                
        pca = PCA(n_components=2)
        path_2d = pca.fit_transform(path)
        ax2d.plot(path_2d[:, 0], path_2d[:, 1], marker='*')
        ax2d.set_title('Explained variance' + str(pca.explained_variance_ratio_))

        pca = PCA(n_components=3)
        path_3d = pca.fit_transform(path)
        ax3d.plot(path_3d[:, 0],path_3d[:, 1],path_3d[:, 2], marker='*')
        ax3d.set_title('Explained variance' + str(pca.explained_variance_ratio_))

        plt.show()


    def angles(self, thetas):
        """
        Pairwise computation of angles along the given path 
        Input: Thetas (num_models x num_parameters)
        Output: Angles in degrees (num_models -2 x 1)
        """
        degrees  = []
        for vec in range(1, len(thetas) - 1):
            left, center, right = thetas[vec-1:vec+2]
            left, right = left - center, right - center
            
            cos_theta = (left @ right)/(np.linalg.norm(left) * np.linalg.norm(right))
            degrees.append((np.arccos(cos_theta)/(2 * np.pi)) * 360)
        return degrees


    def concat_thetas(self, thetas, shifted=False):
        if shifted:
            return np.vstack([t[:-1, :] for t in thetas] + [thetas[-1][-1].reshape(1, -1)])
        else:
            return np.vstack([t[:, :] for t in thetas])


    def split_thetas(self, thetas, N, shifted=False):
        if shifted:
            full = (thetas.shape[0] + (N - 1))//N
            
            shift, groups = 0, []
            for subset in range(full):
                groups.append(thetas[(subset * full) - shift: ((subset + 1) * full) - shift])
                shift+=1
        
        else:            
            
            groups, current = [], 0
            for subset in range(N):
                
                groups.append(thetas[current: current + self.t_shapes[subset]])
                current += self.t_shapes[subset]

        return groups

    def layerwise_norm(self, thetas):
        layer_dict = self.layer_dict

        norms = []
        for layer in layer_dict.keys():
            if 'bias' not in layer:
                start, stop = layer_dict[layer]
                norms.append(np.linalg.norm(thetas[:, start:stop], axis=1))
        
        return norms

    def logging_block(self, updated_thetas, Log_dict, epoch, shifted=False):
        wide_net = self.make_wide_net(self.concat_thetas(updated_thetas, shifted)).to(self.device)
                
        test_loss_surface = ComputeLoss(model=self.basic_net, model_params=self.mp, dataset=self.ds,
                                        part='test', bs=self.bs, device=self.device)

        N = len(updated_thetas)
        print('Logging block', N)
        
        train_loss, train_gradients, train_acc = self.evaluate(wide_net, gradients=True, accuracy=True, mode=0.02)
        test_loss, test_gradients, test_acc = test_loss_surface.evaluate(wide_net, gradients=True, accuracy=True, mode=0.02)

        splited_gradients_train = self.split_thetas(train_gradients, N, shifted)
        splited_gradients_test = self.split_thetas(test_gradients, N, shifted)

        fro = lambda x: np.linalg.norm(x, axis=1)

        norms = [[], []] #norms[0] - ||grad||2, #norms[1] - ||tangent||2
        norms_layerwise = [[], []]

        for idx, train_grad, test_grad, theta in zip(range(len(self.pairs)), splited_gradients_train,
                                                splited_gradients_test, updated_thetas):
                
                
            
            
            orth_grad_train = self.corrected_gradients(train_grad, theta, orthogonal=True,
                                                       memory_block=idx, lr=self.lr)  
            
            corrected_train = self.corrected_gradients(train_grad, theta, orthogonal=False,
                                                       memory_block=idx, lr=self.lr) 

            orth_grad_test = self.corrected_gradients(test_grad, theta, orthogonal=True,
                                                       memory_block=idx, lr=self.lr)  
            
            corrected_test = self.corrected_gradients(test_grad, theta, orthogonal=False,
                                                       memory_block=idx, lr=self.lr) 

            norms[0].append([fro(train_grad), fro(orth_grad_train), fro(corrected_train)])
            norms[1].append([fro(test_grad), fro(orth_grad_test), fro(corrected_test)])

            norms_layerwise[0].append([self.layerwise_norm(train_grad), \
                                       self.layerwise_norm(orth_grad_train), self.layerwise_norm(corrected_train)])
            norms_layerwise[1].append([self.layerwise_norm(test_grad), \
                                       self.layerwise_norm(orth_grad_test), self.layerwise_norm(corrected_test)])

        logs = {'train': {'acc': train_acc,
                          'loss': train_loss,
                          'norm': norms[0],
                          'norm_layerwise': norms_layerwise[0]},
                'test': {'acc': test_acc,
                         'loss': test_loss,
                         'norm': norms[1],
                         'norm_layerwise': norms_layerwise[1]},
                'chain': self.chain_distance(self.concat_thetas((updated_thetas)))}

        Log_dict[epoch] = logs
        np.save(Log_dict['log_dir'] + 'Stats.npy', Log_dict)
        torch.save(wide_net.state_dict(), Log_dict['log_dir'] + 'Net.torch')

        return logs


    def parallel_batch(self, updated_thetas, batch, label, lr, shifted=False):
        self.t_shapes = [u.shape[0] for u in updated_thetas]
        # updated_thetas - list (number of sectors x (number of points x |theta|))
        N = len(updated_thetas)
        
        wide_net = self.make_wide_net(self.concat_thetas(updated_thetas, shifted)).to(self.device)
                
        _, gradients_thetas = self.evaluate_on_batch(batch.to(self.device), label.to(self.device), wide_net)
        
        
        del wide_net
        
        # gradients_thetas_splited - list (number of sectors x (number of points x |theta|))
        gradients_thetas_splited = self.split_thetas(gradients_thetas, N, shifted) 
        
        new_point = []
        # Compute gradient update
        for index, gradients_thetas, thetas in zip(range(N), gradients_thetas_splited, updated_thetas):
                        
            gradients_update = self.corrected_gradients(gradients_thetas, thetas,
                                                        orthogonal=False,
                                                        memory_block=index,
                                                        lr=lr)  
                        
            new_point.append(self.make_step(thetas, gradients_update))
            
        return new_point
    
    def insert_log_dict(self, dictionary):
        self.Log_dict = dictionary
        self.ld = dictionary['log_dir']

    def find_path_mini_batch(self, pairs, lr, steps=10, points=50, alpha=None,
                             log_parameters=None, restart=None, relax_boundaries=True):
        self.checking = True
        self.epoch, self.index = 0, 0
        
        try:
            Log_dict = self.Log_dict
        except:
            pass
        
        self.pairs = pairs
        lr, decay, decay_each, decay_from = lr['lr'], lr['decay'], lr['decay_each'], lr['decay_from']

        # Gradient cache (momentum) inizialization     
        number_of_points = points * len(pairs)
        print('points * len(pairs):', number_of_points)
        
        basic_net = self.basic_net()
        self.gradient_cache = torch.zeros(size = (number_of_points, basic_net.flatten_parameters().shape[0]))
        self.gradient_cache = [g.numpy() for g in self.gradient_cache.split(points)]
        
        print('length of gradient cache:', len(self.gradient_cache))        
        # Gradient cache end
        
        if not restart:
            Log_dict = {'log_dir': log_parameters['log_dir']}
            self.ld = Log_dict['log_dir']

            if not os.path.isdir(log_parameters['log_dir']):
                os.mkdir(log_parameters['log_dir'])

            full_loss, updated_thetas = [], []
            initial_path = []

            for pair_idx, pair in enumerate(pairs):
                start, end = pair

                initial_loss, gradient, thetas =  self.straight_path(start, end, num_points=points, N=points)
                initial_path.append(thetas)
                
                print(self.gradient_cache[pair_idx].shape)
                                
                gradients_update = self.corrected_gradients(gradient, thetas,
                                                            orthogonal=False,
                                                            memory_block=pair_idx,
                                                            lr=lr)
                
                full_loss.append(initial_loss.reshape(1, -1))
                updated_thetas.append(self.make_step(thetas, gradients_update))

            Log_dict['initial_path'] = initial_path
            del initial_path
            
            initial_loss = np.hstack(full_loss).reshape(-1).tolist() + [initial_loss[-1]]

            Log_dict['initial_loss'] = full_loss
            bounds = range(0, steps)
                
        if isinstance(restart, dict):
            Log_dict = np.load(restart['stat_file'], allow_pickle=True).item()
            Log_dict['log_dir'] = log_parameters['log_dir']
            self.ld = log_parameters['log_dir']
            
            if not os.path.isdir(log_parameters['log_dir']):
                os.mkdir(log_parameters['log_dir'])
            
            start_epoch = max([k for k in Log_dict.keys() if type(k) != str]) + 1
            end_epoch = start_epoch +  restart['epochs']
            
            updated_thetas = Log_dict['final_path']
            bounds = range(start_epoch, end_epoch)
        
        
        elif isinstance(restart, list):            
            if not os.path.isdir(Log_dict['log_dir']):
                os.mkdir(Log_dict['log_dir'])
            
            start_epoch = 0
            end_epoch = 121
            
            updated_thetas = restart
            bounds = range(start_epoch, end_epoch)
            
        for step in tqdm(bounds, total=len(bounds), leave=False):
            self.epoch = step
            self.lr = lr
            
            loader = self.init_loaders(step)
            print(f'New epoch started! {step}')
            for idx, (batch, label) in tqdm(enumerate(loader), total=len(loader), leave=False):
                self.index = idx
                    
                updated_thetas = self.parallel_batch(updated_thetas, batch, label, lr=lr, shifted=False)
            
            
            print('After Epoch', len(updated_thetas))
            print('Grad Cache', self.gradient_cache[0])
            
            if (step+1) >= decay_from:
                if (step+1) % decay_each == 0:
                    lr = lr * decay

            if step % log_parameters['log_each'] == 0:
                between_train_loss = self.evalutae_between_thetas(updated_thetas, alpha=3)

                logs = self.logging_block(updated_thetas, Log_dict, step)
                Log_dict[f'train_loss_between{step}'] = between_train_loss
                    
                if log_parameters['plot_logs'] == True:
                    
                    fig, ax = plt.subplots(figsize=(20, 10))

                    # #Loss
                    if step <= 1:
                        ax.plot(initial_loss, label='Initial Loss')
                    
                    ax.plot(logs['train']['loss'], label='Current Loss Train')
                    ax.plot(logs['test']['loss'], linestyle='-.', label='Current Loss Test', linewidth=2)
                    ax.set(xlabel = 'Point', ylabel = 'Loss', title = f'Step: {step}')
                    ax.legend(loc = 'upper left')

                    # #Acc
                    ax2 = ax.twinx()  

                    ax2.set_ylabel('1 - Accuracy (Error rate)') 
                    ax2.plot(1 - logs['test']['acc'], label='Test error rate', linewidth = 2, c = 'purple', linestyle='--')
                    ax2.plot(1 - logs['train']['acc'], label='Train error rate', linewidth = 2, c = 'purple')

                    ax2.legend(loc = 'upper right')
                    
                    plt.savefig(Log_dict['log_dir'] + str(step) + '.png')                    

                    plt.show()
                    # # Plot PCA
                    self.plot_pca(self.concat_thetas(updated_thetas))
                
                
                # Block of insertion/rejection of points
                N_segments = len(updated_thetas)
                save_borders = [[] for _ in range(N_segments)]

                ratio = np.split(between_train_loss, N_segments)

                for index, line in enumerate(ratio):
                    max2_distances = np.argsort(line.max(axis=1))[::-1][:2]
                    
                    print(line[max2_distances])
                    
                    updated_thetas[index] = self.insert_thetas(max2_distances, updated_thetas[index], None)

                    # Save borders and drop them
                    save_borders[index].append([updated_thetas[index][0], updated_thetas[index][-1]])
                    updated_thetas[index] = updated_thetas[index][1:-1]
                
                Log_dict['final_path'] = updated_thetas
                Log_dict[f'borders{self.epoch}'] = save_borders
                
                np.save(Log_dict['log_dir'] + 'Stats.npy', Log_dict)    
                    
        
    def insert_thetas(self, moving_indexes, updated_thetas, momentum=None):
            pairs = [[value, value+1] for idx, value in enumerate(moving_indexes)]

            values_for_insertion = []

            for p in pairs:
                left, right = updated_thetas[p[0]], updated_thetas[p[1]]    
                intermediate_point = .5 * left + .5 * right

                if momentum:
                    left_momentum, right_momentum = momentum[p[0]], momentum[p[1]]
                    intermediate_momentum = .5 * left_momentum + .5 * right_momentum
                    values_for_insertion.append([intermediate_point, intermediate_momentum])
                else:
                    values_for_insertion.append([intermediate_point])

            ctr = 0
            updated_thetas = updated_thetas.tolist()

            if momentum:
                momentum = momentum.tolist()

            for index, p in enumerate(pairs):
                insertion_point = p[1] + ctr

                updated_thetas.insert(insertion_point, values_for_insertion[index][0])

                if momentum:
                    momentum.insert(insertion_point, values_for_insertion[index][1])
                ctr+=1

            if momentum:
                return np.array(updated_thetas), np.array(momentum)

            return np.array(updated_thetas)
    
    
    def evaluate_thetas(self, thetas, dataset_ratio=1/6):
        if isinstance(thetas, str):
            thetas = np.load(thetas)

        wide_net = self.make_wide_net(thetas).to(self.device)
        train_loss, train_grads = self.evaluate(wide_net, dataset_ratio=dataset_ratio)            

        return train_loss
    
    def center_path(self, stats):
    
        final_path = np.load(stats, allow_pickle=True).item()['final_path']

        new_path = []
        points_dict = {5:2, 6: 2, 7: 2, 8: 2, 9: 3, 10: 2, 11: 2, 12: 2, 13: 2}
        self.points_dict = points_dict
    
    
        # Find 
        for segment in final_path:
            for pair in range(segment.shape[0] - 1):
                if pair in points_dict.keys():
                    
                    sector = [segment[pair]]
                    
                    for ratio in np.linspace(0, 1, num=points_dict[pair]+2)[1:-1]:
                        sector.append((1 - ratio) * segment[pair] + ratio * segment[pair+1])
                    
                    sector.append(segment[pair + 1])
                    new_path.append(np.vstack(sector))

        return new_path
    
    
    def evalutae_between_thetas(self, stats, alpha, fix_conv1=False, fix_head=False, dataset_ratio=1/6, plot=True):
        points = np.linspace(0, 1, num=alpha+2)[1:-1]
        
        if isinstance(stats,str):
            final_path = np.load(stats, allow_pickle=True).item()['final_path']
        
        elif isinstance(stats, list):
            final_path = stats
            del stats
            
        between_train_loss = []
        
        for ratio in points:
            print('Ratio', ratio)
            updated_thetas = []
            
            # Find 
            for segment in final_path:            
                    
                updated_thetas.append(np.vstack([(1 - ratio) * segment[pair] + ratio * segment[pair+1]
                                       for pair in range(segment.shape[0] - 1)]))
                
            
            wide_net = self.make_wide_net(self.concat_thetas(updated_thetas, False)).to(self.device)        
            train_loss, train_gradients, train_acc = self.evaluate(wide_net, gradients=True, accuracy=True,
                                                                   mode=(1/alpha),
#                                                                    dataset_ratio=(1/alpha))
                                                                   dataset_ratio=dataset_ratio)
            
            between_train_loss.append(train_loss.reshape(-1, 1))
            
    
            
        between_train_loss = np.hstack(between_train_loss)
        
        if plot:
            fig = plt.figure(figsize=(20, 10))
            plt.plot(between_train_loss.reshape(-1))
            plt.savefig(f'{self.ld}BetweenLoss{self.epoch}.png')
        
        return between_train_loss
    
    def complete_evaluation(self, thetas, alpha, save_path=None, dataset_ratio=1/5):
        between_train_loss = self.evalutae_between_thetas(stats=thetas, alpha=alpha, save_path=save_path,
                                                          fix_conv1=False, fix_head=False, dataset_ratio=dataset_ratio)
        
        train_loss = self.evaluate_thetas(np.vstack(thetas), dataset_ratio=dataset_ratio)
        between_train_loss, train_loss = np.split(between_train_loss, 10), np.split(train_loss, 10)
        
        loss=np.vstack([np.hstack([between_train_loss[i], train_loss[i][1:].reshape(-1, 1)]).reshape(-1) for i in range(10)])
        
        return loss