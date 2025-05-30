import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

class MMProbe():
    def __init__(self, acts, labels):
        pos_activations = acts[labels==1]
        neg_activations = acts[labels==0]
        pos_mean = pos_activations.mean(axis=0)
        neg_mean = neg_activations.mean(axis=0)
        self.theta_mm = pos_mean - neg_mean
        self.theta_mm = self.theta_mm / np.linalg.norm(self.theta_mm)
    
    def project(self, x):
        return x @ self.theta_mm

    def predict(self, x, threshold=0.5):
        return (self.predict_proba(x)[:,1] > threshold).astype(np.int32)

    def predict_proba(self, x):
        pos_prob = torch.sigmoid(torch.from_numpy(self.project(x))).detach().cpu().numpy()
        neg_prob = 1-pos_prob
        return np.stack([neg_prob, pos_prob]).T


class TTPD():
    def __init__(self):
        self.t_g = None
        self.polarity_direction = None
        self.lr = None
    
    @staticmethod
    def learn_truth_direction(acts, labels, polarities):
        all_polarities_zero = np.allclose(polarities, [0], atol=1e-8)
        if all_polarities_zero:
            X = labels.reshape(-1, 1)
        else:
            X =  np.column_stack([labels, labels*polarities])
        solution = np.linalg.inv(X.T @ X) @ X.T @ acts
        if all_polarities_zero:
            t_g = solution.flatten()
            # t_p = None
        else:
            t_g = solution[0, :]
            # t_p = solution[1, :]
        return t_g
    
    @staticmethod
    def learn_polarity_direction(acts, polarities, seed):
        polarities[polarities==-1] = 0
        lr_po = LogisticRegression(penalty=None, fit_intercept=False, random_state=seed, max_iter=10000)
        lr_po.fit(acts, polarities)
        return lr_po.coef_

    def project_acts(self, acts, token_counts=None):
        proj_t_g = acts @ self.t_g
        proj_p = acts @ self.polarity_direction.T
        # proj_t_g = proj_t_g * np.sqrt(token_counts)
        acts_2d = np.concatenate((proj_t_g[:, None], proj_p), axis=1)
        return acts_2d

    @staticmethod
    def from_data(acts_centered, acts, labels, polarities, seed, token_counts=None):
        probe = TTPD()
        probe.t_g = TTPD.learn_truth_direction(acts_centered, labels, polarities)
        probe.polarity_direction = TTPD.learn_polarity_direction(acts, polarities, seed)
        acts_2d = probe.project_acts(acts, token_counts)
        probe.lr = LogisticRegression(penalty=None, fit_intercept=False, random_state=seed, max_iter=10000)
        probe.lr.fit(acts_2d, labels)
        return probe
    
    def predict(self, acts, token_counts=None):
        acts_2d = self.project_acts(acts, token_counts)
        return self.lr.predict(acts_2d)
    
    def predict_proba(self, acts, token_counts=None):
        acts_2d = self.project_acts(acts, token_counts)
        return self.lr.predict_proba(acts_2d)
