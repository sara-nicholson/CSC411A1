import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

BATCHES = 50

class BatchSampler(object):
    '''
    A (very) simple wrapper to randomly sample batches without replacement.

    You shouldn't need to touch this.
    '''
    
    def __init__(self, data, targets, batch_size):
        self.num_points = data.shape[0]
        self.features = data.shape[1]
        self.batch_size = batch_size

        self.data = data
        self.targets = targets

        self.indices = np.arange(self.num_points)

    def random_batch_indices(self, m=None):
        '''
        Get random batch indices without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        if m is None:
            indices = np.random.choice(self.indices, self.batch_size, replace=False)
        else:
            indices = np.random.choice(self.indices, m, replace=False)
        return indices 

    def get_batch(self, m=None):
        '''
        Get a random batch without replacement from the dataset.

        If m is given the batch will be of size m. Otherwise will default to the class initialized value.
        '''
        indices = self.random_batch_indices(m)
        X_batch = np.take(self.data, indices, 0)
        y_batch = self.targets[indices]
        return X_batch, y_batch    


def load_data_and_init_params():
    '''
    Load the Boston houses dataset and randomly initialise linear regression weights.
    '''
    print('------ Loading Boston Houses Dataset ------')
    X, y = load_boston(True)
    features = X.shape[1]

    # Initialize w
    w = np.random.randn(features)
    print("Loaded...")
    print("Total data points: {0}\nFeature count: {1}".format(X.shape[0], X.shape[1]))
    print("Random parameters, w: {0}".format(w))
    print('-------------------------------------------\n\n\n')

    return X, y, w


def cosine_similarity(vec1, vec2):
    '''
    Compute the cosine similarity (cos theta) between two vectors.
    '''
    dot = np.dot(vec1, vec2)
    sum1 = np.sqrt(np.dot(vec1, vec1))
    sum2 = np.sqrt(np.dot(vec2, vec2))

    return dot / (sum1 * sum2)


#TODO: implement linear regression gradient
def lin_reg_gradient(X, y, w):
    '''
    Compute gradient of linear regression model parameterized by w
    '''
    m = len(X)
    first = np.dot(np.dot(X.T,X),w)
    second = np.dot(X.T,y)
    gradient = (2/m)*(first - second)
    return gradient

def main():
    # Load data and randomly initialise weights
    #np.random.seed(42)
    X, y, w = load_data_and_init_params()
    # Create a batch sampler to generate random batches from data
    batch_sampler = BatchSampler(X, y, BATCHES)
    
    #calculate true gradient
    true_grad = lin_reg_gradient(X,y,w)
    
    #calcualte batch gradients
    K = 500
    total = 0
    for i in range(K):
        X_b, y_b = batch_sampler.get_batch()
        batch_grad = lin_reg_gradient(X_b,y_b,w)
        total += batch_grad
    mb_grad = (1/K)*total
    #calcualte cosime similarity
    cos_sim = cosine_similarity(true_grad,mb_grad)
    print("Cosine similarity: {}".format(cos_sim))
    
    #calculate squared mean distance
    true_norm = np.sum((true_grad**2))
    mb_norm = np.sum((mb_grad**2))
    dist = true_norm+mb_norm-2*true_grad.dot(mb_grad.transpose())
    print("Squared mean distance: {}".format(dist))
    
    sample_var_compare(X,y,K,w)
    #question6
    #generate random w to be paramter examined
def sample_var_compare(X,y,K,w):
    
    m = 400
    all_grad = []
    var = []
    
    j = np.random.randint(0,13)
    for i in range(m):
        batch_sampler  = BatchSampler(X,y,i+1)
        batch_gradj = []
        var = []
        for k in range(K):
            X_b,y_b = batch_sampler.get_batch()
            batch_grad = lin_reg_gradient(X_b,y_b,w)
            batch_gradj.append(batch_grad[j])
        all_grad.append(batch_gradj)
        sample_avg = (np.mean(all_grad))
     
        var_calc = np.sum((np.subtract(all_grad,sample_avg)**2),axis=1)
        var.append(var_calc)
    
    plt.plot(np.log(range(m)),np.log(np.asarray(var)).T)
    plt.xlabel("Log(m)")
    plt.ylabel("Log(variance)")
         
            
if __name__ == "__main__":
    main()