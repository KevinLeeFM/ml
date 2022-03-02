from ml.linreg import *

linear_test = {
    'col1': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'col2': [0, 1, 2, 3, 4, 5, 6, 7, 8],
}

quadratic_test = {
    'col1': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'col2': [0, 1, 4, 9, 16, 25, 36, 49, 64],
}

# some variables with high covariance and a little bit of noise
multivar_test = {
    'x1': [0.1, 1, 2, 2.9, 4, 5, 6.1, 7, 8.1],
    'x2': [0, 2, 4, 6, 8.1, 9.9, 12.3, 13.7, 15.9],
    'y': [0, 1, 2, 3, 4, 5, 6, 7, 8],
}

batch = pd.DataFrame(data=multivar_test)
y = batch.loc[:,"y"]
print("Batch: \n", batch)

lr = LinearReg(cons(), id(0), id(1))
alpha = 0.0005
eps = 0.1
maxepoch = 10000

print("Converged!" if graddes(lr, batch, y, alpha, eps, maxepoch) else "Failed to converge.")