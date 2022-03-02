from ml.linreg import *

# it works, but the accuracy is atrocious

batch = pd.read_csv('examples/real_estate.csv')
print("Batch: \n", batch)
y = batch.loc[:,"Y house price of unit area"]
print("Y: \n", y)

# as of this time, normalization wasn't implemented so we had to stick with numbers on the same magnitude
lr = LinearReg(cons(), id(2), id(4))
alpha = 0.000005
eps = 10
maxepoch = 10000

print("Converged!" if graddes(lr, batch, y, alpha, eps, maxepoch) else "Failed to converge.")
