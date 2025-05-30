import numpy as np

import torch
import numpy as np
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition import qLogExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim.initializers import initialize_q_batch_nonneg
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

from solver_FFTPV import sel_em_TPV

torch.set_default_dtype(torch.float64)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Inputs
InAsEQE = np.loadtxt('/home/yigithankose/FFTPV/InAs_EQE.txt')
E_ideal = np.loadtxt('/home/yigithankose/FFTPV/E_ideal.txt')
EQE = InAsEQE

bounds = torch.tensor([[5e-9,5e-9,5e-9,1e19,1e19,1e19],[1e-6,1e-6,1e-6,1e21,1e21,1e21]], device=device)

#Functions
def example_obj(ind):
    z = []
    for x in ind:
        var = []
        em = sel_em_TPV(x[0],x[1],x[2],x[3],x[4],x[5],EQE)
        for i in range(666):
            diff = [(em[3*i]-E_ideal[3*i])**2,(em[3*i+1]-E_ideal[3*i+1])**2,(em[3*i+2]-E_ideal[3*i+2])**2,(em[3*i+3]-E_ideal[3*i+3])**2]
            var.append(3e12/8*(diff[0]+3*diff[1]+3*diff[2]+diff[3]))
        var2 = -sum(var)
        z.append(var2)
    return torch.tensor(z)

def generate_initial_data(n):
    train_x = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n,6, device=device, dtype=torch.double)
    train_obj = example_obj(train_x).unsqueeze(-1)
    best_value = train_obj.max().item()
    return train_x, train_obj, best_value

def get_next_points_analytic(init_x, init_y, best_init_y, bounds, n_points):
    model = SingleTaskGP(init_x, init_y,input_transform=Normalize(d=6),outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    EI = qLogExpectedImprovement(model=model, best_f=best_init_y)

    new_point_analytic, _ = optimize_acqf(
    acq_function=EI,
    bounds=bounds,
    q=n_points,
    num_restarts=200,
    raw_samples=1024,
    options={"batch_limit":5, "maxiter":400}
    )
    return new_point_analytic

init_x, init_y, best_init_y = generate_initial_data(20)

evaluated_x = []
evaluated_y = []
best_x_list = []
best_y_list = []

#Optimization loop
for i in range(100):
    print(i)
    new_candidates = get_next_points_analytic(init_x,init_y,best_init_y,bounds,1)
    new_results = example_obj(new_candidates).unsqueeze(-1)

    init_x = torch.cat([init_x,new_candidates])
    init_y = torch.cat([init_y,new_results])

    best_init_y = init_y.max().item()
    index = torch.argmax(init_y)
    best_init_x = init_x[index]

    new_x_array = new_candidates.cpu().numpy()
    new_y_array = new_results.cpu().numpy()

    evaluated_x.append(new_x_array)
    evaluated_y.append(new_y_array)

    best_x_list.append(best_init_x.cpu().numpy())
    best_y_list.append(best_init_y)

np.savetxt("X_list.txt", np.vstack(evaluated_x))
np.savetxt("Y_list.txt", np.vstack(evaluated_y))

np.savetxt("Best_X_list.txt", np.vstack(best_x_list))
np.savetxt("Best_Y_list.txt", np.array(best_y_list))