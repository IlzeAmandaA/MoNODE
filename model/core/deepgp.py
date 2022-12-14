from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.variational import VariationalStrategy, IndependentMultitaskVariationalStrategy, UnwhitenedVariationalStrategy, CholeskyVariationalDistribution, MeanFieldVariationalDistribution, DeltaVariationalDistribution
from gpytorch.means import ConstantMean,  LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal
import torch
from gpytorch.models      import ApproximateGP

class DeepGP(ApproximateGP): #DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant',u_var='diag', whitened= True):

        # variational distribution
        if u_var=='chol': 
            U_VAR_DIST = CholeskyVariationalDistribution
        elif u_var=='diag': 
            U_VAR_DIST = MeanFieldVariationalDistribution
        elif u_var=='delta': 
            U_VAR_DIST = DeltaVariationalDistribution

        # inducing point parameterization
        if whitened:
            VAR_CLS = VariationalStrategy
        else:
            VAR_CLS = UnwhitenedVariationalStrategy

        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = U_VAR_DIST(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        base_variational_strategy = VAR_CLS(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        variational_strategy = IndependentMultitaskVariationalStrategy(base_variational_strategy, num_tasks=output_dims)

        super(DeepGP, self).__init__(variational_strategy) #, input_dims, output_dims)

        # mean and covariance modules
        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)

        self.covar_module = ScaleKernel(
            RBFKernel(input_dims),
            batch_shape=batch_shape)

    def forward(self, x):
        mean_x = self.mean_module(x) # 6, 420 --> N*T + N_inducing 
        covar_x = self.covar_module(x) #6, 420, 420
        return MultivariateNormal(mean_x, covar_x)
