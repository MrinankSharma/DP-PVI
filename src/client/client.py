import ray
import torch

def zero_init_func(tensor):
    return torch.Tensor(tensor).fill_(0)


@ray.remote
class Client(object):
    def __init__(self, model_class, data, model_parameters, model_hyperparameters, hyperparameters, metadata):
        self.set_hyperparameters(hyperparameters)
        self.set_metadata(metadata)

        self.data = data
        self.model = model_class(model_parameters, model_hyperparameters)

    def set_hyperparameters(self, **kwargs):
        pass

    def set_metadata(self, **kwargs):
        pass

    def compute_update(self, model_parameters=None, model_hyperparameters=None):
        pass

@ray.remote
class BasicClient(Client):
    def __init__(self, lambda_prior, model_class, data, model_parameters, model_hyperparameters, hyperparameters, metadata):
        super().__init__(model_class, data, model_parameters, model_hyperparameters, hyperparameters, metadata)

        self.t_i = {}
        for key in model_parameters.keys():
            self.t_i[key] = self.t_i_init_func(model_parameters[key])

        self.lambda_i = lambda_prior


    def set_hyperparameters(self, privacy_function, t_i_init_func = zero_init_func, **kwargs):
        super().set_hyperparameters(kwargs)

        self.privacy_function = privacy_function
        self.t_i_init_func = t_i_init_func

    def compute_update(self, model_parameters=None, model_hyperparameters=None):
        super().compute_update()

        t_i_old = self.t_i
        lambda_old = self.model.parameters

        lambda_new = self.model.fit(self.data,
                                      t_i_old,
                                      model_parameters,
                                      model_hyperparameters)

        delta_lambda_i = self.model.subtract_params(lambda_new,
                                                    lambda_old)

        delta_lambda_i_tilde = self.privacy_function(delta_lambda_i)

        lambda_new = self.model.add_parameters(lambda_old, delta_lambda_i_tilde)

        t_i_new = self.model.add_parameters(
            self.model.subtract_params(lambda_new,
                                       lambda_old),
            t_i_old
        )

        self.t_i = t_i_new

        return delta_lambda_i_tilde

