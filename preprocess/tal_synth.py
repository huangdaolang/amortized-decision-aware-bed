import numpy as np
import gpytorch
import torch
from attrdict import AttrDict
import matplotlib.pyplot as plt


class TALSynth(object):
    def __init__(self, dim_x=5, n_dec=4, std=0.1):
        self.dim_x = dim_x
        self.n_dec = n_dec
        self.std = std

    def sample(self, batch_size=16, n_context=None, n_query=300, min_n_context=10, max_n_context=100):

        batch = AttrDict()
        n_context = n_context or torch.randint(low=min_n_context, high=max_n_context, size=[1]).item()
        assert n_context > self.n_dec, "n_context should be larger than n_dec"
        n_test = 1

        batch.context_x = torch.zeros([batch_size, n_context, self.dim_x])
        batch.context_d = torch.zeros([batch_size, n_context, 1], dtype=torch.long)
        batch.context_y = torch.zeros([batch_size, n_context, 1])
        batch.query_x = torch.zeros([batch_size, n_query, self.dim_x])
        batch.query_d = torch.zeros([batch_size, n_query, 1], dtype=torch.long)
        batch.query_y = torch.zeros([batch_size, n_query, 1])
        batch.target_x = torch.zeros([batch_size, n_test, self.dim_x])
        batch.target_d = torch.zeros([batch_size, n_test, 1], dtype=torch.long)
        batch.target_y = torch.zeros([batch_size, n_test, 1])

        for i in range(batch_size):
            # length and variance of RBF kernel
            self.l = torch.sqrt(torch.tensor(self.dim_x)) * (0.25 + 0.75 * torch.rand(self.n_dec, self.dim_x))
            self.v = 0.1 + torch.rand(self.n_dec, )

            self.kernels = []
            for j in range(self.n_dec):
                kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=self.dim_x))
                kernel.outputscale = self.v[j]
                kernel.base_kernel.lengthscale = self.l[j, :]
                self.kernels.append(kernel)

            context, query, target = self.sample_a_set(self.dim_x, self.std, self.n_dec, n_context, n_query)

            batch.context_x[i] = context.x
            batch.context_d[i] = context.d
            batch.context_y[i] = context.y
            batch.query_x[i] = query.x
            batch.query_d[i] = query.d
            batch.query_y[i] = query.y
            batch.target_x[i] = target.x
            batch.target_d[i] = target.d
            batch.target_y[i] = target.y

        return batch

    def gen_decision(self, n_context, n_query, n_dec):

        n_total = n_context + n_query + 1

        p = torch.ones(n_dec, ) / n_dec

        initial_decisions = torch.arange(n_dec)

        if n_total > n_dec:
            extra_decisions = torch.multinomial(p, n_total - n_dec, replacement=True)
            d = torch.cat((initial_decisions, extra_decisions), dim=0)
        else:
            d = initial_decisions[:n_total]

        d = d[torch.randperm(d.size(0))]

        return d

    def sample_a_set(self, dim, std, n_dec, n_context, n_query):

        n_total = n_context + n_query + 1

        x = torch.rand(n_total, dim)
        d = self.gen_decision(n_context, n_query, n_dec)
        y = torch.zeros(n_total, n_dec)

        for i in range(n_dec):
            K = self.kernels[i](x).to_dense()

            y[:, i] = torch.distributions.MultivariateNormal(torch.zeros(n_total),
                                                             covariance_matrix=K + 0.01 ** 2 * torch.eye(n_total)).sample() + std * torch.randn(n_total)

        idx_total = torch.arange(n_total)

        idx_context = torch.zeros(n_context, dtype=torch.int64)

        for i in range(n_dec):
            idx_d = torch.where(d == i)[0]
            idx_context[i] = idx_d[torch.randint(len(idx_d), (1,))]
        idx_total = idx_total[~(idx_total.unsqueeze(1) == idx_context[:n_dec]).any(dim=1)]
        idx_tmp = torch.randperm(len(idx_total))[:n_context - n_dec]
        idx_context[n_dec:] = idx_total[idx_tmp]
        idx_total = idx_total[~(idx_total.unsqueeze(1) == idx_context[n_dec:]).any(dim=1)]
        idx_total = idx_total[torch.randperm(len(idx_total))]
        idx_qu = idx_total[:n_query]
        idx_te = idx_total[n_query:]

        train_x = x[idx_context, :]
        train_d = torch.reshape(d[idx_context], (n_context, 1))
        train_y = torch.reshape(y[idx_context, train_d.flatten()], (n_context, 1))

        query_x = x[idx_qu, :]
        query_d = torch.reshape(d[idx_qu], (n_query, 1))
        query_y = torch.reshape(y[idx_qu, query_d.flatten()], (n_query, 1))

        target_x = x[idx_te, :]
        target_d = torch.reshape(torch.argmax(y[idx_te, :], dim=1), (1, 1))
        target_y = torch.reshape(y[idx_te, target_d.flatten()], (1, 1))

        context_set = AttrDict({
            'x': train_x,
            'y': train_y,
            'd': train_d
        })

        query_set = AttrDict({
            'x': query_x,
            'y': query_y,
            'd': query_d
        })

        target = AttrDict({
            'x': target_x,
            'y': target_y,
            'd': target_d
        })

        return context_set, query_set, target


def plot(sampler, batch):

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'purple']

    markers = {'context': 's', 'query': 'o', 'target': '*'}

    for i, kernel in enumerate(sampler.kernels):

        for point_type, marker in markers.items():
            mask = (batch[f'{point_type}_d'].squeeze(-1) == i).any(dim=0)

            if point_type == 'target':
                ax.scatter(batch[f'{point_type}_x'][0, mask, 0].numpy(),
                           batch[f'{point_type}_y'][0, mask, 0].numpy(),
                           c=colors[i], label=f'GP {i + 1} - {point_type.capitalize()}', marker=marker, zorder=100, s=500)
            elif point_type == 'context':
                ax.scatter(batch[f'{point_type}_x'][0, mask, 0].numpy(),
                           batch[f'{point_type}_y'][0, mask, 0].numpy(),
                           c=colors[i], label=f'GP {i + 1} - {point_type.capitalize()}', marker=marker, zorder=100, s=20)
            else:
                ax.scatter(batch[f'{point_type}_x'][0, mask, 0].numpy(),
                           batch[f'{point_type}_y'][0, mask, 0].numpy(),
                           c=colors[i], label=f'GP {i + 1} - {point_type.capitalize()}', marker=marker, zorder=10, s=10)

    ax.set_title("Combined GPs with Different Points")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sampler = TALSynth(dim_x=1, n_dec=4, std=0.1)
    batch = sampler.sample(batch_size=1, n_context=10, n_query=1000)
    plot(sampler, batch)