from architectures.tnp import TNP
import torch
import torch.nn as nn
from attrdict import AttrDict


class TNDP(TNP):
    """
    A Transformer Neural Process (TNP) model for decision-making.

    Args:
        dim_x (int): Dimension of input features
        dim_y (int): Dimension of output features
        dim_embedding (int): Dimension of the embedding space
        num_layers (int): Number of transformer layers
        embedding_depth (int): Depth of embedding MLPs
        num_heads (int): Number of attention heads
        dim_feedforward (int): Dimension of feedforward network
        n_decision (int): Number of possible decisions, in toy case, it is 1.
        dropout (float, optional): Dropout rate. Defaults to 0.0
        horizon (int, optional): Number of steps. Defaults to 20
        embedding_way (str, optional): Method to combine embeddings ('sum' or 'concat'). Defaults to 'sum'
    """
    def __init__(
            self,
            dim_x: int,
            dim_y: int,
            dim_embedding: int,
            num_layers: int,
            embedding_depth: int,
            num_heads: int,
            dim_feedforward: int,
            n_decision: int,
            dropout: float = 0.0,
            horizon: int = 20,
            embedding_way: str = 'sum'
    ):
        super().__init__(
            dim_x,
            dim_y,
            dim_embedding,
            num_layers,
            embedding_depth,
            num_heads,
            dim_feedforward,
            dropout
        )
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_embedding = dim_embedding
        self.num_layers = num_layers
        self.embedding_depth = embedding_depth
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.n_decision = n_decision
        self.dropout = dropout
        self.T = horizon
        self.embedding_way = embedding_way

        self.x_embedder = self.build_mlp(self.dim_x, self.dim_embedding, self.dim_embedding, self.embedding_depth)
        self.y_embedder = self.build_mlp(self.dim_y, self.dim_embedding, self.dim_embedding, self.embedding_depth)
        self.d_embedder = nn.Embedding(self.n_decision, self.dim_embedding)  # for toy case, this can be removed
        self.T_embedder = nn.Embedding(self.T, self.dim_embedding)

        self.query_head = nn.Sequential(
            nn.Linear(self.dim_embedding, self.dim_feedforward),
            nn.ReLU(),
            nn.Linear(self.dim_feedforward, 1)
        )

        self.pred_head = nn.Sequential(
            nn.Linear(self.dim_embedding, self.dim_feedforward),
            nn.ReLU(),
            nn.Linear(self.dim_feedforward, dim_y * 2)
        )

    def construct_input(self, batch):
        batch_size = batch.context_x.shape[0]

        # context set
        embedding_xc = self.x_embedder(batch.context_x)
        embedding_dc = self.d_embedder(batch.context_d.squeeze(-1))
        embedding_yc = self.y_embedder(batch.context_y)

        # query set
        embedding_xq = self.x_embedder(batch.query_x)
        embedding_dq = self.d_embedder(batch.query_d.squeeze(-1))

        # prediction set
        embedding_xt = self.x_embedder(batch.target_x)
        embedding_xt_expanded = embedding_xt.repeat(1, self.n_decision, 1)

        decision_indices = torch.arange(0, self.n_decision).long()
        decision_embeddings = self.d_embedder(decision_indices)
        decision_embeddings = decision_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)

        if self.embedding_way == 'sum':
            embedding_context = embedding_xc + embedding_dc + embedding_yc
            embedding_query = embedding_xq + embedding_dq
            embedding_target = embedding_xt_expanded + decision_embeddings

            # global information
            embedding_t = self.T_embedder(batch.t).unsqueeze(0).repeat(batch_size, 1, 1)
            embedding_x_star = self.x_embedder(batch.target_x)

        elif self.embedding_way == 'concat':
            embedding_context = torch.cat([embedding_xc, embedding_dc, embedding_yc], dim=2)

            embedding_yq = torch.zeros([batch_size, embedding_xq.shape[1], self.dim_embedding])
            embedding_query = torch.cat([embedding_xq, embedding_dq, embedding_yq], dim=2)

            embedding_yt = torch.zeros([batch_size, self.n_decision, self.dim_embedding])
            embedding_target = torch.cat([embedding_xt_expanded, decision_embeddings, embedding_yt], dim=2)

            raise NotImplementedError("embeddings for t and x* are not implemented yet.")
        else:
            raise ValueError("Invalid embedding way")

        # concatenate all
        embedding_all = torch.cat([embedding_t, embedding_x_star, embedding_context,
                                   embedding_query, embedding_target], dim=1)

        return embedding_all

    def create_mask(self, context_x, query_x):
        num_context = context_x.shape[1]
        num_query = query_x.shape[1]
        num_target = self.n_decision  # [(x*, d1), (x*, d2), ..., (x*, dN)]
        num_global = 2  # 1 for step t, another 1 for lambda

        num_all = num_context + num_query + num_target + num_global
        mask = torch.zeros(num_all, num_all).fill_(float('-inf'))
        mask = mask.type(context_x.dtype)

        # Global Information
        mask[0:num_global, 0:num_global] = 0.0  # GI attend GI

        # Context Data
        mask[num_global:num_global + num_context, num_global:num_global + num_context] = 0.0  # CD attend all CD
        mask[num_global:num_global + num_context, 0:num_global] = 0.0  # CD attend GI

        # Query Data
        mask[num_global + num_context:num_global + num_context + num_query, 0:num_global] = 0.0  # QD attend GI
        mask[num_global + num_context:num_global + num_context + num_query, num_global:num_global + num_context] = 0.0  # QD attend all CD
        mask[num_global + num_context:num_global + num_context + num_query, num_global + num_context:num_global + num_context + num_query].fill_diagonal_(
            0.0)  # QD attend itself

        # Target Data
        # mask[num_global + num_context + num_query:, 0:num_global] = 0  # TD attend GI
        mask[num_global + num_context + num_query:, num_global:num_global + num_context] = 0.0  # TD attend all CD
        mask[num_global + num_context + num_query:, num_global + num_context + num_query:].fill_diagonal_(0.0)  # TD attend itself

        return mask, [num_global, num_context, num_query, num_target]

    def forward(self, batch):
        inputs = self.construct_input(batch)
        mask, dims = self.create_mask(batch.context_x, batch.query_x)

        out = self.encoder(inputs, mask=mask)
        query_out = out[:, dims[0]+dims[1]:dims[0]+dims[1]+dims[2]]
        decision_out = out[:, dims[0]+dims[1]+dims[2]:]

        query_out = self.query_head(query_out)
        decision_out = self.pred_head(decision_out)  # [batch_size, n_decision, dim_y*2]
        outs = AttrDict(query_out=query_out, decision_out=decision_out)

        return outs

    def predict(self, batch):
        inputs = self.construct_input(batch)
        mask, dims = self.create_mask(batch.context_x, batch.query_x)
        out = self.encoder(inputs, mask=mask)
        query_out = out[:, dims[0] + dims[1]:dims[0] + dims[1] + dims[2]]
        decision_out = out[:, dims[0] + dims[1] + dims[2]:]

        query_out = self.pred_head(query_out)
        decision_out = self.pred_head(decision_out)  # [batch_size, n_decision, dim_y*2]
        outs = AttrDict(query_out=query_out, decision_out=decision_out)

        return outs


if __name__ == "__main__":
    decision_tnp = TNDP(3, 3, 64, 2, 2, 2, 64, 3)
    mask, dims = decision_tnp.create_mask(torch.randn(1, 3), torch.randn(1, 3))
    print(mask)
    print(dims)
