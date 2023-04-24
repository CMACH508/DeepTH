import torch
import torch.nn as nn
from graphEncoder import GraphAttentionEncoder
from torch.distributions import Categorical


class ValueDecoder(nn.Module):
    def __init__(self, dimension):
        super(ValueDecoder, self).__init__()
        self.value = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.ReLU(),
            nn.Linear(dimension, 1)
        )

    def forward(self, x):
        return self.value(x)


class DeConvolution(nn.Module):
    def __init__(self, hidden_dim):
        super(DeConvolution, self).__init__()
        self.deConv = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU()
        )

    def forward(self, x):
        return self.deConv(x)


class PolicyDecoder(nn.Module):
    def __init__(self, dimension, grid):
        super(PolicyDecoder, self).__init__()
        self.dimension = dimension
        self.out = grid
        self.linear = nn.Sequential(
            nn.Linear(self.dimension, 2 * 2 * self.out),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            DeConvolution(self.out),
            DeConvolution(self.out // 2),
            DeConvolution(self.out // 4),
            DeConvolution(self.out // 8)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.out // 16, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(-1, self.out, 2, 2)
        x = self.conv(self.deconv(x))
        x = x.reshape(-1, self.out ** 2)
        x = self.softmax(x)
        return x


class Reconstruction(nn.Module):
    def __init__(self, dimension, grid):
        super(Reconstruction, self).__init__()
        self.dimension = dimension
        self.out = grid
        self.linear = nn.Sequential(
            nn.Linear(self.dimension, 2 * 2 * self.out),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            DeConvolution(self.out),
            DeConvolution(self.out // 2),
            DeConvolution(self.out // 4),
            DeConvolution(self.out // 8)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.out // 16, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(-1, self.out, 2, 2)
        x = self.conv(self.deconv(x))
        x = x.reshape(-1, 1, self.out, self.out)
        return x


class Projection(nn.Module):
    def __init__(self, grid):
        super(Projection, self).__init__()
        self.grid = grid
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.linear = nn.Sequential(
            nn.Linear(self.grid ** 2, self.grid * 2),
            nn.ReLU(),
            nn.Linear(self.grid * 2, self.grid),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(-1, self.grid ** 2)
        x = self.linear(x)
        return x


class Network(nn.Module):
    def __init__(self, dimension, grid):
        super(Network, self).__init__()
        self.dimension = dimension
        self.grid = grid
        self.graphEncoder = GraphAttentionEncoder(n_heads=8, embed_dim=self.dimension, n_layers=3, node_dim=7)
        self.integratedInput = nn.Sequential(
            nn.Linear(self.dimension * 2, self.dimension),
            nn.ReLU()
        )
        self.valueDecoder = ValueDecoder(self.dimension)
        self.policyDecoder = PolicyDecoder(self.dimension, self.grid)
        self.reconstrucion = Reconstruction(self.dimension, self.grid)
        self.projection = Projection(self.grid)

    def encoder(self, nodes, adjMatrix):
        nodeEmbedding, graphEmbedding = self.graphEncoder(nodes, adjMatrix)
        return nodeEmbedding, graphEmbedding

    def decoder(self, nodeEmbedding, graphEmbedding, macroID):
        macroID = macroID.reshape(-1, 1).repeat(1, self.dimension)[:, None, :]
        currentEmbedding = nodeEmbedding.gather(dim=1, index=macroID).squeeze(1)
        embedding = self.integratedInput(torch.cat((currentEmbedding, graphEmbedding), dim=1))
        value = self.valueDecoder(embedding)
        policy = self.policyDecoder(embedding)
        return value, policy

    def evaluate(self, nodes, adjMatrix, macroID, mask):
        nodeEmbedding, graphEmbedding = self.encoder(nodes, adjMatrix)
        value, policy = self.decoder(nodeEmbedding, graphEmbedding, macroID)
        mask = mask.reshape(-1, self.grid ** 2)
        #policy_sample = torch.pow(policy, 0.1)
        #policy_sample += 0.000001
        policy[mask == 1] = 0
        policy = policy / policy.sum(dim=-1, keepdim=True)
        policy = Categorical(policy)
        action = policy.sample()
        logits = policy.log_prob(action)
        entropy = policy.entropy()
        proj = self.projection(self.reconstrucion(graphEmbedding))
        return action, logits, entropy, value, proj

    def playTest(self, nodes, adjMatrix, macroID, mask):
        nodeEmbedding, graphEmbedding = self.encoder(nodes, adjMatrix)
        _, policy = self.decoder(nodeEmbedding, graphEmbedding, macroID)
        mask = mask.reshape(-1, self.grid ** 2)
        policy[mask == 1] = -1
        action = torch.argmax(policy, dim=-1)
        return action

    def project(self, canvas):
        proj = self.projection(canvas)
        return proj.detach()

    def reconProject(self, nodes, adjMatrix):
        nodeEmbedding, graphEmbedding = self.encoder(nodes, adjMatrix)
        proj = self.projection(self.reconstrucion(graphEmbedding))
        return proj

    def forward(self, nodes, adjMatrix, macroID, action):
        nodeEmbedding, graphEmbedding = self.encoder(nodes, adjMatrix)
        value, policy = self.decoder(nodeEmbedding, graphEmbedding, macroID)
        policy = Categorical(policy)
        logits = policy.log_prob(action)
        entropy = policy.entropy()
        proj = self.projection(self.reconstrucion(graphEmbedding))
        return logits, entropy, value, proj
