import torch
import numpy as np
from network import Network
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from chipManagerMulti import ChipManagerMulti
import dreamplace.PlaceDB as PlaceDB
import dreamplace.Params as Params
import dreamplace.Placer as Placer
import glob
from selfplay import reward_macro


class ChipSet(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return {
            'node': self.data['nodes'][item],
            'canvas': self.data['canvas'][item],
            'nextMacro': self.data['nextMacros'][item],
            'action': self.data['actions'][item],
            'logprob': self.data['logprobs'][item],
            'reward': self.data['rewards'][item]
        }

    def __len__(self):
        return len(self.data['rewards'])


class PPO:
    def __init__(self, gpu_num, lr, K_epochs, eps_clip, grid):
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.gpu_num = gpu_num
        self.grid = grid
        self.net = Network(dimension=32, grid=self.grid).cuda(self.gpu_num)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr)
        self.net_critic = Network(dimension=32, grid=self.grid).cuda(self.gpu_num)
        self.net_critic.load_state_dict(self.net.state_dict())
        self.batch_size = 128

    def select_action_test(self, node, adjMatrix, nextMacro, mask):
        with torch.no_grad():
            node = torch.FloatTensor(node).cuda(self.gpu_num)
            adjMatrix = torch.FloatTensor(adjMatrix).cuda(self.gpu_num)
            nextMacro = torch.LongTensor(nextMacro).cuda(self.gpu_num)
            mask = torch.FloatTensor(mask).cuda(self.gpu_num)
            action = self.net_critic.playTest(node, adjMatrix, nextMacro, mask)
            action = action.cpu().numpy()
        return action

    def playAGameTest(self, chip, report=False):
        gameNum = 1
        fileName = './processed_data/' + chip + '_clustered.npy'
        data = np.load(fileName, allow_pickle=True).item()
        width = data['area'][1] - data['area'][0]
        height = data['area'][3] - data['area'][2]
        manager = ChipManagerMulti(chip, width, height, 32, 32, 128, 128, data['width'], data['height'], gameNum, 1.0, data['adjMatrix'], data['macroIndex'])
        adjMatrix = data['adjMatrix'][None, :, :]
        for index in manager.placeSequence:
            mask = manager.legalMap(index)
            node = manager.macroFeature()
            nextMacro = [[index]] * gameNum
            action = self.select_action_test(node, adjMatrix, nextMacro, mask)
            x = action // manager.y_bins
            y = action % manager.y_bins
            manager.placeItems(index, x, y)
        paramFile = './test/ispd2005/' + chip + '.json'
        params = Params.Params()
        params.load(paramFile)
        placedb = PlaceDB.PlaceDB()
        placedb(params)
        params.plot_flag = 1
        params.gpu = 1
        params.random_center_init_flag = 1
        placedb.write_pl_placed_position('./benchmarks/ispd2005/' + chip + '/' + chip + '_macro_place_test.pl', data['macroIndex'], (manager.place_xs_legalization[0][20:] - 0.5 * manager.items_width[20:]).astype(np.int), (manager.place_ys_legalization[0][20:] - 0.5 * manager.items_height[20:]).astype(np.int))
        fr = open('./benchmarks/ispd2005/' + chip + '/' + chip + '_macro_place_test.aux', 'w')
        line = 'RowBasedPlacement :  ' + chip + '.nodes  ' + chip + '.nets  ' + chip + '.wts  ' + chip + '_macro_place_test.pl  ' + chip + '.scl'
        fr.write(line)
        fr.close()
        params.aux_input = './benchmarks/ispd2005/' + chip + '/' + chip + '_macro_place_test.aux'
        metrics, _, _ = Placer.place(params)
        length = metrics[-1].hpwl.data.item()
        overflow = metrics[-1].overflow.data.item()
        macroLength = reward_macro(data, manager)
        if report:
            manager.render(0)
        return length, overflow, macroLength[0]

    def playAGameTestLarger(self, chip, report=False):
        gameNum = 1
        fileName = './processed_data/' + chip + '_clustered.npy'
        data = np.load(fileName, allow_pickle=True).item()
        width = data['area'][1] - data['area'][0]
        height = data['area'][3] - data['area'][2]
        manager = ChipManagerMulti(chip, width, height, 32, 32, 256, 256, data['width'], data['height'], gameNum, 1.0, data['adjMatrix'], data['macroIndex'])
        adjMatrix = data['adjMatrix'][None, :, :]
        for index in manager.placeSequence:
            mask = manager.legalMap(index)
            node = manager.macroFeature()
            nextMacro = [[index]] * gameNum
            action = self.select_action_test(node, adjMatrix, nextMacro, mask)
            x = action // manager.y_bins
            y = action % manager.y_bins
            manager.placeItems(index, x, y)
        paramFile = './test/ispd2005/' + chip + '.json'
        params = Params.Params()
        params.load(paramFile)
        placedb = PlaceDB.PlaceDB()
        placedb(params)
        params.plot_flag = 1
        params.gpu = 1
        params.random_center_init_flag = 1
        placedb.write_pl_placed_position('./benchmarks/ispd2005/' + chip + '/' + chip + '_macro_place_test.pl', data['macroIndex'], (manager.place_xs_legalization[0][20:] - 0.5 * manager.items_width[20:]).astype(np.int), (manager.place_ys_legalization[0][20:] - 0.5 * manager.items_height[20:]).astype(np.int))
        fr = open('./benchmarks/ispd2005/' + chip + '/' + chip + '_macro_place_test.aux', 'w')
        line = 'RowBasedPlacement :  ' + chip + '.nodes  ' + chip + '.nets  ' + chip + '.wts  ' + chip + '_macro_place_test.pl  ' + chip + '.scl'
        fr.write(line)
        fr.close()
        params.aux_input = './benchmarks/ispd2005/' + chip + '/' + chip + '_macro_place_test.aux'
        metrics, _, _ = Placer.place(params)
        length = metrics[-1].hpwl.data.item()
        overflow = metrics[-1].overflow.data.item()
        macroLength = reward_macro(data, manager)
        if report:
            manager.render(0)
        return length, overflow, macroLength[0]


    def playAGameTest_Init(self, chip, report=False):
        gameNum = 1
        fileName = './processed_data/' + chip + '_clustered.npy'
        data = np.load(fileName, allow_pickle=True).item()
        width = data['area'][1] - data['area'][0]
        height = data['area'][3] - data['area'][2]
        manager = ChipManagerMulti(chip, width, height, 32, 32, 128, 128, data['width'], data['height'], gameNum, 1.0, data['adjMatrix'], data['macroIndex'])
        adjMatrix = data['adjMatrix'][None, :, :]
        for index in manager.placeSequence:
            mask = manager.legalMap(index)
            node = manager.macroFeature()
            nextMacro = [[index]] * gameNum
            action = self.select_action_test(node, adjMatrix, nextMacro, mask)
            x = action // manager.y_bins
            y = action % manager.y_bins
            manager.placeItems(index, x, y)
        paramFile = './test/ispd2005/' + chip + '.json'
        params = Params.Params()
        params.load(paramFile)
        placedb = PlaceDB.PlaceDB()
        placedb(params)
        params.plot_flag = 1
        params.gpu = 1
        params.random_center_init_flag = 0
        placedb.write_pl_placed_free_position('./benchmarks/ispd2005/' + chip + '/' + chip + '_macro_place_test.pl', data['macroIndex'], (manager.place_xs_legalization[0][20:] - 0.5 * manager.items_width[20:]).astype(np.int), (manager.place_ys_legalization[0][20:] - 0.5 * manager.items_height[20:]).astype(np.int))
        fr = open('./benchmarks/ispd2005/' + chip + '/' + chip + '_macro_place_test.aux', 'w')
        line = 'RowBasedPlacement :  ' + chip + '_free.nodes  ' + chip + '.nets  ' + chip + '.wts  ' + chip + '_macro_place_test.pl  ' + chip + '.scl'
        fr.write(line)
        fr.close()
        params.aux_input = './benchmarks/ispd2005/' + chip + '/' + chip + '_macro_place_test.aux'
        metrics, _, _ = Placer.place(params)
        length = metrics[-1].hpwl.data.item()
        overflow = metrics[-1].overflow.data.item()
        return length, overflow

    def update(self, buffer, adjMatrix):
        self.net.train()
        trainData = {
            'nodes': np.array(buffer['nodes']),
            'canvas': np.array(buffer['canvas']),
            'nextMacros': np.array(buffer['nextMacros']),
            'actions': np.array(buffer['actions']),
            'logprobs': np.array(buffer['logprobs']),
            'rewards': np.array(buffer['rewards'])
        }
        adjMatrix = adjMatrix.repeat(self.batch_size, axis=0).astype(np.float)
        adjMatrix = torch.FloatTensor(adjMatrix).cuda()
        trainData = ChipSet(trainData)
        models = glob.glob('./model/*.model')
        trainLoader = DataLoader(trainData, batch_size=self.batch_size, shuffle=True, num_workers=8)
        project_mse = nn.MSELoss()
        for epoch in range(self.K_epochs):
            for batch in trainLoader:
                if len(batch['reward']) != self.batch_size:
                    continue
                node = torch.FloatTensor(batch['node'].float()).cuda()
                canvas = torch.FloatTensor(batch['canvas'].float()).cuda()
                macroID = torch.LongTensor(batch['nextMacro'].long()).cuda()
                actions = torch.LongTensor(batch['action'].long())[:, None].cuda()
                logprobs = torch.FloatTensor(batch['logprob'].float())[:, None].cuda()
                rewards = torch.FloatTensor(batch['reward'].float())[:, None].cuda()
                logits, entropy, value, proj = self.net(node, adjMatrix, macroID, actions)
                canvas_proj = self.net_critic.project(canvas)
                projectLoss = project_mse(proj, canvas_proj)
                ratios = torch.exp(logits - logprobs.to(logits.device).detach())
                rewards = rewards.to(value.device)
                advantages = rewards - value.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                policyLoss = - torch.mean(torch.min(surr1, surr2))
                entropyLoss = - torch.mean(entropy)
                valueLoss = torch.mean(0.5 * (rewards - value).pow(2))
                loss = policyLoss + 0.5 * valueLoss + 0.01 * projectLoss + 0.00001 * entropyLoss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                fr = open('loss.txt', 'a')
                fr.write(str(policyLoss.data.item()) + '\t' + str(valueLoss.data.item()) + '\t' + str(projectLoss.data.item()) + '\t' + str(entropyLoss.data.item()) + '\n')
                fr.close()
            self.save('./modelStep/model' + str(len(models)) + '_' + str(epoch + 1) + '.model')

    def save(self, checkpoint_path):
        torch.save(self.net.module.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        if checkpoint_path != None:
            self.net_critic.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
            self.net.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
            self.net = torch.nn.DataParallel(self.net, device_ids=[0, 1, 2, 3])






