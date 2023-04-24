import torch
import torch.nn as nn
import numpy as np
from network import Network
from chipManagerMulti import ChipManagerMulti
import glob
import multiprocessing
from multiprocessing import Process
import dreamplace.PlaceDB as PlaceDB
import dreamplace.Params as Params
import dreamplace.Placer as Placer


def reward_cal(data, manager):
    x = (manager.place_xs_legalization).astype(np.int64)
    y = (manager.place_ys_legalization).astype(np.int64)
    wls = np.zeros(manager.num)
    congestion = np.zeros((manager.num, manager.chip_width, manager.chip_height))
    for net_id in range(len(data['nets'])):
        pins = np.array(data['nets'][net_id]).astype(np.int64)
        nodes = data['pin2node'][pins]
        for i in range(manager.num):
            right = int(np.max(x[i, nodes] + data['off_x'][pins]))
            left = int(np.min(x[i, nodes] + data['off_x'][pins]))
            bottom = int(np.min(y[i, nodes] + data['off_y'][pins]))
            up = int(np.max(y[i, nodes] + data['off_y'][pins]))
            hpwl_x = right - left
            hpwl_y = up - bottom
            wls[i] += (hpwl_x + hpwl_y)
            if hpwl_x != 0 and hpwl_y != 0:
                congestion[i, left: right + 1, bottom: up + 1] += ((hpwl_x + hpwl_y) / (hpwl_x * hpwl_y))
    congestion = np.array([np.mean(congestion[i]) for i in range(manager.num)])
    congestion2 = np.zeros(manager.num)
    for i in range(manager.num):
        con = list(congestion[i].flatten())
        con.sort(reverse=True)
        congestion2[i] = np.mean(con[: manager.chip_width])
    return wls, congestion, congestion2


def reward_macro(data, manager):
    x = (manager.place_xs_legalization).astype(np.int64)
    y = (manager.place_ys_legalization).astype(np.int64)
    wls = np.zeros(manager.num)
    for net_id in range(len(data['macroNets'])):
        pins = np.array(data['macroNets'][net_id]).astype(np.int64)
        nodes = data['pin2node'][pins]
        for i in range(manager.num):
            right = int(np.max(x[i, nodes] + data['off_x'][pins]))
            left = int(np.min(x[i, nodes] + data['off_x'][pins]))
            bottom = int(np.min(y[i, nodes] + data['off_y'][pins]))
            up = int(np.max(y[i, nodes] + data['off_y'][pins]))
            hpwl_x = right - left
            hpwl_y = up - bottom
            wls[i] += (hpwl_x + hpwl_y)
    return wls


def select_action(net, gpu_num, node, adjMatrix, nextMacro, mask, canvas):
    with torch.no_grad():
        node = torch.FloatTensor(node).cuda(gpu_num)
        adjMatrix = torch.FloatTensor(adjMatrix).cuda(gpu_num)
        nextMacro = torch.LongTensor(nextMacro).cuda(gpu_num)
        mask = torch.FloatTensor(mask).cuda(gpu_num)
        canvas = torch.FloatTensor(canvas).cuda(gpu_num)
        action, logits, _, _, proj = net.evaluate(node, adjMatrix, nextMacro, mask)
        canvas_proj = net.projection(canvas)
        intrinsic_reward = (proj - canvas_proj).pow(2).mean(1)
        action = action.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        intrinsic_reward = intrinsic_reward.detach().cpu().numpy()
    return action.reshape(-1), logits.reshape(-1), intrinsic_reward


def playGame(gpu_num, chip, grid, gameNum=1, thread=0):
    models = glob.glob('./model/*.model')
    net = Network(dimension=32, grid=grid).cuda(gpu_num)
    if len(models) > 0:
        modelName = './model/model' + str(len(models) - 1) + '.model'
        net.load_state_dict(torch.load(modelName, map_location={'cuda:0': 'cuda:' + str(gpu_num)}))
    fileName = './processed_data/' + chip + '_clustered.npy'
    data = np.load(fileName, allow_pickle=True).item()
    width = data['area'][1] - data['area'][0]
    height = data['area'][3] - data['area'][2]
    manager = ChipManagerMulti(chip, width, height, 32, 32, 128, 128, data['width'], data['height'], gameNum, 1.0, data['adjMatrix'], data['macroIndex'])
    adjMatrix = data['adjMatrix'][None, :, :].repeat(gameNum, axis=0)
    buffer = {
        'actions': np.array([]),
        'nodes': np.array([]),
        'canvas': np.array([]),
        'nextMacros': np.array([]),
        'logprobs': np.array([]),
        'rewards': np.array([]),
    }
    index = manager.placeSequence[0]
    mask = manager.legalMap(index)
    node = manager.macroFeature()
    nextMacro = np.array([[index]] * gameNum)
    action, logit, intrinsic_reward = select_action(net, gpu_num, node, adjMatrix, nextMacro, mask, manager.density[:, None, :, :])
    buffer['actions'] = action.copy()
    buffer['logprobs'] = logit.copy()
    buffer['nextMacros'] = nextMacro.copy()
    buffer['nodes'] = node.copy()
    buffer['canvas'] = manager.density[:, None, :, :].copy()
    buffer['rewards'] = np.array([])
    x = action // manager.y_bins
    y = action % manager.y_bins
    manager.placeItems(index, x, y)
    reward = np.zeros((manager.num_macros, gameNum))
    step = 0
    for index in manager.placeSequence[1:]:
        mask = manager.legalMap(index)
        node = manager.macroFeature()
        nextMacro = np.array([[index]] * gameNum)
        action, logit, intrinsic_reward = select_action(net, gpu_num, node, adjMatrix, nextMacro, mask, manager.density[:, None, :, :])
        buffer['actions'] = np.append(buffer['actions'], action)
        buffer['logprobs'] = np.append(buffer['logprobs'], logit)
        buffer['nodes'] = np.vstack((buffer['nodes'], node.copy()))
        buffer['canvas'] = np.vstack((buffer['canvas'], manager.density[:, None, :, :].copy()))
        buffer['nextMacros'] = np.append(buffer['nextMacros'], nextMacro)
        reward[step, :] = intrinsic_reward.copy()
        step += 1
        x = action // manager.y_bins
        y = action % manager.y_bins
        manager.placeItems(index, x, y)
    node = manager.macroFeature()
    node = torch.FloatTensor(node).cuda(gpu_num)
    adjMatrix = torch.FloatTensor(adjMatrix).cuda(gpu_num)
    canvas = torch.FloatTensor(manager.density[:, None, :, :]).cuda(gpu_num)
    proj = net.reconProject(node, adjMatrix)
    canvas_proj = net.projection(canvas)
    intrinsic_reward = (proj - canvas_proj).pow(2).mean(1)
    intrinsic_reward = intrinsic_reward.detach().cpu().numpy()
    reward[step, :] = intrinsic_reward.copy()
    length = []
    overflow = []
    macroLength = reward_macro(data, manager) / 5000000
    for i in range(gameNum):
        paramFile = './test/ispd2005/' + chip + '.json'
        params = Params.Params()
        params.load(paramFile)
        placedb = PlaceDB.PlaceDB()
        placedb(params)
        params.plot_flag = 0
        params.gpu = 1
        params.random_center_init_flag = 1
        placedb.write_pl_placed_position('./benchmarks/ispd2005/' + chip + '/' + chip + '_macro_place_' + str(thread) + '.pl', data['macroIndex'], (manager.place_xs_legalization[i][20:] - 0.5 * manager.items_width[20:]).astype(np.int), (manager.place_ys_legalization[i][20:] - 0.5 * manager.items_height[20:]).astype(np.int))
        fr = open('./benchmarks/ispd2005/' + chip + '/' + chip + '_macro_place_' + str(thread) + '.aux', 'w')
        line = 'RowBasedPlacement :  ' + chip + '.nodes  ' + chip + '.nets  ' + chip + '.wts  ' + chip + '_macro_place_' + str(thread) + '.pl  ' + chip + '.scl'
        fr.write(line)
        fr.close()
        params.aux_input = './benchmarks/ispd2005/' + chip + '/' + chip + '_macro_place_' + str(thread) + '.aux'
        metrics, _, _ = Placer.place(params)
        length.append(metrics[-1].hpwl.data.item())
        overflow.append(metrics[-1].overflow.data.item())
    baseline = 175375600.0
    reward = 0.5 * (reward - np.min(reward) )/ (np.max(reward) - np.min(reward))
    final_reward = np.zeros((manager.num_macros, gameNum))
    final_reward[-1, :] = - 100 * (np.array(length) - baseline) / baseline + reward[-1, :] - macroLength
    for step in range(manager.num_macros - 2, -1, -1):
        final_reward[step, :] = 0.997 * final_reward[step + 1, :] + reward[step, :]
    buffer['rewards'] = final_reward.flatten()
    fileName = './data/data' + str(len(models)) + '_' + str(thread) + '.npy'
    np.save(fileName, buffer)
    line = ''
    for i in range(gameNum):
        line += (str(length[i]) + '\t' + str(overflow[i]) + '\n')
    fr = open('Chip_' + chip + '.txt', 'a')
    fr.write(line)
    fr.close()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    devices = [1, 2, 3]
    parallel = 6
    chip = 'adaptec3'
    jobs = [Process(target=playGame, args=(devices[i % 3], chip, 32, 18, i)) for i in range(parallel)]
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()
