from PPO import PPO
import glob
import numpy as np
import os

if __name__ == '__main__':
    models = glob.glob('./model/*.model')
    if len(models) < 51:
        lr = 0.001
    elif len(models) < 101:
        lr = 0.0001
    else:
        lr = 0.00002
    player = PPO(gpu_num=0, lr=lr, grid=32, K_epochs=1, eps_clip=0.00001)
    if len(models) > 0:
        player.load('./model/model' + str(len(models) - 1) + '.model')
    else:
        player.load(None)
    buffer = {
        'actions': np.array([]),
        'nodes': np.array([]),
        'canvas': np.array([]),
        'nextMacros': np.array([]),
        'logprobs': np.array([]),
        'rewards': np.array([])
    }
    for i in range(6):
        file = './data/data' + str(len(models)) + '_' + str(i) + '.npy'
        data = np.load(file, allow_pickle=True).item()
        buffer['actions'] = np.append(buffer['actions'], data['actions'])
        buffer['logprobs'] = np.append(buffer['logprobs'], data['logprobs'])
        buffer['rewards'] = np.append(buffer['rewards'], data['rewards'])
        buffer['nextMacros'] = np.append(buffer['nextMacros'], data['nextMacros'])
        if len(buffer['nodes']) == 0:
            buffer['nodes'] = data['nodes'].copy()
            buffer['canvas'] = data['canvas'].copy()
        else:
            buffer['nodes'] = np.vstack((buffer['nodes'], data['nodes']))
            buffer['canvas'] = np.vstack((buffer['canvas'], data['canvas']))
        os.remove(file)
    fileName = './processed_data/adaptec3_clustered.npy'
    chipInfo = np.load(fileName, allow_pickle=True).item()
    adjMatrix = chipInfo['adjMatrix'][None, :, :]
    player.update(buffer, adjMatrix)
    player.save('./model/model' + str(len(models)) + '.model')
