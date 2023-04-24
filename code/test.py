from PPO import PPO
import glob


if __name__ == "__main__":
    models = glob.glob('./model/*.model')
    for i in range(1):
        player = PPO(gpu_num=0, grid=32, lr=0.0001, K_epochs=1, eps_clip=0.00001)
        player.load('./modelStep/model' + str(len(models) - 1) + '_' + str(i + 1) + '.model')
        fr = open('test.txt', 'a')
        length, overflow, macroLength = player.playAGameTest('adaptec3')
        fr.write(str(len(models)) + '\t' + str(i) + '\t' + str(length) + '\t' + str(overflow) + '\t' + str(macroLength) + '\n')
        fr.close()
