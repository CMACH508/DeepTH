import numpy as np
import dreamplace.PlaceDB as PlaceDB
import dreamplace.Params as Params
import dreamplace.Placer as Placer


def macroConnection(chip):
    paramFile = './test/ispd2005/' + chip + '.json'
    params = Params.Params()
    params.load(paramFile)
    placedb = PlaceDB.PlaceDB()
    placedb(params)
    macroIndex = np.array([i for i in range(placedb.num_physical_nodes)])
    macroIndex = macroIndex[placedb.node_size_y[: placedb.num_physical_nodes] > 12]
    macro2index = {}
    nets = []
    for i in range(len(macroIndex)):
        macro2index[macroIndex[i]] = i
    for i in range(len(placedb.net2pin_map)):
        nodes = [placedb.pin2node_map[pin] for pin in placedb.net2pin_map[i] if placedb.pin2node_map[pin] in macroIndex]
        pins = [pin for pin in placedb.net2pin_map[i] if placedb.pin2node_map[pin] in macroIndex]
        if len(set(nodes)) > 1:
            nets.append(pins)
    return nets

def place(chip, free=False):
    paramFile = './test/ispd2005/' + chip + '.json'
    params = Params.Params()
    params.load(paramFile)
    if free:
        params.aux_input = "benchmarks/ispd2005/" + chip + "/" + chip + "_free.aux"
    params.plot_flag = 1
    metrics, _, _ = Placer.place(params)
    return metrics[-1].hpwl.item()


if __name__ == "__main__":
    for _ in range(20):
        length = place('adaptec4', True)
        fr = open('adaptec4_free.txt', 'a')
        fr.write(str(length) + '\n')
        fr.close()
    #print(place('bigblue1', False))
    '''
    chips = ['bigblue3']
    for chip in chips:
        print(chip)
        file = './processed_data/Global_' + chip + '.npy'
        data = np.load(file, allow_pickle=True).item()
        new_data = {}
        new_data['area'] = data['area']
        new_data['width'] = data['width']
        new_data['height'] = data['height']
        new_data['off_x'] = data['off_x']
        new_data['off_y'] = data['off_y']
        new_data['adjMatrix'] = data['adjMatrix']
        new_data['nets'] = data['nets']
        new_data['net2pin'] = data['net2pin']
        new_data['pin2node'] = data['pin2node']
        if 'macroNets' not in data.keys():
            new_data['macroNets'] = macroConnection(chip)
        else:
            new_data['macroNets'] = data['macroNets']
        file = './processed_data/' + chip + '.npy'
        data2 = np.load(file, allow_pickle=True).item()
        new_data['macroIndex'] = data2['macroIndex']
        file = './processed_data/' + chip + '_clustered.npy'
        np.save(file, new_data)
    '''