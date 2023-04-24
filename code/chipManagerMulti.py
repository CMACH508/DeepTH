import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpathes


def plotCube(ax, x, y, width, height):
    xs = [x, x + width, x + width, x, x]
    ys = [y, y, y + height, y + height, y]
    ax.plot(xs, ys, c='k', linewidth=0.5)


class ChipManagerMulti:
    def __init__(self, name, chip_width, chip_height, x_bins, y_bins, x_bins_legalization, y_bins_legalization, items_width, items_height, num, density_tr, adj_matrix, macro_index):
        self.name = name
        self.density_tr = density_tr
        self.num = num
        self.num_macros = len(items_width)
        self.chip_width = int(chip_width)
        self.chip_height = int(chip_height)
        self.x_bins = x_bins
        self.y_bins = y_bins
        self.x_bin_size = self.chip_width / self.x_bins
        self.y_bin_size = self.chip_height / self.y_bins
        self.bin_area = self.x_bin_size * self.y_bin_size
        self.x_bins_legalization = x_bins_legalization
        self.y_bins_legalization = y_bins_legalization
        self.x_bin_size_legalization = self.chip_width / self.x_bins_legalization
        self.y_bin_size_legalization = self.chip_height / self.y_bins_legalization
        self.bin_factor_x = self.x_bins_legalization // self.x_bins
        self.bin_factor_y = self.y_bins_legalization // self.y_bins
        self.items_width = items_width
        self.items_height = items_height
        self.place_xs = np.zeros((num, self.num_macros))
        self.place_ys = np.zeros((num, self.num_macros))
        self.place_xs_legalization = np.zeros((num, self.num_macros))
        self.place_ys_legalization = np.zeros((num, self.num_macros))
        self.density = np.zeros((num, self.x_bins, self.y_bins))
        self.canvas_legalization = np.zeros((num, self.x_bins_legalization, self.y_bins_legalization))
        self.canvas = np.zeros((num, self.chip_width, self.chip_height))
        self.is_placed = np.zeros(self.num_macros)
        self.node_type = np.zeros(self.num_macros)
        self.node_type[20:] = 1
        self.adj_matrix = adj_matrix
        self.connections = np.sum(self.adj_matrix, axis=0)
        self.area = self.items_width * self.items_height
        isolated_macros = np.array([i for i in range(20, self.num_macros) if np.sum(self.adj_matrix[i]) == 0])
        connected_macros = np.array([i for i in range(20, self.num_macros) if np.sum(self.adj_matrix[i]) != 0])
        if len(isolated_macros) == 0:
            self.placeSequence = connected_macros[np.argsort(-self.area[connected_macros])]
        else:
            self.placeSequence = np.append(connected_macros[np.argsort(-self.area[connected_macros])], isolated_macros[np.argsort(-self.area[isolated_macros])])
        self.placeSequence = np.append([i for i in range(20)], self.placeSequence)
        self.baseFeature = np.hstack((self.items_width[:, None], self.items_height[:, None], self.connections[:, None], self.node_type[:, None]))[None, :, :].repeat(self.num, axis=0)
        self.macros = macro_index.copy()
        self.macro_index = {}
        for i in range(len(macro_index)):
            self.macro_index[macro_index[i]] = i
        self.cluster_mask = self.processMask(8)

    def __deepcopy__(self, memodict={}):
        new_chip = ChipManagerMulti(self.name, self.chip_width, self.chip_height, self.x_bins, self.y_bins, self.x_bins_legalization, self.y_bins_legalization, self.items_width, self.items_height, self.num, self.density_tr, self.adj_matrix, self.macros)
        new_chip.place_xs = self.place_xs.copy()
        new_chip.place_ys = self.place_ys.copy()
        new_chip.place_xs_legalization = self.place_xs_legalization.copy()
        new_chip.place_ys_legalization = self.place_ys_legalization.copy()
        new_chip.density = self.density.copy()
        new_chip.canvas_legalization = self.canvas_legalization.copy()
        new_chip.canvas = self.canvas.copy()
        new_chip.is_placed = self.is_placed.copy()
        return new_chip

    def processMask(self, d):
        mask = np.zeros((self.x_bins, self.y_bins))
        mask[:d, :] = 1
        mask[-d:, :] = 1
        mask[:, :d] = 1
        mask[:, -d:] = 1
        return mask

    def macroFeature(self):
        placedFeature = self.is_placed[None, :, None].repeat(self.num, axis=0)
        feature = np.concatenate([self.baseFeature, placedFeature, self.place_xs[:, :, None], self.place_ys[:, :, None]], axis=2)
        return feature

    def placeItem(self, gameIndex, item, x, y):
        self.place_xs[gameIndex, item] = x
        self.place_ys[gameIndex, item] = y
        if self.area[item] <= self.bin_area:
            self.density[int(gameIndex), x, y] += (self.area[item] / self.bin_area)
        else:
            x_bins = self.items_width[item] // self.x_bin_size
            y_bins = self.items_height[item] // self.y_bin_size
            left = np.max((0, int(x - x_bins // 2)))
            right = np.min((self.x_bins, int(left + x_bins + 1)))
            bottom = np.max((0, int(y - y_bins // 2)))
            up = np.min((self.y_bins, int(bottom + y_bins + 1)))
            self.density[int(gameIndex), left: right, bottom: up] = 1
        position_x, position_y = self.legalization(gameIndex, item, x, y)
        if position_x == -1 and position_y == -1:
            print('Macro ' + str(item) + ' can not be placed.')
        else:
            self.place_xs_legalization[gameIndex, item] = position_x
            self.place_ys_legalization[gameIndex, item] = position_y
            left = int(np.floor(position_x - 0.5 * self.items_width[item]))
            right = int(np.ceil(position_x + 0.5 * self.items_width[item]))
            bottom = int(np.floor(position_y - 0.5 * self.items_height[item]))
            up = int(np.ceil(position_y + 0.5 * self.items_height[item]))
            self.canvas[gameIndex, left: right + 1, bottom: up + 1] = 1

    def placeLargerCellCluster(self, gameIndex, item, x, y):
        score = self.area[item] / self.bin_area - (self.density_tr - self.density[int(gameIndex), x, y])
        self.density[int(gameIndex), x, y] = self.density_tr
        for distance in range(1, int(np.max([self.x_bins - x, x])) + int(np.max([self.y_bins - y, y]))):
            for i in range(distance + 1):
                j = distance - i
                for y_sign in [1, -1]:
                    if y + y_sign * j < 0 or y + y_sign * j >= self.y_bins:
                        continue
                    for x_sign in [1, -1]:
                        if x + x_sign * j < 0 or x + x_sign * i >= self.x_bins:
                            continue
                        fillin = np.min((score, self.density_tr - self.density[int(gameIndex), x + x_sign * i, y + y_sign * j]))
                        score = score - fillin
                        self.density[int(gameIndex), x + x_sign * i, y + y_sign * j] += fillin
                        if score <= 0:
                            return

    def placeCellCluster(self, gameIndex, item, x, y):
        self.place_xs[gameIndex, item] = x
        self.place_ys[gameIndex, item] = y
        if self.area[item] <= self.bin_area:
            self.density[int(gameIndex), x, y] += (self.area[item] / self.bin_area)
        else:
            self.placeLargerCellCluster(gameIndex, item, x, y)

    def placeItems(self, item, xs, ys):
        self.is_placed[item] = 1
        for i in range(self.num):
            if self.node_type[item] == 1:
                self.placeItem(i, item, xs[i], ys[i])
            else:
                self.placeCellCluster(i, item, xs[i], ys[i])

    def legalMap(self, item):
        if self.area[item] > self.bin_area:
            mask = np.zeros((self.num, self.x_bins, self.y_bins))
            for k in range(self.num):
                for i in range(self.x_bins):
                    for j in range(self.y_bins):
                        left = int(np.max([i - 1, 0]))
                        right = int(np.min([i + 2, self.x_bins]))
                        bottom = int(np.max([j - 1, 0]))
                        up = int(np.min([j + 2, self.y_bins]))
                        mask[k, i, j] = np.sum(self.density_tr - self.density[k, left:right, bottom: up])
            mask = (mask < self.area[item] / self.bin_area).astype(np.int)
            for k in range(self.num):
                if np.sum(1 - mask[k]) == 0:
                    mask[k] = (self.density[k] > 0.8 * self.density_tr).astype(np.int)
        else:
            mask = self.density_tr - self.density.copy()
            mask = (mask < self.area[item] / self.bin_area).astype(np.int)
        if item < 20:
            for k in range(self.num):
                mask_k = (mask[k] + self.cluster_mask > 0).astype(np.int)
                if np.sum(1 - mask_k) == 0:
                    mask_k = mask[k]
                mask[k] = mask_k.copy()
        return mask

    def isLegal(self, gameIndex, item, x, y):
        left = int(np.floor(x - 0.5 * self.items_width[item]))
        right = int(np.ceil(x + 0.5 * self.items_width[item]))
        bottom = int(np.floor(y - 0.5 * self.items_height[item]))
        up = int(np.ceil(y + 0.5 * self.items_height[item]))
        if left >= 0 and right < self.chip_width and bottom >= 0 and up < self.chip_height:
            return np.sum(self.canvas[gameIndex, left: right + 1, bottom: up + 1]) == 0
        else:
            return False

    def legalization(self, gameIndex, item, x, y):
        x = (x + 0.5) * self.bin_factor_x
        y = (y + 0.5) * self.bin_factor_y
        for distance in range(int(np.max([self.x_bins_legalization - x, x])) + int(np.max([self.y_bins_legalization - y, y]))):
            for i in range(np.min((distance + 1, self.x_bins_legalization))):
                j = distance - i
                if j < 0 or j >= self.y_bins_legalization:
                    continue
                for y_sign in [1, -1]:
                    for x_sign in [1, -1]:
                        center_x = (x + x_sign * i + 0.5) * self.x_bin_size_legalization
                        center_y = (y + y_sign * j + 0.5) * self.y_bin_size_legalization
                        if self.isLegal(gameIndex, item, center_x, center_y):
                            return center_x, center_y
        return -1, -1

    def isFinished(self):
        return np.sum(self.is_placed) == self.num_macros

    def render(self, gameIndex):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, self.chip_width)
        ax.set_ylim(0, self.chip_height)
        rect = mpathes.Rectangle((0, 0), self.chip_width, self.chip_height, color='silver')
        ax.add_patch(rect)
        for item in range(self.num_macros):
            placedWidth = self.items_width[item]
            placedHeight = self.items_height[item]
            plotCube(ax, self.place_xs[gameIndex, item], self.place_ys[gameIndex, item], placedWidth, placedHeight)
            rect = mpathes.Rectangle((self.place_xs[gameIndex, item], self.place_ys[gameIndex, item]), placedWidth, placedHeight,
                                     color='b')
            ax.add_patch(rect)
        #for item in self.placedItems[:20]:
        #    placedWidth = self.items_width[item]
        #    placedHeight = self.items_height[item]
        #    plotCube(ax, self.place_xs[gameIndex, item], self.place_ys[gameIndex, item] - self.items_width[item] // 2, placedWidth - self.items_height[item] // 2, placedHeight)
        #    rect = mpathes.Rectangle((self.place_xs[gameIndex, item] - self.items_width[item] // 2, self.place_ys[gameIndex, item] - self.items_height[item] // 2), placedWidth, placedHeight, color='r')
        #    ax.add_patch(rect)
        plt.savefig(self.name + '_placement_' + str(gameIndex) + '.png')
        plt.close()
