from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import PIL
import random

img = PIL.Image.open('C:/Users/833161/OneDrive - Loudoun County Public Schools/Documents/Python Scripts/generative/syntheticData/data-viz.jpg')
w = img.width
h = img.height
img_arr = np.array(img)

starting_means = []

for i in range(5):
    rand_i = random.randint(0, h-1)
    rand_j = random.randint(0, w-1)

    starting_means.append(img_arr[rand_i][rand_j])

groups = [[1], [1, 2], [3], [4], [5]]

def get_closest_mean(means, pixel):
    diffs = {}
    for i in range(len(means)):
        diffs.update({means[i][0] - pixel[0] + means[i][1] - pixel[1] + means[i][2] - pixel[2]: i+1})
    
    print(diffs)
    print(diffs[min(diffs.keys())] - 1)
    return diffs[min(diffs.keys())] - 1

def calculate_new_means(arr):
    new_means = []
    for group in arr:
        global_rgb = [0, 0, 0]
        for i in range(len(group)):
            r, g, b = group[i][1]
            global_rgb[0] += r
            global_rgb[1] += g
            global_rgb[2] += b
        
            new_means.append([global_rgb[0] // len(group), global_rgb[1] // len(group), global_rgb[2] // len(group)])
    
    return new_means

def recursive_K_Means(means, img_arr, pixel_groups):
    print([len(pixel_groups[0]), len(pixel_groups[1]), len(pixel_groups[2]), len(pixel_groups[3]), len(pixel_groups[4])])
    if len(pixel_groups[0]) == len(pixel_groups[1]) == len(pixel_groups[2]) == len(pixel_groups[3]) == len(pixel_groups[4]):
        return img_arr
    else:
        pixel_groups = [[], [], [], [], []]
        for i in range(len(img_arr)):
            for j in range(len(img_arr[i])):
                index = get_closest_mean(means, img_arr[i][j])
                try:
                    pixel_groups[index].append([(i, j), img_arr[i][j]])
                except:
                    print(index)

        means = calculate_new_means(pixel_groups)

        recursive_K_Means(means, img_arr, pixel_groups)

final_img_arr = recursive_K_Means(starting_means, img_arr, groups)

img_data = PIL.Image.fromarray(final_img_arr)

img_data.save('final_img.png')

'''
r = []
g = []
b = []
for i in range(len(img_arr)):
    for j in range(len(img_arr[i])):
        pixel = img_arr[i][j]
        if j % 1 == 0:
            r.append(pixel[0])
            g.append(pixel[1])
            b.append(pixel[2])

ax = plt.axes(projection='3d')
ax.scatter3D(r, g, b)
plt.show()
'''