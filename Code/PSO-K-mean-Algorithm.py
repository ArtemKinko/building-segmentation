from PIL import Image
import random
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class point:
    x = 0
    y = 0
    cr1 = 0
    cr2 = 0
    cr3 = 0

    def __init__(self, x, y, cr1, cr2, cr3):
        self.x = x
        self.y = y
        self.cr1 = cr1
        self.cr2 = cr2
        self.cr3 = cr3


def PSO_Image(image_path):
    image = Image.open(image_path, 'r')
    print(image.size)

    # структура пикселей
    pixels_list = list(image.getdata())
    pixels_array = []
    for i in range(image.size[1]):
        image_row = []
        for j in range(image.size[0]):
            image_row.append(pixels_list[i * image.size[0] + j])
        pixels_array.append(image_row)
        # print(image_row)

    # разбиваем цветовые интервалы на регионы
    regions_num = 10
    regions = []

    # находим левый и правый пределы у каждого цветового канала
    for color in range(3):
        color_list = [pixel[color] for pixel in pixels_list]
        average = math.floor((max(color_list) - min(color_list)) / regions_num)
        region = []
        left_border = 0
        for _ in range(regions_num - 1):
            region.append([left_border, left_border + average])
            left_border += average
        region.append([left_border, max(color_list)])
        regions.append(region)
    print(regions)

    # инициализация роя
    particles_num = 50  # количество частиц
    particles = [point(random.randint(0, image.size[0]),
                       random.randint(0, image.size[1]),
                       0, 0, 0) for _ in range(particles_num)]
    #
    # # задание параметров
    # iteration_num = 100  # количество итераций
    #
    # inertia_max = 1
    # inertia_min = 0
    #
    #
    #
    # def get_inertia(current_iteration):
    #     return inertia_max - current_iteration * (inertia_max - inertia_min) / iteration_num
    #
    # for iteration in range(iteration_num):
    #     inertia = get_inertia(iteration)
    #     print(inertia)

    # k-means
    k_regions = 3  # количество регионов
    data = []
    for i in range(len(pixels_array)):
        for j in range(len(pixels_array[i])):
            # data.append([i, j, pixels_array[i][j][0], pixels_array[i][j][1], pixels_array[i][j][2]])
            data.append([pixels_array[i][j][0], pixels_array[i][j][1], pixels_array[i][j][2]])
    # print(data)

    inertia = []
    kmeans = KMeans(n_clusters=k_regions)
    kmeans.fit(data)
    inertia.append(kmeans.inertia_)

    cluster_centers = kmeans.cluster_centers_
    colors = [(255, 0, 0),
              (0, 255, 0),
              (0, 0, 255),
              (255, 255, 0),
              (255, 0, 255),
              (0, 255, 255),
              (0, 0, 0),
              (255, 255, 255),
              (100, 20, 100),
              (55, 255, 0),
              (0, 55, 255),
              (255, 255, 100),
              (55, 0, 55),
              (0, 100, 100),
              (100, 100, 0),
              (100, 100, 100),
              (255, 0, 0),
              (0, 255, 0),
              (0, 0, 255),
              (255, 255, 0),
              (255, 0, 255),
              (0, 255, 255)
              ]

    # colors = [(math.floor(cluster_centers[i][0]),
    #            math.floor(cluster_centers[i][1]),
    #            math.floor(cluster_centers[i][2])) for i in range(k_regions)]

    # изменение цветов на изображении
    new_pixels = []
    for i in range(image.size[1]):
        for j in range(image.size[0]):
            pixel = pixels_array[i][j]

            min_value = 0
            current_cluster = -1
            # проверяем, к какому кластеру относится пиксель
            for cluster_num in range(k_regions):
                # для 5 - 5
                # value = pow(i - cluster_centers[cluster_num][0], 2) + \
                #     pow(j - cluster_centers[cluster_num][1], 2) + \
                #     pow(pixel[0] - cluster_centers[cluster_num][2], 2) + \
                #     pow(pixel[1] - cluster_centers[cluster_num][3], 2) + \
                #     pow(pixel[2] - cluster_centers[cluster_num][4], 2)
                # для 5 - 3
                # value = pow(pixel[0] - cluster_centers[cluster_num][2], 2) + \
                #         pow(pixel[1] - cluster_centers[cluster_num][3], 2) + \
                #         pow(pixel[2] - cluster_centers[cluster_num][4], 2)
                # для 3 - 3
                value = pow(pixel[0] - cluster_centers[cluster_num][0], 2) + \
                        pow(pixel[1] - cluster_centers[cluster_num][1], 2) + \
                        pow(pixel[2] - cluster_centers[cluster_num][2], 2)
                if current_cluster == -1:
                    current_cluster = cluster_num
                    min_value = value
                else:
                    if value < min_value:
                        current_cluster = cluster_num
                        min_value = value
            new_pixels.append(colors[current_cluster])
    new_image = Image.new("RGB", (image.size[0], image.size[1]))
    new_image.putdata(new_pixels)
    new_image.save(image_path + "NEW.png")

    # print(kmeans.cluster_centers_)





PSO_Image('../Dataset/l1.png')
