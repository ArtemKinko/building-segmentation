from PIL import Image
import random
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time


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
    start_time = time.time()

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
    new_pixels_array = []
    new_pixels = []
    for i in range(image.size[1]):
        temp_pixel_array = []
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
            temp_pixel_array.append(colors[current_cluster])
        new_pixels_array.append(temp_pixel_array)
    elapsed_time = time.time() - start_time
    print(
        "Execution time: " + str(int(elapsed_time / 60)) + " minute(s) and " + str(
            int(elapsed_time % 60)) + " seconds")

    new_image = Image.new("RGB", (image.size[0], image.size[1]))
    new_image.putdata(new_pixels)
    new_image.save(image_path + "NEW.png")


    # черно-белое изображение
    # координаты точно здания
    x_color = 180
    y_color = 700
    color = new_pixels_array[x_color][y_color]
    print(color)
    black_pixels = []
    black_pixels_array = []
    for i in range(image.size[1]):
        temp_black_pixel_array = []
        for j in range(image.size[0]):
            pixel = new_pixels_array[i][j]
            if pixel[0] == color[0] and pixel[1] == color[1] and pixel[2] == color[2]:
                black_pixels.append([0, 0, 0])
                temp_black_pixel_array.append([0, 0, 0])
            else:
                black_pixels.append([255, 255, 255])
                temp_black_pixel_array.append([255, 255, 255])
        black_pixels_array.append(temp_black_pixel_array)

    # каждые 100x100 пикселей считаем количество черных
    pixel_scale = 100
    colors = []
    for i in range(0, image.size[1], pixel_scale):
        temp_colors = []
        for j in range(0, image.size[0], pixel_scale):

            pixel_num = 0
            black_pixel_num = 0


            # ОБРАБОТАТЬ Out Of Range НА ГРАНИЦАХ


            for x in range(i, i + pixel_scale):
                for y in range(j, j + pixel_scale):
                    pixel = black_pixels_array[x][y]
                    if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                        black_pixel_num += 1
                    pixel_num += 1
            percent = black_pixel_num / pixel_num
            color = int(255 * percent)
            temp_colors.append(color)
        colors.append(temp_colors)

    # закрашиваем изображение соответствующим цветом
    picture_pixels_array = []
    for i in range(0, image.size[1]):
        temp_picture_pixels_array = []
        for j in range(0, image.size[0]):
            current_color = colors[math.floor(i / 100)][math.floor(j / 100)]
            temp_picture_pixels_array.append([current_color, 50, 50])
        picture_pixels_array.append(temp_picture_pixels_array)



    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    plt.imshow(pixels_array)
    a.set_title('Original Image')
    a = fig.add_subplot(2, 2, 2)
    plt.imshow(new_pixels_array)
    a.set_title('Segmented Image')
    a = fig.add_subplot(2, 2, 3)
    plt.imshow(black_pixels_array)
    a.set_title('Segmented Image with 2 colors')
    a = fig.add_subplot(2, 2, 4)
    plt.imshow(picture_pixels_array)
    a.set_title('Picture')
    plt.show()

    # print(kmeans.cluster_centers_)





PSO_Image('../Dataset/l1.png')
