from PIL import Image
import random
import cv2 as cv
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import time


class particle:
    red = 0
    green = 0
    blue = 0

    def __init__(self, cr1, cr2, cr3):
        self.red = cr1
        self.green = cr2
        self.blue = cr3


class swarm:
    pixel_image = []  # список из пикселей всего изображения
    particles_num = 50  # количество частиц
    clusters_num = 3  # число кластеров
    particles = []  # список частиц длиной по количеству кластеров, каждый элемент которого
    # представляет теоретический центр кластера
    z_max = 2 ** 24 - 1  # константа, необходимая для вычисления целевой функции
    omega_1 = 0.4  # коэффициент учета максимального евклидово расстояния от частиц до кластеров
    omega_2 = 0.1  # коэффициент учета минимального евклидово расстояния между парами кластерных центров

    def create_random_particle(self):
        return particle(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    def get_euclidean_between_particle_and_pixel(self, particle1, pixel2):
        # получаем евклидово расстояние между двумя частицами
        return math.sqrt((particle1.red - pixel2[0]) ** 2 +
                         (particle1.green - pixel2[1]) ** 2 +
                         (particle1.blue - pixel2[2]) ** 2)

    def generate_cluster_link(self, current_particle, image):
        # определяем, к какому кластеру принадлежит каждый пиксель
        link_list = []
        for pixel in image:
            best_link = 0
            max_d = self.get_euclidean_between_particle_and_pixel(current_particle[0], pixel)
            for i in range(self.clusters_num):
                current_distance = self.get_euclidean_between_particle_and_pixel(current_particle[i], pixel)
                if current_distance <= max_d:
                    best_link = i
                    max_d = current_distance
            link_list.append(best_link)
        return link_list

    def calculate_euclidean_particles(self, current_particle):
        # ищем максимальное среднее евклидово расстояние от пикселей до кластеров
        max_euclidean = -1
        for cluster in range(self.clusters_num):
            summary_distance = 0
            count_pixels = 0
            link_list = self.generate_cluster_link(current_particle, self.pixel_image)
            for i in range(len(self.pixel_image)):
                if link_list[i] == cluster:
                    summary_distance += self.get_euclidean_between_particle_and_pixel(current_particle[cluster],
                                                                                      self.pixel_image[i])
                    count_pixels += 1
            print("Count pixels for cluster", cluster, "is:", count_pixels)
            if summary_distance >= max_euclidean:
                max_euclidean = summary_distance
        return max_euclidean

    def get_euclidean_between_particles(self, particle1, particle2):
        # получаем евклидово расстояние между двумя частицами
        return math.sqrt((particle1.red - particle2.red) ** 2 +
                         (particle1.green - particle2.green) ** 2 +
                         (particle1.blue - particle2.blue) ** 2)
    def calculate_euclidean_clusters(self, current_particle):
        for i in range()


    def __init__(self, particles_num, cluster_num, image):
        self.pixel_image = self.pixel_image
        self.particles_num = particles_num
        self.clusters_num = cluster_num
        self.pixel_image = image
        self.particles = [[swarm.create_random_particle(self) for _ in range(cluster_num)]
                          for _ in range(particles_num)]


def PSO_Image(image_path):
    image = Image.open(image_path, 'r')
    start_time = time.time()

    # структура пикселей
    pixels_list = list(image.getdata())
    pixels_array = []

    image_rs = []
    image_gs = []
    image_bs = []
    for i in range(image.size[1]):
        image_row = []
        for j in range(image.size[0]):
            image_row.append(pixels_list[i * image.size[0] + j])
            image_rs.append(pixels_list[i * image.size[0] + j][0])
            image_gs.append(pixels_list[i * image.size[0] + j][1])
            image_bs.append(pixels_list[i * image.size[0] + j][2])
        pixels_array.append(image_row)

    # ---------------------- PSO-K-MEANS

    particle_swarm = swarm(5, 3, pixels_list)
    print(particle_swarm.calculate_euclidean_particles(particle_swarm.particles[0]))

    # инициализация роя
    # количество частиц
    particles_num = 50
    particles = [point(random.randint(0, 255),
                       random.randint(0, 255),
                       random.randint(0, 255)) for _ in range(particles_num)]

    # инициализация скоростей
    speeds = []

    # количество кластеров
    k_regions = 6

    # ускорения (личная и глобальная составляющие)
    acceleration_c1 = 0.3  # от 0 до 1
    acceleration_c1 = 0.7  # от 0 до 1

    # максимальное количество итераций
    max_iterations = 100

    best_local_points = particles.copy()

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
    colors_rgb = [(255, 0, 0),
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
            new_pixels.append(colors_rgb[current_cluster])
            temp_pixel_array.append(colors_rgb[current_cluster])
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
    # x_color = 100
    # y_color = 100
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

    # каждые n x n пикселей считаем количество черных
    pixel_scale = 50
    colors = []
    for i in range(0, image.size[1], pixel_scale):
        temp_colors = []
        for j in range(0, image.size[0], pixel_scale):

            pixel_num = 0
            black_pixel_num = 0

            if (i + pixel_scale) >= image.size[1]:
                for x in range(i, image.size[1]):
                    if (j + pixel_scale) >= image.size[0]:
                        for y in range(j, image.size[0]):
                            pixel = black_pixels_array[x][y]
                            if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                                black_pixel_num += 1
                            pixel_num += 1
                    else:
                        for y in range(j, j + pixel_scale):
                            pixel = black_pixels_array[x][y]
                            if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                                black_pixel_num += 1
                            pixel_num += 1
                percent = black_pixel_num / pixel_num
                color = int(255 * percent)
                temp_colors.append(color)

            else:
                for x in range(i, i + pixel_scale):
                    if (j + pixel_scale) >= image.size[0]:
                        for y in range(j, image.size[0]):
                            pixel = black_pixels_array[x][y]
                            if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                                black_pixel_num += 1
                            pixel_num += 1
                    else:
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
    picture_pixels = []
    picture_pixels_array = []
    for i in range(0, image.size[1]):
        temp_picture_pixels_array = []
        for j in range(0, image.size[0]):
            current_color = colors[math.floor(i / pixel_scale)][math.floor(j / pixel_scale)]
            temp_picture_pixels_array.append([255, 255 - current_color, 255 - current_color])
            picture_pixels.append((255 - current_color * 3, 255 - current_color * 3, 255))
        picture_pixels_array.append(temp_picture_pixels_array)

    mask_image = Image.new("RGB", (image.size[0], image.size[1]))
    mask_image.putdata(picture_pixels)
    mask_image.save(image_path + "MASK.png")

    cv_image = cv.imread(image_path)
    cv_mask = cv.imread(image_path + "MASK.png")
    dst = cv.addWeighted(cv_image, 1, cv_mask, 0.3, 0)

    # отображаем маску плотности
    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    plt.imshow(picture_pixels_array)
    a.set_title('Маска плотности для области ' + str(pixel_scale) + ' на ' + str(pixel_scale) + ' пикселей')

    # отображаем шкалу распределения плотности
    mask_line_pixels_array = []
    for percent in range(100):
        color = int(255 * percent / 100)
        temp_mask_line_pixels_array = []
        for _ in range(20):
            temp_mask_line_pixels_array.append([255, 255 - color, 255 - color])
        mask_line_pixels_array.append(temp_mask_line_pixels_array)
    a = fig.add_subplot(1, 2, 2)
    a.invert_yaxis()
    a.get_xaxis().set_visible(False)
    plt.imshow(mask_line_pixels_array)
    a.set_title('Шкала плотности распределения в процентах (%)')
    plt.show()

    # markers_color = ['r', 'g', 'b']
    # fig = plt.figure()
    # a = fig.add_subplot(projection='3d')
    # for i in range(k_regions):
    #     current_color = colors_rgb[i]
    #     image_rs = []
    #     image_gs = []
    #     image_bs = []
    #     for x in range(0, image.size[1]):
    #         for y in range(0, image.size[0]):
    #             if new_pixels_array[x][y] == current_color:
    #                 image_rs.append(pixels_array[x][y][0])
    #                 image_gs.append(pixels_array[x][y][1])
    #                 image_bs.append(pixels_array[x][y][2])
    #     a.scatter(image_rs, image_gs, image_bs, alpha=0.7, color=markers_color[i])
    #
    #
    #
    # a.set_xlabel('Красная составляющая')
    # a.set_ylabel('Зеленая составляющая')
    # a.set_zlabel('Синяя составляющая')
    # plt.show()

    fig = plt.figure()
    a = fig.add_subplot(2, 2, 1)
    plt.imshow(pixels_array)
    a.set_title('Оригинальное изображение')
    a = fig.add_subplot(2, 2, 2)
    plt.imshow(new_pixels_array)
    a.set_title('Сегментированное изображение')
    a = fig.add_subplot(2, 2, 3)
    plt.imshow(black_pixels_array)
    a.set_title('Сегментированное изображение (2 цвета)')
    a = fig.add_subplot(2, 2, 4)
    plt.imshow(dst)
    a.set_title('Оригинальное изображение с наложенной маской плотности')
    plt.show()

    # print(kmeans.cluster_centers_)


PSO_Image('../Dataset/l1.png')
