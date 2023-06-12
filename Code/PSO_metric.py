from PIL import Image
import matplotlib.pyplot as plt


def get_pixel_array_from_image(segmented_image):
    segmented_pixels_list = list(segmented_image.getdata())
    segmented_pixels_array = []

    image_rs = []
    image_gs = []
    image_bs = []
    bad_list = 0
    for i in range(segmented_image.size[1]):
        image_row = []
        for j in range(segmented_image.size[0]):
            image_row.append(segmented_pixels_list[i * segmented_image.size[0] + j])
            if type(image_row[j]) is int:
                bad_list = 1
                pixel = segmented_pixels_list[i * segmented_image.size[0] + j]
                image_rs.append(pixel)
                image_gs.append(pixel)
                image_bs.append(pixel)
                image_row[j] = (pixel, pixel, pixel)
            else:
                image_rs.append(segmented_pixels_list[i * segmented_image.size[0] + j][0])
                image_gs.append(segmented_pixels_list[i * segmented_image.size[0] + j][1])
                image_bs.append(segmented_pixels_list[i * segmented_image.size[0] + j][2])
        segmented_pixels_array.append(image_row)
    if bad_list:
        segmented_pixels_list = [segmented_pixels_array[i][j] for i in range(len(segmented_pixels_array))
                                 for j in range(len(segmented_pixels_array[i]))]

    return segmented_pixels_list, segmented_pixels_array


def get_percents_from_image(segmented_image, segmented_pixels_array):
    pixel_scale = 50  # размер подсчитывающего окна
    percents = []
    for i in range(0, segmented_image.size[1], pixel_scale):
        temp_percents = []
        for j in range(0, segmented_image.size[0], pixel_scale):

            pixel_num = 0
            black_pixel_num = 0

            if (i + pixel_scale) >= segmented_image.size[1]:
                for x in range(i, segmented_image.size[1]):
                    if (j + pixel_scale) >= segmented_image.size[0]:
                        for y in range(j, segmented_image.size[0]):
                            pixel = segmented_pixels_array[x][y]
                            if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                                black_pixel_num += 1
                            pixel_num += 1
                    else:
                        for y in range(j, j + pixel_scale):
                            pixel = segmented_pixels_array[x][y]
                            if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                                black_pixel_num += 1
                            pixel_num += 1
                percent = black_pixel_num / pixel_num
                temp_percents.append(percent)

            else:
                for x in range(i, i + pixel_scale):
                    if (j + pixel_scale) >= segmented_image.size[0]:
                        for y in range(j, segmented_image.size[0]):
                            pixel = segmented_pixels_array[x][y]
                            if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                                black_pixel_num += 1
                            pixel_num += 1
                    else:
                        for y in range(j, j + pixel_scale):
                            pixel = segmented_pixels_array[x][y]
                            if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                                black_pixel_num += 1
                            pixel_num += 1
                percent = black_pixel_num / pixel_num
                temp_percents.append(percent)
        percents.append(temp_percents)

    list_percents = [percents[i][j] for i in range(len(percents)) for j in range(len(percents[i]))]

    return list_percents


def get_total_percent(segmented_list, real_segmented_list):
    correct_pixel_count = 0
    for i in range(len(segmented_list)):
        if segmented_list[i][0] == real_segmented_list[i][0]:
            correct_pixel_count += 1
    return correct_pixel_count / len(segmented_list)


def get_iou(segmented_list, real_segmented_list):
    overlap_area = 0
    union_area = 0
    for i in range(len(segmented_list)):
        if segmented_list[i][0] == real_segmented_list[i][0] and real_segmented_list[i][0] == 255:
            overlap_area += 1
        if segmented_list[i][0] == 255 or real_segmented_list[i][0] == 255:
            union_area += 1
    return overlap_area / union_area

def get_f1_metric(segmented_list, real_segmented_list):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(segmented_list)):
        if segmented_list[i][0] == 255 and real_segmented_list[i][0] == 255:
            tp += 1
        if segmented_list[i][0] == 0 and real_segmented_list[i][0] == 0:
            tn += 1
        if segmented_list[i][0] == 255 and real_segmented_list[i][0] == 0:
            fp += 1
        if segmented_list[i][0] == 0 and real_segmented_list[i][0] == 255:
            fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = 2 * recall * precision / (recall + precision)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return precision, recall, f_measure, accuracy

def calculate_metric(segmented_image_path, real_segmented_image_path):
    segmented_image = Image.open(segmented_image_path, 'r')
    real_segmented_image = Image.open(real_segmented_image_path, 'r')
    segmented_list, segmented_pixels_array = get_pixel_array_from_image(segmented_image)
    real_segmented_list, real_segmented_pixels_array = get_pixel_array_from_image(real_segmented_image)

    segmented_percents = get_percents_from_image(segmented_image, segmented_pixels_array)
    print(segmented_percents)

    real_segmented_percents = get_percents_from_image(real_segmented_image, real_segmented_pixels_array)
    print(real_segmented_percents)

    difference_percents = [(segmented_percents[i] - real_segmented_percents[i]) ** 2
                           for i in range(len(segmented_percents))]
    print(difference_percents)

    total_percent = get_total_percent(segmented_list, real_segmented_list)
    print("Percent of correspondence is", total_percent * 100)

    iou = get_iou(segmented_list, real_segmented_list)
    print("Interception over Union is", iou)

    precision, recall, f_measure, accuracy = get_f1_metric(segmented_list, real_segmented_list)
    print("Precision is", precision)
    print("Recall is", recall)
    print("F-measure is", f_measure)
    print("Accuracy is", accuracy)

    # plt.plot([i for i in range(len(segmented_percents))], segmented_percents)
    # plt.plot([i for i in range(len(segmented_percents))], real_segmented_percents)
    plt.plot([i for i in range(len(segmented_percents))], difference_percents)
    plt.show()


calculate_metric("../Dataset/l1.pngBLACK.png", "../Dataset/l1_real_mask.png")
