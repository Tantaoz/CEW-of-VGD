from Read_Write import *
import cv2
import matplotlib.pyplot as plt
import numpy as np


'''水印处理'''


def Watermark_deal(img_deal):
    h, w = img_deal.shape
    for i in range(0, h):
        for j in range(0, w):
            if img_deal[i][j] < 100:
                img_deal[i][j] = 0
            else:
                img_deal[i][j] = 255
    return img_deal


'''二值化图像'''


def Erzhi_watermark(Arnold_img):
    Erzhi_list = [y for x in Arnold_img for y in x]
    # Erzhi_list=Arnold_img.flatten()
    for i in range(0, len(Erzhi_list)):
        if Erzhi_list[i] == 255:
            Erzhi_list[i] = 1
        else:
            Erzhi_list[i] = 0
    return Erzhi_list


# 水印置乱
def Arnold_Encrypt(image):
    shuffle_times, a, b = 100, 50, 50
    arnold_image = np.zeros(shape=image.shape)
    h, w = image.shape
    N = h  # 或N=w
    # 3：遍历像素坐标变换
    for time in range(shuffle_times):
        for ori_x in range(h):
            for ori_y in range(w):
                # 按照公式坐标变换
                new_x = (1 * ori_x + b * ori_y) % N
                new_y = (a * ori_x + (a * b + 1) * ori_y) % N
                arnold_image[new_x, new_y] = image[ori_x, ori_y]
    return arnold_image


def Embed_Watermark(Lst_WaterMark, X_sum, Y_sum, wm):
    # 最大最小归一化，映射函数，QIM嵌入
    X_sum_Min = min(X_sum)
    Y_sum_Min = min(Y_sum)
    X_sum_Max = max(X_sum)
    Y_sum_Max = max(Y_sum)
    block_x = [list() for i in range(wm)]
    block_y = [list() for i in range(wm)]
    # 归一化
    for i in range(0, len(X_sum)):
        if X_sum[i] == X_sum_Min or X_sum[i] == X_sum_Max:
            continue
        X_sum[i] = ((X_sum[i] - X_sum_Min) / (X_sum_Max - X_sum_Min)) * 1e7  # 归一化放大
        index = int(X_sum[i]) % 4096
        block_x[index].append(X_sum[i])
        if Y_sum[i] == Y_sum_Min or Y_sum[i] == X_sum_Max:
            continue
        Y_sum[i] = ((Y_sum[i] - Y_sum_Min) / (Y_sum_Max - Y_sum_Min)) * 1e7  # 归一化放大
        indey = int(Y_sum[i]) % 4096
        block_y[indey].append(Y_sum[i])

    List_Fea = len(Lst_WaterMark) * [0]
    # 投票机制构造
    for j in range(0,len(block_x)):
        Ux, sx, Vx = np.linalg.svd([block_x[j]])
        Uy, sy, Vy = np.linalg.svd([block_y[j]])

        if sx>=sy:
            List_Fea[j] = 255
        else:
            List_Fea[j] = 0

    # for j in range(0, wm):
    #     if len(block_x[j]) > len(block_y[j]):
    #         List_Fea[j] = 255
    #     else:
    #         List_Fea[j] = 0
    return List_Fea, block_x, block_y


"""构造零水印图像"""


def XOR(List_Fea, Lst_WaterMark):
    List_Zero = len(Lst_WaterMark) * [0]
    for m in range(0, len(List_Zero)):
        if List_Fea[m] == Lst_WaterMark[m]:
            List_Zero[m] = 0
        else:
            List_Zero[m] = 255
    return List_Zero




if __name__ == '__main__':
    img = cv2.imread(r"E:\Tent_GISMap.jpg", 0)  # 读取水印图像
    img_deal = Watermark_deal(img)  # 水印图像处理（0，255）
    # img_arnold=Arnold_Encrypt(img_deal)
    # arnold_image = Arnold_Encrypt(img_deal)
    Lst_WaterMark = img_deal.flatten()  # 降维
    wm = 4096
    fn_r =r"D:\Watermark_Experiment\国家基础地理信息系统1：400万数据\主要公路.shp"
    XLst, YLst, feature_num = Read_Shapfile(fn_r)  # 读取原始矢量数据
    X_sum, Y_sum = GetSum(XLst, YLst)
    List_Fea, block_x, block_y = Embed_Watermark(Lst_WaterMark, X_sum, Y_sum, wm)
    List_Zero = XOR(List_Fea, Lst_WaterMark)
    # De_Zero = XOR(List_Fea, List_Zero)
    Array_Z = np.array(List_Zero).reshape(64,64)
    Array_T = np.array(List_Fea).reshape(64, 64)

    # 存储零水印图像

    plt.subplot(221)
    plt.imshow(Array_T, 'gray')
    plt.title("Fea_image")

    plt.subplot(222)
    plt.imshow(Array_Z, 'gray')
    plt.title("zero_image")
    plt.show()
    cv2.imwrite(r"E:\image_PPPP.jpg", Array_Z)
