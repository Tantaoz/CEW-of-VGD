from Em_mark import *

'''反置乱水印'''


def Arnold_Decrypt(image):
    shuffle_times, a, b =  100, 50, 50
    decode_image = np.zeros(shape=image.shape)
    h, w = image.shape[0], image.shape[1]
    N = h
    for time in range(shuffle_times):
        for ori_x in range(h):
            for ori_y in range(w):
                # 按照公式坐标变换
                new_x = ((a * b + 1) * ori_x + (-b) * ori_y) % N
                new_y = ((-a) * ori_x + ori_y) % N
                decode_image[new_x, new_y] = image[ori_x, ori_y]
    return decode_image


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
    for j in range(0,wm):
        Ux, sx, Vx = np.linalg.svd([block_x[j]])
        Uy, sy, Vy = np.linalg.svd([block_y[j]])
        if sx >= sy:
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


# NC值
def NC(ori_img, decode_img):
    h, w = ori_img.shape
    S = 0
    for i in range(0, h):
        for j in range(0, w):
            if ori_img[i][j] == decode_img[i][j]:
                S += 1
            else:
                S += 0
    nc = S / (h * w)
    return nc


if __name__ == '__main__':
    img1 = cv2.imread("E:\Tent_GISMap.jpg", 0)  # 读取原始水印图像
    img2 = cv2.imread(r"E:\image_Point.jpg", 0)  # 读取零水印图像
    img_or = Watermark_deal(img1)  # 水印图像处理（0，255）
    img_zero= Watermark_deal(img2)  # 水印图像处理（0，255）
    wm = 4096
    List_Zero = img_zero.flatten()  # 降维
    fn_r =  r"E:\opendirve1.7\gis_osm_pofw_free_1.shp"
    XLst, YLst, feature_num = Read_Shapfile(fn_r)  # 读取原始矢量数据
    X_sum, Y_sum = GetSum(XLst, YLst)
    List_Fea, block_x, block_y = Embed_Watermark(List_Zero, X_sum, Y_sum, wm)

    WaterMark = XOR(List_Fea, List_Zero)
    Re_mark = np.array(WaterMark).reshape(64, 64)
    # Re_mark=Arnold_Decrypt(Re_mark)  # 反置乱

    nc = NC(img_or, Re_mark)
    print(nc)
    # 显示反置乱水印
    plt.subplot(222)
    plt.imshow(Re_mark, 'gray')
    plt.title("Decode_image")
    # cv2.imwrite("Decode_image.jpg", Re_mark)

    plt.subplot(221)
    plt.imshow(img_or, 'gray')
    plt.title("ori_or")
    plt.show()
