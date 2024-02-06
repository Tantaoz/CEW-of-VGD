import os
import math
from osgeo import ogr
import sys

'''读入矢量数据（shp）'''


def Read_Shapfile(fn_ori):
    ds = ogr.Open(fn_ori, 0)
    if ds is None:
        sys.exit("Could not open {0}.".format(fn_ori))
    layer = ds.GetLayer(0)
    feature_num = layer.GetFeatureCount(0)
    X_Lst, Y_Lst = [], []
    for i in range(0, feature_num):
        feature = layer.GetFeature(i)
        geometry = feature.GetGeometryRef()
        if geometry.GetGeometryName() == 'POLYGON':
            geometry = geometry.GetGeometryRef(0)
        x, y = [0] * geometry.GetPointCount(), [0] * geometry.GetPointCount()
        for j in range(geometry.GetPointCount()):
            x[j] = geometry.GetX(j)
            y[j] = geometry.GetY(j)
        X_Lst.append(x), Y_Lst.append(y)
    ds.Destroy()
    return X_Lst, Y_Lst, feature_num


'''写出矢量数据'''


def write_encrytpion_shp(ori_shp, outputfile, En_X, En_Y):
    ds = ogr.Open(ori_shp, 0)
    if ds is None:
        sys.exit('Could not open {0}.'.format(ori_shp))
    '''1.创建数据源'''
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.access(outputfile, os.F_OK):
        driver.DeleteDataSource(outputfile)
    '''2.复制一个新的图层'''
    layer = ds.GetLayer(0)
    newds = driver.CreateDataSource(outputfile)
    pt_layer = newds.CopyLayer(layer, 'a')  # 第1个参数是OGR的Layer对象，第2个参数是要生成图层的名称。对于Shapefile来说，这个名称是没有用的，但必须给这个字符串赋变量值。
    newds.Destroy()
    nds = ogr.Open(outputfile, 1)
    nlayer = nds.GetLayer(0)
    for i in range(nlayer.GetFeatureCount(0)):
        feature = nlayer.GetFeature(i)
        # geometry = feature.GetGeometryRef().GetGeometryRef(0)
        geometry = feature.GetGeometryRef()
        if geometry.GetGeometryName() == 'POLYGON':
            geometry = geometry.GetGeometryRef(0)
        for k in range(geometry.GetPointCount()):
            geometry.SetPoint_2D(k, En_X[i][k], En_Y[i][k])
        nlayer.SetFeature(feature)
    nds.Destroy()


def GetSum(XLst, YLst):
    X_sum, Y_sum = [], []
    for i in range(len(XLst)):
        for j in range(len(XLst[i])):
            X_sum.append(XLst[i][j])
    for m in range(len(YLst)):
        for n in range(len(YLst[m])):
            Y_sum.append(YLst[m][n])
    return X_sum, Y_sum




def DeHenon(XList, YList, x0, y0, a, b, t0):
    N = len(XList)
    xi = x0
    yi = y0
    Lx, Ly, RxLi, RyLi = [], [], [], []
    # 混沌序列生成
    for k in range(0, t0 + N):
        xi, yi = (1 - a * (math.sin(xi) ** 2) + yi) % 1, (b * xi) % 1
        Lx.append(xi)
        Ly.append(yi)
    xLi = Lx[t0:]
    yLi = Ly[t0:]
    # 向下取整
    for p in xLi:
        x = int(N * p) % N
        RxLi.append(x)  # 取整后的混沌序列
    for q in yLi:
        y = int(N * q) % N
        RyLi.append(y)  # 取整后的混沌序列
    r = 1
    for i in reversed(RxLi):
        XList[i], XList[N - r] = XList[N - r], XList[i]
        r += 1
    s = 1
    for k in reversed(RyLi):
        XList[k], XList[N - s] = XList[N - s], XList[k]

    return XList, YList


"""点解密"""


def En_point(X_sumen, Y_sumen, XLst, YLst):
    XLst1, YLst1 = XLst, YLst
    t = 0
    for i in range(feature_num):
        for j in range(len(XLst[i])):
            XLst1[i][j] = X_sumen[t]
            YLst1[i][j] = Y_sumen[t]
            t += 1
    return XLst1, YLst1


if __name__ == '__main__':
    fn_r = r"E:\Map\En_Line.shp"
    fn_r = r"E:\Map\En_Line.shp"
    XLst, YLst, feature_num = Read_Shapfile(fn_r)  # read vector data
    X_sum, Y_sum = GetSum(XLst, YLst)
    x0, y0 = 0.35, 0.55  # 实验初始值
    a, b, t0 = 3.14, 1.25, 50  # 参数，t0为迭代次数
    X_sumen, Y_sumen = DeHenon(X_sum, Y_sum, x0, y0, a, b, t0)
    RXLst, RYLst = En_point(X_sumen, Y_sumen, XLst, YLst)
    fn_w = r"E:\Map\De_Line.shp"
    write_encrytpion_shp(fn_r, fn_w, RXLst, RYLst)  # Write out the encrypted vector data
    print("finish")
