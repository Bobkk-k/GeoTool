import os
from pathlib import Path
import tempfile
from osgeo import gdal, ogr, osr
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────
# 帮助函数：判断两个 SpatialReference 是否相同
# ──────────────────────────────────────────────────────────────
def _is_same_srs(srs1: osr.SpatialReference, srs2: osr.SpatialReference) -> bool:
    return bool(srs1.IsSame(srs2))


# ──────────────────────────────────────────────────────────────
# 帮助函数：转投影 shapefile，返回新路径
# ──────────────────────────────────────────────────────────────
def _reproject_shp(shp_path: str, target_srs: osr.SpatialReference) -> str:
    driver = ogr.GetDriverByName("ESRI Shapefile")
    tf = tempfile.NamedTemporaryFile(suffix=".shp", delete=False)
    tmp_shp = tf.name
    tf.close()

    # ogr2ogr in-memory 等价
    src_ds = driver.Open(shp_path, 0)
    src_layer = src_ds.GetLayer()
    src_srs = src_layer.GetSpatialRef()

    dst_ds = driver.CreateDataSource(tmp_shp)
    dst_layer = dst_ds.CreateLayer(
        src_layer.GetName(), srs=target_srs, geom_type=src_layer.GetGeomType()
    )

    coord_trans = osr.CoordinateTransformation(src_srs, target_srs)

    # 拷贝字段
    src_layer_defn = src_layer.GetLayerDefn()
    for i in range(src_layer_defn.GetFieldCount()):
        fld_defn = src_layer_defn.GetFieldDefn(i)
        dst_layer.CreateField(fld_defn)

    # 拷贝要素
    for feat in src_layer:
        geom = feat.GetGeometryRef().Clone()
        geom.Transform(coord_trans)

        dst_feat = ogr.Feature(dst_layer.GetLayerDefn())
        dst_feat.SetGeometry(geom)
        for i in range(src_layer_defn.GetFieldCount()):
            dst_feat.SetField(i, feat.GetField(i))
        dst_layer.CreateFeature(dst_feat)
        dst_feat = None

    # 清理
    src_ds, dst_ds = None, None
    return tmp_shp


# ──────────────────────────────────────────────────────────────
# 帮助函数：计算与分辨率对齐的裁剪范围 (minX, minY, maxX, maxY)
# 若希望多次调用保持 h,w 不变，请把返回值缓存后重复传入
# ──────────────────────────────────────────────────────────────
def get_aligned_bounds(shp_path: str, res: float):
    drv = ogr.GetDriverByName("ESRI Shapefile")
    ds = drv.Open(shp_path)
    layer = ds.GetLayer()

    minx, maxx, miny, maxy = layer.GetExtent()  # 注意顺序
    # 与网格对齐：向外扩展到 res 的整数倍并 snap 到像元角
    minx = (minx // res) * res
    miny = (miny // res) * res
    maxx = ((maxx + res - 1e-9) // res) * res
    maxy = ((maxy + res - 1e-9) // res) * res
    ds = None
    return (minx, miny, maxy, maxy) if False else (minx, miny, maxx, maxy)


# ──────────────────────────────────────────────────────────────
# 核心函数：clip_raster
# ──────────────────────────────────────────────────────────────
def clip_raster(
    shp_path: str,
    raster_path: str,
    out_path: str,
    target_epsg: int | str,
    target_res: float,
    resample: str = "nearest",
    lock_grid: bool = False,
    dstNodata: float = -9999,
):
    """
    参数
    ----
    shp_path        : 待裁剪矢量文件路径
    raster_path     : 待裁剪栅格路径
    out_path        : 输出栅格路径
    target_epsg     : 目标坐标系 EPSG 号（int 或 'EPSG:4326' 格式皆可）
    target_res      : 目标分辨率（单位同坐标系，正值；x,y 分辨率相同比较常见）
    resample        : 'nearest' | 'bilinear'
    lock_grid       : (minx, miny, maxx, maxy)。若不为 None，则强制输出栅格
                       使用该范围和 target_res，从而保证同一 shapefile 多次裁剪
                       h、w 一致；可用 get_aligned_bounds() 先算好。
    """
    # 1. 构造目标 SRS
    target_srs = osr.SpatialReference()
    epsg_int = int(str(target_epsg).split(":")[-1])
    target_srs.ImportFromEPSG(epsg_int)

    # 2. 若 shp/raster 坐标系不同，先重投影到临时文件
    drv = ogr.GetDriverByName("ESRI Shapefile")
    shp_ds = drv.Open(shp_path, 0)
    shp_layer = shp_ds.GetLayer()
    shp_srs = shp_layer.GetSpatialRef()

    if not _is_same_srs(shp_srs, target_srs):
        shp_path = _reproject_shp(shp_path, target_srs)

    # Raster SRS
    ras_ds = gdal.Open(raster_path)
    ras_srs_wkt = ras_ds.GetProjection()
    ras_srs = osr.SpatialReference(wkt=ras_srs_wkt)

    tmp_raster = raster_path
    if not _is_same_srs(ras_srs, target_srs):
        # 临时 reproject
        tf = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        tmp_raster = tf.name
        tf.close()

        gdal.Warp(
            tmp_raster,
            ras_ds,
            dstSRS=target_srs.ExportToWkt(),
            resampleAlg=resample,
        )
        ras_ds = None  # 关闭重新打开
        ras_ds = gdal.Open(tmp_raster)

    # 3. Clip + (可选)Resample 一步到位
    res_alg = {
        "nearest": gdal.GRA_NearestNeighbour,
        "bilinear": gdal.GRA_Bilinear,
    }[resample.lower()]

    warp_opts = gdal.WarpOptions(
        cutlineDSName=shp_path,
        cropToCutline=True,
        dstSRS=target_srs.ExportToWkt(),
        xRes=target_res,
        yRes=target_res,
        resampleAlg=res_alg,
        creationOptions=["COMPRESS=LZW"],
        multithread=True,
        dstNodata=dstNodata
    )

    # 若要求固定网格，则添加 extent & -tap
    if lock_grid:
        minx, miny, maxx, maxy = get_aligned_bounds(shp_path,target_res)
        warp_opts = gdal.WarpOptions(
            cutlineDSName=shp_path,
            cropToCutline=False,  # 先裁到锁定范围再用掩模
            dstSRS=target_srs.ExportToWkt(),
            xRes=target_res,
            yRes=target_res,
            outputBounds=(minx, miny, maxx, maxy),
            targetAlignedPixels=True,
            resampleAlg=res_alg,
            creationOptions=["COMPRESS=LZW"],
            multithread=True,
            dstNodata=dstNodata
        )

    gdal.Warp(out_path, ras_ds, options=warp_opts)

    # 清理
    ras_ds, shp_ds = None, None
    # 临时文件可按需删除；这里保留以便调试

def batch_clip_raster(folder_path,shp_path,output_folder,suffix,target_epsg,target_res,resample,lock_grid,dstNodata):
    for root, dirs, files in os.walk(folder_path):
        for file in tqdm(files):
            if file.endswith(".tif"):
                raster_path = os.path.join(root, file)
                file_clip = file.replace(".tif",f"_{suffix}.tif")
                out_path = os.path.join(output_folder, file_clip)
                clip_raster(shp_path, raster_path, out_path, target_epsg, target_res, resample, lock_grid,dstNodata)

def clip_vector(
    clip_shp: str,
    clipped_shp: str,
    out_path: str,
    target_epsg: int | str,
):

    # 1) 构造目标 SRS
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(int(str(target_epsg).split(":")[-1]))

    drv = ogr.GetDriverByName("ESRI Shapefile")

    # -- 裁剪多边形 --
    clip_ds  = drv.Open(clip_shp)
    clip_lyr = clip_ds.GetLayer()
    if not _is_same_srs(clip_lyr.GetSpatialRef(), target_srs):
        clip_shp = _reproject_shp(clip_shp, target_srs)
        clip_ds  = drv.Open(clip_shp)
        clip_lyr = clip_ds.GetLayer()

    # 将 clip layer 合并为一个 MultiPolygon，提高效率
    union_geom = None
    for feat in clip_lyr:
        geom = feat.GetGeometryRef().Clone()
        union_geom = geom if union_geom is None else union_geom.Union(geom)
    clip_ds = None      # 关闭

    # -- 被裁剪图层 --
    tgt_ds  = drv.Open(clipped_shp)
    tgt_lyr = tgt_ds.GetLayer()
    if not _is_same_srs(tgt_lyr.GetSpatialRef(), target_srs):
        clipped_shp = _reproject_shp(clipped_shp, target_srs)
        tgt_ds  = drv.Open(clipped_shp)
        tgt_lyr = tgt_ds.GetLayer()

    # 2) 创建输出 DataSource & Layer
    out_ds  = drv.CreateDataSource(out_path, options=["ENCODING=UTF-8"])
    out_lyr = out_ds.CreateLayer(
        tgt_lyr.GetName(),
        srs=target_srs,
        geom_type=tgt_lyr.GetGeomType(),
    )
    tgt_defn = tgt_lyr.GetLayerDefn()
    for i in range(tgt_defn.GetFieldCount()):
        out_lyr.CreateField(tgt_defn.GetFieldDefn(i))

    # 3) 设置空间过滤器 + 拷贝要素
    tgt_lyr.SetSpatialFilter(union_geom)
    for feat in tgt_lyr:
        geom = feat.GetGeometryRef()
        # 再做一次精确相交检测（可选）
        if not geom.Intersects(union_geom):
            continue
        new_feat = ogr.Feature(out_lyr.GetLayerDefn())
        new_feat.SetGeometry(geom.Clone())
        for i in range(tgt_defn.GetFieldCount()):
            new_feat.SetField(i, feat.GetField(i))
        out_lyr.CreateFeature(new_feat)
        new_feat = None

    # 4) 清理
    tgt_ds, out_ds = None, None
    print(f"[OK] 裁剪完成 → {out_path}")

# ──────────────────────────────────────────────────────────────
# 使用示例
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # name = 'changning'
    # name = ['changning','kangding','wenchuan','jiuzhaigou','lushan','maerkang','luding']
    # shps = [rf"E:/SchoolWork/研究区shp/四川/长宁/changning.shp",
    #        rf"E:/SchoolWork/研究区shp/四川/康定/kangding.shp",
    #        rf"E:/SchoolWork/研究区shp/四川/汶川/wenchuan.shp",
    #        rf"E:/SchoolWork/研究区shp/四川/九寨沟/九寨沟/jiuzhaigou.shp",
    #        rf"E:/SchoolWork/研究区shp/四川/芦山/lushan.shp",
    #        rf"E:/SchoolWork/研究区shp/四川/马尔康/maerkang.shp",
    #        rf"E:/SchoolWork/RiskAssess-using/product/luding/VII_proj.shp"]
    name = ['jinggu','yangbi','changning','kangding','maerkang']
    shps = ["E:\SchoolWork\RiskAssess-using\product\jinggu\jinggu_proj.shp",
            "E:\SchoolWork\RiskAssess-using\product\yangbi\yangbi_proj.shp",
        rf"E:/SchoolWork/RiskAssess-using/product/changning/changningVI.shp",
        rf"E:/SchoolWork/研究区shp/四川/长宁/changning.shp",
        rf"E:/SchoolWork/研究区shp/四川/康定/kangding.shp",
        rf"E:/SchoolWork/研究区shp/四川/马尔康/maerkang.shp"]

    ras_folder = r"E:/SchoolWork/newLSAT-using/product/BigData"
    ras = r"E:\SchoolWork\RiskAssess-using\product\YunnanBigData\roads_Density.tif"
    epsg = 32647
    res = 30  # metre/pixel
    for i, shp in enumerate(shps[:2]):
        print(shp)
        out = rf"E:/SchoolWork/RiskAssess-using/product/{name[i]}"

        if os.path.exists(out):
            print("输出目录已存在")
        else:
            os.makedirs(out)
        print(name[i])
        clip_raster(shp, ras, os.path.join(out,f'{name[i]}_denseroads.tif'), epsg, res, resample="bilinear",lock_grid=True,dstNodata=-9999)


 
    # batch_clip_raster(ras_folder,shp,out,name,epsg,res,"nearest",True,-9999)
    # output_shp = rf"E:/SchoolWork/newLSAT-using/output/{name}/{name}_lsp.shp"
    # clip_vector(shp, r"E:\SchoolWork\TssTool\data\Sichuanlandslidepoints\landslidepoint.shp", output_shp, epsg)

