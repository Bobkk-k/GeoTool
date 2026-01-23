import argparse
import json
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

def project_study_area(study_area_path,target_srs,output_path):
        # 定义目标分辨率和坐标系


    study_area_ds = ogr.Open(study_area_path)
    study_area_layer = study_area_ds.GetLayer()
    study_area_srs = study_area_layer.GetSpatialRef()

    if study_area_srs.IsSame(target_srs):
        return study_area_path
    
    drv = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(output_path):
        drv.DeleteDataSource(output_path)

    # ③ 打开源数据集，创建目标图层
    src_ds   = drv.Open(study_area_path, 0)          # 只读
    src_lyr  = src_ds.GetLayer()
    geomtype = src_lyr.GetGeomType()                 # 保持原来几何类型
    dst_ds   = drv.CreateDataSource(output_path)
    dst_lyr  = dst_ds.CreateLayer(src_lyr.GetName(),srs=target_srs,geom_type=geomtype)

    # ④ 复制字段结构
    src_defn = src_lyr.GetLayerDefn()
    for i in range(src_defn.GetFieldCount()):
        dst_lyr.CreateField(src_defn.GetFieldDefn(i))
    dst_defn = dst_lyr.GetLayerDefn()

    # ⑤ 坐标转换器
    coord_trans = osr.CoordinateTransformation(study_area_srs, target_srs)

    # ⑥ 遍历要素并写入
    for src_feat in src_lyr:
        geom = src_feat.GetGeometryRef().Clone()
        geom.Transform(coord_trans)

        dst_feat = ogr.Feature(dst_defn)
        dst_feat.SetGeometry(geom)

        # 复制属性字段
        for i in range(dst_defn.GetFieldCount()):
            field_name = dst_defn.GetFieldDefn(i).GetNameRef()
            dst_feat.SetField(field_name, src_feat.GetField(field_name))

        dst_lyr.CreateFeature(dst_feat)
        dst_feat = None  # 释放

    # ⑦ 关闭数据集
    src_ds = None
    dst_ds = None

    # ⑧ 返回重投影后 Shapefile 的主文件路径
    return output_path


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
    ras_path: str,
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
        proj_path = os.path.join(os.path.dirname(shp_path),f"{os.path.splitext(os.path.basename(shp_path))[0]}_proj.shp")
        shp_path = project_study_area(shp_path,target_srs,proj_path)

    # Raster SRS
    ras_ds = gdal.Open(ras_path)
    ras_srs_wkt = ras_ds.GetProjection()
    ras_srs = osr.SpatialReference(wkt=ras_srs_wkt)

    tmp_raster = ras_path
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

def batch_clip_raster(folder_path,shp_path,output_folder,target_epsg,target_res,resample,lock_grid,dstNodata):
    for root, dirs, files in os.walk(folder_path):
        for file in tqdm(files):
            if file.endswith(".tif"):
                raster_path = os.path.join(root, file)
                file_clip = file.replace(".tif",f"_clip.tif")
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
        proj_path = os.path.join(os.path.dirname(clip_shp),f"{os.path.splitext(os.path.basename(clip_shp))[0]}_proj.shp")
        clip_shp = project_study_area(clip_shp,target_srs,proj_path)

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
        proj_path = os.path.join(os.path.dirname(clip_shp),f"{os.path.splitext(os.path.basename(clipped_shp))[0]}_proj.shp")
        clipped_shp = project_study_area(clipped_shp,target_srs,proj_path)
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

def get_pure_basename(file_path):
    """提取无任何扩展名的纯文件名（适配多层后缀，如 roads_density.fds.tif）"""
    basename = os.path.basename(file_path)
    while os.path.splitext(basename)[1]:
        basename = os.path.splitext(basename)[0]
    return basename

def load_and_validate_config(config_path):
    """加载JSON配置文件并校验必要参数"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        raise Exception(f"配置文件不存在：{config_path}")
    except json.JSONDecodeError:
        raise Exception(f"配置文件格式错误（非有效JSON）：{config_path}")

    # 校验核心参数
    required_keys = {
        "clip_type": ["raster", "vector"],
        "path_type": ["folder", "file"],
        "shp": str,                # 单个裁剪范围shp
        "epsg": int                # 投影编码
    }

    for key, expected in required_keys.items():
        if key not in config:
            raise Exception(f"配置文件缺少必选参数：{key}")
        
        if key == "clip_type" and config[key] not in expected:
            raise Exception(f"clip_type只能是 'raster' 或 'vector'，当前值：{config[key]}")
        
        if isinstance(expected, type) and not isinstance(config[key], expected):
            raise Exception(f"{key} 类型错误，期望 {expected.__name__}，实际 {type(config[key]).__name__}")

    if len(config["obj_list"]) == 0:
        raise Exception("obj_list列表不能为空，请至少指定一个输入路径")

    # 按需校验栅格专属参数
    if config["clip_type"] == "raster":
        raster_keys = ["res"]
        for key in raster_keys:
            if key not in config:
                raise Exception(f"裁剪类型为raster时，缺少必选参数：{key}")
        config.setdefault("resample", "bilinear")
        config.setdefault("lock_grid", True)
        config.setdefault("dstNodata", -9999)

    # 可选参数：output_root（自定义输出根目录），无则设为None
    config.setdefault("output_root", None)
    if config["path_type"] == "folder":
        if "obj_folder" not in config:
            raise Exception(f"配置文件缺少必选参数：obj_folder")
    elif config["path_type"] == "file":
        if "obj_list" not in config:
            raise Exception(f"配置文件缺少必选参数：obj_list")
    return config

# ======================== 核心执行逻辑 =========================
def main():
    parser = argparse.ArgumentParser(description="基于JSON配置文件批量裁剪栅格/矢量数据（单shp裁剪范围）")
    parser.add_argument("-c", "--config", required=True, help="JSON配置文件的绝对路径（如：E:/config.json）")
    args = parser.parse_args()

    # 加载配置
    try:
        config = load_and_validate_config(args.config)
    except Exception as e:
        print(f"配置文件加载失败：{e}")
        return

    # 提取基础参数
    clip_type = config["clip_type"]
    path_type = config["path_type"]
    clip_shp = config["shp"]         # 单个裁剪范围shp
    epsg = config["epsg"]
    output_root = config["output_root"]  # 自定义输出根目录（可选）
    if clip_type == "raster":
        res = config["res"]
        resample = config["resample"]
        lock_grid = config["lock_grid"]
        dstNodata = config["dstNodata"]
        if path_type == "file":
            # 统一处理单/多对象裁剪
            obj_list = config["obj_list"]    # 输入路径列表（栅格/矢量路径）
            for idx, input_path in enumerate(obj_list):
                print(f"\n========== 处理第 {idx+1} / {len(obj_list)} 个对象：{input_path} ==========")
                print(f"当前配置参数：{config}\n")
                # 提取输入路径的纯basename（无扩展名）和输入目录
                input_basename = get_pure_basename(input_path)
                input_dir = os.path.dirname(input_path)
                
                # ========== 核心调整：双选择的输出目录逻辑 ==========
                # 1. 判断是否有自定义输出根目录
                if output_root and isinstance(output_root, str):
                    if not os.path.exists(output_root):
                        print(f"创建自定义输出根目录：{output_root}")
                        os.makedirs(output_root)
                    else:
                        print(f"自定义输出根目录已存在：{output_root}")
                    out_dir = output_root
                else:
                    print(f"未指定自定义输出根目录，默认使用输入目录：{input_dir}")
                    out_dir = input_dir

                    
                # 构建输出栅格路径：basename + "_clip.tif"
                out_tif = os.path.join(out_dir, f"{input_basename}_clip.tif")
                    
                # 调用clip_raster
                clip_raster(
                    shp_path=clip_shp,
                    ras_path=input_path,       # 输入栅格路径（来自obj_list）
                    out_path=out_tif,
                    target_epsg=epsg,
                    target_res=res,
                    resample=resample,
                    lock_grid=lock_grid,
                    dstNodata=dstNodata
                )

                # 4.3 执行矢量裁剪
        elif path_type == "folder":
            obj_folder = config["obj_folder"]    # 输入路径列表（栅格/矢量路径）
            if output_root and isinstance(output_root, str):
                if not os.path.exists(output_root):
                    print(f"创建自定义输出根目录：{output_root}")
                    os.makedirs(output_root)
                else:
                    print(f"自定义输出根目录已存在：{output_root}")
                out_dir = output_root
            else:
                print(f"未指定自定义输出根目录，默认使用输入目录：{input_dir}")
                out_dir = input_dir
            batch_clip_raster(obj_folder,clip_shp,out_dir,epsg,res,"nearest",True,-9999)

    elif clip_type == "vector":
        # 构建输出矢量路径：basename + "_clip.shp"
        out_shp = os.path.join(out_dir, f"{input_basename}_clip.shp")
        
        # 调用clip_vector
        clip_vector(
            clip_shp=clip_shp,
            clipped_shp=input_path,  # 输入矢量路径（来自obj_list）
            output_path=out_shp,
            target_epsg=epsg
        )
        


    print("\n========== 所有裁剪任务执行完成 ==========")

if __name__ == "__main__":
    main()
 
    # batch_clip_raster(ras_folder,shp,out,name,epsg,res,"nearest",True,-9999)
    # output_shp = rf"E:/SchoolWork/newLSAT-using/output/{name}/{name}_lsp.shp"
    # clip_vector(shp, r"E:\SchoolWork\TssTool\data\Sichuanlandslidepoints\landslidepoint.shp", output_shp, epsg)

