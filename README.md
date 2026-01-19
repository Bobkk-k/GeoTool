# GeoTool
Using GDAL to implement some geo operations


## Clip 功能
可以使用Clip.py实现栅格/矢量裁剪功能

修改clip_config.json文件

裁剪栅格示例：
```json
{
    "clip_type": "raster",
    "obj_list": [
        "/path/to/raster1.tif",
        "/path/to/raster2.tif",
    ],
    "shp": "/path/to/clip.shp",
    "epsg": 32647,
    "res": 30,
    "output_root": null,
    "resample": "bilinear",
    "lock_grid": true,
    "dstNodata": -9999
}
```

裁剪矢量示例：
```json
{
    "clip_type": "vector",
    "obj_list": [
        "/path/to/vector_be_clipped.shp"
    ],
    "shp": "/path/to/clip.shp",
    "epsg": 32647,
    "output_root": null,
}
```