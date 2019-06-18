# topohack
Taking ICESat-2 to the mountains: a workflow using satellite laser altimetry to resolve topography over complex terrain  

### Data Science Objective
Compare and evaluate ICESat-2 data with high resolution DEMs (airborne lidar/satellite stereo) collected at lower latitudes over bare ground. 

### Datasets
- ICESat-2 ATL06 (20 m resolution)
- ICESat-2 ATL03 (geolocated points along track)
- Bareground data ([LULC for US](https://www.mrlc.gov/data/legends/national-land-cover-database-2011-nlcd2011-legend), [Global Bareground Data Product for areas outside US](https://glad.umd.edu/dataset/global-2010-bare-ground-30-m) (

### Validation Datasets
- [WADNR Lidar](http://lidarportal.dnr.wa.gov/)
- [ASO Lidar](https://nsidc.org/data/aso)
- [Himat DEMs](https://nsidc.org/the-drift/data-update/high-mountain-asia-8-meter-digital-elevation-models-now-available/)
- DEM from H.P. and Justin (add)


### Tools
- Python: geopandas, rasterio, numpy, scipy, pandas, [pygeotools](https://github.com/dshean/pygeotools), [demcoreg](https://github.com/dshean/demcoreg)
- [NASA Ames Stereo Pipeline](https://github.com/NeoGeographyToolkit/StereoPipeline)

### Tasks
- Learn how to download the ICESat-2 data by lat lon bounding box
- Explore ATL03 and ATL06 products
- Learn how to subset ATL03 and ATL06 data based on flags
- Intersect ICESat-2 tracks with [RGI glacier polygons](https://www.glims.org/RGI/) to get a sense of bare ground coverage near glaciers.
- Evaluate/Compare the topography resolved by ICESat-2 profiles along steep mountains with topographic profiles returned from  high-resolution DEMs.
- Get a sense of Snow accumulation (depth) by comapring Snow-off DEM over Grand Mesa with winter (October to February) IceSat-2 collects. 
### Study Site:
- Depends on ICESat-2 coverage.
- Potential study sites: Cascades and Olympic Ranges in Western Washington, Rocky Montains, High Mountain Asia, Grand Mesa
### Other Questions 
Are notebooks available that show how the comparison between IceBridge lidar and ICESat-2 tracks was done?
