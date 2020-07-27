import h5py
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, Polygon
import rasterio
# from rasterio import features
# import rasterstats as rs

# TODO sort functions alphabetically and add docstring


def ATL_to_dict(filename, dataset_dict, check_name='land_ice_segments',
                ancillary=False):
    """
    Read selected datasets from any ATL file. Wrap for each ATL type.

    From Ben.

    Input arguments:
        filename: ATl file to read
        dataset_dict: A dictinary describing the fields to be read
                keys give the group names to be read,
                entries are lists of datasets within the groups
        pairs: Beam pairs to search for
        beams: names of right and left beams
        name_fmt: How beam names are stored
        check_name: name_fmt/check_name is checked to exist in each file
        ancillary: bool for whether to read non-beam data (ancillary and orbital).
                Use this to avoid a re-read of the hdf5 if you need the metadata.
    Output argument:
        D: dictionary containing ATL06 data.  Each dataset in
           dataset_dict has its own entry in D.  Each dataset
           in D6 contains a list of numpy arrays containing the
           data. If ancillary data, there is an additional dictionary,
           which contains the file-wide properties.
    """
    D=[]
    name_fmt='gt%d%s'
    pairs=[1, 2, 3]
    beams=['l','r']
    # open the HDF5 file
    with h5py.File(filename) as h5f:
        if ancillary:
            anc={}
            for group in dataset_dict.keys():
                for dataset in dataset_dict[group]:
                    DS = '/%s/%s' % (group, dataset)
                    try:
                        anc[dataset] = np.array(h5f[DS])[0]
                    except KeyError:
                        # This should just mean that it is a beam attribute
                        pass
 
        # loop over beam pairs
        for pair in pairs:
            # loop over beams
            for beam_ind, beam in enumerate(beams):
                # check if a beam exists, if not, skip it
                if '/' + (name_fmt % (pair, beam)) + '/' + check_name not in h5f:
                    continue
                # loop over the groups in the dataset dictionary
                temp={}
                for group in dataset_dict.keys():
                    for dataset in dataset_dict[group]:
                        DS='/' + (name_fmt % (pair, beam)) + '/%s/%s' % (group, dataset)
                        # since a dataset may not exist in a file, we're going to try to read it, and if it doesn't work, we'll move on to the next:
                        try:
                            temp[dataset]=np.array(h5f[DS])
                            # some parameters have a _FillValue attribute.  If it exists, use it to identify bad values, and set them to np.NaN
                            if '_FillValue' in h5f[DS].attrs:
                                fill_value=h5f[DS].attrs['_FillValue']
                                bad = temp[dataset]==fill_value
                                temp[dataset]=np.float64(temp[dataset])
                                temp[dataset][bad]=np.NaN
                        except KeyError as e:
                            pass

                if len(temp) > 0:
                    # it's sometimes convenient to have the beam and the pair as part of the output
                    # data structure: This is how we put them there.
                    # Changed by DL because there is no requirement that a particular
                    # value is in the dict...
                    temp['pair']=np.zeros_like(list(temp.values())[0])+pair
                    temp['beam']=np.zeros_like(list(temp.values())[0])+beam_ind
                    #temp['filename']=filename
                    D.append(temp)
    if ancillary:
        D.append(anc)
    return D


def ATL_2_gdf(ATL_fn, dataset_dict, check_name='land_ice_segments',
                ancillary=False):
    """Convert ATL hdf5 to geopandas dataframe.

    All arguments passed to ATL_to_dict (from Ben).

    If ancillary data, and there are such data requested, there will be
    columns with all values identical containing those data.
    """
    if ('latitude' in dataset_dict[check_name]):
        dataset_dict[check_name].append('latitude')
    if ('longitude' in dataset_dict[check_name]):
        dataset_dict[check_name].append('longitude')
    #use Ben's Scripts to convert to dict
    data_dict = ATL_to_dict(ATL_fn, dataset_dict, check_name=check_name,
                ancillary=ancillary)

    if ancillary:
        anc = data_dict.pop()

    #this will give us 6 tracks
    for i, track in enumerate(data_dict):
        #1 track
        #convert to datafrmae
        df = pd.DataFrame(track)
        df['p_b'] = str(track['pair'][0])+'_'+str(track['beam'][0])
        df['geometry'] = df.apply(point_covert,axis=1)
        if i==0:
            df_final = df.copy()
        else:
            df_final = df_final.append(df)

    if ancillary:
        for key, val in anc.items():
            df_final[key] = val
    gdf_final = gpd.GeoDataFrame(df_final,geometry='geometry',crs={'init':'epsg:4326'})

    return gdf_final


def ATL06_to_dict(filename, dataset_dict, ancillary=False):
    """Call ATL_to_dict with defaults for ATL06."""
    check_name = 'land_ice_segments'
    return ATL_to_dict(filename, dataset_dict, check_name=check_name,
                       ancillary=ancillary)


def ATL08_to_dict(filename, dataset_dict, ancillary=False):
    """Call ATL_to_dict with defaults for ATL08."""
    check_name = 'land_segments'
    return ATL_to_dict(filename, dataset_dict, check_name=check_name,
                       ancillary=ancillary)


def ATL06_2_gdf(ATL_fn, dataset_dict, ancillary=False):
    """Wrap ATL_2_gdf with defaults for ATL_06."""
    check_name = 'land_ice_segments'
    return ATL_2_gdf(ATL_fn, dataset_dict, check_name=check_name,
                     ancillary=ancillary)


def ATL08_2_gdf(ATL_fn, dataset_dict, ancillary=False):
    """Wrap ATL_2_gdf with defaults for ATL_08."""
    check_name = 'land_segments'
    return ATL_2_gdf(ATL_fn, dataset_dict, check_name=check_name,
                     ancillary=ancillary)


def dem2polygon(dem_file_name):
    """
        Take DEM and return polygon geodataframe matching the extent and coordinate system of the input DEM.
        
        Input parameters:
        dem_file_name: Absolute path to a DEM file
        
        Output parameters:
        dem_polygon: A polygon geodataframe matching the extent and coordinate system of the input DEM.

    """
    
    # read in dem using rasterio
    dem = rasterio.open(dem_file_name)
    
    # extact total bounds of dem
    bbox = dem.bounds
    
    # convert to corner points
    p1 = Point(bbox[0], bbox[3])
    p2 = Point(bbox[2], bbox[3])
    p3 = Point(bbox[2], bbox[1])
    p4 = Point(bbox[0], bbox[1])
    
    # extract corner coordinates
    np1 = (p1.coords.xy[0][0], p1.coords.xy[1][0])
    np2 = (p2.coords.xy[0][0], p2.coords.xy[1][0])
    np3 = (p3.coords.xy[0][0], p3.coords.xy[1][0])
    np4 = (p4.coords.xy[0][0], p4.coords.xy[1][0])
    
    # convert to polygon
    bb_polygon = Polygon([np1, np2, np3, np4])

    # create geodataframe
    dem_polygon_gdf = gpd.GeoDataFrame(gpd.GeoSeries(bb_polygon), columns=['geometry'])
    
    dem_polygon_gdf.crs = dem.crs
    
    return dem_polygon_gdf

def get_ndv(ds):
    no_data = ds.nodatavals[0]
    if no_data is None:
        #this means no data is not set in tif tag, nead to cheat it from raster
        ndv = ds.read(1)[0,0]
    else:
        ndv = no_data
    return ndv


def point_covert(row):
    geom = Point(row['longitude'],row['latitude'])
    return geom


def sample_near_nbor(ds,geom):
    """
    sample values from raster at the given ICESat-2 points
    using nearest neighbour algorithm
    Inputs are a rasterio dataset and a geodataframe of ice_sat2 points
    """
    # reproject the shapefile to raster projection
    x_min,y_min,x_max,y_max = ds.bounds
    geom = geom.to_crs(ds.crs)
    #filter geom outside bounds
    geom = geom.cx[x_min:x_max,y_min:y_max]
    X = geom.geometry.x.values
    Y = geom.geometry.y.values
    xy = np.vstack((X,Y)).T
    sampled_values = np.array(list(ds.sample(xy)))
    no_data = get_ndv(ds)
    sample = np.ma.fix_invalid(np.reshape(sampled_values,np.shape(sampled_values)[0]))
    sample = np.ma.masked_equal(sample,no_data)
    x_atc = np.ma.array(geom.x_atc.values,mask = sample.mask)
    return x_atc, sample

def buffer_sampler(ds,geom,buffer,val='median',ret_gdf=False):
    """
    sample values from raster at the given ICESat-2 points
    using a buffer distance, and return median/mean or a full gdf ( if return gdf=True)
    Inputs = rasterio dataset, Geodataframe containing points, buffer distance, output value = median/mean (default median)
    and output format list of x_atc,output_value arrays (default) or  full gdf
    """
    import rasterstats as rs
    ndv = get_ndv(ds)
    array = ds.read(1)
    gt = ds.transform
    stat = val
    geom = geom.to_crs(ds.crs)
    x_min,y_min,x_max,y_max = ds.bounds
    geom = geom.cx[x_min:x_max, y_min:y_max]
    geom['geometry'] = geom.geometry.buffer(buffer)
    json_stats = rs.zonal_stats(geom,array,affine=gt,geojson_out=True,stats=stat,nodata=ndv)
    gdf = gpd.GeoDataFrame.from_features(json_stats)
    if val =='median':
        gdf = gdf.rename(columns={'median':'med'})
        call = 'med'
    else:
        gdf = gdf.rename(columns={'mean':'avg'})
        call = 'avg'
    if ret_gdf:
        out_file = gdf
    else:
        out_file = [gdf.x_atc.values,gdf[call].values]
    return out_file
def concat_gdf(gdf_list):
    """
    concatanate geodataframes into 1 geodataframe
    Assumes all input geodataframes have same projection
    Inputs : list of geodataframes in same projection
    Output : 1 geodataframe containing everything having the same projection
    """
    #from https://stackoverflow.com/questions/48874113/concat-multiple-shapefiles-via-geopandas
    gdf = pd.concat([gdf for gdf in gdf_list]).pipe(gpd.GeoDataFrame)
    gdf.crs = (gdf_list[0].crs)
    return gdf
