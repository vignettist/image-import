import pandas as pd
import time
import matplotlib
import datetime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pywt
import scipy.sparse
import scipy.misc
from scipy.ndimage import filters
import urllib2
import pymongo
from bson.objectid import ObjectId
import json
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import itertools
from sklearn import cluster as clus
from shapely.geometry import MultiPoint

def find_common_location(cluster):
    locations = [v['results'][0]['address_components'] for v in cluster['geolocation'] if v['status'] != u'ZERO_RESULTS']
    types = [u'country', u'administrative_area_level_1', 'metro_areas', u'administrative_area_level_2', u'locality', u'sublocality_level_1', u'neighborhood', u'sublocality_level_2']
    try:
        metro_areas = [lookup_usa_address(v) for v in locations]
    except:
        metro_areas = [lookup_usa_address(v) for v in [v['results'][1]['address_components'] for v in cluster['geolocation'] if v['status'] != u'ZERO_RESULTS']]
    name = ''
    
    o = 1
    country = ''
    
    for j in range(len(types)):
        flag = True

        if j == 2:
            name_list = metro_areas
        else:
            name_list = [[v['long_name'] for v in locations[i] if types[j] in v['types']][0] for i in range(len(locations)) if [v['long_name'] for v in locations[i] if types[j] in v['types']]]
        
        if name_list:
            if j == 0:
                country = name_list[0]
            if (name_list[1:] != name_list[:-1]):
                flag = False
        else:
            flag = False
            if j == 0:
                country = ''
        

        if not flag:

            if j == 0:
                if len(name_list) > 0:
                    name = list(set(name_list))
                else:
                    name = ''

            # if the country is the US, return a list of states. should also do this for other large countries
            elif (j == 1) and (country == 'United States'):
                name = list(set(name_list))
            else:
                if ((j == 3) and (metro_areas[0] is not None)):
                    name = metro_areas[0]
                else:
                    o = 1
                    name = None
                    while name is None:
                        name = [v['long_name'] for v in locations[0] if types[j-o] in v['types']]
                        if len(name) > 0:
                            name = name[0]
                        else:
                            name = None
                            
                        o += 1

            break

    return name

msa_list = pd.read_excel('msa_list.xls', skiprows=2)

def lookup_usa_address(geolocation):
    state_name = [v['long_name'] for v in geolocation if u'administrative_area_level_1' in v['types']]

    if state_name:
        state_name = state_name[0]
        state_list = msa_list[(msa_list['State Name']) == state_name]

        if not state_list.empty:
            # special case for DC because its the only US address without an "administrative_area_level_2"
            if state_name == 'District of Columbia':
                county_name = 'District of Columbia'
            else:
                county_name = [v['long_name'] for v in geolocation if u'administrative_area_level_2' in v['types']][0]

            match = state_list[state_list['County/County Equivalent'] == county_name]

            if len(match) > 0:
                cbsa_title = match['CBSA Title'].iloc[0]
                cbsa_title = cbsa_title.split(',')[0]
                cbsa_title = cbsa_title.split('-')[0]

                return 'the ' + cbsa_title + ' area'
            else:
                return None
        else:
            return None
    else:
        return None

def cluster(images, stretch_nights=5.0, timescale=12.0, algorithm='meanshift', threshold=1.7):
    image_df = pd.DataFrame(images)
    image_df['unix_time'] = [(image_df['datetime'][i]['utc_timestamp'] - datetime.datetime(1970,1,1)).total_seconds()/(60*60*24) for i in range(len(image_df))]

    if stretch_nights:
        time = 0
        times = []
        control_time = 0

        for i in range(len(image_df['datetime'])):
            times.append(time)

            if (i < len(image_df['datetime'])-1):
                current_time = image_df['datetime'].iloc[i]['utc_timestamp']

                while (image_df['datetime'].iloc[i+1]['utc_timestamp'] - current_time).total_seconds() > (60 * 60):
                    local_time = current_time + datetime.timedelta(0, image_df['datetime'].iloc[i]['tz_offset'])

                    if (local_time.hour > 2) and (local_time.hour < 7):
                        time += stretch_nights / timescale
                        control_time += 1.0 / timescale
                    else:
                        time += 1.0 / timescale
                        control_time += 1.0/timescale

                    current_time += datetime.timedelta(0, 60*60)

                local_time = current_time + datetime.timedelta(0, image_df['datetime'].iloc[i]['tz_offset'])

                if (local_time.hour > 2) and (local_time.hour < 7):
                    time += stretch_nights * (image_df['datetime'].iloc[i+1]['utc_timestamp'] - current_time).total_seconds()/(60*60*timescale);
                    control_time += 1.0 * (image_df['datetime'].iloc[i+1]['utc_timestamp'] - current_time).total_seconds()/(60*60*timescale);
                else:
                    time += 1.0 * (image_df['datetime'].iloc[i+1]['utc_timestamp'] - current_time).total_seconds()/(60*60*timescale);
                    control_time += 1.0 * (image_df['datetime'].iloc[i+1]['utc_timestamp'] - current_time).total_seconds()/(60*60*timescale);

        times = np.array(times)
        times *= (control_time / time)
        times -= np.mean(times)
        image_df['cluster_time'] = times
    else:
        image_df['cluster_time'] = image_df['unix_time'] / timescale

    image_df = image_df.dropna(axis=0, subset=['latitude'])

    latlontimes = np.zeros((len(image_df['latitude']), 3))
    latlontimes[:,0] = image_df['latitude'] 
    latlontimes[:,1] = image_df['longitude']
    latlontimes[:,2] = image_df['cluster_time']   

    if (algorithm == "birch"):
        cl = clus.Birch(threshold=(threshold * (1 - timescale/150.0)), n_clusters=None).fit(latlontimes)
    elif (algorithm == 'meanshift'):
        cl = clus.MeanShift(bandwidth=threshold).fit(latlontimes)
    elif (algorithm == 'dbscan'):
        cl = clus.DBSCAN(eps=threshold, min_samples=3).fit(latlontimes)
    elif (algorithm == 'affinity'):
        cl = clus.AffinityPropagation(damping=threshold).fit(latlontimes)

    labels = cl.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print(str(np.sum(labels == -1 )) + " unclassified images")
    print(algorithm + ", threshold=" + str(threshold) + ", timescale=" + str(timescale) + ", stretch nights=" + str(stretch_nights))
    print("    " + str(n_clusters_) + " clusters")

    # process the clusters to make sure that they all make sense

    timespans = np.zeros(n_clusters_)
    start_times = np.zeros(n_clusters_)

    for i in set(labels):
        cluster_mask = (labels == i)
        new_index = image_df[cluster_mask].index
        cluster = image_df.loc[new_index]
        cluster = cluster.sort('unix_time')
        cluster.index = range(len(cluster))

        start_time = cluster['unix_time'][0] * 4
        end_time = cluster['unix_time'][len(cluster) - 1] * 4

        start_location = (cluster['latitude'][0], cluster.longitude[0])
        end_location = (cluster.latitude[len(cluster) - 1], cluster.longitude[len(cluster) - 1])

        timespan = (end_time - start_time)
        timespans[i] = timespan
        start_times[i] = start_time



    corder = np.argsort(start_times)
    start_times = start_times[corder]
    timespans = timespans[corder]
    # sorted_labels = labels[corder]
    sorted_clusters = np.arange(n_clusters_)[corder]


    clusters_structure = []

    stdevs = np.zeros(len(set(labels)))

    for i in range(n_clusters_):
        cluster_struct = {}

        cluster_mask = (labels == sorted_clusters[i])
        new_index = image_df[cluster_mask].index
        cluster = image_df.loc[new_index]
        cluster = cluster.sort('unix_time')
        cluster.index = range(len(cluster))

        cluster_struct['cluster'] = cluster
        cluster_struct['start_time'] = start_times[i]
        cluster_struct['length'] = timespans[i]
        cluster_struct['locations'] = [[cluster.longitude[j], cluster.latitude[j]] for j in range(len(cluster))]

        cluster_struct['times'] = [cluster.datetime[j] for j in range(len(cluster))]

        points = MultiPoint(cluster_struct['locations'])
        geom = {}
        geom['type'] = 'Polygon'
        
        try:
            p = points.convex_hull
            p = p.buffer(np.sqrt(p.area) * 0.33)
            geom['coordinates'] = list(p.simplify(0.0005).exterior.coords)
        except:
            p = points.buffer(0.005)
            p = p.convex_hull
            geom['coordinates'] = list(p.simplify(0.0005).exterior.coords)
        
        centroid = {'type': 'Point'}
        centroid['coordinates'] = list(p.centroid.coords)[0]

        cluster_struct['boundary'] = geom
        cluster_struct['centroid'] = centroid
        
        clusters_structure.append(cluster_struct)

    return clusters_structure
