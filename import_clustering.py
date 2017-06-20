import pandas as pd
import time
import matplotlib
import datetime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.sparse
import scipy.misc
from scipy.ndimage import filters
import urllib2
import pymongo
from bson.objectid import ObjectId
import json
import matplotlib.pyplot as plt
import itertools
from sklearn import cluster as clus
from shapely.geometry import MultiPoint
from geopy.distance import vincenty

def find_common_location(cluster):
    cluster = cluster.dropna(axis=0, subset=['geolocation'])
    locations = [v['results'][0]['address_components'] for v in cluster['geolocation'] if v['status'] != u'ZERO_RESULTS']
    types = [u'country', u'administrative_area_level_1', 'metro_areas', u'administrative_area_level_2', u'locality', u'sublocality_level_1', u'neighborhood', u'sublocality_level_2']
    try:
        metro_areas = [lookup_usa_address(v) for v in locations]
    except:
        try:
            metro_areas = [lookup_usa_address(v) for v in [v['results'][1]['address_components'] for v in cluster['geolocation'] if v['status'] != u'ZERO_RESULTS']]
        except:
            try:
                metro_areas = [lookup_usa_address(v) for v in [v['results'][2]['address_components'] for v in cluster['geolocation'] if v['status'] != u'ZERO_RESULTS']]
            except:
                metro_areas = None

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
    country_name = [v['long_name'] for v in geolocation if u'country' in v['types']]
    if country_name[0] == 'United States':
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

    # keep track of images that are missing location data
    images_missing_location = image_df[image_df.isnull()['latitude']]
    # remove images without location
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

    timespans = np.zeros(n_clusters_)
    start_times = np.zeros(n_clusters_)
    end_times = np.zeros(n_clusters_)
    clusters = []

    for i in set(labels):
        cluster_mask = (labels == i)
        new_index = image_df[cluster_mask].index
        cluster = image_df.loc[new_index]
        # cluster = cluster.sort('unix_time')
        cluster.index = range(len(cluster))
        clusters.append(list(cluster['_id']))

        start_times[i] = cluster['unix_time'][0]
        end_times[i] = cluster['unix_time'][len(cluster) - 1]

    unclustered_images = pd.DataFrame()
    # now try to fit the photos missing locations into our clusters
    for k in images_missing_location.index:
        in_cluster = False
        distance_from_end = np.zeros(n_clusters_)
        distance_from_start = np.zeros(n_clusters_)
        for i in set(labels):
            if (images_missing_location.loc[k]['unix_time'] >= start_times[i]) and (images_missing_location.loc[k]['unix_time'] <= end_times[i]):
                # easy, this just belongs in that cluster
                if not in_cluster:
                    clusters[i].append(images_missing_location.loc[k]['_id'])
                    in_cluster = True
            
            distance_from_end[i] = images_missing_location.loc[k]['unix_time'] - end_times[i]
            distance_from_start[i] = images_missing_location.loc[k]['unix_time'] - start_times[i]

        # the photo didn't fit cleanly inside a cluster, but maybe it's still pretty close
        if (not in_cluster):
            min_distance_from_end = np.argmin(np.abs(distance_from_end))
            min_distance_from_start = np.argmin(np.abs(distance_from_start))

            min_distance = min(np.abs(distance_from_start[min_distance_from_start]), np.abs(distance_from_end[min_distance_from_end]))
            if (min_distance < 1):
                if np.abs(distance_from_end[min_distance_from_end]) < np.abs(distance_from_start[min_distance_from_start]):
                    clusters[min_distance_from_end].append(images_missing_location.loc[k]['_id'])
                else:
                    clusters[min_distance_from_start].append(images_missing_location.loc[k]['_id'])
            else:
                unclustered_images.append(images_missing_location.loc[k])

    # so hopefully that got a lot of the photos, but there will still be some that are pretty distant from every other cluster
    # TODO: generate clusters from remaining unclustered images
    return clusters

def make_cluster_details(images_list, logical_images=None, db=None):
    if logical_images is None:
        query = []
        for i in range(len(images_list)):
            query.append({'_id': images_list[i]})
        logical_images = db.logical_images.find({'$or': query})

    cluster = [li for li in logical_images if li['_id'] in images_list]
    cluster = pd.DataFrame(cluster)
    cluster['unix_time'] = [(cluster['datetime'][i]['utc_timestamp'] - datetime.datetime(1970,1,1)).total_seconds()/(60*60*24) for i in range(len(cluster))]
    cluster = cluster.sort('unix_time')
    cluster.index = range(len(cluster))

    start_location = (cluster['latitude'][0], cluster.longitude[0])
    end_location = (cluster.latitude[len(cluster) - 1], cluster.longitude[len(cluster) - 1])

    db_cluster = {}
    db_cluster['photos'] = images_list
    db_cluster['start_time'] = cluster.iloc[0]['datetime']
    db_cluster['end_time'] = cluster.iloc[-1]['datetime']

    db_cluster['times'] = [cluster.datetime[j] for j in range(len(cluster))]

    # make geoJSON
    linestring = {}
    linestring['type'] = "LineString"
    linestring['coordinates'] =  [[cluster.longitude[j], cluster.latitude[j]] for j in range(len(cluster))]
    db_cluster['locations'] = linestring

    start_location = {}
    start_location['type'] = "Point"
    start_location['coordinates'] = db_cluster['locations']['coordinates'][0]
    db_cluster['start_location'] = start_location

    end_location = {}
    end_location['type'] = "Point"
    end_location['coordinates'] = db_cluster['locations']['coordinates'][-1]
    db_cluster['end_location'] = end_location

    db_cluster['faces'] = list(itertools.chain.from_iterable(list(cluster['openfaces'])))
    db_cluster['location'] = find_common_location(cluster)

    db_cluster = outline_clusters(db_cluster)

    # find top three images:
    cluster = cluster.drop([u'inception_classification', u'inception_pool', u'syntactic_fingerprint'], axis=1)

    if len(cluster) > 3:
        div_one = len(cluster)/3
        div_two = 2*len(cluster)/3

        top_images = []

        images_dict = cluster.to_dict(orient='records')

        top_images.append(images_dict[np.argmax(cluster[:div_one]['interest_score'])])
        top_images.append(images_dict[np.argmax(cluster[div_one:div_two]['interest_score'])])
        top_images.append(images_dict[np.argmax(cluster[div_two:]['interest_score'])])

        db_cluster['top_images'] = top_images
    else:
        db_cluster['top_images'] = cluster.to_dict(orient='records')

    return db_cluster

# calculate some distance metrics on the cluster, its centroid, and its convex hull
def outline_clusters(cluster):
    points = MultiPoint(cluster['locations']['coordinates'])
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
    
    n = len(cluster['locations']['coordinates'])
    distance = 0
    if (n > 1):
        for j in range(n-1):
            distance += vincenty((cluster['locations']['coordinates'][j][1], cluster['locations']['coordinates'][j][0]), (cluster['locations']['coordinates'][j+1][1], cluster['locations']['coordinates'][j+1][0])).meters/1000

    cluster['boundary'] =  geom
    cluster['centroid'] = centroid
    cluster['distance'] = distance

    return cluster