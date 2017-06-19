from math import radians, cos, sin, asin, sqrt
import numpy as np
import pymongo
import pandas as pd
import sklearn
from shapely.geometry import MultiPoint

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r  * 1000

def cluster_places(db, USERNAME, USER_ID):
    images = db.logical_images
    image_list = list(images.find({}))

    latlons = np.array([[i['longitude'], i['latitude']] for i in image_list if 'latitude' in i.keys() and i['latitude'] is not None])
    ids = [i['_id'] for i in image_list if 'latitude' in i.keys() and i['latitude'] is not None]

    precomputed_distance = np.zeros((len(latlons), len(latlons)))

    for i in range(len(latlons)):
        for j in range(len(latlons)):
            y_meters = (latlons[i,1] - latlons[j,1]) * 111111
            x_meters = (latlons[i,0] - latlons[j,0]) * (111111 * np.cos((latlons[i,1] + latlons[j,1]) * np.pi / 360))
            
            precomputed_distance[i,j] = np.sqrt(x_meters**2 + y_meters**2)

    clus = sklearn.cluster.DBSCAN(eps=80, min_samples=4, metric='precomputed')
    clus.fit(precomputed_distance)

    labels = clus.labels_
    unique_labels = set(labels)

    for i in unique_labels:
        if i != -1:
            cluster_center = np.median(latlons[labels == i, :], axis=0)
            cluster_latlons = latlons[labels == i, :]
            max_distance = 0
            
            for j in range(len(cluster_latlons)):
                d = haversine(cluster_center[0], cluster_center[1], cluster_latlons[j,0], cluster_latlons[j,1])
                if (d > max_distance):
                    max_distance = d
            
            cluster_radius = max(50, max_distance + 20)

            image_cluster_list = []
            
            for j in range(len(latlons)):
                if haversine(cluster_center[0], cluster_center[1], latlons[j,0], latlons[j,1]) < cluster_radius:
                    image_cluster_list.append(ids[j])
                        
            place = {}
            place['location'] = {'type': 'Point', 'coordinates': list(cluster_center)}
            place['radius'] = cluster_radius
            place['images'] = list(image_cluster_list)
            place['user_id'] = USER_ID
            place['username'] = USERNAME
            
            place_id = db.places.insert_one(place).inserted_id
            
            for image in image_cluster_list:
                db.logical_images.update_one({'_id': image}, {'$set': {'place': {'place_id': place_id}}})
                
                cluster = list(db.clusters.find({'photos': {'$in': [image]}}))
                
                if len(cluster) == 1:
                    cluster = cluster[0]
                    if 'places' in cluster.keys():
                        places = cluster['places']
                    else:
                        places = []
                    
                    place_ids = [p['place_id'] for p in places]
                    
                    if place_id not in place_ids:
                        places.append({'place_id': place_id})
                        db.clusters.update_one({'_id': cluster['_id']}, {'$set': {'places': places}})
                else:
                    print cluster
                    print "?"