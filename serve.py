#!flask/bin/python
from flask import Flask, jsonify
import pymongo
from shapely.geometry import MultiPoint
from bson import ObjectId
from flask import make_response
import pandas as pd
from sklearn import cluster as clus
import datetime
import numpy as np
import copy
import import_clustering

app = Flask(__name__)
client = pymongo.MongoClient("localhost", 3001)
db = client.meteor

# This route will generate the full cluster info based on a cluster stub that
# contains only a list of photos to be included in that cluster
@app.route('/generate/<string:cluster_id>', methods=['PUT'])
def generate_cluster(cluster_id):
    clusters = list(db.clusters.find({'_id': ObjectId(cluster_id)}));

    if (len(clusters) > 1):
        return jsonify({'error': 'too many clusters'}), 500

    if (len(clusters) == 0):
        return jsonify({'error': 'no cluster with id'}), 500

    cluster = clusters[0]

    cluster_details = import_clustering.make_cluster_details(cluster['photos'], db=db)

    db.clusters.update_one({'_id': ObjectId(cluster_id)}, {'$set': cluster_details})

    return jsonify({'response': 'okay'})

@app.route('/update/<string:cluster_id>', methods=['PUT'])
def update_cluster(cluster_id):
    print(cluster_id)
    clusters = list(db.clusters.find({'_id': ObjectId(cluster_id)}));

    if (len(clusters) > 1):
        return jsonify({'error': 'too many clusters'}), 500

    if (len(clusters) == 0):
        return jsonify({'error': 'no cluster with id'}), 500

    cluster = clusters[0]

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
    
    print(centroid)

    db.clusters.update_one({'_id': cluster['_id']}, {'$set': {'boundary': geom, 'centroid': centroid}})

    return jsonify({'response': 'okay'})

@app.route('/split/<string:cluster_id>', methods=['GET'])
def split_cluster(cluster_id):
    print("splitting cluster")
    print(cluster_id)

    clusters = list(db.clusters.find({'_id': ObjectId(cluster_id)}))

    if (len(clusters) > 1):
        return jsonify({'error': 'too many clusters'}), 500

    if (len(clusters) == 0):
        return jsonify({'error': 'no cluster with id'}), 500

    cluster = clusters[0]

    # now do a k-means split with k=2 on this cluster

    stretch_nights = 5.0
    timescale = 12.0

    cluster_df = pd.DataFrame()

    time = 0
    times = []
    control_time = 0

    for i in range(len(cluster['times'])):
        times.append(time)

        if (i < len(cluster['times'])-1):
            current_time = cluster['times'][i]['utc_timestamp']

            while (cluster['times'][i]['utc_timestamp'] - current_time).total_seconds() > (60 * 60):
                local_time = current_time + datetime.timedelta(0, cluster['times'][i]['tz_offset'])

                if (local_time.hour > 2) and (local_time.hour < 7):
                    time += stretch_nights / timescale
                    control_time += 1.0 / timescale
                else:
                    time += 1.0 / timescale
                    control_time += 1.0/timescale

                current_time += datetime.timedelta(0, 60*60)

            local_time = current_time + datetime.timedelta(0, cluster['times'][i]['tz_offset'])

            if (local_time.hour > 2) and (local_time.hour < 7):
                time += stretch_nights * (cluster['times'][i+1]['utc_timestamp'] - current_time).total_seconds()/(60*60*timescale);
                control_time += 1.0 * (cluster['times'][i+1]['utc_timestamp'] - current_time).total_seconds()/(60*60*timescale);
            else:
                time += 1.0 * (cluster['times'][i+1]['utc_timestamp'] - current_time).total_seconds()/(60*60*timescale);
                control_time += 1.0 * (cluster['times'][i+1]['utc_timestamp'] - current_time).total_seconds()/(60*60*timescale);

    times = np.array(times)
    times *= (control_time / time)
    times -= np.mean(times)
    cluster_df['cluster_time'] = times

    latlontimes = np.zeros((len(cluster['times']), 3))
    latlontimes[:,0] = [cluster['locations']['coordinates'][i][1] for i in range(len(cluster['locations']['coordinates']))]
    latlontimes[:,1] = [cluster['locations']['coordinates'][i][0] for i in range(len(cluster['locations']['coordinates']))]
    latlontimes[:,2] = cluster_df['cluster_time']

    cl = clus.KMeans(n_clusters=2).fit(latlontimes)

    print(cl.labels_)

    cluster_one = [photo for (photo, i) in zip(cluster['photos'], range(len(cluster['photos']))) if cl.labels_[i] == cl.labels_[0]]
    cluster_two = [photo for (photo, i) in zip(cluster['photos'], range(len(cluster['photos']))) if cl.labels_[i] != cl.labels_[0]]
    
    result_one = db.clusters.insert_one({'photos': cluster_one, 'user_id': cluster['user_id'], 'username': cluster['username']})
    result_two = db.clusters.insert_one({'photos': cluster_two, 'user_id': cluster['user_id'], 'username': cluster['username']})
    
    print(str(result_one.inserted_id), str(result_two.inserted_id))

    db.clusters.delete_one({'_id': ObjectId(cluster_id)})
    generate_cluster(str(result_one.inserted_id))
    generate_cluster(str(result_two.inserted_id))

    return jsonify({'response': 'okay'});

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.errorhandler(500)
def not_found(error):
    return make_response(jsonify({'error': 'Internal server error'}), 500)

if __name__ == '__main__':
    app.run(debug=True, port=3122)