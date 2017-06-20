import os
import json
from dateutil import parser
from wand.image import Image
from pytz import timezone
from timezonefinder import TimezoneFinder
import urllib2
import PIL
import PIL.Image
import PIL.ExifTags
import datetime
import time
import copy
from import_classify import Classifier
import import_faces
import import_cc_faces
import import_social_analysis
import import_clustering
from pymongo import MongoClient
import cv2
import numpy as np
import itertools
import place_cluster

client = MongoClient('127.0.0.1', 3001)
db = client.meteor

PREFIX_DIR = "/Users/loganw/Documents/vignette-photos/loganw/"
SEARCH_DIRECTORY = "/Users/loganw/Desktop/Walla Walla/"
USER_ID = "YchQu2tWYJpspmkbf"
USERNAME = "bilal"

extract_metadata_step = False
tf_step = False
face_step = False
social_interest_step = False
run_logical_images_step = False
run_generate_clusters_step = False
run_cluster_details_step = True
run_place_clustering_step = False

f = open("APIKEY.txt", "r")
GOOGLE_API_KEY = f.readline().strip()
f.close()

def getExifFromImage(img):
    if img._getexif() is None:
        return None
    else:
        return {
            PIL.ExifTags.TAGS[k]: v
            for k, v in img._getexif().items()
            if k in PIL.ExifTags.TAGS }

def getExifTime(exif):
    if exif is None:
        return None
    else:
        if 'DateTimeOriginal' in exif and not exif['DateTimeOriginal'] == '':
            return datetime.datetime.strptime(exif['DateTimeOriginal'],'%Y:%m:%d %H:%M:%S')
        elif 'DateTime' in exif and not exif['DateTime'] == '':
            # print 'DateTime found'
            return datetime.datetime.strptime(exif['DateTime'],'%Y:%m:%d %H:%M:%S')
        else:
            return None

def convertGPSToDecimal(gps):
    return float(gps[0][0])/gps[0][1] + float(gps[1][0])/(60.0*gps[1][1]) + float(gps[2][0])/(60.0*60.0*gps[2][1])

def getGPS(exif):
    lat = None
    lon = None
    alt = None
    gps_time = None
    gps_direction = None

    if 'GPSInfo' in exif and 2 in exif['GPSInfo']:
        lat = convertGPSToDecimal(exif['GPSInfo'][2])
        if exif['GPSInfo'][1] == 'S':
            lat = lat * -1

        lon = convertGPSToDecimal(exif['GPSInfo'][4])
        if exif['GPSInfo'][3] == 'W':
            lon = lon * -1

        if 6 in exif['GPSInfo']:
            alt = float(exif['GPSInfo'][6][0]) / exif['GPSInfo'][6][1] 
        else:
            alt = None

        if 7 in exif['GPSInfo']:
            try:
                gps_hours = exif['GPSInfo'][7][0][0] / exif['GPSInfo'][7][0][1]
                gps_minutes = exif['GPSInfo'][7][1][0] / exif['GPSInfo'][7][1][1]
                gps_seconds = int(exif['GPSInfo'][7][2][0] / exif['GPSInfo'][7][2][1])
                gps_microseconds = int(((float(exif['GPSInfo'][7][2][0]) / float(exif['GPSInfo'][7][2][1])) - gps_seconds) * 1e6)
                if gps_seconds < 60:
                    gps_time = datetime.time(gps_hours, gps_minutes, gps_seconds, gps_microseconds)
                    gps_time_str = gps_time.strftime('%H:%M:%S.%f')
                else:
                    gps_time = None
            except ZeroDivisionError:
                gps_time = None
        else:
            gps_time = None

        # GPS track direction
        if 17 in exif['GPSInfo']:
            gps_direction = exif['GPSInfo'][17][0] / exif['GPSInfo'][17][1]
        else:
            gps_direction = None

    return (lat, lon, alt, gps_time, gps_direction)

def orient_and_resize(image_filename):
    resized_images = {}
    
    img = Image(filename=image_filename)
    img.auto_orient()

    (w,h) = img.size
    w = int(w)
    h = int(h)

    filename_base = image_filename
    filename_base = filename_base.replace("/", "-")
    filename_base = filename_base.replace(" ", "_")

    with img.clone() as img_clone:
        img_clone.resize(1280, int((1280.0/w)*h))
        fname = "Resized/" + filename_base + "_1280.jpg"
        resized_images['1280'] = fname
        img_clone.save(filename=PREFIX_DIR + fname)

    with img.clone() as img_clone:
        img_clone.resize(640, int((640.0/w)*h))
        fname = "Resized/" + filename_base + "_640.jpg"
        resized_images['640'] = fname
        img_clone.save(filename=PREFIX_DIR + fname)

    with img.clone() as img_clone:
        img_clone.resize(320, int((320.0/w)*h))
        fname = "Resized/" + filename_base + "_320.jpg"
        resized_images['320'] = fname
        img_clone.save(filename=PREFIX_DIR + fname)

    with img.clone() as img_clone:
        img_clone.resize(160, int((160.0/w)*h))
        fname = "Resized/" + filename_base + "_160.jpg"
        resized_images['160'] = fname
        img_clone.save(filename=PREFIX_DIR + fname)

    img.close()
    return ((w, h), resized_images)

def getExposureTime(exif):
    if 'ExposureTime' in exif:
        exposure_time = float(exif['ExposureTime'][0]) / float(exif['ExposureTime'][1])
    else:
        exposure_time = 'null'

    return exposure_time

def getFlashFired(exif):
    if 'Flash' in exif:
        flash_fired = exif['Flash']
    else:
        flash_fired = 'null'

    return flash_fired

def get_image_metadata(image_filename, tz=None):
    print image_filename
    img = PIL.Image.open(image_filename)
    exif = getExifFromImage(img)
    img.close()

    image_entry = {}    
    image_entry["original_uri"] = image_filename

    exif_time = getExifTime(exif)

    # without information on when the image was taken, it's not useful. this is the minimum required metadata.
    if exif_time is not None:

        # extract information from GPS metadata
        (lat, lon, alt, gps_time, gps_direction) = getGPS(exif)
        # gps_time = gps_time.isoformat() if isinstance(gps_time, time) else gps_time

        image_entry["latitude"] = lat
        image_entry["longitude"] = lon
        image_entry["altitude"] = alt
        image_entry["direction"] = gps_direction

        image_entry["exposure_time"] = getExposureTime(exif)
        image_entry["flash_fired"] = getFlashFired(exif)

        timezonefinder = TimezoneFinder()

        try:
            tz = timezone(timezonefinder.timezone_at(lng=lon, lat=lat))
        except ValueError:
            print "No lat lon, using previous timezone"
        except TypeError:
            print "No lat lon, using previous timezone"
        except AttributeError:
            try:
                print "searching area"
                tz = timezone(timezonefinder.closest_timezone_at(lng=lon, lat=lat))
            except AttributeError:
                print "failed to find timezone, using previous"

        if gps_time is not None:
            # if image time matches GPS time, use the more precise GPS time
            if (exif_time.minute == gps_time.minute) and (abs(exif_time.second - gps_time.second) < 2):
                exif_time = exif_time.replace(second=gps_time.second, microsecond=gps_time.microsecond)

        # is_dst is only used when the time is ambiguous (thanks, Daylight Savings Time!)
        tz_offset = tz.utcoffset(exif_time, is_dst=False)
        tz_name = tz.tzname(exif_time, is_dst=False)

        # convert to UTC
        exif_time = exif_time - tz_offset

        image_entry["datetime"] = {"utc_timestamp": exif_time, "tz_offset": tz_offset.total_seconds(), "tz_name": tz_name};

        # geocode location

        if lat is not None:
            geolocation_url = "https://maps.googleapis.com/maps/api/geocode/json?latlng=" + str(lat) + "," + str(lon) + "&key=" + GOOGLE_API_KEY
            geolocation = json.loads(urllib2.urlopen(geolocation_url).read())
            image_entry["geolocation"] = geolocation
    else:
        print 'skipping processing'

    # orient and resize images

    (size, resized_images) = orient_and_resize(image_entry["original_uri"])

    image_entry["resized_uris"] = resized_images
    image_entry["size"] = size

    return (image_entry, tz)

def calc_ratio(img1, img2):
    # Initiate SIFT detector
    sift = cv2.SURF(500)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    
    ratio = None
    ransac_ratio = None
    homography_difference = None
    
    try:
        flann = cv2.FlannBasedMatcher(index_params,search_params)

        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        ratio = float(len(good)) / (len(kp1) + len(kp2) - len(matches))
        ransac_ratio = None

        if len(good)>10:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

            ransac_ratio = np.sum(mask) / (len(kp1) + len(kp2) - len(matches))

            h,w = img1.shape[:2]
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            
            homography_difference = np.sqrt(np.sum(np.square(pts - dst)))
    except:
        pass

    return (ratio, ransac_ratio, homography_difference, len(kp1) + len(kp2))

if extract_metadata_step:

    ##########################################
    # walk 
    ##########################################
    print("WALKING DIRECTORIES")
    image_list = []
    previous_tz = timezone("US/Pacific")
    for dir_name, subdir_list, file_list in os.walk(SEARCH_DIRECTORY):
        print('Found directory: %s' % dir_name)

        for filename in file_list:
            if filename[-3:].lower() == 'jpg':
                (new_image, previous_tz) = get_image_metadata(dir_name + "/" + filename, tz=previous_tz)
                new_image['user_id'] = USER_ID
                new_image['username'] = USERNAME
                db.images.insert_one(new_image)

if tf_step:

    ##########################################
    # TensorFlow run semantic analysis
    ##########################################

    print("TF CLASSIFIER")
    image_list = list(db.images.find({'datetime': {'$exists': True}, 'user_id': USER_ID}))
    TF = Classifier(PREFIX_DIR)
    images = TF.run_inference_on_images(image_list)
    db.images.drop()
    db.images.insert_many(images)

if face_step:

    ##########################################
    # Face detection/identification
    ##########################################

    print("OPENFACE CLASSIFIER")
    images = list(db.images.find({'datetime': {'$exists': True}, 'user_id': USER_ID}))
    ff = import_faces.FaceFinder()
    t = time.time()
    for i in range(len(images)):
        if (i % 100) == 0:
            elapsed = time.time() - t
            t = time.time()
            
            print( str(i) + '/' + str(len(images)) + ', ' + str(elapsed/100.0) + ' seconds per image')

        faces = ff.getFaces(PREFIX_DIR + images[i]['resized_uris']['1280'])
        images[i]['openfaces'] = faces

    db.images.drop()
    db.images.insert_many(images)

if social_interest_step:

    ##########################################
    # Old face detection
    ##########################################

    print("HAAR CASCADE CLASSIFIER")
    images = list(db.images.find({'datetime': {'$exists': True}, 'user_id': USER_ID}))
    t = time.time()
    for i in range(len(images)):
        if (i % 100) == 0:
            elapsed = time.time() - t
            t = time.time()
            
            print( str(i) + '/' + str(len(images)) + ', ' + str(elapsed/100.0) + ' seconds per image')
        cc_faces = import_cc_faces.detect_faces(PREFIX_DIR + images[i]['resized_uris']['1280'], images[i]['size'])  
        images[i]['faces'] = cc_faces

    ##########################################
    # Social interest analysis (dependent on old face classification method -- retrain?)
    ##########################################

    print("SOCIAL INTEREST ANALYZER")
    t = time.time()
    for i in range(len(images)):
        
        if (i % 100) == 0:
            t_per_im = (time.time() - t)/100
            print('    ' + str(t_per_im) + ' seconds per image')
            t = time.time()

        images[i]['interest_score'] = import_social_analysis.interest_score(images[i])

    ##########################################
    # Insert images into database
    ##########################################

    db.images.drop()
    db.images.insert_many(images)

if run_logical_images_step:

    images = list(db.images.find({'datetime': {'$exists': True}, 'user_id': USER_ID}))

    ##########################################
    # Duplicate classification
    ##########################################

    print("Classifying duplicates")

    # sort images by time
    sorted_images = sorted(images, key=lambda x: x['datetime']['utc_timestamp'])

    logical_images = []
    logical_image = [sorted_images[0]]

    img2 = cv2.imread(PREFIX_DIR + sorted_images[0]['resized_uris']['320'])

    for i in range(len(sorted_images) - 1):
        if (i % 100) == 0:
            print(i)
        
        img1 = img2
        img2 = cv2.imread(PREFIX_DIR + sorted_images[i+1]['resized_uris']['320'])
        time_difference = sorted_images[i+1]['datetime']['utc_timestamp'] - sorted_images[i]['datetime']['utc_timestamp']
        time_difference = np.abs(time_difference.seconds)

        h1,w1 = img1.shape[:2]
        h2,w2 = img2.shape[:2]
        
        if img1.shape[:2] == img2.shape[:2]:
            (r, rr, hd, lkp) = calc_ratio(img1, img2)
        elif (float(img1.shape[0]) / img1.shape[1]) - (float(img2.shape[1]) / img2.shape[0]) < 0.1:
            img2r = np.rot90(img2, k=1)
            (r, rr, hd, lkp) = calc_ratio(img1, img2r)
            
            img2r = np.rot90(img2, k=-1)
            (r2, rr2, hd2, lkp2) = calc_ratio(img1, img2r)
            
            if (hd2 < hd):
                (r, rr, hd, lkp) = (r2, rr2, hd2, lkp2)
            
        else:
            (r, rr, hd, lkp) = calc_ratio(img1, img2)
        
        save = True
        sure = False
        
        if (time_difference < 2):
                sure = True
        
        if (r > 0.09):
            if (hd is not None) and (hd < 250):
                sure = True
            if (time_difference < 5) and (r > 0.3):
                sure = True
            
        if lkp < 50:
            sure = True
        
        if sure:
            logical_image.append(sorted_images[i+1])
        else:
            logical_images.append(logical_image)
            logical_image = [sorted_images[i+1]]

    logical_images.append(logical_image)

    summarized_logical_images = []

    for i in range(len(logical_images)):
        for j in range(len(logical_images[i])):
            try:
                logical_images[i][j].pop('all_photos')
            except:
                pass
        
        li = copy.deepcopy(logical_images[i])
        
        interest_scores = [l['interest_score'] for l in li]
        longitudes = [l['longitude'] for l in li if np.isreal(l['longitude']) and l['longitude'] is not None]
        longitudes = [l for l in longitudes if not np.isnan(l)]
        latitudes = [l['latitude'] for l in li if np.isreal(l['latitude']) and l['latitude'] is not None]
        latitudes = [l for l in latitudes if not np.isnan(l)]
        geolocations = [l['geolocation'] for l in li if np.isreal(l['latitude']) and l['latitude'] is not None]

        med_lat = np.median(latitudes)
        med_lon = np.median(longitudes)
        
        v = np.argmax(interest_scores)
        li_dict = copy.deepcopy(li[v])
        
        if not np.isnan(med_lon) and med_lon is not None:
            li_dict['longitude'] = med_lon
            li_dict['latitude'] = med_lat

            distances = np.sqrt([(la - med_lat)**2 + (lo - med_lon)**2 for la, lo in zip(latitudes, longitudes)])
            d = np.argmin(distances)

            li_dict['geolocation'] = geolocations[d]

            li_dict['location'] = {'coordinates': [med_lon, med_lat], 'type': 'Point'}
            li_dict['location']['coordinates'][1] = med_lat
            
        li_dict['datetime'] = copy.deepcopy(li[0]['datetime'])
        
        li_dict['all_photos'] = copy.deepcopy(li)
        li_dict['all_photos'].pop(v)
        
        if '_id' in li_dict.keys():
            li_dict.pop('_id')

        summarized_logical_images.append(li_dict)

    ##########################################
    # Insert logical images into database
    ##########################################

    db.logical_images.insert_many(summarized_logical_images)

summarized_logical_images = None

if run_generate_clusters_step:

    print("CLUSTERING IMAGES")
    summarized_logical_images = list(db.logical_images.find({'user_id': USER_ID}))

    ##########################################
    # Clustering
    ##########################################

    clusters = import_clustering.cluster(summarized_logical_images)

    for cluster in clusters:
        db_cluster['photos'] = cluster
        db_cluster['user_id'] = USER_ID
        db_cluster['username'] = USERNAME

        db.clusters.insert_one(db_cluster)

if run_cluster_details_step:

    ##########################################
    # Extract relevant details and insert clusters into database
    ##########################################

    if summarized_logical_images is None:
        summarized_logical_images = list(db.logical_images.find({'user_id': USER_ID}))
    
    clusters = list(db.clusters.find({'user_id': USER_ID}))

    for cluster in clusters:
        db_cluster = import_clustering.make_cluster_details(cluster['photos'], logical_images=summarized_logical_images)
        cluster.update(db_cluster)
        db.clusters.update_one({'_id': cluster['_id']}, {'$set': cluster})

if run_place_clustering_step:

    ##########################################
    # Cluster places together
    ##########################################

    place_cluster.cluster_places(db, USERNAME, USER_ID)
