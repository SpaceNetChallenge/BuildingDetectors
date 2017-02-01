import numpy as np
np.random.seed(227)
import json
from shapely.geometry import Polygon, shape, Point
import gdal, ogr
import pandas as pd
import scipy.misc
import skimage.draw
import skimage.io
import sys, os, re, datetime, multiprocessing
from keras.models import *
from keras.layers import *
from keras.layers.advanced_activations import *
from keras.layers.normalization import *
from keras.optimizers import *
import keras.preprocessing.image
import xgboost as xgb

img_shape = (408, 440)
default_nthread=4

def draw_polygon(polygon):
    p = re.findall(r'[0-9.]+ [0-9.]+ [0-9.]+', polygon)
    sp = [pp.split(' ') for pp in p]
    x = [round(float(pp[0])) for pp in sp]
    y = [round(float(pp[1])) for pp in sp]
    yy, xx = skimage.draw.polygon(y, x, img_shape)
    return yy, xx

def CreateLabel(fn, gtf):
    polygons = gtf.ix[gtf['ImageId'] == fn]
    polygons = polygons.ix[polygons['BuildingId'] >= 0, 'PolygonWKT_Pix']    

    label_fill = np.zeros(img_shape)
    label_edge = np.zeros(img_shape)
    for polygon in polygons:
        p = re.findall(r'[0-9.\-]+ [0-9.\-]+ [0-9.\-]+', polygon)
        sp = [pp.split(' ') for pp in p]
        x = [round(float(pp[0])) for pp in sp]
        y = [round(float(pp[1])) for pp in sp]
        yy, xx = skimage.draw.polygon(y, x)
        label_fill[yy, xx] = 1
        yy, xx = skimage.draw.polygon_perimeter(y, x)
        label_edge[yy, xx] = 1
    return label_fill, label_edge

def CreateData(path, fn, gtf=None):
    ds3 = gdal.Open(path+'3band/'+'3band_'+fn+'.tif')
    ds8 = gdal.Open(path+'8band/'+'8band_'+fn+'.tif')
    if gtf is not None:
        data = np.zeros(img_shape + (13,), dtype='uint16')
        data[:, :, 11], data[:, :, 12] = CreateLabel(fn, gtf)
    else:
        data = np.zeros(img_shape + (11,), dtype='uint16')
    
    data3 = ds3.ReadAsArray()
    data3shape = data3.shape[1:]
    data3shape = (min(data3shape[0], img_shape[0]), min(data3shape[1], img_shape[1]))
    for ch in range(3):
        data[:data3shape[0], :data3shape[1], ch] = data3[ch, :data3shape[0], :data3shape[1]]
    data8 = ds8.ReadAsArray()
    for ch in range(8):
        vmax = data8[ch].max()
        data[:data3shape[0], :data3shape[1], 3 + ch] = scipy.misc.imresize((data8[ch] / vmax * 255).astype('uint8'), data3shape, interp='bilinear') / 255 * vmax
    return data

def GetFileList(path):
    return [fn[6:-4] for fn in sorted(os.listdir(path + '3band/')) if fn.endswith('.tif')]

def CreateTrain(path, gtf_path):
    os.makedirs('train', exist_ok=True)
    file_list = GetFileList(path)
    gtf = pd.read_csv(gtf_path)
    for i, fn in enumerate(file_list):
        print(i, fn)
        CreateData(path, fn, gtf).dump('train/{}.npz'.format(fn))
    
def create_model():
    def conv(nb_filters, normalize, border_mode='same'):
        def ret(x):
            y = x
            for nb_filter in nb_filters:
                y = Convolution2D(nb_filter, 3, 3, border_mode=border_mode)(y)
                if normalize: y = BatchNormalization()(y)
                y = LeakyReLU(alpha=0.3)(y)
            return y
        return ret

    input = Input(shape=img_shape+(11,), dtype='float32')
    x1 = ZeroPadding2D((4, 4))(input)
    x1 = conv([16] * 2, False)(x1)
    x2 = conv([16] * 2, True)(x1)

    npooling = 4
    p = x2
    pi = []
    for npool in range(npooling):
        pi.append(p)
        p = MaxPooling2D((2, 2))(p)
        p = conv([24] * 3, True)(p)
    for npool in range(npooling):
        p = UpSampling2D((2, 2))(p)
        p = conv([24] * 1, True)(p)
        p = merge([p, pi[-1-npool]], mode='concat', concat_axis=3)
    x3 = p
    x4 = conv([32] * 3, True, border_mode='valid')(x3)
    output_area = Convolution2D(1, 3, 3, border_mode='valid')(x4)
    output_area = Activation('sigmoid', name='area')(output_area)
    output_edge = Convolution2D(1, 3, 3, border_mode='valid')(x4)
    output_edge = Activation('sigmoid', name='edge')(output_edge)

    model = Model(input=input, output=[output_area, output_edge])
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    return model

def get_val_and_generator():
    path = 'train/'
    files = [path + f for f in sorted(os.listdir(path)) if f.endswith('.npz')]
    tr = []
    val = []
    for i, f in enumerate(files):
        if i % 50 == 0: val.append(f)
        else: tr.append(f)
    val_img = np.concatenate([np.load(f)[None, :] for f in val])

    def gen():
        batch_size = 8
        rnd = np.random.RandomState(0)
        while True:
            rnd.shuffle(tr)
            for i in range(0, len(tr), batch_size):
                t = tr[i:min(i+batch_size, len(tr))]
                img = np.concatenate([np.load(f)[None, :] for f in t])
                yield img[:, :, :, :-2], [img[:, :, :, -2:-1], img[:, :, :, -1:]]

    return (val_img[:, :, :, :-2], [val_img[:, :, :, -2:-1], val_img[:, :, :, -1:]]), gen()

def train(nepoch=100):   
    os.makedirs('model', exist_ok=True)
    save_interval = 1

    model = create_model()
    validation_data, gen = get_val_and_generator()
    with open('log_keras.csv'.format(), 'w') as logfile: 
      logfile.write("i, loss\n")
      for i in range(1, nepoch + 1):        
        loss = model.fit_generator(gen, validation_data=validation_data, samples_per_epoch=1000, nb_epoch=1, verbose=1, pickle_safe=True)

        logline = "{:>3}, {}\n".format(i, loss.history)
        print(logline, end="")
        logfile.write(logline)

        model.save_weights('model/2_{}.h5'.format(i), overwrite=True)

    return model

#CreateGeoJSON and FixGeoJSON functions were copied from https://gist.github.com/hagerty and modified. 
#
#Copyright (c) 2016 In-Q-Tel
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

def CreateGeoJSON ( fn, cluster ):
    memdrv = gdal.GetDriverByName ('MEM')
    src_ds = memdrv.Create('',cluster.shape[1],cluster.shape[0],1)
    src_ds.SetGeoTransform([0, 1, 0, 0, 0, 1])
    band = src_ds.GetRasterBand(1)
    band.WriteArray(cluster)
    dst_layername = "BuildingID"
    drv = ogr.GetDriverByName("geojson")

    if os.path.exists('./geojson/' + fn + dst_layername + ".geojson"):
        drv.DeleteDataSource('./geojson/' + fn + dst_layername + ".geojson")

    dst_ds = drv.CreateDataSource ( './geojson/' + fn + dst_layername + ".geojson")
    dst_layer = dst_ds.CreateLayer( dst_layername, srs=None )
    dst_layer.CreateField( ogr.FieldDefn("DN", ogr.OFTInteger) )
    gdal.Polygonize( band  , None, dst_layer, 0, ['8CONNECTED=8'], callback=None )
    return

def FixGeoJSON( fn ):
    buf_dist = 0.0
    dst_layername = "BuildingID"
    drv = ogr.GetDriverByName("geojson")
    dst_ds = drv.Open ( './geojson/' + fn + dst_layername + ".geojson")
    dst_layer = dst_ds.GetLayer(0)
    if os.path.exists('./geojson/buffer' + fn + dst_layername + ".geojson"):
        drv.DeleteDataSource('./geojson/buffer' + fn + dst_layername + ".geojson")
    adst_ds = drv.CreateDataSource ( './geojson/buffer' + fn + dst_layername + ".geojson")
    adst_layer = adst_ds.CreateLayer( dst_layername, srs=None )
    adst_layer.CreateField( ogr.FieldDefn("DN", ogr.OFTInteger) )

    for i in range(dst_layer.GetFeatureCount()):
        f = dst_layer.GetFeature(i)
        clusternumber = f.GetField("DN")
        f.SetGeometry(f.GetGeometryRef().Buffer(buf_dist))
        if 0 == f.GetField("DN"):
            dst_layer.DeleteFeature(i) #not supported by geoJSON driver now
        else:
            adst_layer.CreateFeature(f)
    return

#################################################################################################

def ParseGeoJSON( fn ):    
    with open('./geojson/buffer' + fn + "BuildingID" + ".geojson") as f:
        polygon_list = json.load(f)['features']
        if len(polygon_list) == 0:
            yield '{},-1,POLYGON EMPTY,1'.format(fn)
        else:
            check_img = np.zeros(img_shape)
            for polygon in polygon_list:
                dn = polygon['properties']['DN']
                coords_raw = polygon['geometry']['coordinates'][0]
                rr, cc = skimage.draw.polygon([int(p[1]) for p in coords_raw], [int(p[0]) for p in coords_raw], img_shape)
                check_img[rr,cc] = 1

                coords = ','.join((str(int(co[0])) + ' ' + str(int(co[1])) + ' 0' for co in coords_raw))
                yield '{},{},"POLYGON (({}))",{}'.format(fn, dn, coords, dn)

def FindAllClusters( intensity, edge ):
    precision = 1e9

    pixels = np.vstack([[[i, j, 0] for i in range(intensity.shape[0])] for j in range(intensity.shape[1])])
    intensity[intensity < 0.45] = 0
    pixels[:, 2] = (intensity[pixels[:, 0], pixels[:, 1]] * precision).astype('int32')
    pixels = pixels[pixels[:, 2] > 0.50 * precision]
    pixels = pixels[pixels[:, 2].argsort()[::-1]]
    
    cluster = 0 * intensity
    features = []
    cluster_size_list = []

    k = 0
    for ip, p in enumerate(pixels):
        if cluster[p[0], p[1]] > 0: continue
        feature = []
        k = k + 1
        
        current = np.array([p])
        cluster_size = 0
        max_next_size = 0

        next_size_duplicates_list = []
        next_size_duplicates_ratio_list = []
        next_size_filled_list = []
        next_size_filled_ratio_list = []
        next_size_list = []
        next_size_diff_list = []
        next_size_ratio_list = []

        intensity_list = []
        edge_list = []
        valid_intensity_list = []
        valid_edge_list = []
        invalid_intensity_list = []
        invalid_edge_list = []
        while current.shape[0] > 0:
            cluster[current[:, 0], current[:, 1]] = k
            cluster_size += current.shape[0]
            next = np.concatenate([current + d for d in [(1,0,0), (0,1,0), (-1,0,0), (0,-1,0)]])
            next = next[np.lexsort([next[:, 0], next[:, 1], next[:, 2]], axis=0)]
            duplicates = np.all(next[:-1, :2] == next[1:, :2], axis=1)
            duplicates = np.concatenate([duplicates, [False]])

            size = next.shape[0]
            next = next[-duplicates]
            next_size_duplicates_list.append(size - next.shape[0])
            next_size_duplicates_ratio_list.append((size - next.shape[0]) / (size + 0.1))
            next = next[(0 <= next[:, 0]) & (next[:, 0] < intensity.shape[0]) & (0 <= next[:, 1]) & (next[:, 1] < intensity.shape[1])]
            size = next.shape[0]
            next = next[cluster[next[:, 0], next[:, 1]] == 0]
            next_size_filled_list.append(size - next.shape[0])
            next_size_filled_ratio_list.append((size - next.shape[0]) / (size + 0.1))

            nextintencity = intensity[next[:, 0], next[:, 1]] * precision
            intensity_list.append(nextintencity)
            nextedge = edge[next[:, 0], next[:, 1]]
            edge_list.append(nextedge)

            idx = (0.0 < nextintencity) & (nextintencity < next[:, 2] * 1.02)

            valid_intensity_list.append(nextintencity[idx])
            invalid_intensity_list.append(nextintencity[-idx])
            valid_edge_list.append(nextedge[idx])
            invalid_edge_list.append(nextedge[-idx])

            idx_keepmax = next[:, 2] < nextintencity
            nextintencity[idx_keepmax] = next[idx_keepmax, 2]
            next[:, 2] = nextintencity
            size = next.shape[0]
            next = next[idx]
            next_size_diff_list.append(size - next.shape[0])
            next_size_ratio_list.append((size - next.shape[0]) / (size + 0.1))
            next_size_list.append(next.shape[0])

            max_next_size = max(max_next_size, next.shape[0])

            current = next

        if cluster_size < 100:
            cluster[cluster == k] = 0
            k = k - 1
        else:
            feature.append(k)
            feature.append(p[2])
            feature.append(len(next_size_list))
            feature.append(max_next_size)
            feature.append(cluster_size)
            for size_list in [next_size_duplicates_list,
                                next_size_duplicates_ratio_list,
                                next_size_filled_list,
                                next_size_filled_ratio_list,
                                next_size_list,
                                next_size_diff_list,
                                next_size_ratio_list]:
                    n = len(size_list)
                    for i in range(4):
                        li = size_list[n * i // 4: n * (i + 1) // 4]
                        feature.append(sum(li) / (len(li) + 0.1))
                    li = list(sorted(size_list))
                    for i in range(1, 5):
                        feature.append(li[n * i // 5])
                    feature.append(sum(li) / n)
            feature.append(np.std(next_size_list))
            for value_list in [intensity_list,
                                edge_list,
                                valid_intensity_list,
                                valid_edge_list,
                                invalid_intensity_list,
                                invalid_edge_list]:
                    np_value_list = np.concatenate(value_list)
                    n = np_value_list.shape[0]
                    for i in range(4):
                        li = np_value_list[n * i // 4: n * (i + 1) // 4]
                        feature.append(li.sum() / (li.shape[0] + 0.1))
                    np_value_list.sort()
                    for i in range(1, 5):
                        feature.append(np_value_list[n * i // 5] if n > 0 else -1)
                    feature.append(np_value_list.sum() / n)
            features.append(feature)
    
    features = np.array(features)
    if features.shape[0] > 0:
        stats = np.concatenate([np.percentile(features, q, axis=0) for q in [25, 50, 75]] + [features.mean(axis=0)])   
        stats = np.tile(stats, (features.shape[0], 1))
        features = np.concatenate([features, stats], axis=1)
    return k, cluster, features

def pred_proc(arg):
    path = arg[0]
    filenames = arg[1]

    model_file = '2_96'
    model = create_model()
    model.load_weights('model/{}.h5'.format(model_file))

    for fn in filenames:    
        dt1 = datetime.datetime.now()
        area, edge = model.predict(CreateData(path, fn)[np.newaxis, :])
        area = area[0, :, :, 0]
        edge = edge[0, :, :, 0]
        area.dump('area/{}.npz'.format(fn))
        edge.dump('edge/{}.npz'.format(fn))
        dt2 = datetime.datetime.now()
        print(fn, int((dt2-dt1).total_seconds()))

def predict(path, nthread=default_nthread):
    all_files = GetFileList(path)
    n = len(all_files)
    filelist_group = [all_files[n*i//nthread:n*(i+1)//nthread] for i in range(nthread)]

    if True:
        with multiprocessing.Pool(nthread) as pool:
            pool.map(pred_proc, [(path, filelist) for filelist in filelist_group])
    else:
        pred_proc((path, all_files))

def train_building_proc(arg):
    pred_proc(arg)

    path = arg[0]
    filenames = arg[1]
    ith = arg[2]
    gtf_path = arg[3]

    gtf = pd.read_csv(gtf_path)

    feature_list = []
    label_list = []
    for fn in filenames:
        area = np.load('area/{}.npz'.format(fn))
        edge = np.load('edge/{}.npz'.format(fn))
        cluster_count, cluster, features = FindAllClusters(area, edge)
        if cluster_count == 0: continue

        truth_polygons = gtf.ix[(gtf['ImageId'] == fn) & (gtf['BuildingId'] >= 0), 'PolygonWKT_Pix']
        truth = []
        for polygon in truth_polygons:
            img = np.zeros(img_shape, dtype=int)
            rr, cc = draw_polygon(polygon)
            img[rr, cc] = 1
            truth.append((img, img.sum()))
        labels = []
        for k in range(1, 1 + cluster_count):
            p = (cluster == k)
            ps = p.sum()
            maxv = 0
            for t, ts in truth:
                v = (p & t).sum()
                v = v / (ps + ts - v)
                maxv = max(maxv, v)
            labels.append(maxv)
            print(fn, k, maxv)
        feature_list.append(features)
        label_list.append(np.array(labels))
    feature_list = np.concatenate(feature_list)
    label_list = np.concatenate(label_list)
    print(feature_list.shape, label_list.shape)
    np.concatenate([feature_list, label_list[:, np.newaxis]], axis=1).dump('train_building_{}.npz'.format(ith))

def train_building(path, gtf_path, create_data=True, nthread=default_nthread):
    if create_data:
        all_files = GetFileList(path)
        n = len(all_files)
        filelist_group = [all_files[n*i//nthread:n*(i+1)//nthread] for i in range(nthread)]

        with multiprocessing.Pool(nthread) as pool:
            pool.map(train_building_proc, [(path, filelist[1], filelist[0], gtf_path) for filelist in enumerate(filelist_group)]) 

        np.concatenate([np.load('train_building_{}.npz'.format(ith)) for ith in range(nthread)]).dump('train_building.npz')
    
    tr = np.load('train_building.npz')
    val_idx = np.array(range(tr.shape[0])) % 100 == 0
    label = tr[:, -1]
    label = 1 / (1 + np.exp(4.0 - 8.0 * label))
    val_label = label[val_idx]
    label = label[-val_idx]
    val = tr[val_idx, :-1]
    tr = tr[-val_idx, :-1]
    dtrain = xgb.DMatrix(tr, label=label)
    dval = xgb.DMatrix(val, label=val_label)
    params = {
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'eta': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'max_depth': 6,
        'seed': 0,
        'nthread': default_nthread,
        'silent': 1,
    }
    model = xgb.train(params, dtrain, evals=[(dtrain, 'train'), (dval, 'val')], num_boost_round=1000, verbose_eval=True)
    model.save_model('model/xgb.model')

def test_proc(arg):
    pred_proc(arg)

    path = arg[0]
    filenames = arg[1]

    xgbmodel = xgb.Booster({'nthread': 1})
    xgbmodel.load_model(arg[2])

    minconfidence = 0.22
    for fn in filenames:
        dt2 = datetime.datetime.now()
        area = np.load('area/{}.npz'.format(fn))
        edge = np.load('edge/{}.npz'.format(fn))
        cluster_count, cluster_raw, features = FindAllClusters(area, edge)
        cluster = np.zeros(img_shape, dtype=int)
        if cluster_count > 0:
            confidence = pd.Series(xgbmodel.predict(xgb.DMatrix(features)))
            rank = confidence.rank(method='first')
            maxrank = rank.max()
            minrank = rank[confidence >= minconfidence].min()
            rank = rank - maxrank + 255
            rank[confidence < minconfidence] = 0
            rank[rank < 0] = 0            
            for i in range(cluster_count):
                cluster[cluster_raw == (i + 1)] = rank[i]

        CreateGeoJSON(fn, cluster)
        FixGeoJSON(fn)
        dt3 = datetime.datetime.now()
        print(fn, cluster_count, int((dt3-dt2).total_seconds()))

def test(path, nthread=default_nthread):
    nsplit = 20
    all_files = GetFileList(path)
    n = len(all_files)
    filelist_group = [all_files[n*i//nsplit:n*(i+1)//nsplit] for i in range(nsplit)]

    with multiprocessing.Pool(nthread) as pool:
        pool.map(test_proc, [(path, filelist, "model/xgb.model") for filelist in filelist_group])  

def merge_results(path, out_filepath):
    with open(out_filepath, 'w') as fw:
        fw.write('ImageId,BuildingId,PolygonWKT_Pix,Confidence\n')
        for fn in GetFileList(path):
            lines = ParseGeoJSON(fn)
            for li in lines: fw.write(li + '\n')

def run():
    os.makedirs('./geojson', exist_ok=True) 
    os.makedirs('./area', exist_ok=True)
    os.makedirs('./edge', exist_ok=True)

    command = sys.argv[1]
    if command == 'train':
        step = sys.argv[2]
        train_dir = sys.argv[3]
        if not train_dir.endswith('/'): train_dir += '/'
        gtf_path = sys.argv[4]
        if step == '1': CreateTrain(train_dir, gtf_path)
        if step == '2': train()
        if step == '3': train_building(train_dir, gtf_path)        

    if command == 'test':
        test_dir = sys.argv[2]
        out_filepath = sys.argv[3]
        if not test_dir.endswith('/'): test_dir += '/'
        test(test_dir)
        merge_results(test_dir, out_filepath)

if __name__ == '__main__':
    run()