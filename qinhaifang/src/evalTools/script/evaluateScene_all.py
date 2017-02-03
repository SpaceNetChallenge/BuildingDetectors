from spaceNet import evalTools as eT
from spaceNet import geoTools as gT
import numpy as np
import sys
import multiprocessing
import time
import os
import os.path
import re
import logging
import logging.config
from multiprocessing import Pool
import setting

def convert_fp_worker(test_file):
    """docstring for convert_worker"""
    try:
        image_id = '_'.join(test_file.split('.')[0].split('_')[:-1])
        print('image id{}'.format(image_id))
        predict_geojson_file = os.path.join(setting.PREDICT_PIXEL_GEO_JSON_DIR, '{}_predict.geojson'.format(image_id))
	truth_geojson_file = os.path.join(setting.PIXEL_GEO_JSON_DIR, '{}_Pixel.geojson'.format(image_id))
	#truth_geojson_file = os.path.join(setting.PIXEL_GEO_JSON_DIR_4X, '{}_Pixel.geojson'.format(image_id))
        truth_fp = truth_geojson_file
        print('truth_fp:{}'.format(truth_fp))
	test_fp = predict_geojson_file
        print('test_fp:{}'.format(test_fp))
	# initialize scene counts
        #test_fp = '/data/building_extraction/SpaceNet/data/predict_pixelGeoJson/3band_013022223132_Public_img2052_predict.geojson'
        #truth_fp = '/data/building_extraction/SpaceNet/data/predict_pixelGeoJson/3band_013022223132_Public_img2052_predict.geojson'
        true_pos_counts = []
    	false_pos_counts = []
    	false_neg_counts = []

    	t0 = time.time()
    	# Start Ingest Of Truth and Test Case
    	sol_polys = gT.importgeojson(truth_fp, removeNoBuildings=True)
    	prop_polys = gT.importgeojson(test_fp)
    	# Speed up search by preprocessing ImageId and polygonIds

    	test_image_ids = set([item['ImageId'] for item in prop_polys if item['ImageId'] > 0])
    	prop_polysIdList = np.asarray([item['ImageId'] for item in prop_polys if item["ImageId"] > 0 and \
                                   item['BuildingId']!=-1])
    	prop_polysPoly = np.asarray([item['poly'] for item in prop_polys if item["ImageId"] > 0 and \
                                   item['BuildingId']!=-1])
    	sol_polysIdsList = np.asarray([item['ImageId'] for item in sol_polys if item["ImageId"] > 0 and \
                                   item['BuildingId']!=-1])
    	sol_polysPoly = np.asarray([item['poly'] for item in sol_polys if item["ImageId"] > 0 and \
                                   item['BuildingId']!=-1])
        bad_count = 0
    	F1ScoreList = []
	ResultList = []

    	eval_function_input_list = eT.create_eval_function_input((test_image_ids,
                                                         (prop_polysIdList, prop_polysPoly),
                                                         (sol_polysIdsList, sol_polysPoly)))
        # Calculate Values
    	if parallel==False:
            result_list = []
            for eval_input in eval_function_input_list:
                #print(eval_input)
                #return 1, 1, 1
                result_list.append(eT.evalfunction(eval_input))
    	else:
            result_list = p.map(eT.evalfunction, eval_function_input_list)
    	result_sum = np.sum(result_list, axis=0)
        #if result_sum == 0:
        #    return 0,0,0
    	true_pos_total = result_sum[1]
    	false_pos_total = result_sum[2]
    	false_neg_total = result_sum[3]
    	print('True_Pos_Total', true_pos_total)
    	print('False_Pos_Total', false_pos_total)
    	print('False_Neg_Total', false_neg_total)
        if float(true_pos_total) < 1:
	    precision = 0
            recall = 0
            F1ScoreTotal = 0
        else:
    	    precision = float(true_pos_total) / (float(true_pos_total) + float(false_pos_total))
    	    recall = float(true_pos_total) / (float(true_pos_total) + float(false_neg_total))
    	    F1ScoreTotal = 2.0 * precision*recall / (precision + recall)
    	print('precision', precision)
    	print('recall',  recall)
    	print('F1Total', F1ScoreTotal)

    	print(result_list)
    	print(np.mean(result_list))
        #eval_worker(truth_geojson_file,predict_geojson_file);
        return precision,recall,F1ScoreTotal
    except Exception as e:
        logging.warning('Eval Exception[{}] image_id[{}]'.format(e, image_id))
        return 0,0,0



if __name__ == "__main__":

    # load Truth and Test File Locations
    if len(sys.argv) >= 1:
	test_fp_path = sys.argv[1]
        #truth_fp_path = sys.argv[2]
    else:
        # a test
        test_fp = '/data/building_extraction/SpaceNet/data/predict_pixelGeoJson/013022232200_Public_img7001_predict.geojson'
        truth_fp = '/data/building_extraction/SpaceNet/data/pixelGeoJson/013022232200_Public_img7001_Pixel.geojson'

    parallel=False
    test_file_list = os.listdir(test_fp_path)
    prec_sum = 0
    rec_sum = 0
    F1_Score_sum = 0
    case = 0
    for test_file in test_file_list:
        #print(test_file)
        pre, rec, F1 = convert_fp_worker(test_file)
        print('{},{},{}'.format(pre,rec,F1))
	if pre == 2:
            print('jump!')
	    continue
	else:
	    prec_sum = prec_sum + pre
	    rec_sum = rec_sum + rec
	    #F1_Score_sum = F1_Score_sum + F1
            case += 1
        if case % 100 == 0:
            logging.info('Eval {}'.format(case))

    prec_sum = prec_sum/case
    rec_sum = rec_sum/case
    F1_Score_sum = 2.0 * prec_sum*rec_sum / (prec_sum + rec_sum)
    print('case:{}'.format(case))
    print('precision_ap:{}'.format(prec_sum))
    print('recall_ap:{}'.format(rec_sum))
    print('F1_Score_ap:{}'.format(F1_Score_sum))


