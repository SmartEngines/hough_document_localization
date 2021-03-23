import sys
import metrics
import json
import numpy as np
import argparse


THRESHOLD = 0.9
RESIDUALS_THRESHOLD = 0.017


def read_json(fin):
    with open(fin, encoding='utf-8') as f:
        return json.load(f)


def process_name(name):
    pos = max(name.rfind('/'), name.rfind('\\'))
    name = name[pos + 1: len(name)]
    pos = name.find('.')
    name = name[0: pos]

    return name


def process_item(item, statistics):	
    stat = {}
    if not item['system_result_quad_exists']:
        if 'mean_iou' in statistics:
            stat['mean_iou'] = 0.0
        if 'iou_gt' in statistics:
            stat['iou_gt'] = 0.0
        if 'iou' in statistics:
            stat['iou'] = 0.0
        if 'minD' in statistics:
            stat['minD'] = np.nan

        return stat

    system_quad = item['system_result_quad']
    true_quad = item['ground_truth_quad']
    template_size = item['template_size']
    image_size = item['size']

    if 'mean_iou' in statistics:
        mean_iou_stat = metrics.mean_intersection_over_union(system_quad, true_quad, image_size)
        stat['mean_iou'] = mean_iou_stat

    if 'iou_gt' in statistics:
        normed_iou_stat = metrics.intersection_over_union_with_ground_truth_normalization(system_quad,
                                                                                          true_quad,
                                                                                          template_size)
        stat['iou_gt'] = normed_iou_stat

    if 'iou' in statistics:
        iou_stat = metrics.intersection_over_union(system_quad, true_quad)
        stat['iou'] = iou_stat

    if 'minD' in statistics:
        residual_stat = metrics.residual_metric(system_quad, true_quad, template_size)
        stat['minD'] = residual_stat

    return stat


def add_to_stats(stats, one_item_stat):
    for key in one_item_stat.keys():
        if key in stats.keys():
            stats[key].append(one_item_stat[key])
        else:
            stats[key] = []
            stats[key].append(one_item_stat[key])


def process(results, runlist, stat_types):
    print('Calulating following statistic(s):')
    for stat in stat_types:
        print('   ' + stat)
    print('on subset of MIDV-500:')
    print(runlist)	

    results = read_json(results)
    runlist = open(runlist, 'r')
    current_images = set(runlist.read().splitlines())
    stats = {}
    for result in results:
        result_name = result['origin_image_path']
        result_name = process_name(result_name)
        if result_name in current_images:
            one_stat = process_item(result, stat_types)
            add_to_stats(stats, one_stat)
    return stats


def process_whole_report(results, stat_types):
    print('Calulating following statistic(s):')
    for stat in stat_types:
        print('   ' + stat)
    print('on full MIDV-500 dataset')    
    results = read_json(results)
    stats = {}
    for result in results:
        one_stat = process_item(result, stat_types)
        add_to_stats(stats, one_stat)
    return stats


def stat_output(stats, stat_types):
    length, avg, thr_num = 0, 0.0, 0.0
    if 'mean_iou' in stat_types:
        length = len(stats['mean_iou'])
        avg = np.mean(np.array(stats['mean_iou']))
        for st in stats['mean_iou']:
            if st > THRESHOLD - 1e-10:
                thr_num += 1
        thr_num /= length

        print('\n')
        print('Metric: Mean intersection over union of foreground and background')
        print('   Number of images                : ', length)
        print('   Average value                   : ', avg)
        print('   Percentage of correct (*) images: ', thr_num * 100, '%')
        print("* A result is considered to be correct if its score is more than THRESHOLD=%.2f." % THRESHOLD,
              "To change THRESHOLD see help")
    length, avg, thr_num = 0, 0.0, 0.0
    if 'iou_gt' in stat_types:
        length = len(stats['iou_gt'])
        avg = np.mean(np.array(stats['iou_gt']))
        for st in stats['iou_gt']:
            if st > THRESHOLD - 1e-10:
                thr_num += 1
        thr_num /= length

        print('\n')
        print('Metric: Intersection over union in ground truth quad coordinate system')
        print('   Number of images                : ', length)
        print('   Average value                   : ', avg)
        print('   Percentage of correct (*) images: ', thr_num * 100, '%')
        print("* A result is considered to be correct if its score is more than THRESHOLD=%.2f." % THRESHOLD,
              "To change THRESHOLD see help")
    length, avg, thr_num = 0, 0.0, 0.0
    if 'iou' in stat_types:
        length = len(stats['iou'])
        avg = np.mean(stats['iou'])
        for st in stats['iou']:
            if st > THRESHOLD - 1e-10:
                thr_num += 1
        thr_num /= length

        print('\n')
        print('Metric: Intersection over union')
        print('   Number of images                : ', length)
        print('   Average value                   : ', avg)
        print('   Percentage of correct (*) images: ', thr_num * 100, '%')
        print("* A result is considered to be correct if its score is more than THRESHOLD=%.2f." % THRESHOLD,
              "To change THRESHOLD see help")
    length, avg, thr_num = 0, 0.0, 0.0
    if 'minD' in stat_types:
        length = len(stats['minD'])
        avg = np.nanmean(stats['minD'])

        for st in stats['minD']:
            if not np.isnan(st):
                if st < RESIDUALS_THRESHOLD + 1e-10:
                    thr_num += 1
        thr_num /= length

        print('\n')
        print('Metric: Minimum corner distance in result quad coordinate system')
        print('   Number of images                 : ', length)
        print('   Average value (*)                : ', avg)
        print('   Percentage of correct (**) images: ', thr_num * 100, '%')
        print("* Note: average value is calculated only for the images having a resulting quadrilateral\n"
              "** A result is considered to be correct if its deviation"
              " is less than RESIDUALS_THRESHOLD=%.3f." % RESIDUALS_THRESHOLD,
              "To change RESIDUALS_THRESHOLD see help\n")


def create_parser():
    parser = argparse.ArgumentParser(description='This script calculate accuracy on MIDV-500 dataset',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('report', type=str,
                        help='path to file containing output data in JSON format. '
                             'It should contain following fields for every image:\n'
                             '\'origin_image_path\'         - path to source image (either absolute or relative)\n'
                             '\'size\'                      - image size as an array of numbers [width, height]\n'
                             '\'template_size\'             - size of a template as an array of numbers '
                             '[width, height]\n'
                             '\'ground_truth_quad\'         - markup quadrilateral coordinates '
                             'as an array of 4 arrays of 2 doubles '
                             'e.g. [[0.0, 0.0], [100.0, 0.0], [100.0, 200.0], [0.0, 200.0]]\n'
                             '\'system_result_quad_exists\' - if there is a system quadrilateral output\n'
                             '\'system_result_quad\'        - output quadrilateral coordinates '
                             '(see \'ground_truth_quad\')\n'
                             '\n'
                             'e.g.:\n'
                             '{'
                             '"origin_image_path": "D:/images\\01_alb_id/CA\\CA01_01.tif",\n'
                             '"size": [ 1080, 1920],\n'
                             '"template_size": [ 856, 540],\n'
                             '"ground_truth_quad": [\n'
                             '   [ 97.0, 672.0],\n'
                             '   [ 904.0, 643.0],\n'
                             '   [ 931.0, 1142.0],\n'
                             '   [ 122.0, 1185.0]]\n'
                             '"system_result_quad_exists": true,\n'
                             '"system_result_quad": [\n'
                             '   [ 45.1, 674.3],\n'
                             '   [ 903.8, 643.4],\n'
                             '   [ 930.4, 1141.7],\n'
                             '   [ 116.2, 1185.0]]\n'
                             '}')

    parser.add_argument('--runlist', type=str,
                        help='path to a list of images to be checked. If None given, whole report file will be checked')
    parser.add_argument('--metrics', dest='metrics', nargs='+', type=str,
                        choices={'iou', 'mean_iou', 'iou_gt', 'minD'},
                        help='what metrics should be used to calculate statistics. Possible variants: \n'
                             '\'iou\'       - IoU (intersection over union) \n'
                             '\'mean_iou\'  - mean IoU of foreground and background \n'
                             '\'iou_gt\' - IoU in ground-truth quad coordinate system \n'
                             '\'minD\'      - minimum by all relabling quads from maximum corner distance in result quad coordinate system\n'
                             "Note: this value is calculated only for the images having a resulting quadrilateral")
    parser.add_argument('--hardChangeThreshold', dest='changed_thr_const', type=float,
                        help='changes threshold constant (default=0.9)')
    parser.add_argument('--hardChangeResidualsThreshold', dest='changed_residuals_const', type=float,
                        help='changes residuals Threshold constant (default=0.017)')
    return parser


if __name__ == "__main__":
    parse = create_parser()
    args = parse.parse_args()    
    if args.metrics is None:
    	args.metrics = ['iou_gt', 'minD']
        
    if args.changed_thr_const:
        THRESHOLD = args.changed_thr_const
    if args.changed_residuals_const:
        RESIDUALS_THRESHOLD = args.changed_residuals_const
    if args.runlist:
        runlst = args.runlist
        computed_stat = process(args.report, runlst, args.metrics)
        stat_output(computed_stat, args.metrics)
    else:
        computed_stat = process_whole_report(args.report, args.metrics)
        stat_output(computed_stat, args.metrics)
