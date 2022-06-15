# Results of experiments for Advanced Hough-based method for on-device document localization

This repository contains document localization results of the system proposed in [[this article](http://www.computeroptics.ru/KO/PDF/KO45-5/450509.pdf)], metrics for accuracy measuremrents and runlists of [[MIDV-500 dataset](https://doi.org/10.18287/2412-6179-2019-43-5-818-824)].

To calculate the statistics use run_stat.py. For our experiments the code was executed used Python 3.9.2 and following python modules: numpy 1.19.5, opencv-python 4.5.1.48, Polygon3 3.0.8.

Description of each source code file and data directory follows.

## Code overview

1\. `metrics.py` - python module containing implementations for basic functions required for calculating statistics in the experiments;

2\. `run_stat.py` - python script for statistics calculation.


## Overview of data directories

1\. `data_midv500/` - directory contains file `report.json` with data for every image in the following structure: 
```
{
//path to source image (either absolute or relative)
"origin_image_path": "./images/01_alb_id/CA/CA01_01.tif",
// image size as an array of numbers [width, height]
"size": [1080, 1920],
//size of a template as an array of numbers
"template_size": [ 856, 540],
//ground-truth quadrilateral coordinates of the document as an array of 4 arrays of 2 doubles
"ground_truth_quad": [
   [ 97.0, 672.0],
   [ 904.0, 643.0],
   [ 931.0, 1142.0],
   [ 122.0, 1185.0]]
//if there is a system quadrilateral outpu
"system_result_quad_exists": true,
//resulting quadrilateral coordinates of the sytem proposed in [this article]
"system_result_quad": [
   [ 45.1, 674.3],
   [ 903.8, 643.4],
   [ 930.4, 1141.7],
   [ 116.2, 1185.0]]
}
```


2\. `runlists_midv500/` - directory with runlists contain image identificators subsets of the MIDV-500 dataset: 

- `doc_area_in_frame_more_than_ninety_percent.lst` is a list of images having the percentage of document area inside the frame 90 or higher

- `four_vertices_in_frame_closed.lst` is a list of images having 4 vertices lying inside the frame rectangle engorged by a small value eps in every direction (that is inside of a convex hull of the following points: (-eps, -eps), (image width + eps, -eps), (image width + eps, image height + eps), (-eps, image height + eps))

- `at_least_three_vertices_in_frame_closed.lst` is a list of images having at least 3 vertices lying inside the frame rectangle engorged by a small value eps in every direction (that is inside of a convex hull of the following points: (-eps, -eps), (image width + eps, -eps), (image width + eps, image height + eps), (-eps, image height + eps))

- `at_least_three_vertices_in_frame_half_closed.lst` is a list of images having at least 3 vertices lying inside the frame rectangle engorged by a small value eps in top and left directions (that is inside of a convex hull of the following points: (-eps, -eps), (image width, -eps), (image width, image height), (-eps, image height))

## Citing work 
If you end up using our code or results in your research, please consider citing:
``` bash
@article{tropin2021advanced,
  title={Advanced Hough-based method for on-device document localization},
  author={Tropin, DV and Ershov, AM and Nikolaev, DP and Arlazarov, VV},
  journal={Computer Optics},
  volume={45},
  number={5},
  pages={702--712},
  year={2021}, 
  doi={10.18287/2412-6179-CO-895}
}
```



