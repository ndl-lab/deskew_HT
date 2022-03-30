# !/usr/bin/env python3

# Copyright (c) 2022, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/

import os
import cv2
import argparse
import numpy as np
from alyn3.deskew import Deskew
from alyn3.skew_detect import SkewDetect

os.environ["OPENCV_IO_ENABLE_JASPER"] = "true"


def deskew_image(input, output, r_angle=0,
                 skew_max=4.0, acc_deg=0.5, roi_w=1.0, roi_h=1.0,
                 method=1, gray=1.0, quality=100, short=None,
                 log=None):

    image_name = os.path.basename(input)
    print('process: '+image_name)
    d = Deskew(input, output,
               r_angle=r_angle,
               skew_max=skew_max,
               acc_deg=acc_deg,
               method=method,
               gray=gray,
               quality=quality,
               short=short,
               roi_w=roi_w,
               roi_h=roi_h)
    res = d.run()

    if log:
        with open(log, mode='a') as f:
            line = '{}\t{:.6f}\n'.format(
                res['Image File'], (-res['Estimated Angle']))
            f.write(line)


def deskew_dir(input_dir_path, output_dir_path, r_angle=0,
               skew_max=4.0, acc_deg=0.5, roi_w=1.0, roi_h=1.0,
               method=1, gray=1.0, quality=100, short=None,
               log=None):
    image_list = os.listdir(input_dir_path)

    for image_name in image_list:
        input_path = os.path.join(input_dir_path, image_name)
        if(os.path.isdir(input_path)):
            continue
        print('process: '+str(image_name))
        output_path = os.path.join(output_dir_path, image_name)
        d = Deskew(input_path, output_path,
                   r_angle=r_angle,
                   skew_max=skew_max,
                   acc_deg=acc_deg,
                   method=method,
                   gray=gray,
                   quality=quality,
                   short=short,
                   roi_w=roi_w,
                   roi_h=roi_h)
        res = d.run()

        if log:
            with open(log, mode='a') as f:
                line = '{}\t{:.6f}\n'.format(
                    res['Image File'], (-res['Estimated Angle']))
                f.write(line)


def add_detected_lines(input_path, output_path,
                       skew_max=4.0, acc_deg=0.5,
                       roi_w=1.0, roi_h=1.0,
                       bgr=[0, 0, 255]):
    line_len = 4000

    print("Add the detected lines to "+os.path.basename(input_path))
    sd = SkewDetect(input_path, skew_max=skew_max, acc_deg=acc_deg,
                    roi_w=roi_w, roi_h=roi_h)
    acc, ang_rad, distance = sd.determine_line(input_path)
    img = cv2.imread(input_path)
    if len(acc) == 0:
        print('Image file:{} has no lines detected'.format(input_path))
    else:
        max_val = max(acc)
        for val, theta, rho in zip(acc[::-1], ang_rad[::-1], distance[::-1]):
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho + int(img.shape[1] * (0.5-roi_w/2.0))
            y0 = b * rho + int(img.shape[0] * (0.5-roi_h/2.0))
            x1 = int(x0 + line_len*(-b))
            y1 = int(y0 + line_len*(a))
            x2 = int(x0 - line_len*(-b))
            y2 = int(y0 - line_len*(a))
            tmp_bgr = bgr.copy()
            tmp_bgr[0] = 255.0 * (1.0 - val / max_val)
            tmp_bgr[1] = tmp_bgr[0]
            # print(tmp_bgr)
            cv2.line(img, (x1, y1), (x2, y2), tmp_bgr, 2)

    cv2.imwrite(output_path, img)


def add_detected_lines_dir(input_dir_path, output_dir_path,
                           skew_max=4.0, acc_deg=0.1,
                           roi_w=1.0, roi_h=1.0,
                           bgr=[0, 0, 255]):
    # Hough変換で検知したLineを元画像に書き加える
    # Add the lines detected by Hough Transform to the input images
    image_list = os.listdir(input_dir_path)

    for image_name in image_list:
        input_path = os.path.join(input_dir_path, image_name)
        if(os.path.isdir(input_path)):
            continue
        output_path = os.path.join(output_dir_path, image_name)
        add_detected_lines(input_path, output_path,
                           skew_max=skew_max, acc_deg=acc_deg,
                           roi_w=roi_w, roi_h=roi_h,
                           bgr=[0, 0, 255])


def parse_args():
    usage = 'python3 {} INPUT [-o OUTPUT] [-s SKEW_MAX] [-a ANGLE_ACC] [-m METHOD]'.format(
        __file__)
    argparser = argparse.ArgumentParser(
        usage=usage,
        description='Deskew image(when INPUT is an image) or images in INPUT(when INPUT is a directory).',
        formatter_class=argparse.RawTextHelpFormatter)
    argparser.add_argument(
        'input',
        help='input image file or directory path',
        type=str)
    argparser.add_argument(
        '-o',
        '--out',
        default='out.jpg',
        help='output file or directory path',
        type=str)
    argparser.add_argument(
        '-l',
        '--log',
        default=None,
        help='estimated skew log file path\n'
             'output format:\n'
             'Image_file_path <tab> Estimated_skew_angle[deg]')
    argparser.add_argument(
        '-s',
        '--skew_max',
        default=4.0,
        help='maximum expected skew angle[deg], default: 4.0',
        type=float)
    argparser.add_argument(
        '-a',
        '--angle_acc',
        default=0.5,
        help='estimated skew angle accuracy[deg], default: 0.5',
        type=float)
    argparser.add_argument(
        '-rw',
        '--roi_width',
        default=1.0,
        help='horizontal cropping ratio of the region of interest \n'
             'to the whole image. (0.0, 1.0] default: 1.0(whole image)',
        type=float)
    argparser.add_argument(
        '-rh',
        '--roi_height',
        default=1.0,
        help='vertical cropping ratio of the region of interest \n'
             'to the whole image. (0.0, 1.0] default: 1.0(whole image)',
        type=float)
    argparser.add_argument(
        '-m',
        '--method',
        default=1,
        help='interpolation method.\n'
             '0: Nearest-neighbor  1: Bi-linear(default)\n'
             '2: Bi-quadratic      3: Bi-cubic\n'
             '4: Bi-quartic        5: Bi-quintic\n',
        type=int)
    argparser.add_argument(
        '-g',
        '--gray',
        default=1.0,
        dest='gray',
        help='gray value outside the input image boundaries.\n'
             '[0.0(black), 1.0(white)], default: 1.0',
        type=float)
    argparser.add_argument(
        '-q', '--quality',
        default=100,
        dest='quality',
        help='output jpeg image quality.\n'
             '1 is worst quality and smallest file size,\n'
             'and 100 is best quality and largest file size.\n'
             '[1, 100], default: 100',
        type=int)
    argparser.add_argument(
        '--short',
        default=None,
        dest='short',
        help='the length of the short side of the output image.',
        type=int)
    argparser.add_argument(
        '-v',
        '--version',
        version='deskew version 1.0.0',
        action='version')
    argparser.add_argument(
        '--debug',
        action='store_true')

    return argparser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    input = args.input
    output = args.out
    print('input directory/image: '+input)

    if(os.path.isdir(input)):  # directory
        if output[-4:] == '.jpg':
            output = output[:-4]  # 'out'
        print('output: '+output)
        os.makedirs(output, exist_ok=True)
        deskew_dir(input, output,
                   r_angle=0,
                   skew_max=args.skew_max,
                   acc_deg=args.angle_acc,
                   roi_w=args.roi_width,
                   roi_h=args.roi_height,
                   method=args.method,
                   gray=args.gray,
                   quality=args.quality,
                   short=args.short,
                   log=args.log)
        if args.debug:
            print('[Debug] Dump input images with detected lines')
            os.makedirs(output+'_withL', exist_ok=True)
            add_detected_lines_dir(input, output+'_withL',
                                   roi_w=args.roi_width,
                                   roi_h=args.roi_height,
                                   skew_max=args.skew_max,
                                   acc_deg=args.angle_acc)
    else:  # single image
        print('output: '+output)
        deskew_image(input, output,
                     r_angle=0,
                     skew_max=args.skew_max,
                     acc_deg=args.angle_acc,
                     roi_w=args.roi_width,
                     roi_h=args.roi_height,
                     method=args.method,
                     gray=args.gray,
                     quality=args.quality,
                     short=args.short,
                     log=args.log)
        if args.debug:
            print('[Debug] Dump input image with detected lines')
            add_detected_lines(input, output+'_withL.jpg',
                               roi_w=args.roi_width,
                               roi_h=args.roi_height,
                               skew_max=args.skew_max,
                               acc_deg=args.angle_acc)
