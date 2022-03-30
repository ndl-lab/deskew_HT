""" Calculates skew angle """
"""
This code is based on the following file:
https://github.com/kakul/Alyn/blob/master/alyn/skew_detect.py
"""
import os
import optparse

import numpy as np
# import matplotlib.pyplot as plt
from skimage import io
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
import cv2


class SkewDetect:

    piby4 = np.pi / 4

    def __init__(
        self,
        input_file=None,
        output_file=None,
        sigma=0.50,
        display_output=None,
        num_peaks=20,
        skew_max=4.0,
        acc_deg=0.5,
        roi_w=1.0,
        roi_h=1.0,
    ):

        self.sigma = sigma
        self.input_file = input_file
        self.output_file = output_file
        self.display_output = display_output
        self.num_peaks = num_peaks
        self.skew_max = skew_max
        self.acc_deg = acc_deg
        self.roi_w = roi_w
        self.roi_h = roi_h

    def write_to_file(self, wfile, data):

        for d in data:
            wfile.write(d + ': ' + str(data[d]) + '\n')
        wfile.write('\n')

    def get_max_freq_elem(self, arr):

        max_arr = []
        freqs = {}
        for i in arr:
            if i in freqs:
                freqs[i] += 1
            else:
                freqs[i] = 1

        sorted_keys = sorted(freqs, key=freqs.get, reverse=True)
        max_freq = freqs[sorted_keys[0]]

        for k in sorted_keys:
            if freqs[k] == max_freq:
                max_arr.append(k)

        return max_arr

    def compare_sum(self, value):
        if value >= 44 and value <= 46:
            return True
        else:
            return False

    def display(self, data):

        for i in data:
            print(str(i) + ": " + str(data[i]))

    def calculate_deviation(self, angle):

        angle_in_degrees = np.abs(angle)
        deviation = np.abs(SkewDetect.piby4 - angle_in_degrees)

        return deviation

    def run(self):

        if self.display_output:
            if self.display_output.lower() == 'yes':
                self.display_output = True
            else:
                self.display_output = False

        if self.input_file is None:
            print("Invalid input, nothing to process.")
        else:
            self.process_single_file()

    def check_path(self, path):

        if os.path.isabs(path):
            full_path = path
        else:
            full_path = os.getcwd() + '/' + str(path)
        return full_path

    def process_single_file(self):

        file_path = self.check_path(self.input_file)
        res = self.determine_skew(file_path)

        if self.output_file:
            output_path = self.check_path(self.output_file)
            wfile = open(output_path, 'w')
            self.write_to_file(wfile, res)
            wfile.close()

        return res

    def determine_skew(self, img_file):

        img_ori = io.imread(img_file, as_gray=True)
        height, width = img_ori.shape
        img = img_ori[int(height*(0.5-self.roi_h/2.0)):int(height*(0.5+self.roi_h/2.0)),
                      int(width * (0.5-self.roi_w/2.0)):int(width * (0.5+self.roi_w/2.0))]

        img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))

        edges = canny(img, sigma=self.sigma)
        range_rad = np.arange(-np.pi/2, -np.pi/2+np.deg2rad(self.skew_max),
                              step=np.deg2rad(self.acc_deg))
        range_rad = np.concatenate(
            [range_rad,
             np.arange(-np.deg2rad(self.skew_max), np.deg2rad(self.skew_max),
                       step=np.deg2rad(self.acc_deg))],
            axis=0)
        range_rad = np.concatenate(
            [range_rad,
             np.arange(np.pi/2-np.deg2rad(self.skew_max), np.pi/2,
                       step=np.deg2rad(self.acc_deg))],
            axis=0)

        h, a, d = hough_line(edges, theta=range_rad)

        th = 0.2 * h.max()
        _, ap, _ = hough_line_peaks(
            h, a, d, threshold=th, num_peaks=self.num_peaks)

        if len(ap) == 0:
            data = {
                "Image File": img_file,
                "Average Deviation from pi/4": 0.0,
                "Estimated Angle": 0.0,
                "Angle bins": [[], [], [], []],
                "Message": "Bad Quality"}
            return data

        absolute_deviations = [self.calculate_deviation(k) for k in ap]
        average_deviation = np.mean(np.rad2deg(absolute_deviations))
        ap_deg = [np.rad2deg(x) for x in ap]

        for i in range(len(ap_deg)):
            if ap_deg[i] >= 45.0:
                ap_deg[i] -= 90.0
            elif ap_deg[i] <= -45.0:
                ap_deg[i] += 90.0

        bin_0_45 = []
        bin_45_90 = []
        bin_0_45n = []
        bin_45_90n = []

        for ang in ap_deg:

            deviation_sum = (90 - ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_45_90.append(ang)
                continue

            deviation_sum = (ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_0_45.append(ang)
                continue

            deviation_sum = (-ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_0_45n.append(ang)
                continue

            deviation_sum = (90 + ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_45_90n.append(ang)

        angles = [bin_0_45, bin_45_90, bin_0_45n, bin_45_90n]
        lmax = 0

        for j in range(len(angles)):
            tmp_l = len(angles[j])
            if tmp_l > lmax:
                lmax = tmp_l
                maxi = j

        if lmax:
            ans_arr = self.get_max_freq_elem(angles[maxi])  # 最多頻度の角度array
            ans_res = np.mean(ans_arr)    # 同数最多が複数あるかもしれないのでavg

        else:  # angls が空のとき
            ans_arr = self.get_max_freq_elem(ap_deg)
            ans_res = np.mean(ans_arr)

        data = {
            "Image File": img_file,
            "Average Deviation from pi/4": average_deviation,
            "Estimated Angle": ans_res,
            "Angle bins": angles,
            "Message": "Successfully detected lines"}

        if self.display_output:
            self.display(data)

        return data

    def determine_skew_on_memory(self, img_data):

        img_ori = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        height, width = img_ori.shape
        img = img_ori[int(height*(0.5-self.roi_h/2.0)):int(height*(0.5+self.roi_h/2.0)),
                      int(width * (0.5-self.roi_w/2.0)):int(width * (0.5+self.roi_w/2.0))]

        img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))

        edges = canny(img, sigma=self.sigma)
        range_rad = np.arange(-np.pi/2, -np.pi/2+np.deg2rad(self.skew_max),
                              step=np.deg2rad(self.acc_deg))
        range_rad = np.concatenate([range_rad,
                                    np.arange(-np.deg2rad(self.skew_max),
                                              np.deg2rad(self.skew_max),
                                              step=np.deg2rad(self.acc_deg))],
                                   axis=0)
        range_rad = np.concatenate([range_rad,
                                    np.arange(np.pi/2-np.deg2rad(self.skew_max),
                                              np.pi/2,
                                              step=np.deg2rad(self.acc_deg))],
                                   axis=0)

        h, a, d = hough_line(edges, theta=range_rad)

        th = 0.2 * h.max()
        _, ap, _ = hough_line_peaks(
            h, a, d, threshold=th, num_peaks=self.num_peaks)

        if len(ap) == 0:
            data = {
                "Average Deviation from pi/4": 0.0,
                "Estimated Angle": 0.0,
                "Angle bins": [[], [], [], []],
                "Message": "Bad Quality"}
            return data

        absolute_deviations = [self.calculate_deviation(k) for k in ap]
        average_deviation = np.mean(np.rad2deg(absolute_deviations))
        ap_deg = [np.rad2deg(x) for x in ap]

        for i in range(len(ap_deg)):
            if ap_deg[i] >= 45.0:
                ap_deg[i] -= 90.0
            elif ap_deg[i] <= -45.0:
                ap_deg[i] += 90.0

        bin_0_45 = []
        bin_45_90 = []
        bin_0_45n = []
        bin_45_90n = []

        for ang in ap_deg:

            deviation_sum = (90 - ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_45_90.append(ang)
                continue

            deviation_sum = (ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_0_45.append(ang)
                continue

            deviation_sum = (-ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_0_45n.append(ang)
                continue

            deviation_sum = (90 + ang + average_deviation)
            if self.compare_sum(deviation_sum):
                bin_45_90n.append(ang)

        angles = [bin_0_45, bin_45_90, bin_0_45n, bin_45_90n]
        lmax = 0

        for j in range(len(angles)):
            tmp_l = len(angles[j])
            if tmp_l > lmax:
                lmax = tmp_l
                maxi = j

        if lmax:
            ans_arr = self.get_max_freq_elem(angles[maxi])  # 最多頻度の角度array
            ans_res = np.mean(ans_arr)    # 同数最多が複数あるかもしれないのでavg

        else:  # angls が空のとき
            ans_arr = self.get_max_freq_elem(ap_deg)
            ans_res = np.mean(ans_arr)

        data = {
            "Average Deviation from pi/4": average_deviation,
            "Estimated Angle": ans_res,
            "Angle bins": angles,
            "Message": "Successfully detected lines"}

        return data

    def determine_line(self, img_file):

        img_ori = io.imread(img_file, as_gray=True)
        height, width = img_ori.shape
        img = img_ori[int(height*(0.5-self.roi_h/2.0)):int(height*(0.5+self.roi_h/2.0)),
                      int(width * (0.5-self.roi_w/2.0)):int(width * (0.5+self.roi_w/2.0))]
        edges = canny(img, sigma=self.sigma)
        range_rad = np.arange(-np.pi/2, -np.pi/2+np.deg2rad(self.skew_max),
                              step=np.deg2rad(self.acc_deg))
        range_rad = np.concatenate([range_rad,
                                    np.arange(-np.deg2rad(self.skew_max),
                                              np.deg2rad(self.skew_max),
                                              step=np.deg2rad(self.acc_deg))],
                                   axis=0)
        range_rad = np.concatenate([range_rad,
                                    np.arange(np.pi/2-np.deg2rad(self.skew_max), np.pi/2,
                                              step=np.deg2rad(self.acc_deg))],
                                   axis=0)

        h, a, d = hough_line(edges, theta=range_rad)

        th = 0.2 * h.max()
        ac, ap, d = hough_line_peaks(
            h, a, d, threshold=th, num_peaks=self.num_peaks)

        return ac, ap, d


if __name__ == '__main__':

    parser = optparse.OptionParser()

    parser.add_option(
        '-d', '--display',
        default=None,
        dest='display_output',
        help='Display logs')
    parser.add_option(
        '-i', '--input',
        default=None,
        dest='input_file',
        help='Input file name')
    parser.add_option(
        '-o', '--output',
        default=None,
        dest='output_file',
        help='Output file name')
    parser.add_option(
        '-p', '--plot',
        default=None,
        dest='plot_hough',
        help='Plot the Hough Transform')
    parser.add_option(
        '-s', '--sigma',
        default=3.0,
        dest='sigma',
        help='Sigma for Canny Edge Detection',
        type=float)
    options, args = parser.parse_args()
    skew_obj = SkewDetect(
        options.input_file,
        options.output_file,
        options.sigma,
        options.display_output,
        options.num_peaks,
        options.plot_hough)
    skew_obj.run()
