""" Deskews file after getting skew angle """
"""
This code is based on the following file:
https://github.com/kakul/Alyn/blob/master/alyn/deskew.py
"""
import optparse
import numpy as np
import os

from alyn3.skew_detect import SkewDetect
import cv2


class Deskew:

    def __init__(self, input_file, output_file, r_angle=0,
                 skew_max=4.0, acc_deg=0.1, method=1,
                 roi_w=1.0, roi_h=1.0,
                 gray=1.0, quality=100, short=None):
        self.input_file = input_file
        self.output_file = output_file
        self.r_angle = r_angle
        self.method = method
        self.gray = gray
        self.quality = quality
        self.short = short
        self.skew_obj = SkewDetect(self.input_file,
                                   skew_max=skew_max, acc_deg=acc_deg,
                                   roi_w=roi_w, roi_h=roi_h)

    def deskew(self):
        print('input: '+self.input_file)

        res = self.skew_obj.process_single_file()
        angle = res['Estimated Angle']
        rot_angle = angle + self.r_angle

        img = cv2.imread(self.input_file, cv2.IMREAD_COLOR)
        g = self.gray * 255
        rotated = self.rotate_expand(img, rot_angle, g)

        if self.short:
            h = rotated.shape[0]
            w = rotated.shape[1]
            print('origin w,h: {}, {}'.format(w, h))
            if w < h:
                h = int(h*self.short/w+0.5)
                w = self.short
            else:
                w = int(w*self.short/h+0.5)
                h = self.short
            print('resized w,h: {}, {}'.format(w, h))
            rotated = cv2.resize(rotated, (w, h))

        if self.output_file:
            self.save_image(rotated)

        return res

    def deskew_on_memory(self, input_data):
        res = self.skew_obj.determine_skew_on_memory(input_data)
        angle = res['Estimated Angle']
        rot_angle = angle + self.r_angle

        img = input_data
        g = self.gray * 255
        rotated = self.rotate_expand(img, rot_angle, g)

        if self.short:
            h = rotated.shape[0]
            w = rotated.shape[1]
            print('origin w,h: {}, {}'.format(w, h))
            if w < h:
                h = int(h*self.short/w+0.5)
                w = self.short
            else:
                w = int(w*self.short/h+0.5)
                h = self.short
            print('resized w,h: {}, {}'.format(w, h))
            rotated = cv2.resize(rotated, (w, h))

        return rotated

    def save_image(self, img):
        path = self.skew_obj.check_path(self.output_file)
        if os.path.splitext(path)[1] in ['.jpg', '.JPG', '.jpeg', '.JPEG']:
            cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        else:
            cv2.imwrite(path, img)

    def rotate_expand(self, img, angle=0, g=255):
        h = img.shape[0]
        w = img.shape[1]
        angle_rad = angle/180.0*np.pi
        w_rot = int(np.round(h*np.absolute(np.sin(angle_rad)) +
                    w*np.absolute(np.cos(angle_rad))))
        h_rot = int(np.round(h*np.absolute(np.cos(angle_rad)) +
                    w*np.absolute(np.sin(angle_rad))))
        size_rot = (w_rot, h_rot)
        mat = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        mat[0][2] = mat[0][2] - w/2 + w_rot/2
        mat[1][2] = mat[1][2] - h/2 + h_rot/2
        rotated = cv2.warpAffine(img, mat, size_rot, borderValue=(g, g, g))

        return rotated

    def run(self):
        if self.input_file:
            return self.deskew()


def optparse_args():
    parser = optparse.OptionParser()

    parser.add_option(
        '-i',
        '--input',
        default=None,
        dest='input_file',
        help='Input file name')
    parser.add_option(
        '-o', '--output',
        default=None,
        dest='output_file',
        help='Output file name')
    parser.add_option(
        '-r', '--rotate',
        default=0,
        dest='r_angle',
        help='Rotate the image to desired axis',
        type=int)
    parser.add_option(
        '-g', '--gray',
        default=1.0,
        dest='gray',
        help='Gray level outside the input image boundaries.\n'
             'between 0.0(black) and 1.0(white)\n'
             '[0.0, 1.0], default: 1.0',
        type=float)
    parser.add_option(
        '-q', '--quality',
        default=100,
        dest='quality',
        help='output jpeg image quality. i\n'
             '1 is worst quality and smallest file size,\n'
             'and 100 is best quality and largest file size.\n'
             '[1, 100], default: 100',
        type=int)

    return parser.parse_args()


if __name__ == '__main__':
    options, args = optparse_args()
    deskew_obj = Deskew(
        options.input_file,
        options.display_image,
        options.output_file,
        options.r_angle,
        options.gray,
        options.quality)

    deskew_obj.run()
