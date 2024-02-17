import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import cv2
import glob

import os



def get_all_images(folder, ext):

    all_files = []
    #Iterate through all files in folder
    for file in os.listdir(folder):
        #Get the file extension
        _,  file_ext = os.path.splitext(file)

        #If file is of given extension, get it's full path and append to list
        if ext in file_ext:
            full_file_path = os.path.join(folder, file)
            all_files.append(full_file_path)

    #Get list of all files
    return all_files

def detect_img(yolo):
    filepath = 'mAP/input/test/'
    save_path = 'mAP/input/output_images/'
    save_file = 'mAP/input/detection-results/'
    jpg_files = get_all_images(filepath, 'jpg')
    for i in range(0,len(jpg_files)):
        img = jpg_files[i]
        name = img[15:]
        image = Image.open(img)
        r_image,annotations = yolo.detect_image(image)
        # r_image.show()
        r_image.save(save_path+name)
        end = name.rfind(".")
        name = name[:end]
        print(name)
        filename = save_file + name + ".txt"
        print(filename)
        f = open(filename,"w+")
        for j in range(0, len(annotations)):
            data = annotations[j]
            class_name = data[0].split(" ")[0]
            confidence = data[0].split(" ")[1]
            left = data[1]
            top = data[2]
            right = data[3]
            bottom = data[4]
            result = str(class_name)+" "+str(confidence)+" "+str(left)+" "+str(top)+" "+str(right)+" "+str(bottom)
            print(result)
            f.write(result+"\n")

    yolo.close_session()

FLAGS = None

# if __name__ == '__main__':
# class YOLO defines the default value, so suppress any default here
parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
'''
Command line options
'''
parser.add_argument(
    '--model', type=str, dest='model_path', default='model_data/yolo.h5',
    help='path to model weight file, default ' + YOLO.get_defaults("model_path")
)

parser.add_argument(
    '--anchors', type=str, dest='anchors_path',
    help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
)

parser.add_argument(
    '--classes', type=str, dest='classes_path', default= 'model_data/_classes.txt',
    help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
)

parser.add_argument(
    '--gpu_num', type=int,
    help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
)

parser.add_argument(
    '--image', default=False, action="store_true",
    help='Image detection mode, will ignore all positional arguments'
)
'''
Command line positional arguments -- for video detection mode
'''
parser.add_argument(
    "--input", nargs='?', type=str,required=False,default='./path2your_video',
    help = "Video input path"
)

parser.add_argument(
    "--output", nargs='?', type=str, default="",
    help = "[Optional] Video output path"
)

def predict_video(in_file,out_file):
    detect_video(YOLO(**vars(FLAGS)), in_file, out_file)

FLAGS = parser.parse_args()

if FLAGS.image:
    """
    Image detection mode, disregard any remaining command line arguments
    """
    print("Image detection mode")
    if "input" in FLAGS:
        print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
    detect_img(YOLO(**vars(FLAGS)))
elif "input" in FLAGS:
    detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
else:
    print("Must specify at least video_input_path.  See usage with --help.")
