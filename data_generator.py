import numpy as np
import matplotlib.pyplot as plt

from PIL import Image,ImageDraw

import os

import struct
from struct import unpack

import xml.etree.cElementTree as et
import xml.dom.minidom

imgSize = 302

def unpack_drawing(file_handle):
    key_id, = unpack('Q', file_handle.read(8))
    countrycode, = unpack('2s', file_handle.read(2))
    recognized, = unpack('b', file_handle.read(1))
    timestamp, = unpack('I', file_handle.read(4))
    n_strokes, = unpack('H', file_handle.read(2))
    image = []
    for i in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x = unpack(fmt, file_handle.read(n_points))
        y = unpack(fmt, file_handle.read(n_points))
        image.append((x, y))

    return {
        'key_id': key_id,
        'countrycode': countrycode,
        'recognized': recognized,
        'timestamp': timestamp,
        'image': image
    }


def unpack_drawings(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                yield unpack_drawing(f)
            except struct.error:
                break

def DrawImage(xL, yL, im):
    draw = ImageDraw.Draw(im)
    size = xL.size
    for i in range(0, size-1):
        draw.line((xL[i], yL[i], xL[i+1], yL[i+1]), fill = 0, width = 2)

def generateImage(sketch, fname, mapFile, regFile, objectclass, mean):
    im = Image.new('RGB', (302,302), (255,255,255)) #The picture size is 300 * 300, use 302 to create padding area
    for parray in sketch['image']:
        xL = np.asarray(parray[0], dtype = np.int16)
        yL = np.asarray(parray[1], dtype = np.int16)
        DrawImage(xL, yL, im)
    fname = 'D:\quickDraw_Recognition\\' + fname
    channelMean = np.mean(im, axis = (0,1))

    # tempMean = np.zeros((3,302,302))
    # print('processing image')
    # for x in range(im.size[0]):
    #     for y in range(im.size[1]):
    #         r,g,b = im.getpixel((x,y))
    #         tempMean[0][y][x] = r
    #         tempMean[1][y][x] = g
    #         tempMean[2][y][x] = b
    # mean += tempMean
    # print('processing done')

    mean += np.asarray(im, dtype = np.int16)
    im.save(fname,'PNG')

    mapFile.write("%s\t%d\n" % (fname, objectclass))
    regFile.write("|regrLabels\t%f\t%f\t%f\n" % (channelMean[0]/255.0, channelMean[1]/255.0, channelMean[2]/255.0))

#Save Image mean to a xmlfile
def saveMean(fname, data):
    root = et.Element('opencv_storage')
    et.SubElement(root, 'Channel').text = '3'
    et.SubElement(root, 'Row').text = str(imgSize)
    et.SubElement(root, 'Col').text = str(imgSize)
    meanImg = et.SubElement(root, 'MeanImg', type_id='opencv-matrix')
    et.SubElement(meanImg, 'rows').text = '1'
    et.SubElement(meanImg, 'cols').text = str(imgSize * imgSize * 3)
    et.SubElement(meanImg, 'dt').text = 'f'
    et.SubElement(meanImg, 'data').text = ' '.join(['%e' % n for n in np.reshape(data, (imgSize * imgSize * 3))])

    tree = et.ElementTree(root)
    tree.write(fname)
    x = xml.dom.minidom.parse(fname)
    with open(fname, 'w') as f:
        f.write(x.toprettyxml(indent = '  '))


# Select 10 kinds of data...
binaray_files_names = ['ant.bin', 'alarm_clock.bin', 'ambulance.bin', 'angel.bin', 'anvil.bin', 'apple.bin', 'arm.bin', 'backpack.bin', 'basketball.bin', 'bed.bin']
# Unpack them...
print('Unpack binary files into coordinates...')
sketch_data = list()
for x in range(10):
    print('unpack#%d class' % x)
    sketches = unpack_drawings(binaray_files_names[x])
    sketch_of_a_class = list()
    for sketch in sketches:
        sketch_of_a_class.append(sketch)
    sketch_data.append(sketch_of_a_class)
print('Unpack Done')

dataMean = np.zeros((3, imgSize, imgSize))
calculate = np.zeros((imgSize, imgSize, 3))


data_size = 100000 #generate 100000 images

num_in_class = np.zeros(10, dtype = np.int16)
with open('train_map.txt','w') as mapFile:
    with open('regfile.txt','w') as regFile:
        n = 0
        for x in range(data_size):
            fname = ('%06d.png' % x)
            object_class = np.random.randint(low = 0, high = 10)
            index = num_in_class[object_class]
            generateImage(sketch_data[object_class][index], fname, mapFile, regFile, object_class, calculate)
            num_in_class[object_class] = num_in_class[object_class] + 1
            n = n + 1
            if n % 100 == 0:
                print('produced 100 images')

for x in range(imgSize):
    for y in range(imgSize):
        dataMean[0][y][x] = calculate[x][y][0]
        dataMean[1][y][x] = calculate[x][y][1]
        dataMean[2][y][x] = calculate[x][y][2]

dataMean = dataMean/data_size
saveMean('image_mean.xml', dataMean)