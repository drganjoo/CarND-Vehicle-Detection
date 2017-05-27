import csv
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

data_folder = 'C:\\Users\\fahad\\Downloads\\object-detection-crowdai.tar\\object-detection-crowdai'
output_folder = 'D:\\carnd\\CarND-Vehicle-Detection\\data\\crowd_ai'

with open('{}/labels.csv'.format(data_folder), 'r') as f:
    reader = csv.reader(f)
    next(reader, None)
    
    index = 1
    for row in reader:
        output_file = '{}/{}-{}.jpeg'.format(output_folder, row[5], index)
        #print(row)
        filename = row[4]
        
        img = mpimg.imread('{}/{}'.format(data_folder, filename))
        
        x = [int(row[0]), int(row[2])]
        y = [int(row[1]), int(row[3])]
        
        if x[0] < x[1]:
            x1, x2 = x[0], x[1]
        else:
            x2, x1 = x[0], x[1]
            
        if y[0] < y[1]:
            y1, y2 = y[0], y[1]
        else:
            y2, y1 = y[0], y[1]
        
        if y2-y1 > 25 and x2-x1 > 25:
            img_car = img[y1:y2,x1:x2]
            img_car_64 = cv2.resize(img_car, (64,64))

            img_car_64bgr = cv2.cvtColor(img_car_64, cv2.COLOR_RGB2BGR)
            print(output_file)
            cv2.imwrite(output_file, img_car_64bgr)
        else:
            print('Discarding...small size')

        index += 1
        
        #cv2.rectangle(img, (x1,y1), (x2,y2), (255, 0, 0), 6)
        
        #f, (ax1, ax2) = plt.subplots(1,2,figsize=(10,5))
        #ax1.imshow(img)
        #ax1.set_title('Row #: {}'.format(index))
        #ax2.imshow(img_car_64)
        #ax2.set_title('{},{},{},{}'.format(x1,y1, x2, y2))
