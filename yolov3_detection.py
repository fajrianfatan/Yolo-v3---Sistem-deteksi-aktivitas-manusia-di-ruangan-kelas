qqqqqq# Penggunaan
# python test4.py --input videos/contoh.mp4 --output output/contoh_output.avi --yolo yolo-coco
import cv2
import numpy as np
from shapely.geometry import Polygon
import time
import imutils
#import argparse

#Argument Parser untuk ekspor video menggunakan terminal
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--input", required=True,
#	help="path to input video")
#ap.add_argument("-o", "--output", required=True,
#	help="path to output video")
#ap.add_argument("-y", "--yolo", required=False,
#	help="base path to YOLO directory")
#ap.add_argument("-c", "--confidence", type=float, default=0.5,
#	help="minimum probability to filter weak detections")
#ap.add_argument("-t", "--threshold", type=float, default=0.3,
#	help="threshold when applyong non-maxima suppression")
#args = vars(ap.parse_args())

def check_intersection(polygon1, polygon2):
    return polygon1.intersects(polygon2)

net = cv2.dnn.readNetFromDarknet(r"C:\Users\User\PycharmProjects\yolotest\yolov3_custom.cfg", r"C:\Users\User\PycharmProjects\yolotest\yolov3_custom_4000.weights")

classes = ['Manusia','Proyektor']

#with open("C:\python\yolotest\images\custom_data\classes.txt", "r") as f:
   # classes = f.read().splitlines()
result = None
(width, hight) = (None, None)
#path video yang akan diproses
#cap = cv2.VideoCapture(args["input"])
cap = cv2.VideoCapture(r"C:\Users\User\PycharmProjects\yolotest\Videotest_all.mp4")

#video frame counter
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = 200
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

while True:
    # read the next frame from the file
    (ret, img) = cap.read()
    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not ret:
        break
    # if the frame dimensions are empty, grab them
    if width is None or hight is None:
        (hight, width) = img.shape[:2]

    #deklarasi variabel untuk hitungan jumlah kemunculan objek dan bounding box intersection
    #bounding box intersection berfungsi untuk menentukan aktivitas mencurigakan manusia terhadap proyektor
    intersectionCounter = 0
    manusiaCounter = 0
    proyektorCounter = 0
    proyektorBoxes = []
    manusiaBoxes = []
    #resize image
    #img = cv2.resize(img,(640,480))
    #hight,width,_ = img.shape
    #penggunaan library time
    start = time.time()
    end = time.time()

    #blob
    blob = cv2.dnn.blobFromImage(img, 1/255,(320,320),swapRB = True,crop= False)
    net.setInput(blob)
    output_layers_name = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_name)

    #deklarasi variabel untuk bounding box
    boxes =[]
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            #penentuan nilai confidence
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3]* hight)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
    #non maximum supression
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,.8,.4)
    font = cv2.FONT_HERSHEY_PLAIN
    #warna bounding box manusia dan proyektor
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))
    #
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            #print(boxes[i])
            label = str(classes[class_ids[i]])
            if label == 'Proyektor':
                proyektorCounter+=1
                #proyektorBoxes.append(boxes[i])
                proyektorPolygon = [(x,y), (w,y), (w,h), (x,h)]
                proyektorBoxes.append(Polygon(proyektorPolygon))
            if label == 'Manusia':
                manusiaCounter+=1
                #manusiaBoxes.append(boxes[i])
                manusiaPolygon = [(x, y), (w, y), (w, h), (x, h)]
                manusiaBoxes.append(Polygon(manusiaPolygon))
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " " + str(float(confidence) * 100) + "%", (x - w, y + (int(0.5 * h))), font, 1, (0, 0, 0), 3)
            cv2.putText(img, label + " " + str(float(confidence) * 100) + "%", (x - w, y + (int(0.5 *h))), font, 1, (0, 255, 0), 1)

    for proyektor in proyektorBoxes:
        for manusia in manusiaBoxes:
            status = check_intersection(proyektor, manusia)
            if status:
                intersectionCounter += 1
    cv2.putText(img, 'Proyektor' + " : " + str(proyektorCounter), (20, 40), font, 2, (0, 0, 0), 3)
    cv2.putText(img, 'Proyektor' + " : " + str(proyektorCounter), (20, 40), font, 2, (255, 255, 255), 1)
    cv2.putText(img, 'Manusia' + " : " + str(manusiaCounter), (20, 70), font, 2, (0, 0, 0), 3)
    cv2.putText(img, 'Manusia' + " : " + str(manusiaCounter), (20, 70), font, 2, (255, 255, 255), 1)
    cv2.putText(img, 'Sentuhan' + " : " + str(intersectionCounter), (20, 100), font, 2, (0, 0, 0), 3)
    cv2.putText(img, 'Sentuhan' + " : " + str(intersectionCounter), (20, 100), font, 2, (0, 255, 255), 1)

    #penentuan status aktivitas manusia di ruangan kelas
    if intersectionCounter > 0:
        cv2.putText(img, 'Status : Terdapat aktivitas mencurigakan' , (20, 140), font, 2, (0, 0, 0), 3)
        cv2.putText(img, 'Status : Terdapat aktivitas mencurigakan', (20, 140), font, 2, (0, 0, 255), 1)
    else:
        cv2.putText(img, 'Status : Tidak ada aktivitas mencurigakan', (20, 140), font, 2, (0, 0, 0), 3)
        cv2.putText(img, 'Status : Tidak ada aktivitas mencurigakan', (20, 140), font, 2, (255, 0, 0), 1)

    #inisiasi video yang telah diproses untuk diekspor
    if result is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        result = cv2.VideoWriter('result.avi', fourcc, 20, (img.shape[1], img.shape[0]), True)
        #fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        #result = cv2.VideoWriter(args["output"], fourcc, 20, (img.shape[1], img.shape[0]), True)
        # some information on processing single frame
    #time counter per frame
    if total > 0:
        elap = (end - start)
        print("[INFO] single frame took {:.4f} seconds".format(elap))
        print("[INFO] estimated total time to finish: {:.4f}".format(
            elap * total))
    #ekspor hasil video
    result.write(img)
    #menampilkan video pada jendela baru & tombol exit
    cv2.imshow('Yolo v3 Deteksi Aktivitas Manusia di Ruangan Kelas', img)
    if cv2.waitKey(1) == ord('q'):
        break
#rilis hasil ekspor
result.release()

cap.release()