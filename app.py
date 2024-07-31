from flask import Flask, jsonify, request
import os
import cv2
import cvzone
from cvzone.PoseModule import PoseDetector

app = Flask(__name__)

@app.route('/api/tryon', methods=['POST'])
def try_on():
    # Get the image from the request
    file = request.files['image']
    file.save('temp_image.jpg')

    # Initialize OpenCV and PoseDetector
    cap = cv2.VideoCapture('temp_image.jpg')
    detector = PoseDetector()

    # Define parameters
    shirtFolderPath = "Resources/Shirts"
    listShirts = os.listdir(shirtFolderPath)
    fixedRatio = 360/190
    shirtRatioHeightWidth = 350 / 343

    # Process the image
    success, img = cap.read()
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)
    if lmList:
        lm11 = (lmList[11][0:2])
        lm12 = (lmList[12][0:2])
        imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[0]), cv2.IMREAD_UNCHANGED)
        widthOfShirt = int((lm11[0]-lm12[0])*fixedRatio)
        imgShirt = cv2.resize(imgShirt, (widthOfShirt, int(widthOfShirt*shirtRatioHeightWidth)))
        currentScale = (lm11[0]-lm12[0])/190
        offset = int(90*currentScale), int(60*currentScale)
        try:
            img = cvzone.overlayPNG(img, imgShirt, (lm12[0]-offset[0], lm12[1]-offset[1]))
        except:
            pass

    # Save the processed image
    output_path = 'output_image.jpg'
    cv2.imwrite(output_path, img)
    return jsonify({"message": "Image processed successfully", "output_image": output_path})

if __name__ == '__main__':
    app.run(debug=True)
