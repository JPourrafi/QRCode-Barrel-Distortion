from ultralytics import YOLO
import cv2
import time

ncnn_model=YOLO("qr1_ncnn_model")

video = cv2.VideoCapture(0) 
while(True): 
    ret, frame = video.read() 
    fps=time.time()
    if ret:
        results = ncnn_model(frame)
        for r in results:
            boxes = r.boxes.xyxy
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            fps=1/(time.time()-fps)
            cv2.putText(frame, f"fps:{fps}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

vid.release() 
cv2.destroyAllWindows()