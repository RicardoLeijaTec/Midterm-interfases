#!/usr/bin/env python3
import cv2 
import numpy as np 
import rclpy
from rclpy.node import Node 
from std_msgs.msg import Int32MultiArray

import ctypes
from ctypes.util import find_library
lib = ctypes.cdll.LoadLibrary(find_library("multiplier"))
times_hundred = lib.hundred_times_x

rclpy.init()
node = Node("color_detection_node")
pub = node.create_publisher(Int32MultiArray, '/coordinates', 10)

def coords_publisher(coordinates):
  print('[INFO] Coordinates:', coordinates)
  msg = Int32MultiArray()
  msg.data = coordinates
  pub.publish(msg)


def run():
  video_capture = cv2.VideoCapture(0)
  while True:
    _, frame = video_capture.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low = np.array([90, 190, 120])
    high = np.array([135, 255, 255])

    filtered_vid = cv2.inRange(hsv, low, high)
    contours, _ = cv2.findContours(filtered_vid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cont in contours:
      aproximation = cv2.approxPolyDP(cont,
        0.02 * cv2.arcLength(cont, True), True)
      contour_area = cv2.contourArea(cont)
      if contour_area > 500:
        cv2.drawContours(frame, [cont], 0, (0,0,0), 5)
        if len(aproximation) == 4:
            M = cv2.moments(cont)
            if M['m00'] != 0:
              cy = int(M['m01'] / M['m00'])
              cx = int(M['m10'] / M['m00'])
              coordinate = [times_hundred(cx), times_hundred(cy)]
              coords_publisher(coordinate)
              cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
              cv2.putText(frame, "Obj middle", (cx - 20, cy - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
              cv2.drawContours(frame, [cont], -1, (0, 255, 0), 2)
    cv2.imshow("video", frame)
    cv2.imshow("filtered vid", filtered_vid)
    if (cv2.waitKey(30) == 27):
      break
    
  try:
    rclpy.spin(node)
  except:
    pass
  finally:
    cv2.destroyAllWindows()
    video_capture.release()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
  run()