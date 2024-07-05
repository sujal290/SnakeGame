import random
import cvzone
import cv2
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Creating webcam object using cv2 & here using camera 3rd
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # width set
cap.set(4, 720)   # height set

# this is to detect one hand on camera
detector = HandDetector(detectionCon=0.8, maxHands=1)


class SnakeGame:
    def __init__(self, path_food):
        self.path_food = path_food
        self.points = []
        self.lengths = []
        self.current_length = 0
        self.allowed_length = 150
        self.previous_head = 0, 0
        self.img_food = cv2.imread(self.path_food, cv2.IMREAD_UNCHANGED)
        self.h_food, self.w_food, _ = self.img_food.shape
        self.food_point = 0, 0
        self.random_food_location()
        self.score = 0
        self.game_over = False

    def reset_game(self):
        self.points = []
        self.lengths = []
        self.current_length = 0
        self.allowed_length = 150
        self.previous_head = 0, 0
        self.img_food = cv2.imread(self.path_food, cv2.IMREAD_UNCHANGED)
        self.h_food, self.w_food, _ = self.img_food.shape
        self.food_point = 0, 0
        self.random_food_location()
        self.score = 0
        self.game_over = False

    def random_food_location(self):
        self.food_point = random.randint(100, 1000), random.randint(100, 600)

    def update(self, img_main, current_head):
        if self.game_over:
            cvzone.putTextRect(img_main, "Game Over", [300, 400], scale=7, thickness=5, offset=20)
            cvzone.putTextRect(img_main, f'Your Score: {self.score}', [300, 550], scale=7, thickness=5, offset=20)
        else:
            px, py = self.previous_head
            cx, cy = current_head

            self.points.append([cx, cy])
            distance = math.hypot(cx - px, cy - py)
            self.lengths.append(distance)
            self.current_length += distance
            self.previous_head = cx, cy

            # Length Reduction of snake.
            if self.current_length > self.allowed_length:
                for i, length in enumerate(self.lengths):
                    self.current_length -= length
                    self.lengths.pop(i)
                    self.points.pop(i)

                    if self.current_length < self.allowed_length:
                        break

            # check if snake ate the Food
            rx, ry = self.food_point
            if rx - self.w_food//2 < cx < rx + self.w_food//2 and ry - self.h_food//2 < cy < ry + self.h_food//2:
                self.random_food_location()
                self.allowed_length += 50
                self.score += 1
                print(self.score)

            # Drawing snake
            if self.points:
                for i, point in enumerate(self.points):
                    if i != 0:
                        cv2.line(img_main, tuple(self.points[i-1]), tuple(self.points[i]), (0, 0, 255), 20)
                cv2.circle(img_main, tuple(self.points[-1]), 20, (200, 0, 200), cv2.FILLED)

            # Draw Food
            img_main = cvzone.overlayPNG(img_main, self.img_food, (rx - self.w_food // 2, ry - self.h_food // 2))

            cvzone.putTextRect(img_main, f'Your Score: {self.score}', [50, 80], scale=3, thickness=3, offset=10)

            # check for collision
            pts = np.array(self.points[:-2], np.int32)
            if len(pts) > 2:  # Ensure there are enough points to form a polygon
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(img_main, [pts], False, (0, 200, 0), 3)
                minDist = cv2.pointPolygonTest(pts, (cx, cy), True)
                if minDist is not None and -1 <= minDist <= 1:
                    print("Hit")
                    self.game_over = True

        return img_main


game = SnakeGame("Donut.png")

# For running webcam
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        lm_list = hands[0]['lmList']
        point_index = lm_list[8][0:2]
        img = game.update(img, point_index)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        game.reset_game()