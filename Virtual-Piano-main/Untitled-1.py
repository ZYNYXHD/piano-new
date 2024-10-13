import cv2
import mediapipe as mp
import pyglet
import time

# Constants
wCam, hCam = 1280, 480  # Increased camera resolution for two octaves
w, h = 40, 150  # Width and height of white keys
playlist = [
    './tones/tone1.mp3', './tones/tone2.mp3', './tones/tone3.mp3', './tones/tone4.mp3',
    './tones/tone5.mp3', './tones/tone6.mp3', './tones/tone7.mp3', './tones/tone1.mp3',
    './tones/tone2.mp3', './tones/tone3.mp3', './tones/tone4.mp3', './tones/tone5.mp3',
    './tones/tone6.mp3', './tones/tone7.mp3'
]

# Define positions for the keys (2 octaves)
# White keys (C, D, E, F, G, A, B in two cycles)
white_key_positions = [(i * 50 + 60, 0) for i in range(14)]  # 14 white keys

# Black keys positions (C#, D#, F#, G#, A#)
black_key_positions = [
    (wx + 35, 0)  # Place black keys above the respective white keys
    for wx in [i * 50 + 60 for i in range(14) if i % 7 != 2 and i % 7 != 6]
]

# Initialize camera
cap = cv2.VideoCapture(0) 
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Load all songs
songs = [pyglet.media.load(song) for song in playlist]


def findHands(img, draw=True):
    """Detects hands in the image and draws landmarks if draw parameter is set to True."""
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks and draw:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    return img, results


def findPositions(img, results, draw=True):
    """Finds the positions of landmarks on detected hands."""
    lmList = []

    if results.multi_hand_landmarks:
        for myHand in results.multi_hand_landmarks:
            xList = []
            yList = []
            lList = []

            for id, lm in enumerate(myHand.landmark):
                hi, wi, c = img.shape
                cx, cy = int(lm.x * wi), int(lm.y * hi)
                xList.append(cx)
                yList.append(cy)
                lList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            lmList.append(lList)

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

    return img, lmList


def playMusic(p1, p2):
    """Plays music based on the hand position on the virtual piano."""
    for i, (wx, wy) in enumerate(white_key_positions):
        if (wx - 15 < p1 < wx + 15) and (wy < p2 < wy + h):
            cv2.rectangle(img, (wx, wy), (wx + w, wy + h), (255, 0, 255), -1)
            song = songs[i % len(songs)]  # Map to white keys
            song.play()
            time.sleep(0.1)
            break  # Exit after playing one note to avoid multiple triggers

    for i, (bx, by) in enumerate(black_key_positions):
        if (bx - 15 < p1 < bx + 15) and (by < p2 < by + h // 1.5):
            cv2.rectangle(img, (bx, by), (bx + w - 20, by + int(h // 1.5)), (0, 0, 0), -1)
            song = songs[i + 7]  # Map to black keys
            song.play()
            time.sleep(0.1)
            break  # Exit after playing one note


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Mirror the camera feed
    img, results = findHands(img)
    img, lmlist = findPositions(img, results)

    # Draw white keys
    for wx, wy in white_key_positions:
        cv2.rectangle(img, (wx, wy), (wx + w, wy + h), (255, 255, 255), -1)

    # Draw black keys
    for bx, by in black_key_positions:
        cv2.rectangle(img, (bx, by), (bx + w - 20, by + int(h // 1.5)), (0, 0, 0), -1)

    # Check if hand landmarks are detected and play music accordingly
    if len(lmlist) >= 1:
        p1, p2 = lmlist[0][8][1:2]  # Index finger
        playMusic(p1, p2)

    # Show the image with piano keys
    cv2.imshow("Image", img)
    cv2.waitKey(1)
