import cv2
import sys

def main():
    # Open the first video file
    print(sys.argv[1], sys.argv[2])
    video1 = cv2.VideoCapture(sys.argv[1])

    # Open the second video file
    video2 = cv2.VideoCapture(sys.argv[2])

    # Get the frames per second (fps) and frame size of the first video
    fps = video1.get(cv2.CAP_PROP_FPS)
    frame_size = (int(video1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Create a VideoWriter object to write the output video
    output = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

    # Loop through the frames of the first video and write them to the output video
    while True:
        ret, frame = video1.read()
        if not ret:
            break
        output.write(frame)

    # Loop through the frames of the second video and write them to the output video
    while True:
        ret, frame = video2.read()
        if not ret:
            break
        output.write(frame)

    # Release the video objects and the output video
    video1.release()
    video2.release()
    output.release()


main()
