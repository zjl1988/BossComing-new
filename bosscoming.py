import face_recognition
import cv2
import jpush as jpush
from conf import app_key, master_secret
from jpush import common

# This is a super simple (but slow) example of running face recognition on live video from your webcam.
# There's a second example that's a little more complicated but runs faster.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.


#push notification
_jpush = jpush.JPush(app_key, master_secret)
push = _jpush.create_push()
push.audience = jpush.all_
push.notification = jpush.notification(alert="Boss Coming")
push.platform = jpush.all_

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("pictures/zjl1.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces([obama_face_encoding], face_encoding,tolerance=0.5)

        name = "Unknown"
        if match[0]:
            name = "Boss"
            try:
                response = push.send()
            except common.Unauthorized:
                raise common.Unauthorized("Unauthorized")
            except common.APIConnectionException:
                raise common.APIConnectionException("conn")
            except common.JPushFailure:
                print ("JPushFailure")
            except:
                print ("Exception")


        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)


        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()