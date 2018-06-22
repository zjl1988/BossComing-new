import face_recognition
import cv2
import freetype
import copy
import numpy as np
import time
# This is a super simple (but slow) example of running face recognition on live video from your webcam.
# There's a second example that's a little more complicated but runs faster.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.




class put_chinese_text(object):
    def __init__(self, ttf):
        self._face = freetype.Face(ttf)

    def draw_text(self, image, pos, text, text_size, text_color):
        '''
        draw chinese(or not) text with ttf
        :param image:     image(numpy.ndarray) to draw text
        :param pos:       where to draw text
        :param text:      the context, for chinese should be unicode type
        :param text_size: text size
        :param text_color:text color
        :return:          image
        '''
        self._face.set_char_size(text_size * 64)
        metrics = self._face.size
        ascender = metrics.ascender/64.0

        #descender = metrics.descender/64.0
        #height = metrics.height/64.0
        #linegap = height - ascender + descender
        ypos = int(ascender)

        img = self.draw_string(image, pos[0], pos[1]+ypos, text, text_color)
        return img

    def draw_string(self, img, x_pos, y_pos, text, color):
        '''
        draw string
        :param x_pos: text x-postion on img
        :param y_pos: text y-postion on img
        :param text:  text (unicode)
        :param color: text color
        :return:      image
        '''
        prev_char = 0
        pen = freetype.Vector()
        pen.x = x_pos << 6   # div 64
        pen.y = y_pos << 6

        hscale = 1.0
        matrix = freetype.Matrix(int(hscale)*0x10000, int(0.2*0x10000),\
                                 int(0.0*0x10000), int(1.1*0x10000))
        cur_pen = freetype.Vector()
        pen_translate = freetype.Vector()

        image = copy.deepcopy(img)
        for cur_char in text:
            self._face.set_transform(matrix, pen_translate)

            self._face.load_char(cur_char)
            kerning = self._face.get_kerning(prev_char, cur_char)
            pen.x += kerning.x
            slot = self._face.glyph
            bitmap = slot.bitmap

            cur_pen.x = pen.x
            cur_pen.y = pen.y - slot.bitmap_top * 64
            self.draw_ft_bitmap(image, bitmap, cur_pen, color)

            pen.x += slot.advance.x
            prev_char = cur_char

        return image

    def draw_ft_bitmap(self, img, bitmap, pen, color):
        '''
        draw each char
        :param bitmap: bitmap
        :param pen:    pen
        :param color:  pen color e.g.(0,0,255) - red
        :return:       image
        '''
        x_pos = pen.x >> 6
        y_pos = pen.y >> 6
        cols = bitmap.width
        rows = bitmap.rows

        glyph_pixels = bitmap.buffer

        for row in range(rows):
            for col in range(cols):
                if glyph_pixels[row*cols + col] != 0:
                    img[y_pos + row][x_pos + col][0] = color[0]
                    img[y_pos + row][x_pos + col][1] = color[1]
                    img[y_pos + row][x_pos + col][2] = color[2]
# Get a reference to webcam #0 (the default one)
if __name__ == '__main__':
    video_capture = cv2.VideoCapture(0)

    # Load a sample picture and learn how to recognize it.
    obama_image = face_recognition.load_image_file("pictures/zjl1.jpg")
    zhaodebin_image = face_recognition.load_image_file("pictures/zhaodeb.jpg")
    wuchuanjin_image = face_recognition.load_image_file("pictures/wuchuanjin.jpg")
    yangyinjian_image = face_recognition.load_image_file("pictures/杨银剑.png")
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
    print(obama_face_encoding.size)
    zhaodebin_face_encoding = face_recognition.face_encodings(zhaodebin_image)[0]
    wuchuanjin_face_encoding=face_recognition.face_encodings(wuchuanjin_image)[0]
    yangyinjian_face_encoding=face_recognition.face_encodings(yangyinjian_image)[0]
    ret, frame = video_capture.read()
    ft = put_chinese_text('msyh.ttf')
    while ret:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Find all the faces and face enqcodings in the frame of video
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces([obama_face_encoding,zhaodebin_face_encoding,wuchuanjin_face_encoding,yangyinjian_face_encoding], face_encoding,tolerance=0.4)

            name = "未识别"
            if match[0]:
                name = "张吉利"
                print(name+"comming")
            if match[1]:
                name="赵德滨"
                print(name+"comming")
            if match[2]:
                name = "吴传金"
                print(name + "comming")
            if match[3]:
                name = "杨银剑"
                print(name + "comming")
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            #font = cv2.FONT_HERSHEY_DUPLEX

            #cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            img = np.zeros([300, 600, 3])

            color_ = (0, 255, 0)
            pos = (left, bottom - 35)
            text_size = 24
            pos2 = (3,3)

            image = ft.draw_text(frame, pos, name, text_size, color_)
            text_size2 = 24
            text="正常"
            if name != "未识别":
                text=name+" 通缉级别:红色 危险！"

            image2 = ft.draw_text(img, pos2, text, text_size2, color_)

        # Display the resulting image
        cv2.imshow(u'Video', image)
        cv2.imshow('Details', image2)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()