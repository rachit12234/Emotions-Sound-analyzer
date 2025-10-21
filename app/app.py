import kivy
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
import cv2

import faceDetection
from faceDetection import CameraOn

class gridLayout(GridLayout):
    def __init__(self , **kwargs):
        super(gridLayout , self ).__init__(**kwargs)

        self.cols = 1

        self.top_grid = GridLayout()
        self.top_grid.cols = 2
        
        self.camera_view = Image()
        self.top_grid.add_widget(self.camera_view)
        self.top_grid.add_widget(Label(text = "Audio Graphs"))

        self.add_widget(self.top_grid)

        self.add_widget(Label(text = "Button & analysis of audio"))
        Clock.schedule_interval(self.update_camera, 1.0 / 30.0)


    def update_camera(self,dt):
        frame = CameraOn()
        if frame is not None:
            buf = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.camera_view.texture = texture

class MyApp(App):
    def build (self):
        return gridLayout()

if __name__ == '__main__':
    MyApp().run()