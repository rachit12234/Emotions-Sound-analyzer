import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button

class gridLayout(GridLayout):
    def __init__(self , **kwargs):
        super(gridLayout , self ).__init__(**kwargs)

        self.cols = 1

        self.top_grid = GridLayout()
        self.top_grid.cols = 2
        

        self.top_grid.add_widget(Label(text = "Camera"))
        self.top_grid.add_widget(Label(text = "Audio Graphs"))

        self.add_widget(self.top_grid)
        
        self.add_widget(Label(text = "Button & analysis of audio"))

        

class MyApp(App):
    def build (self):
        return gridLayout()

if __name__ == '__main__':
    MyApp().run()