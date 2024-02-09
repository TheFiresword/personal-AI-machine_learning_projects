import traffic
import arcade
import arcade.gui
import cv2
import sys
import os
import numpy as np
import tkinter
from tkinter import filedialog

a = ["20km/h maximum speed limit",
     "30km/h max speed limit",
     "50km/h max speed limit",
     "60km/h max speed limit",
     "70km/h max speed limit",
     "80km/h max speed limit",
     "End of 80km/h speed limit",
     "100km/h max speed limit",
     "120km/h max speed limit",
     "No passing",
     "No passing for vehicles over 3.5t",
     "Priority",
     "Priority road",
     "Yield",
     "Stop",
     "All vehicles prohibited",
     "Vehicles over 3.5t prohibited",
     "Do not enter",
     "General danger",
     "Curve(left)",
     "Curve(right)",
     "Double curve(left)",
     "Rough road",
     "Slippery when wet or dirty",
     "Road narrows(right side)",
     "Road work",
     "Traffic signals",
     "Pedestrians",
     "Watch for children",
     "Bicycle crossing",
     "Beware of ice/snow",
     "Wild animal crossing",
     "End of all restrictions",
     "Mandatory direction of travel(right)",
     "Mandatory direction of travel(left)",
     "Mandatory direction of travel(straight)",
     "Mandatory direction of travel(straigh or right)",
     "Mandatory direction of travel(straigh or left)",
     "Pass by on right",
     "Pass by on left",
     "Roundabout",
     "End of no passing zone",
     "End of no passing zone for vehicles over 3.5t",
     ]

label_mapping = {
    i: a[i] for i in range(traffic.NUM_CATEGORIES)
}

deep_learning_model = traffic.load_model(sys.argv[1])
deep_learning_model.summary()


IMG_BOX_WIDTH = 100
IMG_BOX_HEIGHT = 100

IMG_DIR = os.getcwd().join("gtsrb")


class customWindow(arcade.Window):
    def __init__(self):
        super(customWindow, self).__init__(width=3*IMG_BOX_WIDTH, height=3*IMG_BOX_HEIGHT, title="ui", resizable=True)
        self.reset = None
        self.v_box = None
        self.reset_button = None
        self.image_box = None
        self.choose_image_button = None

        self.result_text = ""

        self.manager = None
        self.sprite_list = arcade.SpriteList()

        arcade.set_background_color(arcade.color.BLACK)

    def setup(self):
        style = {'bg_color': arcade.color.BLACK,
                 'font_color': arcade.csscolor.WHITE,
                 'font_size': 10,
                 'border_color': arcade.csscolor.SKY_BLUE,
                 'border_width': 2
                 }
        self.manager = arcade.gui.UIManager()
        self.manager.enable()
        self.choose_image_button = arcade.gui.UIFlatButton(text="Choose", width=60, style=style)
        self.choose_image_button.on_click = self.choose_img

        self.reset_button = arcade.gui.UIFlatButton(text="Reset", width=60, style=style)

        self.reset_button.on_click = self.on_reset_btn_click

        self.v_box = arcade.gui.UIBoxLayout()
        self.v_box.add(self.choose_image_button.with_space_around(bottom=10))
        self.v_box.add(self.reset_button)

        self.manager.add(
            arcade.gui.UIAnchorWidget(
                anchor_x="right",
                anchor_y="top",
                align_y=-self.height / 5,
                align_x=-10,
                child=self.v_box
            )
        )

    def on_draw(self):
        arcade.start_render()
        self.manager.draw()
        arcade.draw_rectangle_outline(self.width / 3, 2 * self.height / 3 - 15, width=IMG_BOX_WIDTH,
                                      height=IMG_BOX_HEIGHT, color=arcade.csscolor.SKY_BLUE)
        arcade.draw_text(text=self.result_text, start_x=IMG_BOX_WIDTH/2, start_y=IMG_BOX_HEIGHT,
                         color=arcade.csscolor.DARK_GRAY)

        if self.reset:
            self.sprite_list.clear()
            self.reset = False

        self.sprite_list.draw()

    def choose_img(self, event):
        file_path = filedialog.askopenfilename(initialdir=IMG_DIR)
        self.image_box = arcade.Sprite(filename=file_path, center_x=self.width / 3,
                                       center_y=2 * self.height / 3 - 15)
        img_w, img_h = self.image_box.width, self.image_box.height
        self.image_box.scale = (IMG_BOX_WIDTH/img_w + IMG_BOX_HEIGHT/img_h) / 2

        self.sprite_list.clear()
        self.sprite_list.append(self.image_box)
        self.analyze_image(file_path)

    def on_reset_btn_click(self, event):
        self.result_text = ""
        self.reset = True

    def analyze_image(self, path):
        processed_img = cv2.resize(cv2.imread(path), (traffic.IMG_WIDTH, traffic.IMG_HEIGHT))
        result = deep_learning_model.predict(
            [np.array(processed_img).reshape(
                (1, traffic.IMG_WIDTH, traffic.IMG_HEIGHT, 3)
            )])
        self.result_text = str(label_mapping[result.argmax()])


w = customWindow()
w.setup()
arcade.run()
