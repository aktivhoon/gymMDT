"""
2D rendering framework
"""
from __future__ import division
import os
import six
import sys

if "Apple" in sys.version:
    if 'DYLD_FALLBACK_LIBRARY_PATH' in os.environ:
        os.environ['DYLD_FALLBACK_LIBRARY_PATH'] += ':/usr/lib'
        # (JDS 2016/04/15): avoid bug on Anaconda 2.3.0 / Yosemite

from gym import error

try:
    import pyglet
except ImportError as e:
    raise ImportError("""
        HINT: you can install pyglet directly via 'pip install pyglet'. 
        But if you really just want to install all Gym dependencies and not have to think about it, 
        'pip install -e .[all]' or 'pip install gym[all]' will do it.
    """)
    

try:
    from pyglet.gl import *
except ImportError as e:
    raise ImportError(""" 
        Error occured while running `from pyglet.gl import *`",suffix="
        HINT: make sure you have OpenGL install. On Ubuntu, you can run 'apt-get install python-opengl'. 
        If you're running on a server, you may need a virtual frame buffer; something like this should work: 
        'xvfb-run -s \"-screen 0 1400x900x24\" python <your_script.py>'
    """)

import math
import numpy as np

def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error('Invalid display specification: {}. (Must be a string like :0 or None.)'.format(spec))

class Viewer(object):
    def __init__(self, world, display=None):
        display = get_display(display)
        
        self.columns = world.width
        self.rows = world.height

        self.width_size = world.width_size
        self.height_size = world.height_size

        self.width = self.columns * self.width_size
        self.height = self.rows * self.height_size

        self.window = pyglet.window.Window(width=self.width, height=self.height, display=display)
        self.window.on_close = self.window_closed_by_user
        self.geoms = []

        glEnable(GL_BLEND)
        # glEnable(GL_MULTISAMPLE)
        glEnable(GL_LINE_SMOOTH)
        # glHint(GL_LINE_SMOOTH_HINT, GL_DONT_CARE)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glLineWidth(1.0)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.close()
    
    def draw_fixpoint(self, width=2):
        pyglet.gl.glColor4f(0.1,0.1,0.1,1.0)
        center_x = self.width // 2
        center_y = self.height // 2
        pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ('v2i', (center_x - width//2, center_y, center_x + width//2, center_y)))
        pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ('v2f', (center_x, center_y - width//2, center_x, center_y + width//2)))


    def add_geom(self, geom):
        self.geoms.append(geom)

    def render(self, return_rgb_array=False):
        glClearColor(0.95,0.95,0.95,1.0)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        arr = None

        if self.geoms is not None:
            for geom in self.geoms:
                geom.sprite.draw()

        if self.items is not None:
            for item in self.items:
                item.sprite.draw()

        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            # In https://github.com/openai/gym-http-api/issues/2, we
            # discovered that someone using Xmonad on Arch was having
            # a window of size 598 x 398, though a 600 x 400 window
            # was requested. (Guess Xmonad was preserving a pixel for
            # the boundary.) So we use the buffer height/width rather
            # than the requested one.
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1,:,0:3]
        self.window.flip()
        
        return arr

    def get_array(self):
        self.window.flip()
        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        self.window.flip()
        arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        arr = arr.reshape(self.height, self.width, 4)
        return arr[::-1,:,0:3]

class GameObject:
    def __init__(self, img=None):
        if img is not None:
            img = pyglet.image.load(img)
            self.img_width = img.width
            self.img_height = img.height
            self.sprite = pyglet.sprite.Sprite(img)

    def set_pos_(self, x, y):
        self.sprite.x = x - self.img_width // 2
        self.sprite.y = y - self.img_height // 2


def translate_position(position, width_size, height_size):
    x = position[0] * width_size + width_size // 2
    y = position[1] * height_size + height_size // 2
    return x, y

def make_plus(world):
    path = os.path.abspath(os.getcwd())
    obj = GameObject(path + '/images/plus.png')
    x, y = translate_position([2, 2], world.width_size, world.height_size)
    obj.set_pos_(x, y)
    return obj

def make_state(world):
    path = os.path.abspath(os.getcwd())
    if world.s == 1:
        obj = GameObject(path + '/images/1.png')
    elif world.s == 2:
        obj = GameObject(path + '/images/2.png')
    elif world.s == 3:
        obj = GameObject(path + '/images/3.png')
    elif world.s == 4:
        obj = GameObject(path + '/images/4.png')
    x, y = translate_position([2, 3], world.width_size, world.height_size)
    obj.set_pos_(x, y)
    return obj

def call_stimuli_object(object_name):
    path = os.path.abspath(os.getcwd())
    if object_name == 'r':
        obj = GameObject(path + '/images/red.png')
    elif object_name == 'b':
        obj = GameObject(path + '/images/blue.png')
    elif object_name == 'y':
        obj = GameObject(path + '/images/yellow.png')

    return obj

def make_stimuli(state, world):
    left_obj = call_stimuli_object(state.left)
    right_obj = call_stimuli_object(state.right)

    left_x, left_y = translate_position([1, 2], world.width_size, world.height_size)
    left_obj.set_pos_(left_x, left_y)
    
    right_x, right_y = translate_position([3, 2], world.width_size, world.height_size)
    right_obj.set_pos_(right_x, right_y)

    return [left_obj, right_obj]

def make_result(state, world):
    path = os.path.abspath(os.getcwd())
    obj_name = state.item.item_name
    obj = GameObject(path + '/images/' + obj_name + '.png')
    x, y = translate_position([2, 2], world.width_size, world.height_size)
    obj.set_pos_(x, y)
    
    return [obj]

# ================================================================

class SimpleImageViewer(object):
    def __init__(self, display=None):
        self.window = None
        self.isopen = False
        self.display = display
    def imshow(self, arr):
        if self.window is None:
            height, width, channels = arr.shape
            self.window = pyglet.window.Window(width=width, height=height, display=self.display)
            self.width = width
            self.height = height
            self.isopen = True
        assert arr.shape == (self.height, self.width, 3), "You passed in an image with the wrong number shape"
        image = pyglet.image.ImageData(self.width, self.height, 'RGB', arr.tobytes(), pitch=self.width * -3)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0,0)
        self.window.flip()
    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False
    def __del__(self):
        self.close()
