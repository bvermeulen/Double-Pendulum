import tkinter as tk
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as mpl_patches
from matplotlib import lines as mpl_lines
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

FIG_SIZE = (8, 6)
TICK_INTERVAL = 1.5
X_MIN, X_MAX = -5, 5
Y_MIN, Y_MAX = -10, 1.5


class MplMap():
    '''  set up map
    '''
    @classmethod
    def settings(cls, root, fig_size):
        # set the plot outline, including axes going through the origin
        cls.root = root
        cls.fig, cls.ax = plt.subplots(figsize=fig_size)
        cls.ax.set_xlim(X_MIN, X_MAX)
        cls.ax.set_ylim(Y_MIN, Y_MAX)
        cls.ax.set_aspect(1)
        tick_range = np.arange(
            round(X_MIN + (10*abs(X_MIN) % TICK_INTERVAL*10)/10, 1),
            X_MAX + 0.1, step=TICK_INTERVAL)
        cls.ax.set_xticks(tick_range)
        cls.ax.set_yticks([])
        cls.ax.tick_params(axis='x', which='major', labelsize=6)
        cls.ax.spines['left'].set_color('none')
        cls.ax.spines['right'].set_color('none')
        cls.ax.spines['bottom'].set_position('zero')
        cls.ax.spines['top'].set_color('none')
        cls.canvas = FigureCanvasTkAgg(cls.fig, master=cls.root)

    @classmethod
    def get_canvas(cls):
        return cls.canvas


class Pendulum(MplMap):
    '''  class defining methods for Pendulum
    '''
    def __init__(self):
        # Physical constants
        self.g = 9.8
        self.length_of_stick = 2.5
        self.damping_factor = 0.1
        self.mass_of_bob = 2

        # initial state
        self.theta_0 = 0  # 0 degrees
        self.theta_dot_0 = 0  # no initial angular velocity
        self.theta = self.theta_0
        self.color_of_bob = 'green'

        x, y = self.calc_xy()
        self.bob = mpl_patches.Circle((x, y), 0.2 + self.mass * 0.02,
                                      fc=self.color, alpha=1)
        self.bob.set_picker(0)
        self.ax.add_patch(self.bob)
        cv_bob = self.bob.figure.canvas
        cv_bob.mpl_connect('pick_event', self.on_pick)
        cv_bob.mpl_connect('motion_notify_event', self.on_motion)
        cv_bob.mpl_connect('button_release_event', self.on_release)

        self.stick = mpl_lines.Line2D([0, x], [0, y])
        self.ax.add_line(self.stick)

        self.current_object = None
        self.current_dragging = False
        self.break_the_loop = False

        self.bob.figure.canvas.draw()
        self.stick.figure.canvas.draw()

    @property
    def color(self):
        return self.color_of_bob

    @color.setter
    def color(self, color):
        self.color_of_bob = color
        self.bob.set_color(self.color)
        self.blip()

    @property
    def mass(self):
        return self.mass_of_bob

    @mass.setter
    def mass(self, value):
        self.mass_of_bob = value
        self.bob.set_radius(0.2 + self.mass_of_bob * 0.02)
        self.blip()

    @property
    def gravity(self):
        return self.g

    @gravity.setter
    def gravity(self, value):
        self.g = value
        self.blip()

    @property
    def length(self):
        return self.length_of_stick

    @length.setter
    def length(self, value):
        self.length_of_stick = value
        self.blip()

    @property
    def damping(self):
        return self.damping_factor

    @damping.setter
    def damping(self, value):
        self.damping_factor = value
        self.blip()

    def on_pick(self, event):
        if event.artist != self.bob:
            return

        self.current_dragging = True
        self.current_object = event.artist

    def on_motion(self, event):
        if not self.current_dragging:
            return
        if self.current_object is None:
            return

        self.calc_theta(event.xdata, event.ydata)
        self.blip()

    def on_release(self, _):
        self.current_object = None
        self.current_dragging = False

    def start_swing(self):
        self.break_the_loop = False
        self.plot_pendulum()

    def stop_swing(self):
        self.break_the_loop = True

    def calc_theta(self, x, y):
        try:
            self.theta = np.arctan2(x, -y)
        except TypeError:
            pass

    def calc_xy(self):
        x = self.length * np.sin(self.theta)
        y = - self.length * np.cos(self.theta)
        return x, y

    def get_theta_double_dot(self, theta, theta_dot):
        ''' definition of ordinary differential equation for a
            pendulum
        '''
        return -self.damping_factor / self.mass_of_bob * theta_dot\
            - (self.g / self.length) * np.sin(theta)

    def resolve_theta(self):
        ''' create a generator that solves the ODE by calculating
            the integral and yielding time (in ms) and theta.
        '''
        theta_dot = self.theta_dot_0
        _time = 0  # in seconds
        delta_t = 0.001  # some small time step in seconds

        while True:
            theta_double_dot = self.get_theta_double_dot(
                self.theta, theta_dot)
            self.theta += theta_dot * delta_t
            theta_dot += theta_double_dot * delta_t
            yield int(_time * 1000)
            _time += delta_t

    def plot_pendulum(self):
        ''' methods to plot pendulum in matplotlib
        '''
        # note a frame per second (fps) > 20 the actual time
        # may not be able to keep up with model time

        fps = 20
        ms_per_frame = int(1000/fps)

        def current_time_ms():
            return int(round(time.time() * 1000))

        def check_drift(time_ms, running_time_ms):
            # check every 5 seconds
            if time_ms % (5 * fps * ms_per_frame) == 0:
                print(f'time (ms): {time_ms:,}, '
                      f'drift: {running_time_ms - time_ms}')

        actual_start_time_ms = current_time_ms()
        for time_ms in self.resolve_theta():
            if self.break_the_loop:
                break

            if time_ms % ms_per_frame == 0:

                running_time_ms = current_time_ms() - actual_start_time_ms
                check_drift(time_ms, running_time_ms)

                while running_time_ms < time_ms:
                    # wait and update time
                    running_time_ms = current_time_ms() - actual_start_time_ms
                else:
                    self.root.update()
                    self.blip()

    def blip(self):
        x, y = self.calc_xy()
        self.bob.center = (x, y)
        self.stick.set_data([0, x], [0, y])
        self.bob.figure.canvas.draw()
        self.stick.figure.canvas.draw()


class TkHandler():

    def __init__(self, root, canvas, pendulum):
        self.root = root
        self.pendulum = pendulum

        self.root.wm_title("Pendulum")

        sliders_frame = tk.Frame(self.root)
        sliders = {'gravity': [0, 100, 1],  # {key: [min, max, resolution]}
                   'mass':[1, 10, 0.1],
                   'length':[0.5, 10, 0.1],
                   'damping': [0, 2, 0.1],
                  }
        def create_slider(name, _min, _max, resolution):
            slider_frame = tk.Frame(sliders_frame)
            label_slider = tk.Label(slider_frame, text=f'\n{name:<11}: ')
            slider = tk.Scale(slider_frame, from_=_min, to=_max,
                              resolution=resolution,
                              orient=tk.HORIZONTAL,
                              sliderlength=15,
                              length=150,
                              command=lambda value: self._set_value(value, name))
            slider.set(getattr(self.pendulum, name))
            label_slider.pack(side=tk.LEFT)
            slider.pack(side=tk.LEFT)
            slider_frame.pack()

        for key, val in sliders.items():
            print(key, val)
            create_slider(key, *val)

        buttons_frame = tk.Frame(self.root)
        tk.Button(buttons_frame, text='Quit', command=self._quit)\
            .pack(side=tk.LEFT)
        tk.Button(buttons_frame, text='Green',\
            command=lambda *args: self._set_color('green', *args))\
            .pack(side=tk.LEFT)
        tk.Button(buttons_frame, text='Red',\
            command=lambda *args: self._set_color('red', *args))\
            .pack(side=tk.LEFT)
        tk.Button(buttons_frame, text='Start', command=self._start)\
            .pack(side=tk.LEFT)
        tk.Button(buttons_frame, text='Stop', command=self._stop)\
            .pack(side=tk.LEFT)

        # fill the grid
        tk.Grid.rowconfigure(self.root, 0, weight=1)
        tk.Grid.columnconfigure(self.root, 0, weight=1)
        sliders_frame.grid(row=0, column=0, sticky=tk.NW)
        buttons_frame.grid(row=2, column=0, columnspan=2, sticky=tk.W)
        canvas.get_tk_widget().grid(row=0, column=1, rowspan=1, columnspan=1,
                                    sticky=tk.W+tk.E+tk.N+tk.S)

        tk.mainloop()

    def _quit(self):
        self.root.quit()
        self.root.destroy()

    def _set_color(self, color):
        self.pendulum.color = color

    def _set_value(self, value, name):
        value = float(value)
        print(name, value)

        if name == 'gravity':
            self.pendulum.gravity = float(value)

        elif name == 'mass':
            self.pendulum.mass = float(value)

        elif name == 'length':
            self.pendulum.length = float(value)

        elif name == 'damping':
            self.pendulum.damping = float(value)

        else:
            assert False, f'wrong key value given: {name}'

    def _start(self):
        self.pendulum.start_swing()

    def _stop(self):
        self.pendulum.stop_swing()

def main():
    root = tk.Tk()
    MplMap.settings(root, FIG_SIZE)
    TkHandler(root, MplMap.get_canvas(), Pendulum())


if __name__ == "__main__":
    main()
