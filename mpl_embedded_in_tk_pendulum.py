import tkinter as tk
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as mpl_patches
from matplotlib import lines as mpl_lines
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.integrate import ode

FIG_SIZE = (8, 6)
TICK_INTERVAL = 1.5
X_MIN, X_MAX = -10, 10
Y_MIN, Y_MAX = -10, 10


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


class DoublePendulum(MplMap):
    '''  class defining methods for Pendulum
    '''
    def __init__(self):
        # Physical constants and initial settings
        self.g = 9.8
        self.damping1 = 0.0  # damping factor bob1
        self.damping2 = 0.0  # damping factor bob2
        self.length_r1 = 2.0
        self.length_r2 = 4.0
        self.mass_bob1 = 5.0
        self.mass_bob2 = 2.5
        self.color_bob1 = 'green'
        self.color_bob2 = 'red'
        self.plotsize = 1.10 * (self.length_r1 + self.length_r2)  # check this out not used now

        # initial state
        self.theta1_initial = + 120 / 180 * np.pi
        self.theta2_initial = + 180 / 180 * np.pi
        self.theta1_dot_initial = 0
        self.theta2_dot_initial = 0
        self.theta1 = self.theta1_initial
        self.theta2 = self.theta2_initial

        _x1, _y1 = self.calc_xy(self.length_r1, self.theta1_initial)
        self.bob1 = mpl_patches.Circle((_x1, _y1), 0.2 + self.m1 * 0.02,
                                       fc=self.color_bob1, alpha=1, zorder=2)
        self.bob1.set_picker(0)
        self.ax.add_patch(self.bob1)
        self.stick1 = mpl_lines.Line2D([0, _x1], [0, _y1], zorder=2)
        self.ax.add_line(self.stick1)
        cv_bob1 = self.bob1.figure.canvas
        cv_bob1.mpl_connect('pick_event', self.on_pick)
        cv_bob1.mpl_connect('motion_notify_event', self.on_motion)
        cv_bob1.mpl_connect('button_release_event', self.on_release)

        _x2, _y2 = self.calc_xy(self.length_r2, self.theta2_initial)
        _x2 += _x1
        _y2 += _y1
        self.bob2 = mpl_patches.Circle((_x2, _y2), 0.2 + self.m2 * 0.02,
                                       fc=self.color_bob2, alpha=1, zorder=2)
        self.bob2.set_picker(0)
        self.ax.add_patch(self.bob2)
        self.stick2 = mpl_lines.Line2D([_x1, _x2], [_y1, _y2], zorder=2)
        self.ax.add_line(self.stick2)
        cv_bob2 = self.bob2.figure.canvas
        cv_bob2.mpl_connect('pick_event', self.on_pick)
        cv_bob2.mpl_connect('motion_notify_event', self.on_motion)
        cv_bob2.mpl_connect('button_release_event', self.on_release)

        self.x_traces = []
        self.y_traces = []
        self.trace_line, = self.ax.plot([0], [0], color='black',
                                        linewidth=0.2, zorder=1)

        self.current_object = None
        self.current_dragging = False
        self.break_the_loop = False

        self.blip()

    def switch_colors_of_bob(self):
        print('switch color')
        self.color_bob1, self.color_bob2 = self.color_bob2, self.color_bob1
        self.bob1.set_color(self.color_bob1)
        self.bob2.set_color(self.color_bob2)
        self.blip()

    def toggle_trace_visible(self):
        print(self.trace_line.get_visible())
        if self.trace_line.get_visible():
            self.trace_line.set_visible(False)
        else:
            self.trace_line.set_visible(True)
        self.blip()

    def clear_trace(self):
        self.x_traces = []
        self.y_traces = []
        self.trace_line.set_data([0], [0])
        self.blip()

    @property
    def gravity(self):
        return self.g

    @gravity.setter
    def gravity(self, value):
        self.g = value

    @property
    def m1(self):
        return self.mass_bob1

    @m1.setter
    def m1(self, value):
        self.mass_bob1 = value
        self.bob1.set_radius(0.2 + self.mass_bob1 * 0.02)
        self.blip()

    @property
    def m2(self):
        return self.mass_bob2

    @m2.setter
    def m2(self, value):
        self.mass_bob2 = value
        self.bob2.set_radius(0.2 + self.mass_bob2 * 0.02)
        self.blip()

    @property
    def l1(self):
        return self.length_r1

    @l1.setter
    def l1(self, value):
        self.length_r1 = value
        self.calc_positions()
        self.blip()

    @property
    def l2(self):
        return self.length_r2

    @l2.setter
    def l2(self, value):
        self.length_r2 = value
        self.calc_positions()
        self.blip()

    @property
    def k1(self):
        return self.damping1

    @k1.setter
    def k1(self, value):
        self.damping1 = value

    @property
    def k2(self):
        return self.damping2

    @k2.setter
    def k2(self, value):
        self.damping2 = value

    def on_pick(self, event):
        if event.artist != self.bob1 and \
           event.artist != self.bob2:
            return

        self.current_dragging = True
        self.current_object = event.artist

    def on_motion(self, event):
        if not self.current_dragging:
            return
        if self.current_object == self.bob1:
            self.theta1 = self.calc_theta(event.xdata, event.ydata, self.theta1)

        elif self.current_object == self.bob2:
            self.theta2 = self.calc_theta(event.xdata, event.ydata, self.theta2)

        else:
            return

        self.calc_positions()
        self.blip()

    def on_release(self, _):
        self.current_object = None
        self.current_dragging = False

    def start_swing(self):
        self.break_the_loop = False
        self.plot_double_pendulum()

    def stop_swing(self):
        self.break_the_loop = True
        self.x_traces = []
        self.y_traces = []

    def calc_positions(self):
        _x1, _y1 = self.calc_xy(self.l1, self.theta1)
        self.bob1.center = (_x1, _y1)
        self.stick1.set_data([0, _x1], [0, _y1])

        _x2, _y2 = self.calc_xy(self.l2, self.theta2)
        _y2 += _y1
        _x2 += _x1
        self.bob2.center = (_x2, _y2)
        self.stick2.set_data([_x1, _x2], [_y1, _y2])

    def add_to_trace(self):
        _x2, _y2 = self.bob2.center
        self.x_traces.append(_x2)
        self.y_traces.append(_y2)
        self.trace_line.set_data(self.x_traces[:], self.y_traces[:])

    @staticmethod
    def calc_theta(x, y, theta):
        try:
            return np.arctan2(x, -y)
        except TypeError:
            return theta

    @staticmethod
    def calc_xy(length, theta):
        x = length * np.sin(theta)
        y = - length * np.cos(theta)
        return x, y

    def blip(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def get_derivatives_double_pendulum(self, t, state):
        ''' definition of ordinary differential equation for a
            double pendulum. See for derivations at
            https://ir.canterbury.ac.nz/bitstream/handle/10092/12659/chen_2008_report.pdf
        '''
        t1, w1, t2, w2 = state
        dt = t1 - t2
        _sin_dt = np.sin(dt)
        _den1 = (self.m1 + self.m2 * _sin_dt * _sin_dt)

        _num1 = self.m2 * self.l1 * w1 * w1 * np.sin(2*dt)
        _num2 = 2 * self.m2 * self.l2 * w2 * w2 * _sin_dt
        _num3 = 2 * self.g * self.m2 * np.cos(t2) * _sin_dt + \
                2 * self.g * self.m1 * np.sin(t1)
        _num4 = 2 * (self.k1 * w1 - self.k2 * w2 * np.cos(dt))
        w1_dot = (_num1 + _num2 + _num3 + _num4)/ (-2 * self.l1 * _den1)

        _num1 = self.m2 * self.l2 * w2 * w2 * np.sin(2*dt)
        _num2 = 2 * (self.m1 + self.m2) * self.l1 * w1 * w1 * _sin_dt
        _num3 = 2 * self.g * (self.m1 + self.m2) * np.cos(t1) * _sin_dt
        _num4 = 2 * (self.k1 * w1 * np.cos(dt) - \
                    self.k2 * w2 * (self.m1 + self.m2)/ self.m2)
        w2_dot = (_num1 + _num2 + _num3 + _num4)/ (2 * self.l2 *_den1)

        state_differentiated = np.zeros(4)
        state_differentiated[0] = w1
        state_differentiated[1] = w1_dot
        state_differentiated[2] = w2
        state_differentiated[3] = w2_dot

        return state_differentiated

    def plot_double_pendulum(self):
        ''' methods to plot pendulum in matplotlib
        '''
        # note a frame per second (fps) > 24 the actual time
        # may not be able to keep up with model time

        fps = 24
        seconds_per_frame = 1/fps

        def current_time():
            return time.time()

        def check_drift(_time, running_time):
            # check every 5 seconds
            if _time % 5 < seconds_per_frame:
                print(f'time (ms): {1000*_time:,.0f}, '
                      f'drift: {1000*(running_time - _time):,.0f}')

        t_initial = 0
        _time = t_initial

        dp_integrator = ode(self.get_derivatives_double_pendulum).set_integrator('vode')
        state = np.array([self.theta1, self.theta1_dot_initial,
                          self.theta2, self.theta2_dot_initial])
        dp_integrator.set_initial_value(state, t_initial)

        self.add_to_trace()

        actual_start_time = current_time()
        while dp_integrator.successful() and not self.break_the_loop:

            self.theta1, _, self.theta2, _ = state

            self.calc_positions()
            self.add_to_trace()

            running_time = current_time() - actual_start_time
            check_drift(_time, running_time)

            while running_time < _time:
                running_time = current_time() - actual_start_time

            else:
                self.blip()

            state = dp_integrator.integrate(dp_integrator.t + seconds_per_frame)
            _time += seconds_per_frame



class TkHandler():

    def __init__(self, root, canvas, doublependulum):
        self.root = root
        self.pendulum = doublependulum

        self.root.wm_title("Double Pendulum")

        sliders_frame = tk.Frame(self.root)
        sliders = {'gravity':   {'label':'Gravity   ', 'settings': [0, 30, 1]},        # 'settings': [min, max, resolution] # pylint: disable=C0301
                   'mass_bob1': {'label':'Mass bob 1', 'settings': [1, 10, 0.1]},
                   'mass_bob2': {'label':'Mass bob 2', 'settings': [1, 10, 0.1]},
                   'length_r1': {'label':'Length r1 ', 'settings': [0.1, 10, 0.1]},
                   'length_r2': {'label':'Length r2 ', 'settings': [0.1, 10, 0.1]},
                   'damping1':  {'label':'Damping 1 ', 'settings': [0, 1, 0.1]},
                   'damping2':  {'label':'Damping 2 ', 'settings': [0, 1, 0.1]},
                  }

        def create_slider(slider_key, slider_params):
            _min, _max, _resolution = slider_params['settings']

            slider_frame = tk.Frame(sliders_frame)
            label_slider = tk.Label(slider_frame, font=("TkFixedFont"),
                                    text=f'\n{slider_params["label"]:<11s}')
            slider = tk.Scale(slider_frame, from_=_min, to=_max, resolution=_resolution,
                              orient=tk.HORIZONTAL,
                              sliderlength=15,
                              length=150,
                              command=lambda value: self._set_value(value, slider_key))
            slider.set(getattr(self.pendulum, slider_key))
            label_slider.pack(side=tk.LEFT)
            slider.pack(side=tk.LEFT)
            slider_frame.pack()

        for key, slider_params in sliders.items():
            print(key, slider_params)
            create_slider(key, slider_params)

        buttons_frame = tk.Frame(self.root)
        tk.Button(buttons_frame, text='Quit', command=self._quit)\
            .pack(side=tk.LEFT)
        tk.Button(buttons_frame, text='Switch colors',\
            command=lambda *args: self._set_colors(*args))\
            .pack(side=tk.LEFT)
        tk.Button(buttons_frame, text='Trace on/ off',\
            command=lambda *args: self._toggle_trace_visible(*args))\
            .pack(side=tk.LEFT)
        tk.Button(buttons_frame, text='Clear trace',\
            command=lambda *args: self._clear_trace(*args))\
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

    def _set_colors(self):
        self.pendulum.switch_colors_of_bob()

    def _toggle_trace_visible(self):
        self.pendulum.toggle_trace_visible()

    def _clear_trace(self):
        self.pendulum.clear_trace()

    def _set_value(self, value, name):
        value = float(value)
        print(name, value)

        if name == 'gravity':
            self.pendulum.gravity = float(value)

        elif name == 'mass_bob1':
            self.pendulum.m1 = float(value)

        elif name == 'mass_bob2':
            self.pendulum.m2 = float(value)

        elif name == 'length_r1':
            self.pendulum.l1 = float(value)

        elif name == 'length_r2':
            self.pendulum.l2 = float(value)

        elif name == 'damping1':
            self.pendulum.k1 = float(value)

        elif name == 'damping2':
            self.pendulum.k2 = float(value)

        else:
            assert False, f'wrong key value given: {name}'

    def _start(self):
        self.pendulum.start_swing()

    def _stop(self):
        self.pendulum.stop_swing()


def main():
    root = tk.Tk()
    MplMap.settings(root, FIG_SIZE)
    TkHandler(root, MplMap.get_canvas(), DoublePendulum())


if __name__ == "__main__":
    main()
