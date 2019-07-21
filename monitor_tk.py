import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import time
import numpy as np
import matplotlib.pyplot as plt

global stop_monitor

fps = 24
seconds_per_frame = 1 / fps
stop_monitor = False


class Monitor():
    ''' class to monitor value versus time
    '''
    def __init__(self, figsize=(5, 2), monitor_window=10, update_interval=0.1,
                 ymin=-1, ymax=1, ticks=None):
        self.monitor_time_window_s = monitor_window
        self.update_monitor_interval_s = update_interval
        self.fig_monitor, self.ax_monitor = plt.subplots(
            1, 1, figsize=figsize)
        self.ax_monitor.set_ylim(ymin, ymax)
        if ticks is None:
            ticks = [ymin, 0, ymax]
        self.ax_monitor.set_yticks(ticks)
        self.ax_monitor.tick_params(axis='y', which='major', labelsize=6)
        self.ax_monitor.tick_params(axis='x', which='major', labelsize=6)
        self.ax_monitor.grid(True)
        self.fig_monitor.tight_layout()

        self.monitor_plot, = self.ax_monitor.plot(
            [0], [0], color='black', linewidth=0.5)
        self.ax_monitor.set_xlim(self.monitor_time_window_s, 0)

    @property
    def update_time_s(self):
        return self.update_monitor_interval_s

    @property
    def monitor_fig(self):
        return self.fig_monitor

    @property
    def monitor_ax(self):
        return self.ax_monitor

    def set_fig_title(self, title='Monitor Figure'):
        self.fig_monitor.canvas.set_window_title(title)

    def set_ax_title(self, title='Monitor'):
        self.ax_monitor.set_title(title)
        self.fig_monitor.tight_layout()

    def blip(self):
        self.fig_monitor.canvas.draw()
        self.fig_monitor.canvas.flush_events()

    def update(self, _time, value):
        ''' method to plot value in time_window_graphs, time is
            past time, so t=0 is now and t=20 is 20 seconds ago
        '''
        # reset when time is zero
        if _time < self.update_monitor_interval_s:
            self.monitor_values = []
            self.time_values_reversed = deque([])
            self.time_reversed = []

        # build the time_reversed list (in reversed order to represent
        # time passed) and trim the monitor_values so they keep the
        # a finite length
        self.monitor_values.append(value)
        if _time < self.monitor_time_window_s + self.update_monitor_interval_s:
            self.time_values_reversed.appendleft(_time)
            self.time_reversed = list(self.time_values_reversed)
        else:
            self.monitor_values.pop(0)

        self.monitor_plot.set_data(self.time_reversed, self.monitor_values)


def run_monitor(monitor):
    global stop_monitor

    def current_time():
        return time.time()

    print(f'stop_monitor: {stop_monitor}')

    actual_start_time = current_time()
    _time = 0
    while not stop_monitor:
        value = np.sin(_time) - 0.5 + np.random.random_sample()

        if _time % monitor.update_time_s < seconds_per_frame:
            monitor.update(_time, value)
            monitor.blip()

        _time += seconds_per_frame

        # wait for actual time to catch up with model time _time
        running_time = current_time() - actual_start_time
        while running_time < _time:
            running_time = current_time() - actual_start_time


    else:
        print('stop me!!!!')


def _start(monitor):
    global stop_monitor
    stop_monitor = False
    run_monitor(monitor)

def key_start(monitor):
    global stop_monitor
    stop_monitor = False
    run_monitor(monitor)

def _stop():
    global stop_monitor
    stop_monitor = True

def _exit(root):
    global stop_monitor
    stop_monitor = True
    print('quit program ....')
    root.after(1000, root.quit)
    root.after(1000, root.destroy)

def main():
    root = tk.Tk()
    root.title('Press any key to start ...')
    monitor = Monitor(monitor_window=20, ymin=-1.6, ymax=1.6,
                      ticks=[-1.5, -1.0, -0.5, 0.0, +0.5, 1.0, 1.5])
    monitor.set_fig_title('Press any key to start ...')
    monitor.set_ax_title()
    monitor_canvas = FigureCanvasTkAgg(monitor.monitor_fig, master=root)
    monitor.monitor_fig.canvas.mpl_connect(
        'key_press_event', lambda event: key_start(monitor))

    button_frame = tk.Frame(master=root)
    button_start = tk.Button(button_frame,
        text='start', command=lambda: _start(monitor)
    ).pack(anchor=tk.W, side=tk.LEFT)
    button_stop = tk.Button(button_frame,
        text='stop', command=lambda: _stop()
    ).pack(side=tk.LEFT)
    button_exit = tk.Button(button_frame,
        text='exit', command=lambda: _exit(root)
    ).pack(side=tk.LEFT)

    monitor_canvas.get_tk_widget().pack()
    button_frame.pack(side=tk.LEFT)
    # run_monitor(monitor, True)
    tk.mainloop()

if __name__ == "__main__":
    main()
