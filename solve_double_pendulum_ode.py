import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as mpl_patches
from matplotlib import lines as mpl_lines

# Physical constants
g = 9.8
damping_factor = 0.001 * 0
length_r1 = 2.0
length_r2 = 4.0
mass_bob1 = 10.0
mass_bob2 = 5.0
plotsize = 1.10 * (length_r1 + length_r2)

# initial state
theta1_initial = + 120 / 180 * np.pi  # 135 degrees
theta2_initial = + 180 / 180 * np.pi  # 180 degrees
theta1_dot_initial = 0   # no initial angular velocity
theta2_dot_initial = 0   # no initial angular velocity


def calc_xy(length, theta):
    x = length * np.sin(theta)
    y = - length * np.cos(theta)
    return x, y


_x1, _y1 = calc_xy(length_r1, theta1_initial)
bob1 = mpl_patches.Circle((_x1, _y1), 0.05 * plotsize, fc='g', alpha=1, zorder=2)
bob1.set_picker(0)
stick1 = mpl_lines.Line2D([0, _x1], [0, _y1], zorder=2)

_x2, _y2 = calc_xy(length_r2, theta2_initial)
_x2 += _x1
_y2 += _y1
bob2 = mpl_patches.Circle((_x2, _y2), 0.05 * plotsize, fc='r', alpha=1, zorder=2)
stick2 = mpl_lines.Line2D([_x1, _x2], [_y1, _y2], zorder=2)


def get_thetas_double_dot(theta1, theta1_dot, theta2, theta2_dot):
    ''' definition of ordinary differential equation for a
        double pendulum
    '''
    _num1 = -g * (2 * mass_bob1 + mass_bob2) * np.sin(theta1)
    _num2 = -mass_bob2 * g * np.sin(theta1-2*theta2)
    _num3 = -2*np.sin(theta1-theta2) * mass_bob2
    _num4 = theta2_dot * theta2_dot * length_r2 + \
            theta1_dot * theta1_dot * length_r1 * np.cos(theta1-theta2)
    _den = length_r1 * (2 * mass_bob1 + mass_bob2 - \
           mass_bob2 * np.cos(2*theta1-2*theta2))
    theta1_doubledot = (_num1 + _num2 + _num3 * _num4) / _den

    _num1 = 2 * np.sin(theta1-theta2)
    _num2 = (theta1_dot * theta1_dot * length_r1 * (mass_bob1 + mass_bob2))
    _num3 = g * (mass_bob1 + mass_bob2) * np.cos(theta1)
    _num4 = (theta2_dot * theta2_dot * length_r2 * \
            mass_bob2 * np.cos(theta1-theta2))
    _den = length_r2 * (2 * mass_bob1 + mass_bob2 - \
           mass_bob2 * np.cos(2*theta1-2*theta2))
    theta2_doubledot = (_num1 * (_num2 + _num3 + _num4)) / _den

    return theta1_doubledot, theta2_doubledot


def resolve_thetas():
    '''  create a generator that solves the ODE by calculating
         the integral and yielding time (in ms) and theta.
    '''
    theta1 = theta1_initial
    theta1_dot = theta1_dot_initial
    theta2 = theta2_initial
    theta2_dot = theta2_dot_initial
    _time = 0  # in seconds
    delta_t = 0.01  # some small time step in seconds

    while True:
        theta1_double_dot, theta2_double_dot = \
            get_thetas_double_dot(theta1, theta1_dot, theta2, theta2_dot)

        theta1_dot += theta1_double_dot * delta_t
        theta2_dot += theta2_double_dot * delta_t
        theta1_dot *= (1 - damping_factor)
        theta2_dot *= (1 - damping_factor)
        theta1 += theta1_dot * delta_t
        theta2 += theta2_dot * delta_t

        yield int(_time * 1000), theta1, theta2
        _time += delta_t


def plot_double_pendulum(fig, trace_line):
    # note a frame per second (fps) > 10 the actual time
    # may not be able to keep up with model time
    fps = 15
    ms_per_frame = int(1000/fps)
    ms_tolerance = 10

    def current_time_ms():
        return int(round(time.time() * 1000))

    def check_drift(time_ms, running_time_ms):
        # check every 5 seconds
        if time_ms % (5 * fps * ms_per_frame) < ms_tolerance:
            print(f'time (ms): {time_ms:,}, '
                  f'drift: {running_time_ms - time_ms}')
            # fig.canvas.flush_events()


    _x2, _y2 = bob2.center
    x_traces = [_x2]; y_traces = [_y2]
    actual_start_time_ms = current_time_ms()
    for time_ms, theta1, theta2 in resolve_thetas():

        if time_ms % ms_per_frame < ms_tolerance:

            _x1, _y1 = calc_xy(length_r1, theta1)
            bob1.center = (_x1, _y1)
            stick1.set_data([0, _x1], [0, _y1])

            _x2, _y2 = calc_xy(length_r2, theta2)
            _x2 += _x1
            _y2 += _y1
            x_traces.append(_x2)
            y_traces.append(_y2)
            trace_line.set_data(x_traces[:], y_traces[:])
            bob2.center = (_x2, _y2)
            stick2.set_data([_x1, _x2], [_y1, _y2])

            running_time_ms = current_time_ms() - actual_start_time_ms
            check_drift(time_ms, running_time_ms)

            while running_time_ms < time_ms:
                # wait and update time
                running_time_ms = current_time_ms() - actual_start_time_ms

            else:
                # blip the image
                fig.canvas.draw()
                fig.canvas.flush_events()

        if time_ms > 5_000:
            return


def main():
    '''  set plot and action
    '''
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.canvas.set_window_title('Double Pendulum')
    ax.set_title('click on green bob to start ...')
    ax.set_xlim(-plotsize, plotsize)
    ax.set_ylim(-plotsize, plotsize)
    ax.set_aspect(1)
    trace_line, = ax.plot([0], [0], color='black', linewidth=0.2, zorder=1)

    def on_pick(event):
        plot_double_pendulum(fig, trace_line)

    ax.add_patch(bob1)
    ax.add_line(stick1)
    ax.add_patch(bob2)
    ax.add_line(stick2)

    bob1.figure.canvas.mpl_connect('pick_event', on_pick)

    plt.show()


if __name__ == "__main__":
    main()
