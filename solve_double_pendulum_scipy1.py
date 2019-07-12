import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as mpl_patches
from matplotlib import lines as mpl_lines
from scipy.integrate import ode

# Physical constants
g = 9.8
damping_factor = 0.9 * 0
length_r1 = 2.0
length_r2 = 4.0
mass_bob1 = 10.0
mass_bob2 = 5.0
plotsize = 1.10 * (length_r1 + length_r2)

# initial state
theta1_initial = + 120 / 180 * np.pi
theta2_initial = + 180 / 180 * np.pi
theta1_dot_initial = 0
theta2_dot_initial = 0


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


def get_derivatives_double_pendulum(t, state):
    ''' definition of ordinary differential equation for a
        double pendulum
    '''
    theta1, theta1_dot, theta2, theta2_dot = state

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

    state_differentiated = np.zeros(4)
    state_differentiated[0] = theta1_dot
    state_differentiated[1] = theta1_doubledot
    state_differentiated[2] = theta2_dot
    state_differentiated[3] = theta2_doubledot

    return state_differentiated

dp_integrator = ode(get_derivatives_double_pendulum).set_integrator('vode')

def plot_double_pendulum(fig, trace_line):
    # note a frame per second (fps) > 10 the actual time
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

    state = np.array([theta1_initial, theta1_dot_initial,
                      theta2_initial, theta2_dot_initial])
    dp_integrator.set_initial_value(state, t_initial)

    _x2, _y2 = bob2.center
    x_traces = [_x2]
    y_traces = [_y2]

    actual_start_time = current_time()
    while dp_integrator.successful():

        theta1, _, theta2, _ = state

        _x1, _y1 = calc_xy(length_r1, theta1)
        bob1.center = (_x1, _y1)
        stick1.set_data([0, _x1], [0, _y1])

        _x2, _y2 = calc_xy(length_r2, theta2)
        _y2 += _y1
        _x2 += _x1
        bob2.center = (_x2, _y2)
        stick2.set_data([_x1, _x2], [_y1, _y2])

        x_traces.append(_x2)
        y_traces.append(_y2)
        trace_line.set_data(x_traces[:], y_traces[:])

        running_time = current_time() - actual_start_time
        check_drift(_time, running_time)

        while running_time < _time:
            running_time = current_time() - actual_start_time

        else:
            # blip the image
            fig.canvas.draw()
            fig.canvas.flush_events()

        state = dp_integrator.integrate(dp_integrator.t + seconds_per_frame)
        _time += seconds_per_frame

        if _time > 15:
            return


def main():
    '''  set plot and action
    '''
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.canvas.set_window_title('Double Pendulum')
    ax.set_title('s1: click on green bob to start ...')
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
