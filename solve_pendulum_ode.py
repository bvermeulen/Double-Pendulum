import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as mpl_patches
from matplotlib import lines as mpl_lines
import time

# Physical constants
g = 9.8
length = 2.5
damping_factor = 0.1
mass_of_bob = 2

# initial state
THETA_0 = np.pi / 3  # 60 degrees
THETA_DOT_0 = 0  # no initial angular velocity


def calc_xy(theta):
    x = length * np.sin(theta)
    y = - length * np.cos(theta)
    return x, y


x, y = calc_xy(THETA_0)
bob = mpl_patches.Circle((x, y), 0.2, fc='g', alpha=1)
bob.set_picker(0)
stick = mpl_lines.Line2D([0, x], [0, y])


def get_theta_double_dot(theta, theta_dot):
    ''' definition of ordinary differential equation for a
        pendulum
    '''
    return -damping_factor / mass_of_bob * theta_dot\
           - (g / length) * np.sin(theta)


def resolve_theta():
    '''  create a generator that solves the ODE by calculating
         the integral and yielding time (in ms) and theta.
    '''
    theta = THETA_0
    theta_dot = THETA_DOT_0
    _time = 0  # in seconds
    delta_t = 0.001  # some small time step in seconds

    while True:
        theta_double_dot = get_theta_double_dot(
            theta, theta_dot)
        theta += theta_dot * delta_t
        theta_dot += theta_double_dot * delta_t
        yield int(_time * 1000), theta
        _time += delta_t


def plot_pendulum():
    # note a frame per second (fps) > 20 the actual time
    # may not be able to keep up with model time
    fps = 20
    ms_per_frame = int(1000/fps)
    current_time_ms = lambda: int(round(time.time() * 1000))

    def check_drift(time_ms, running_time_ms):
        # check every 5 seconds
        if time_ms % (5 * fps * ms_per_frame) == 0:
            print(f'time (ms): {time_ms:,}, '
                  f'drift: {running_time_ms - time_ms}')

    actual_start_time_ms = current_time_ms()
    for time_ms, theta in resolve_theta():

        if time_ms % ms_per_frame == 0:
            x, y = calc_xy(theta)
            bob.center = (x, y)
            stick.set_data([0, x], [0, y])

            running_time_ms = current_time_ms() - actual_start_time_ms
            check_drift(time_ms, running_time_ms)

            while running_time_ms < time_ms:
                # wait and update time
                running_time_ms = current_time_ms() - actual_start_time_ms
            else:
                # blip the image
                bob.figure.canvas.draw()
                stick.figure.canvas.draw()

        if time_ms > 20_000:
            return


def main():
    '''  set plot and action
    '''
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.canvas.set_window_title('Pendulum')
    ax.set_title('click on green bob to start ...')
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.0, 1.0)
    ax.set_aspect(1)

    def on_pick(event):
        plot_pendulum()

    ax.add_patch(bob)
    ax.add_line(stick)
    bob.figure.canvas.mpl_connect('pick_event', on_pick)

    bob.figure.canvas.draw()
    stick.figure.canvas.draw()

    plt.show()


if __name__ == "__main__":
    main()
