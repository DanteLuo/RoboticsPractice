import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


# backward method for calculating derivative
def backward_derivative_est(t,y,dy=None):
    y_i1 = y[1:]
    y_i0 = y[0:len(y)-1]
    t_i1 = t[1:]
    t_i0 = t[0:len(t)-1]
    delta_t = t_i1-t_i0
    delta_y = y_i1-y_i0

    naive_dy = delta_y/delta_t

    if dy != None:
        plt.plot(t[1:],naive_dy,'or')
        plt.plot(t[1:],dy[1:])
        plt.xlabel('t')
        plt.ylabel('dy/dt')
        plt.legend(['backward estimation','true derivatives'])
        plt.show()

    return naive_dy


# using a polynomial to fit the curve and get the derivatives assumes constant timesteps
# power feature is the power of the highest polynomials and window size is the sliding window size
def regression_derivative_est(t, y, delta_t, dy=None, power_feature=2, window_size=4):

    A = np.zeros([window_size,power_feature+1])
    derivative_coeff_array = np.zeros(power_feature+1)
    derivative_coeff_array[1] = 1
    derivatives = np.zeros([len(y)-window_size+1,1])
    t_window = np.arange((-window_size+1),1)*delta_t

    for power_id in range(power_feature+1):
        A[:,power_id] = t_window**power_id

    A_pinv = np.linalg.pinv(A)

    for ind_y in range(len(y)-window_size+1):
        y_sub = y[ind_y:ind_y+window_size]
        derivatives[ind_y] = derivative_coeff_array.dot(A_pinv.dot(y_sub))

    if dy != None:
        plt.plot(t[window_size-1:],derivatives,'or')
        plt.plot(t[1:],dy[1:])
        plt.xlabel('t')
        plt.ylabel('dy/dt')
        plt.legend(['regression estimation','true derivatives'])
        plt.show()

    return derivatives


def main():
    data = sio.loadmat('DataHW05_Prob5.mat')

    t = data['t']
    y = data['y']
    dy = data['dy']

    print('The size of data is:', dy.shape)

    backward_derivative_est(t,y,dy)

    delta_t = t[1]-t[0]  # remind the delta t is a constant
    regression_derivative_est(t,y,delta_t,dy)

    data_six = sio.loadmat('DataHW05_Prob6.mat')

    t = data_six['t']
    y = data_six['y']
    dy = data_six['dy']
    num_points = len(dy)
    print(len(t))

    print('The size of data is:', dy.shape)

    est_backward = backward_derivative_est(t,y)

    delta_t = t[1]-t[0]  # remind the delta t is a constant
    window_size = 12
    est_regression = regression_derivative_est(t,y,delta_t,window_size=window_size)

    # print the loss of the estimation
    err_backward = np.sqrt(np.sum(np.square(dy[1:]-est_backward)))/(num_points-1)
    err_regression = np.sqrt(np.sum(np.square(dy[window_size-1:]-est_regression)))/(num_points-window_size+1)
    print('The error of the backward estimation is',err_backward,'and the error of regression'
                                                                 ' estimation is',err_regression)

    plt.plot(t[1:], est_backward, 'or')
    plt.plot(t[window_size-1:],est_regression,'ob')
    plt.plot(t[1:], dy[1:])
    plt.xlabel('t')
    plt.ylabel('dy/dt')
    plt.legend(['backward estimation','regression estimation','true derivatives'])
    plt.show()


if __name__ == '__main__':
    main()



