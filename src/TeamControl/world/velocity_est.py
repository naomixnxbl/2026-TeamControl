from collections import deque
FRAME_HIST_LEN = 10
ball_hist = deque(maxlen = FRAME_HIST_LEN)

# Using vx = covariane(t,x) / variance(t)
# Least Squares line fit

def velocity_est(ball_hist, fps = 60):
    # Number of balls captured in ball history
    N = len(ball_hist)

    if N < 2:
        return 0.0, 0.0

    # Constant delta t
    dt = 1.0/fps

    # Single pass for means
    sum_t = 0.0
    sum_x = 0.0
    sum_y = 0.0
    for i in range(N):
        ti = i * dt
        sum_t += ti
        sum_x += ball_hist[i][0]
        sum_y += ball_hist[i][1]

    t_mean = sum_t / N
    x_mean = sum_x / N
    y_mean = sum_y / N

    # Single pass for covariance and variance
    num_x = 0.0
    num_y = 0.0
    den = 0.0
    for i in range(N):
        t_diff = i * dt - t_mean
        num_x += t_diff * (ball_hist[i][0] - x_mean)
        num_y += t_diff * (ball_hist[i][1] - y_mean)
        den += t_diff * t_diff

    if den == 0:
        return 0.0, 0.0

    vx = num_x/den
    vy = num_y/den

    return vx, vy



# When receiving