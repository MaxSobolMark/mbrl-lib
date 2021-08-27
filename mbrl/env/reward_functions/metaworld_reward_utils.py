"""A set of reward utilities written by the authors of dm_control
Copied from Meta-World repo, branch new-reward-functions.
Modified to use Tensorflow instead of Numpy"""
import tensorflow as tf

# The value returned by tolerance() at `margin` distance from `bounds` interval.
_DEFAULT_VALUE_AT_MARGIN = 0.1


def _sigmoids(x, value_at_1, sigmoid):
    """Returns 1 when `x` == 0, between 0 and 1 otherwise.

    Args:
        x: A scalar or numpy array.
        value_at_1: A float between 0 and 1 specifying the output when `x` == 1.
        sigmoid: String, choice of sigmoid type.

    Returns:
        A numpy array with values between 0.0 and 1.0.

    Raises:
        ValueError: If not 0 < `value_at_1` < 1, except for `linear`, `cosine` and
        `quadratic` sigmoids which allow `value_at_1` == 0.
        ValueError: If `sigmoid` is of an unknown type.
    """
    if sigmoid in ('cosine', 'linear', 'quadratic'):
        if not 0 <= value_at_1 < 1:
            raise ValueError(
                '`value_at_1` must be nonnegative and smaller than 1, '
                'got {}.'.format(value_at_1))
    else:
        if not 0 < value_at_1 < 1:
            raise ValueError('`value_at_1` must be strictly between 0 and 1, '
                             'got {}.'.format(value_at_1))

    if sigmoid == 'gaussian':
        scale = tf.sqrt(-2 * tf.math.log(value_at_1))
        return tf.math.exp(-0.5 * (x * scale)**2)

    elif sigmoid == 'hyperbolic':
        scale = np.arccosh(1 / value_at_1)
        return 1 / np.cosh(x * scale)

    elif sigmoid == 'long_tail':
        scale = tf.sqrt(1 / value_at_1 - 1)
        return 1 / ((x * scale)**2 + 1)

    elif sigmoid == 'reciprocal':
        scale = 1 / value_at_1 - 1
        return 1 / (abs(x) * scale + 1)

    elif sigmoid == 'cosine':
        scale = np.arccos(2 * value_at_1 - 1) / np.pi
        scaled_x = x * scale
        return np.where(
            abs(scaled_x) < 1, (1 + np.cos(np.pi * scaled_x)) / 2, 0.0)

    elif sigmoid == 'linear':
        scale = 1 - value_at_1
        scaled_x = x * scale
        return np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

    elif sigmoid == 'quadratic':
        scale = np.sqrt(1 - value_at_1)
        scaled_x = x * scale
        return np.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)

    elif sigmoid == 'tanh_squared':
        scale = np.arctanh(np.sqrt(1 - value_at_1))
        return 1 - np.tanh(x * scale)**2

    else:
        raise ValueError('Unknown sigmoid type {!r}.'.format(sigmoid))


def tolerance(x,
              bounds=(0.0, 0.0),
              margin=0.0,
              sigmoid='gaussian',
              value_at_margin=_DEFAULT_VALUE_AT_MARGIN):
    """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.

    Args:
        x: A scalar or numpy array.
        bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
        the target interval. These can be infinite if the interval is unbounded
        at one or both ends, or they can be equal to one another if the target
        value is exact.
        margin: Float. Parameter that controls how steeply the output decreases as
        `x` moves out-of-bounds.
        * If `margin == 0` then the output will be 0 for all values of `x`
            outside of `bounds`.
        * If `margin > 0` then the output will decrease sigmoidally with
            increasing distance from the nearest bound.
        sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
        'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
        value_at_margin: A float between 0 and 1 specifying the output value when
        the distance from `x` to the nearest bound is equal to `margin`. Ignored
        if `margin == 0`.

    Returns:
        A float or numpy array with values between 0.0 and 1.0.

    Raises:
        ValueError: If `bounds[0] > bounds[1]`.
        ValueError: If `margin` is negative.
    """
    lower, upper = bounds
    if lower > upper:
        raise ValueError(
            f'Lower bound must be <= upper bound. Lower: {lower}; upper: {upper}'
        )


#    if margin < 0.:
#        print('margin: ', margin)
#        print('numpy: ', margin.numpy())
#        raise ValueError('`margin` must be non-negative. Current value: {}'.format(margin))

    in_bounds = tf.math.logical_and(lower <= x, x <= upper)
    d = tf.divide(tf.where(x < lower, lower - x, x - upper),
                  tf.cast(margin, tf.float32))
    value = tf.where(condition=margin == 0,
                     x=tf.where(in_bounds, 1.0, 0.0),
                     y=tf.where(in_bounds, 1.0,
                                _sigmoids(d, value_at_margin, sigmoid)))

    return float(value) if x.shape == () else value


def inverse_tolerance(x, bounds=(0.0, 0.0), margin=0.0, sigmoid='reciprocal'):
    """Returns 0 when `x` falls inside the bounds, between 1 and 0 otherwise.

    Args:
        x: A scalar or numpy array.
        bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
        the target interval. These can be infinite if the interval is unbounded
        at one or both ends, or they can be equal to one another if the target
        value is exact.
        margin: Float. Parameter that controls how steeply the output decreases as
        `x` moves out-of-bounds.
        * If `margin == 0` then the output will be 0 for all values of `x`
            outside of `bounds`.
        * If `margin > 0` then the output will decrease sigmoidally with
            increasing distance from the nearest bound.
        sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
        'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
        value_at_margin: A float between 0 and 1 specifying the output value when
        the distance from `x` to the nearest bound is equal to `margin`. Ignored
        if `margin == 0`.

    Returns:
        A float or numpy array with values between 0.0 and 1.0.

    Raises:
        ValueError: If `bounds[0] > bounds[1]`.
        ValueError: If `margin` is negative.
    """
    bound = tolerance(x,
                      bounds=bounds,
                      margin=margin,
                      sigmoid=sigmoid,
                      value_at_margin=0)
    return 1 - bound


def rect_prism_tolerance(curr, zero, one):
    """Computes a reward if curr is inside a rectangluar prism region.

    The 3d points curr and zero specify 2 diagonal corners of a rectangular
    prism that represents the decreasing region.

    one represents the corner of the prism that has a reward of 1.
    zero represents the diagonal opposite corner of the prism that has a reward
        of 0.
    Curr is the point that the prism reward region is being applied for.

    Args:
        curr(np.ndarray): The point who's reward is being assessed.
            shape is (3,).
        zero(np.ndarray): One corner of the rectangular prism, with reward 0.
            shape is (3,)
        one(np.ndarray): The diagonal opposite corner of one, with reward 1.
            shape is (3,)
    """
    in_range = lambda a, b, c: float(b <= a <= c) if c >= b else float(c <= a
                                                                       <= b)
    in_prism = (in_range(curr[0], zero[0], one[0])
                and in_range(curr[1], zero[1], one[1])
                and in_range(curr[2], zero[2], one[2]))
    if in_prism:
        diff = one - zero
        x_scale = (curr[0] - zero[0]) / diff[0]
        y_scale = (curr[1] - zero[1]) / diff[1]
        z_scale = (curr[2] - zero[2]) / diff[2]
        return x_scale * y_scale * z_scale
        # return 0.01
    else:
        return 1.


def hamacher_product(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """The element-wise hamacher (t-norm) product of a and b.

    computes (a * b) / ((a + b) - (a * b))

    Args:
        a (tf.Tensor): Shape: [n] 1st term of hamacher product.
        b (tf.Tensor): Shape: [n] 2nd term of hamacher product.
    Raises:
        ValueError: a and b must range between 0 and 1

    Returns:
        tf.Tensor: The element-wise hamacher product of a and b.
    """
    #    if not ((0. <= a <= 1.) and (0. <= b <= 1.)):
    #        raise ValueError("a and b must range between 0 and 1")
    a_times_b = tf.multiply(a, b)  # Element-wise
    denominator = a + b - a_times_b
    h_prod = tf.where(denominator > 0, tf.divide(a_times_b, denominator), 0.)
    #    assert 0. <= h_prod <= 1.
    return h_prod


def _reward_pos(obs: tf.Tensor, goal_position: tf.Tensor) -> tf.Tensor:
    """Modified version of _reward_pos that doesn't require theta.

    Args:
        obs (tf.Tensor): Shape: [n, obs_dims]
        goal_position (tf.Tensor): [n, goal_dims]

    Returns:
        [type]: [description]
    """
    hand = obs[:, :3]
    door = obs[:, 4:7] + tf.constant([-0.05, 0, 0])

    threshold = 0.12
    # floor is a 3D funnel centered on the door handle
    radius = tf.norm(hand[:, :2] - door[:, :2], axis=-1)
    floor = tf.where(condition=radius <= threshold,
                     x=0.0,
                     y=0.04 * tf.math.log(radius - threshold) + 0.4)
    # if radius <= threshold:
    #     floor = 0.0
    # else:
    #     floor = 0.04 * tf.math.log(radius - threshold) + 0.4
    # prevent the hand from running into the handle prematurely by keeping
    # it above the "floor"
    above_floor = tf.where(condition=hand[:, 2] >= floor,
                           x=1.0,
                           y=tolerance(
                               floor - hand[:, 2],
                               bounds=(0.0, 0.01),
                               margin=tf.math.maximum(
                                   floor / 2.0,
                                   tf.broadcast_to(0.0, floor.shape)),
                               sigmoid='long_tail',
                           ))
    # above_floor = 1.0 if hand[2] >= floor else tolerance(
    #     floor - hand[2],
    #     bounds=(0.0, 0.01),
    #     margin=tf.math.maximum(floor / 2.0, 0.0),
    #     sigmoid='long_tail',
    # )
    # move the hand to a position between the handle and the main door body
    in_place = tolerance(
        tf.norm(hand - door - tf.constant([0.05, 0.03, -0.01]), axis=-1),
        bounds=(0, threshold / 2.0),
        margin=0.5,
        sigmoid='long_tail',
    )
    ready_to_open = hamacher_product(above_floor, in_place)

    # now actually open the door
    # door_angle = -theta
    # a = 0.2  # Relative importance of just *trying* to open the door at all
    # b = 0.8  # Relative importance of fully opening the door
    # opened = a * float(theta < -np.pi/90.) + b * reward_utils.tolerance(
    opened = tolerance(
        # np.pi/2. + np.pi/6 - door_angle,
        tf.norm(door - goal_position, axis=-1),
        # bounds=(0, 0.5),
        bounds=(0, 0.08),
        # margin=np.pi/3.,
        margin=0.20,
        sigmoid='long_tail',
    )

    return ready_to_open, opened


def _reward_grab_effort(actions: tf.Tensor) -> tf.Tensor:
    """calculates grab-effort part of the reward.

    Args:
        actions (tf.Tensor): Shape: [n, actdim]

    Returns:
        tf.Tensor: [n]
    """
    return (tf.clip_by_value(actions[:, 3], -1, 1) + 1.0) / 2.0
