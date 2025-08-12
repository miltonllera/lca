import numpy as np


def gaussian_field(x, y, sigma, zero_low=True):
    """
    Generate a Gaussian field.

    Args:
        x (int): Width of the field
        y (int): Height of the field
        zero_low (bool, optional): If True, field values range from 0 to 1,
            otherwise -1 to 1. Default: True
        sigma (float, optional): Controls the spread of the Gaussian. Default: 5

    Returns:
        torch.Tensor: Gaussian field with shape [1, 1, x, y]
    """
    xx, yy = np.meshgrid(np.linspace(-1, 1, x), np.linspace(-1, 1, y), indexing="ij")
    if zero_low:
        field = 1 - np.exp(-(xx**2 + yy**2) * sigma)
    else:
        field = 1 - 2 * np.exp(-(xx**2 + yy**2) * sigma)
    return field[None], np.array([sigma])


def sinusoidal_fields(
    x, y, channels, same_direction=False, zero_low=True, freq_list=None, freq_min=1.0, freq_max=10.0
):
    """
    Generate sinusoidal fields.

    Args:
        x (int): Width of the field
        y (int): Height of the field
        channels (int): Number of channels/fields to generate
        same_direction (bool, optional): If True, all fields vary along x-axis,
                                        otherwise alternate between x and y. Default: False
        zero_low (bool, optional): If True, field values range from 0 to 1,
                                   otherwise -1 to 1. Default: True
        freq_list (list or np.array, optional): Custom frequencies to use. Default: None
        freq_min (float, optional): Minimum frequency when auto-generating. Default: 1
        freq_max (float, optional): Maximum frequency when auto-generating. Default: 10

    Returns:
        tuple: (numpy array of fields with shape [1, channels, x, y], np.array of frequencies used)

    """
    xx, yy = np.meshgrid(np.linspace(-1, 1, x), np.linspace(-1, 1, y), indexing="ij")
    freqs = (
        np.array(freq_list)
        if freq_list is not None
        else np.linspace(freq_min, freq_max, channels // 2 if not same_direction else channels)
    )
    a, b = (0.5, 1) if zero_low else (1, 0)
    if same_direction:
        fields = [a * (np.sin(f * xx * 2 * np.pi) + b) for f in freqs]
    else:
        fields = [
            (
                a * (np.sin(f * xx * 2 * np.pi) + b)
                if i % 2 == 0
                else a * (np.cos(f * yy * 2 * np.pi) + b)
            )
            for i, f in enumerate(np.repeat(freqs, repeats=2)[:channels])
        ]

    # print("Frequencies used:", freqs)
    return np.stack(fields), freqs


def radial_fields(x, y, channels, zero_low, freq_min=1.0, freq_max=10.0, freq_list=None):
    """
    Generate radial fields with sinusoidal patterns.

    Args:
        x (int): Width of the field
        y (int): Height of the field
        channels (int): Number of channels/fields to generate
        zero_low (bool, optional): If True, field values range from 0 to 1, otherwise -1 to 1.
                                   Default: True
        freq_min (float, optional): Minimum frequency when auto-generating. Default: 1
        freq_max (float, optional): Maximum frequency when auto-generating. Default: 10
        freq_list (list or torch.Tensor, optional): Custom frequencies to use. Default: None

    Returns:
        tuple: (np.ndarray of fields with shape [1, channels, x, y],
                np.ndarray of frequencies used
            )
    """
    a, b = (0.5, 1.0) if zero_low else (1, 0)

    xx, yy = np.meshgrid(np.linspace(-1, 1, x), np.linspace(-1, 1, y), indexing="ij")
    r = np.sqrt(xx**2 + yy**2)  # Compute radial distance

    freqs = (
        np.array(freq_list)
        if freq_list is not None
        else np.linspace(freq_min, freq_max, channels)
    )

    fields = [a * (np.sin(f * r * 2 * np.pi) + b) for f in freqs]

    return np.stack(fields), freqs


def checkerboard_fields(x, y, channels, scale_list=None):
    """
    Generate checkerboard fields with different scales.

    Args:
        x (int): Width of the field
        y (int): Height of the field
        channels (int): Number of channels/fields to generate
        scale_list (list or torch.Tensor, optional): Custom scale factors to use. Default: None

    Returns:
        torch.Tensor: Checkerboard fields with shape [1, channels, x, y]
    """
    xx, yy = np.meshgrid(np.arange(x), np.arange(y), indexing="ij")

    scales = (
        np.array(scale_list)
        if scale_list is not None
        else np.array([2**i for i in range(channels)])
    )

    fields = [((xx // scale + yy // scale) % 2) * 2 - 1 for scale in scales]

    return np.stack(fields)[None], scales


def directional_fields(x, y, n, zero_low=True, angle_list=None):
    """
    Generate directional fields at different angles.

    Args:
        x (int): Width of the field
        y (int): Height of the field
        n (int): Number of channels/fields to generate
        zero_low (bool, optional): If True, field values range from 0 to 1, otherwise -1 to 1.
                                   Default: True
        angle_list (list or np.ndarray optional): Custom angles to use (in degrees).
                                                  Default: None
    Returns:
        tuple: (torch.Tensor of fields with shape [1, n, x, y], torch.Tensor of angles used)
    """
    if angle_list is not None:
        # Convert input angles from degrees to radians
        angles = np.array(angle_list) * np.pi / 180.0
    else:
        # Generate n equally spaced angles between 0 and 360 degrees
        angles = np.linspace(0, 360 - 360 / n, n) * np.pi / 180.0

    xx, yy = np.meshgrid(np.linspace(-1, 1, x), np.linspace(-1, 1, y), indexing="ij")
    fields = [
        (
            ((np.cos(a) * xx + np.sin(a) * yy) + 1) / 2
            if zero_low
            else (np.cos(a) * xx + np.sin(a) * yy)
        )
        for a in angles
    ]
    return np.stack(fields), angles


def mix_fields(
    x,
    y,
    n,
    zero_low=True,
    same_direction_sin=False,
    gaussian_sigma=5.0,
    sin_freq_min=1.0,
    sin_freq_max=10.0,
    sin_freq_list=None,
    dir_angle_list=None,
):
    """
    Generate a mix of different fields: Gaussian, sinusoidal, and directional.

    Args:
        x (int): Width of the field
        y (int): Height of the field
        n (int): Number of channels/fields to generate for sinusoidal and directional fields
        same_direction_sin (bool, optional): If True, all sinusoidal fields vary along x-axis.
                                             Default: False
        zero_low (bool, optional): If True, field values range from 0 to 1, otherwise -1 to 1.
                                   Default: True
        gaussian_sigma (float, optional): Controls the spread of the Gaussian. Default: 5
        sin_freq_min (float, optional): Minimum frequency for sinusoidal fields. Default: 1
        sin_freq_max (float, optional): Maximum frequency for sinusoidal fields. Default: 10
        sin_freq_list (list or torch.Tensor, optional): Custom frequencies for sinusoidal
                                                        fields. Default: None
        dir_angle_list (list or torch.Tensor, optional): Custom angles for directional fields.
                                                         Default: None

    Returns:
        tuple: (np.ndarray of combined fields, tuple of parameters
            (sinusoidal_params, directional_params))
    """
    gaussian, gaussian_params = gaussian_field(x, y, sigma=gaussian_sigma, zero_low=zero_low)

    sinusoidal, sinusoidal_params = sinusoidal_fields(
        x,
        y,
        n,
        same_direction_sin,
        zero_low,
        freq_list=sin_freq_list,
        freq_min=sin_freq_min,
        freq_max=sin_freq_max,
    )

    directional, directional_params = directional_fields(
        x, y, n, zero_low=zero_low, angle_list=dir_angle_list
    )

    return (
        np.concat([gaussian, sinusoidal, directional], axis=0),
        (gaussian_params, sinusoidal_params, directional_params)
    )
