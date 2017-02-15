import numpy as np


def csem_1d_background(rad, region=None):
    regions = {'upper_mantle', 'transition_zone', 'lower_mantle', 'outer_core', 'inner_core'}
    if region:
        region = region.lower()
        if region not in regions:
            raise ValueError('region must be one of: {}.'.format(', '.join(regions)))

    r_earth = 6371.0
    s_ani = 0.0011
    r_ani = 6191.0
    x = rad / r_earth

    #- Above 80 km depth.
    if rad >= 6291.0 and region == 'upper_mantle':
        rho = 2.6910 + 0.6924 * x
        vph = vpv = 4.1875 + 3.9382 * x
        vsv = 2.1519 + 2.3481 * x
        vsh = vsv + s_ani * (rad - r_ani)
        eta = 0.0
        Qmu = 600.0 - 41411.16 * (1.0 - x)
        Qkappa = 57823.0
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    #- 80-180 km depth.
    elif 6291.0 >= rad >= 6191.0 and region == 'upper_mantle':
        rho = 2.6910 + 0.6924 * x
        vph = vpv = 4.1875 + 3.9382 * x
        vsv = 2.1519 + 2.3481 * x
        vsh = vsv + s_ani * (rad - r_ani)
        eta = 0.0
        Qmu = 80.0 + 4013.76 * (0.987443 - x)
        Qkappa = 57823.0
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    #- 180-320 km depth.
    elif 6191.0 >= rad >= 6051.0 and region == 'upper_mantle':
        rho = 9.1790 - 5.9841 * x
        vph = vpv = 40.5988 - 33.5317 * x
        vsv = vsh = 16.8261 - 12.7527 * x
        eta = 0.0
        Qmu = 143.0
        Qkappa = 57823.0
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    #- 320-400 km depth (400 km discontinuity).
    elif 6051.0 >= rad >= 5971.0 and region == 'upper_mantle':
        rho = 7.1089 - 3.8045 * x
        vph = vpv = 20.3926 - 12.2569 * x
        vsv = vsh = 8.9496 - 4.4597 * x
        eta = 0.0
        Qmu = 143.0
        Qkappa = 57823.0
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    #- 400-600 km depth.
    elif 5971.0 >= rad >= 5771.0 and region == 'transition_zone':
        rho = 11.2494 - 8.0298 * x
        vph = vpv = 39.7027 - 32.6166 * x
        vsv = vsh = 22.3512 - 18.5856 * x
        eta = 0.0
        Qmu = 143.0
        Qkappa = 57823.0
        return rho, vpv, vsv, vph, vsh, eta, Qmu, Qkappa

    #- 600-670 km depth (670 km discontinuity).
    elif 5771.0 >= rad >= 5701.0 and region == 'transition_zone':
        rho = 5.3197 - 1.4836 * x
        vph = vpv = 19.0957 - 9.8672 * x
        vsv = vsh = 9.9839 - 4.9324 * x
        eta = 0.0
        Qmu = 143.0
        Qkappa = 57823.0
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    #- 670-771 km depth.
    elif 5701.0 >= rad >= 5600.0 and region == 'lower_mantle':
        rho = 7.9565 - 6.4761 * x + 5.5283 * x * x - 3.0807 * x * x * x
        vph = vpv = 29.2766 - 23.6026 * x + 5.5242 * x * x - 2.5514 * x * x * x
        vsv = vsh = 22.3459 - 17.2473 * x - 2.0834 * x * x + 0.9783 * x * x * x
        eta = 0.0
        Qmu = 312.0
        Qkappa = 57823.0
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    #- 771-2741 km depth.
    elif 5600.0 >= rad >= 3630.0 and region == 'lower_mantle':
        rho = 7.9565 - 6.4761 * x + 5.5283 * x * x - 3.0807 * x * x * x
        vph = vpv = 24.9520 - 40.4673 * x + 51.4832 * x * x - 26.6419 * x * x * x
        vsv = vsh = 11.1671 - 13.7818 * x + 17.4575 * x * x - 9.2777 * x * x * x
        eta = 0.0
        Qmu = 312.0
        Qkappa = 57823.0
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    #- 2741-2891 km depth (core-mantle boundary).
    elif 3630.0 >= rad >= 3480.0 and region == 'lower_mantle':
        rho = 7.9565 - 6.4761 * x + 5.5283 * x * x - 3.0807 * x * x * x
        vph = vpv = 15.3891 - 5.3181 * x + 5.5242 * x * x - 2.5514 * x * x * x
        vsv = vsh = 6.9254 + 1.4672 * x - 2.0834 * x * x + 0.9783 * x * x * x
        eta = 0.0
        Qmu = 312.0
        Qkappa = 57823.0
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    #- 2891-5149 km depth (inner core boundary).
    elif 3480.0 >= rad >= 1221.5 and region == 'outer_core':
        rho = 12.5815 - 1.2638 * x - 3.6426 * x * x - 5.5281 * x * x * x
        vph = vpv = 11.0487 - 4.0362 * x + 4.8023 * x * x - 13.5732 * x * x * x
        vsv = vsh = 0.0
        eta = 0.0
        Qmu = 0.0
        Qkappa = 57823.0
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    #- 5149-6371 km depth.
    elif rad <= 1221.5 and region == 'inner_core':
        rho = 13.0885 - 8.8381 * x * x
        vph = vpv = 11.2622 - 6.3640 * x * x
        vsv = vsh = 3.6678 - 4.4475 * x * x
        eta = 0.0
        Qmu = 84.6
        Qkappa = 1327.7
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    else:
        raise ValueError(
            'Radius of {} could not be processed. Ensure correct region is '
            'specified ({})'.format(rad, ', '.join(regions)))


def csem_1d_background_no_regions(rad):
    """
    :param rad: distance from core in km
    :return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa:
    """
    r_earth = 6371.0
    s_ani = 0.0011
    r_ani = 6191.0
    x = rad / r_earth

    #- Above 80 km depth.
    if rad >= 6291.0:
        rho = 2.6910 + 0.6924 * x
        vph = vpv = 4.1875 + 3.9382 * x
        vsv = 2.1519 + 2.3481 * x
        vsh = vsv + s_ani * (rad - r_ani)
        eta = 0.0
        Qmu = 600.0 - 41411.16 * (1.0 - x)
        Qkappa = 57823.0
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    #- 80-180 km depth.
    elif 6291.0 > rad >= 6191.0:
        rho = 2.6910 + 0.6924 * x
        vph = vpv = 4.1875 + 3.9382 * x
        vsv = 2.1519 + 2.3481 * x
        vsh = vsv + s_ani * (rad - r_ani)
        eta = 0.0
        Qmu = 80.0 + 4013.76 * (0.987443 - x)
        Qkappa = 57823.0
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    #- 180-320 km depth.
    elif 6191.0 > rad >= 6051.0:
        rho = 9.1790 - 5.9841 * x
        vph = vpv = 40.5988 - 33.5317 * x
        vsv = vsh = 16.8261 - 12.7527 * x
        eta = 0.0
        Qmu = 143.0
        Qkappa = 57823.0
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    #- 320-400 km depth (400 km discontinuity).
    elif 6051.0 > rad >= 5971.0:
        rho = 7.1089 - 3.8045 * x
        vph = vpv = 20.3926 - 12.2569 * x
        vsv = vsh = 8.9496 - 4.4597 * x
        eta = 0.0
        Qmu = 143.0
        Qkappa = 57823.0
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    #- 400-600 km depth.
    elif 5971.0 > rad >= 5771.0:
        rho = 11.2494 - 8.0298 * x
        vph = vpv = 39.7027 - 32.6166 * x
        vsv = vsh = 22.3512 - 18.5856 * x
        eta = 0.0
        Qmu = 143.0
        Qkappa = 57823.0
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    #- 600-670 km depth (670 km discontinuity)
    elif 5771.0 > rad >= 5701.0:
        rho = 5.3197 - 1.4836 * x
        vph = vpv = 19.0957 - 9.8672 * x
        vsv = vsh = 9.9839 - 4.9324 * x
        eta = 0.0
        Qmu = 143.0
        Qkappa = 57823.0
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    #- 670-771 km depth.
    elif 5701.0 > rad >= 5600.0:
        rho = 7.9565 - 6.4761 * x + 5.5283 * x * x - 3.0807 * x * x * x
        vph = vpv = 29.2766 - 23.6026 * x + 5.5242 * x * x - 2.5514 * x * x * x
        vsv = vsh = 22.3459 - 17.2473 * x - 2.0834 * x * x + 0.9783 * x * x * x
        eta = 0.0
        Qmu = 312.0
        Qkappa = 57823.0
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    #- 771-2741 km depth.
    elif 5600.0 > rad >= 3630.0:
        rho = 7.9565 - 6.4761 * x + 5.5283 * x * x - 3.0807 * x * x * x
        vph = vpv = 24.9520 - 40.4673 * x + 51.4832 * x * x - 26.6419 * x * x * x
        vsv = vsh = 11.1671 - 13.7818 * x + 17.4575 * x * x - 9.2777 * x * x * x
        eta = 0.0
        Qmu = 312.0
        Qkappa = 57823.0
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    #- 2741-2891 km depth.
    elif 3630.0 > rad >= 3480.0:
        rho = 7.9565 - 6.4761 * x + 5.5283 * x * x - 3.0807 * x * x * x
        vph = vpv = 15.3891 - 5.3181 * x + 5.5242 * x * x - 2.5514 * x * x * x
        vsv = vsh = 6.9254 + 1.4672 * x - 2.0834 * x * x + 0.9783 * x * x * x
        eta = 0.0
        Qmu = 312.0
        Qkappa = 57823.0
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    #- 2891-5149 km depth (inner core boundary).
    elif 3480.0 > rad >= 1221.5:
        rho = 12.5815 - 1.2638 * x - 3.6426 * x * x - 5.5281 * x * x * x
        vph = vpv = 11.0487 - 4.0362 * x + 4.8023 * x * x - 13.5732 * x * x * x
        vsv = vsh = 0.0
        eta = 0.0
        Qmu = 0.0
        Qkappa = 57823.0
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    #- 5149-6371 km depth.
    elif 1221.5 > rad >= 0.0:
        rho = 13.0885 - 8.8381 * x * x
        vph = vpv = 11.2622 - 6.3640 * x * x
        vsv = vsh = 3.6678 - 4.4475 * x * x
        eta = 0.0
        Qmu = 84.6
        Qkappa = 1327.7
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    # This should never happen
    elif rad < 0.0:
        raise ValueError('Negative radius specified for 1D_prem')

def csem_1d_background_eval_point_cloud(rad):
    """
    Evaluates 1d_prem for an array containing the radii
    :param rad: distance from core in km
    :return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa:
    """
    print('Evaluating PREM')
    g = np.vectorize(csem_1d_background_no_regions)
    rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa = g(rad)
    return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa


def get_region(rad):
    discontinuities = np.array([1221.5, 3480.0, 3630.0, 5600.0, 5701.0,
                                5771.0, 5971.0, 6051.0, 6191.0, 6291.0])
    region = np.digitize(rad, discontinuities)
    return region


def csem_1d_background_regional(rad, region):
    r_earth = 6371.0
    s_ani = 0.0011
    r_ani = 6191.0
    x = rad / r_earth

    #- Above 80 km depth.
    if region == 10:
        rho = 2.6910 + 0.6924 * x
        vph = vpv = 4.1875 + 3.9382 * x
        vsv = 2.1519 + 2.3481 * x
        vsh = vsv + s_ani * (rad - r_ani)
        eta = 0.0
        Qmu = 600.0 - 41411.16 * (1.0 - x)
        Qkappa = 57823.0
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    #- 80-180 km depth.
    elif region == 9:
        rho = 2.6910 + 0.6924 * x
        vph = vpv = 4.1875 + 3.9382 * x
        vsv = 2.1519 + 2.3481 * x
        vsh = vsv + s_ani * (rad - r_ani)
        eta = 0.0
        Qmu = 80.0 + 4013.76 * (0.987443 - x)
        Qkappa = 57823.0
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    #- 180-320 km depth.
    elif region == 8:
        rho = 9.1790 - 5.9841 * x
        vph = vpv = 40.5988 - 33.5317 * x
        vsv = vsh = 16.8261 - 12.7527 * x
        eta = 0.0
        Qmu = 143.0
        Qkappa = 57823.0
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    #- 320-400 km depth (400 km discontinuity).
    elif region == 7:
        rho = 7.1089 - 3.8045 * x
        vph = vpv = 20.3926 - 12.2569 * x
        vsv = vsh = 8.9496 - 4.4597 * x
        eta = 0.0
        Qmu = 143.0
        Qkappa = 57823.0
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    #- 400-600 km depth.
    elif region == 6:
        rho = 11.2494 - 8.0298 * x
        vph = vpv = 39.7027 - 32.6166 * x
        vsv = vsh = 22.3512 - 18.5856 * x
        eta = 0.0
        Qmu = 143.0
        Qkappa = 57823.0
        return rho, vpv, vsv, vph, vsh, eta, Qmu, Qkappa

    #- 600-670 km depth (670 km discontinuity).
    elif region == 5:
        rho = 5.3197 - 1.4836 * x
        vph = vpv = 19.0957 - 9.8672 * x
        vsv = vsh = 9.9839 - 4.9324 * x
        eta = 0.0
        Qmu = 143.0
        Qkappa = 57823.0
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    #- 670-771 km depth.
    elif region == 4:
        rho = 7.9565 - 6.4761 * x + 5.5283 * x * x - 3.0807 * x * x * x
        vph = vpv = 29.2766 - 23.6026 * x + 5.5242 * x * x - 2.5514 * x * x * x
        vsv = vsh = 22.3459 - 17.2473 * x - 2.0834 * x * x + 0.9783 * x * x * x
        eta = 0.0
        Qmu = 312.0
        Qkappa = 57823.0
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    #- 771-2741 km depth.
    elif region == 3:
        rho = 7.9565 - 6.4761 * x + 5.5283 * x * x - 3.0807 * x * x * x
        vph = vpv = 24.9520 - 40.4673 * x + 51.4832 * x * x - 26.6419 * x * x * x
        vsv = vsh = 11.1671 - 13.7818 * x + 17.4575 * x * x - 9.2777 * x * x * x
        eta = 0.0
        Qmu = 312.0
        Qkappa = 57823.0
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    #- 2741-2891 km depth (core-mantle boundary).
    elif region == 2:
        rho = 7.9565 - 6.4761 * x + 5.5283 * x * x - 3.0807 * x * x * x
        vph = vpv = 15.3891 - 5.3181 * x + 5.5242 * x * x - 2.5514 * x * x * x
        vsv = vsh = 6.9254 + 1.4672 * x - 2.0834 * x * x + 0.9783 * x * x * x
        eta = 0.0
        Qmu = 312.0
        Qkappa = 57823.0
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    #- 2891-5149 km depth (inner core boundary).
    elif region == 1:
        rho = 12.5815 - 1.2638 * x - 3.6426 * x * x - 5.5281 * x * x * x
        vph = vpv = 11.0487 - 4.0362 * x + 4.8023 * x * x - 13.5732 * x * x * x
        vsv = vsh = 0.0
        eta = 0.0
        Qmu = 0.0
        Qkappa = 57823.0
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    #- 5149-6371 km depth.
    elif region == 0:
        rho = 13.0885 - 8.8381 * x * x
        vph = vpv = 11.2622 - 6.3640 * x * x
        vsv = vsh = 3.6678 - 4.4475 * x * x
        eta = 0.0
        Qmu = 84.6
        Qkappa = 1327.7
        return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa

    else:
        raise ValueError(
            'Radius of {} and region {} could not be processed. Ensure correct region is '
            'specified, region must lie within 0 (inner core) and 10 (upper mantle)'.format(rad, region))


def csem_1d_background_eval_point_cloud_region_specified(rad, region):
    """
    Evaluates 1d_prem for an array containing the radii
    :param rad: distance from core in km
    :param region: number that specifies region within the 1D model
    :return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa:
    """
    print('Evaluating PREM')
    g = np.vectorize(csem_1d_background_regional)
    rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa = g(rad, region)
    return rho, vpv, vph, vsv, vsh, eta, Qmu, Qkappa