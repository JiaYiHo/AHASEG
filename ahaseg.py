import numpy as np
import scipy.ndimage as ndi

import copy

# %%

def circular_sector(r_range, theta_range, LV_center):
    cx, cy = LV_center
    theta = theta_range/180*np.pi
    z = r_range.reshape(-1, 1).dot(np.exp(1.0j*theta).reshape(1, -1))
    xall = -np.imag(z) + cx
    yall = np.real(z) + cy
    return xall, yall

def get_xall_yall(xall, yall, deg, RV_fov):
    if deg == -1:
        m3_xall = np.round(xall.flatten())
        m3_yall = np.round(yall.flatten())
    else:
        m3_xall = np.round(xall[:, deg].flatten())
        m3_yall = np.round(yall[:, deg].flatten())
    mask = (m3_xall >= 0) & (m3_yall >= 0) & \
           (m3_xall < RV_fov.shape[0]) & (m3_yall < RV_fov.shape[1])
    m3_xall = m3_xall[np.nonzero(mask)].astype(int)
    m3_yall = m3_yall[np.nonzero(mask)].astype(int)
    return m3_xall, m3_yall


def radial_projection(RV, LV_center):

    cx, cy = LV_center
    M = copy.copy(RV)
    sum_curve = np.zeros(360)
    rr = int(max(M.shape))
    xall, yall = circular_sector(np.arange(0, rr, 0.5),
                                 np.arange(0., 360.), LV_center)


    for num in range(0, 360):
        m3_xall, m3_yall = get_xall_yall(xall, yall, num, M)
        radial = np.array([])
        for nn in range(m3_xall.size):
            radial = np.append(radial, RV[m3_xall[nn], m3_yall[nn]])
        sum_curve[num] = radial.sum()

    return sum_curve
# %%

def UP_DN(sumar):
    from scipy.optimize import curve_fit
    from scipy.signal import medfilt
    x = np.arange(sumar.size)
    y = sumar
    maxv = np.argmax(y)
    y = medfilt(y)

    def gauss(x, *p):
        A, mu, sigma = p
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))

    def fit(x, y):
        p0 = [np.max(y), np.argmax(y)+x[0], 1.]
        try:
            coeff, var_matrix = curve_fit(gauss, x, y, p0=p0)
            A, mu, sigma = coeff
        except:
            mu = 0
            sigma = 0
        return mu, sigma

    mu, sigma = fit(x[:maxv], y[:maxv])
    uprank1 = mu - sigma*2.5
    mu, sigma = fit(x[maxv:], y[maxv:])
    downrank1 = mu + sigma*2.5
    if downrank1 == 0:
        downrank1 = 360

    uprank2 = np.nonzero(y > 5)[0][0]
    downrank2 = np.nonzero(y > 5)[0][-1]
    uprank = max(uprank1, uprank2)
    downrank = min(downrank1, downrank2)

    return int(uprank), int(downrank)
#  %%


def degree_calcu(UP, DN, seg_num):
    anglelist = np.zeros(seg_num)
    if seg_num == 4:
        anglelist[0] = DN-180.
        anglelist[1] = UP
        anglelist[2] = DN
        anglelist[3] = UP+180.

    if seg_num == 6:
        anglelist[0] = DN-180.
        anglelist[1] = UP
        anglelist[2] = (UP+DN)/2.
        anglelist[3] = DN
        anglelist[4] = UP+180.
        anglelist[5] = anglelist[2]+180.
    anglelist = (anglelist + 360) % 360

    return anglelist.astype(int)
# %%
def labelit(angles, LV_wall, LV_center):
    LV_seprt = LV_wall * 0
    angles2 = np.append(angles, angles[0])
    rr = int(max(LV_wall.shape))
    angles2 = np.rad2deg(np.unwrap(np.deg2rad(angles2)))

    for ii in range(angles2.size-1):
        xall, yall = circular_sector(np.arange(0, rr, 0.5),
                                     np.arange(angles2[ii], angles2[ii+1],
                                     0.5), LV_center)
        xall, yall = get_xall_yall(xall, yall, -1, LV_wall)
        LV_seprt[xall, yall] = ii + 1

    return LV_seprt*LV_wall



def LVseg(LVbmask, LVwmask, RVbmask, nseg=4):
    LV_center = ndi.center_of_mass(LVwmask)
    sum_curve = radial_projection(RVbmask, LV_center)
    uprank, downrank = UP_DN(sum_curve)
    anglelist = degree_calcu(uprank, downrank, nseg)
    LVSeg_label = labelit(anglelist, LVwmask, LV_center)


    return LVSeg_label
        
