from . import utils
import numpy as np
import scipy.optimize

def txdelay3Plane(param, tiltx, tilty):
    return txdelay3(param, tiltx, tilty)

def txdelay3Diverging(param, tiltx, tilty, omega):
    return txdelay3(param, tiltx, tilty, omega)

def txdelay3Focused(param, x, y, z):
    return txdelay3(x, y, z, param)

def txdelay3(*args):
    """
    TXDELAY3   Transmit delays for a matrix array
    TXDELAY3 returns the transmit time delays for focused, plane or
    diverging beam patterns with a matrix array.

    DELAYS = TXDELAY3(X0,Y0,Z0,PARAM) returns the transmit time delays for
    a pressure field focused at the point (x0,y0,z0). Note: if z0 is
    negative, then the point (x0,y0,z0) is a virtual source. The properties
    of the medium and the array must be given in the structure PARAM (see
    below).

    DELAYS = TXDELAY3([X01 X02],[Y01 Y02],[Z01 Z02],PARAM) returns the
    transmit time delays for a pressure field focused at the line specified
    by the two points (x01,y01,z01) and (x02,y02,z02).

    DELAYS = TXDELAY3(PARAM,TILTx,TILTy) returns the transmit time delays
    for a tilted plane wave. TILTx is the tilt angle about the X-axis.
    TILTy is the tilt angle about the Y-axis. If TILTx = TILTy = 0, then
    the delays are 0.

    DELAYS = TXDELAY3(PARAM,TILTx,TILTy,OMEGA) yields the transmit time
    delays for a diverging wave. The sector is characterized by the angular
    tilts and the solid angle OMEGA subtented by the rectangular aperture
    of the transducer. TILTx is the tilt angle about the X-axis. TILTy is
    the tilt angle about the Y-axis. OMEGA sets the amount of the field of
    view (in [0 2pi]). This syntax is for matrix arrays only, i.e. the
    element positions must form a plaid grid.

    [DELAYS,PARAM] =  TXDELAY3(...) updates the PARAM structure parameters
    including the default values. PARAM will also include PARAM.TXdelay,
    which is equal to DELAYS (in s).

    [...] = TXDELAY3 (no input parameter) simulates a focused pressure
    field generated by a 3-MHz 32x32 matrix array (1st example below).

    Units: X0,Y0,Z0 must be in m; TILTx,TILTy must be in rad. OMEGA is in
            sr (steradian). DELAYS are in s.

    PARAM is a structure that must contain the following fields:
    ------------------------------------------------------------
    1) PARAM.elements: x- and y-coordinates of the element centers
            (in m, REQUIRED). It MUST be a two-row matrix, with the 1st
            and 2nd rows containing the x and y coordinates, respectively.
    2) PARAM.width: element width, in the x-direction
            (in m, required for diverging waves)
    3) PARAM.height: element height, in the y-direction
            (in m, required for diverging waves)
    4) PARAM.c: longitudinal velocity (in m/s, default = 1540 m/s)

    ---
    NOTE #1: X-, Y-, and Z-axes
    Conventional axes are used: For a linear array, the X-axis is PARALLEL
    to the transducer and points from the first (leftmost) element to the
    last (rightmost) element (X = 0 at the CENTER of the transducer). The
    Z-axis is PERPENDICULAR to the transducer and points downward (Z = 0 at
    the level of the transducer, Z increases as depth increases). The
    Y-axis is such that the coordinates are right-handed.
    ---
    NOTE #2: TILTx and TILTy (in radians) describe the tilt angles in the
            trigonometric direction.
    ---

    Example #1:
    ----------
    %-- Generate a focused pressure field with a matrix array
    % 3-MHz matrix array with 32x32 elements
    param.fc = 3e6;
    param.bandwidth = 70;
    param.width = 250e-6;
    param.height = 250e-6;
    % position of the elements (pitch = 300 microns)
    pitch = 300e-6;
    [xe,ye] = meshgrid(((1:32)-16.5)*pitch);
    param.elements = [xe(:).'; ye(:).'];
    % Focus position
    x0 = 0; y0 = -2e-3; z0 = 30e-3;
    % Transmit time delays: 
    dels = txdelay3(x0,y0,z0,param);
    % 3-D grid
    n = 32;
    [xi,yi,zi] = meshgrid(linspace(-5e-3,5e-3,n),linspace(-5e-3,5e-3,n),...
        linspace(0,6e-2,4*n));
    % RMS pressure field
    RP = pfield3(xi,yi,zi,dels,param);
    % Display the pressure field
    slice(xi*1e2,yi*1e2,zi*1e2,20*log10(RP/max(RP(:))),...
        x0*1e2,y0*1e2,z0*1e2)
    shading flat
    colormap(hot), caxis([-20 0])
    set(gca,'zdir','reverse'), axis equal
    alpha color % some transparency
    c = colorbar; c.YTickLabel{end} = '0 dB';
    zlabel('[cm]')

    Example #2:
    ----------
    %-- Generate a pressure field focused on a line
    % 3-MHz matrix array with 32x32 elements
    param.fc = 3e6;
    param.bandwidth = 70;
    param.width = 250e-6;
    param.height = 250e-6;
    % position of the elements (pitch = 300 microns)
    pitch = 300e-6;
    [xe,ye] = meshgrid(((1:32)-16.5)*pitch);
    param.elements = [xe(:).'; ye(:).'];
    % Oblique focus-line @ z = 2.5cm
    x0 = [-1e-2 1e-2]; y0 = [-1e-2 1e-2]; z0 = [2.5e-2 2.5e-2];
    % Transmit time delays: 
    dels = txdelay3(x0,y0,z0,param);
    % 3-D grid
    n = 32;
    [xi,yi,zi] = meshgrid(linspace(-5e-3,5e-3,n),linspace(-5e-3,5e-3,n),...
        linspace(0,6e-2,4*n));
    % RMS pressure field
    RP = pfield3(xi,yi,zi,dels,param);
    % Display the elements
    figure, plot3(xe*1e2,ye*1e2,0*xe,'b.')
    % Display the pressure field
    contourslice(xi*1e2,yi*1e2,zi*1e2,RP,[],[],.5:.5:6,15)
    set(gca,'zdir','reverse'), axis equal
    colormap(hot)
    zlabel('[cm]')
    view(-35,20), box on


    This function is part of <a
    href="matlab:web('https://www.biomecardio.com/MUST')">MUST</a> (Matlab UltraSound Toolbox).
    MUST (c) 2020 Damien Garcia, LGPL-3.0-or-later

    See also PFIELD3, SIMUS3, DASMTX3, GETPARAM, TXDELAY.

    -- Damien Garcia -- 2022/10, last update: 2022/10/29
    website: <a
    href="matlab:web('https://www.biomecardio.com')">www.BiomeCardio.com</a>
    """


    #-- Check the input arguments
    if len(args) == 4: 
        if isinstance(args[3], utils.Param):
            # Origo: TXDELAY3(x0,y0,z0,param)
            param = args[3]
            option = 'Origo'
        elif isinstance(args[0], utils.Param):
            # Diverging wave: TXDELAY3(param,TILTx,TILTx,Omega)
            param = args[0]
            option = 'Diverging Wave'
        else:
            ValueError('Wrong input arguments. PARAM must be a structure.')
    elif len(args) == 3: # Plane wave: TXDELAY3(param,TILTx,TILTy)
        param = args[0]
        option = 'Plane Wave'
    else:
        ValueError('Wrong input arguments.')

    assert isinstance(param, utils.Param),'Wrong input arguments. PARAM must be a structure.'

    #-- Coordinates of the transducer elements (xe,ye)
    assert utils.isfield(param,'elements'), 'PARAM.elements must contain the x- and y-locations of the transducer elements.'
    assert param.elements.shape[0]==2, 'PARAM.elements must have two rows that contain the x (1st row) and y (2nd row) coordinates of the transducer elements.'
    xe = param.elements[0,:]
    ye = param.elements[1,:]
    
    xe = xe.reshape((1, -1), order="F")
    ye = ye.reshape((1, -1), order="F")
    
    #-- Longitudinal velocity (in m/s)
    if not utils.isfield(param,'c'):
        param.c = 1540

    c = param.c


    #-- Positions of the transducer elements
    x, z, THe, h= param.getElementPositions()

    if option == 'Plane Wave':
        # DR : problems checking if it is not a vector, used as a number later, no need for casting into array
        tiltx = args[1] # tiltx = np.array(args[1]).reshape((-1, 1))
        tilty = args[2] # tilty = np.array(args[2]).reshape((-1, 1))
        assert np.isscalar(tiltx+tilty), 'TILTx and TILTy must be two scalars.'
        assert np.all(np.abs(np.array([tiltx,tilty]))<np.pi/2), 'The tilt angles must verify |TILTx| and |TILTy| < pi/2'
        delays = (xe*np.sin(tilty)-ye*np.sin(tiltx))/c
    #-----
    elif option == 'Diverging Wave':
        # DR : problems checking if it is not a vector, used as a number later, no need for casting into array
        tiltx = args[1] # tiltx = np.array(args[1]).reshape((-1, 1))
        tilty = args[2] # tilty = np.array(args[2]).reshape((-1, 1))
        omega = args[3] # omega = np.array(args[3]).reshape((-1, 1))
        assert np.isscalar(tiltx+tilty+omega), 'TILTX, TILTY, and OMEGA must be scalar.'
        assert np.all(np.abs(np.array([tiltx,tilty]))<np.pi/2), 'The tilt angles must verify |TILTx| and |TILTy| < pi/2'
        assert omega<=2*np.pi, 'The solid angle must be smaller than 2pi'
        assert omega>=0, 'The solid angle must be nonnegative'

        # check if the elements are on a plaid grid
        uxe = np.unique(xe)
        uye = np.unique(ye)
        [xep,yep] = np.meshgrid(uxe,uye)
        test = np.all(np.sort(xep.flatten())==np.sort(xe)) & np.all(np.sort(yep.flatten())==np.sort(ye))
        assert test, 'The elements must be on a plaid grid with the "Diverging wave" option, i.e. the element positions must form a plaid grid.'

        # rotation of the point [0,0,-1]
        # TILTx about the x-axis, TILTy about the y-axis
        x = -np.sin(tilty)*np.cos(tiltx)
        y = np.sin(tiltx)
        z = -np.cos(tilty)*np.cos(tiltx)

        # corresponding azimuth and elevation
        # [az,el] = cart2sph(x,y,z);
        az = np.arctan2(y,x)
        el = np.arctan2(z,np.sqrt(x**2 + y**2))

        # dimensions of the matrix array
        l = np.max(xe)-np.min(xe)+param.width # width of the matrix array
        b = np.max(ye)-np.min(ye)+param.height # height of the matrix array


        # we know the azimuth and elevation of the virtual source
        # we need its radial position r for the given solid angle
        def myfun(r, omega=omega, l=l, b=b, az=az, el=el):
            return abs(solidAngle(r,l,b,az,el) - omega) # Define the function to minimize
        r = scipy.optimize.fminbound(myfun, 0, 2*np.pi, xtol=1e-6) # DR : function tolerance not being taken into account.
        # r = scipy.optimize.minimize_scalar(myfun, bounds=(0, 2*np.pi), method='bounded', options={'xatol': 1e-6, 'fatol': 1e-6}) # DR : alternative (not tried)

        # position of the virtual source, and TX delays
        cos_el = np.cos(el)
        x0 = r*cos_el*np.cos(az)
        y0 = r*cos_el*np.sin(az)
        z0 = r*np.sin(el)
        delays = txdelay3(x0,y0,z0,param)

    #-----
    elif option == 'Origo':
        x0 = np.array(args[0]).reshape((-1, 1), order="F")
        y0 = np.array(args[1]).reshape((-1, 1), order="F")
        z0 = np.array(args[2]).reshape((-1, 1), order="F")
        assert len(x0)==len(y0)==len(z0), 'X0, Y0, and Z0 must have the same length.'
        if len(x0)==1: # focus point
            d = np.sqrt((xe-x0)**2 + (ye-y0)**2 + z0**2)
        elif len(x0)==2: # focus line
            X = np.concatenate((xe, ye, np.zeros_like(xe)), axis=0)
            x1 = np.tile(np.concatenate((x0[0],y0[0],z0[0]), axis=0).reshape((-1,1), order="F"), (1, x0.shape[1]))
            x2 = np.tile(np.concatenate((x0[1],y0[1],z0[1]), axis=0).reshape((-1,1), order="F"), (1, x0.shape[1]))
            d = np.linalg.norm(np.cross(X-x1,X-x2, axis=0), axis=0)/np.linalg.norm(x2-x1)
        else:
            ValueError('X0, Y0, and Z0 must have 1 or 2 elements.')
        delays = -d/c*np.sign(z0)

    delays = delays-np.min(delays,-1).reshape((-1, 1))

    param.TXdelay = delays
    return delays


def solidAngle(r, l, b, az, el):
    # Advanced Geometry: Mathematical Analysis of Unified Articles
    # by Harish Chandra Rajpoot
    # Publisher: ‎Notion Press; First Edition (2014)
    # ISBN-10: 9383808152
    # ISBN-13: 978-9383808151

    # https://www.slideshare.net/hcr1991/solid-angle-subtended-by-a-rectangular-plane-at-any-point-in-the-space

    # Khadjavi, A.
    # "Calculation of solid angle subtended by rectangular apertures."
    # JOSA 58.10 (1968): 1417-1418.

    # notations from Harish Chandra Rajpoot
    L1 = l/2 + r*np.cos(el)*np.cos(az)
    L2 = -l/2 + r*np.cos(el)*np.cos(az)
    B1 = b/2 + r*np.cos(el)*np.sin(az)
    B2 = b/2 - r*np.cos(el)*np.sin(az)
    H = r*np.sin(el)

    # Solid angle calculation
    def w(l, b):
        return np.arcsin(l*b/np.hypot(l, H)/np.hypot(b, H))

    O = w(L1, B1) + w(L1, B2) - w(L2, B1) - w(L2, B2)
    return O
