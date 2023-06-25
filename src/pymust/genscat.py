import scipy, scipy.interpolate
from . import utils
import numpy as np

def genscat(roidim,meandist,I = None,g = None):
    """
    %GENSCAT   Generate a distribution of scatterers
    %   [XS,YS,ZS] = GENSCAT([WIDTH HEIGHT],MEANDIST) generates a 2-D
    %   pseudorandom distribution of scatterers such that the mean distance
    %   between a scatterer and its nearest neighbor is approximately MEANDIST.
    %   The vector [WIDTH HEIGHT] (unit = m) specifies the width and height of
    %   the rectangular ROI to which the scatterers belong. The middle of its
    %   lower edge (in the x-z coordinate system) is located at (0,0). In this
    %   2-D syntax, YS is a vector of zeros.
    %
    %   [XS,YS,ZS] = GENSCAT([WIDTH HEIGHT DEPTH],MEANDIST) generates a 3-D
    %   pseudorandom distribution of scatterers such that the mean distance
    %   between a scatterer and its nearest neighbor is approximately MEANDIST.
    %   The vector [WIDTH HEIGHT DEPTH] (unit = m) specifies the width
    %   (x-direction), height (z-direction), and depth (y-direction) of the ROI
    %   box to which the scatterers belong. The center of its lower face (in
    %   the x-y-z coordinate system) is located at (0,0,0).
    %
    %   [XS,YS,ZS,RC] = GENSCAT([...],MEANDIST,I) also returns the reflection
    %   coefficients RC of the scatterers. The RC values follow Rayleigh
    %   distributions whose means are calculated from the image I. I must be a
    %   2-D or 3-D image whose size is given by [WIDTH HEIGHT] (2-D) or
    %   [WIDTH HEIGHT DEPTH] (3-D).
    %
    %   If a 2-D image I is given as an input, you can use [WIDTH NaN] or [Nan
    %   HEIGHT]. The missing value is calculated from the pixel-based size of
    %   the image I while preserving the aspect ratio. Similarly, [WIDTH Nan
    %   NaN], [Nan HEIGHT NaN], or [NaN NaN DEPTH] can be used with a 3-D
    %   image. If I is empty or omitted, then I is assumed to be an image of
    %   ones.
    %
    %   [...] = GENSCAT(...,G) assumes that the dynamic range of the input
    %   image I is G dB (defaut value = 40 dB). If G<=1, it is assumed that the
    %   image I is gamma-compressed, with gamma = G.
    %
    %   Note on MEANDIST:
    %   ---------------- 
    %   We recommend a MEANDIST value less than or equal to the minimum
    %   wavelength (at -6 dB). For a speed of sound c, from the PARAM structure
    %   generated by GETPARAM, you may use
    %       MEANDIST = PARAM.c/(PARAM.fc*(1+PARAM.bandwidth/200)),
    %   or a smaller value. This value is used automatically if you input the
    %   PARAM structure instead of MEANDIST:
    %       [...] = GENSCAT([...],PARAM,I)
    %   Note: by default, PARAM.c = 1540 [m/s] and PARAM.bandwidth = 75 [%],
    %   as in PFIELD.
    %
    %
    %   Example: cardiac scatterers:
    %   ---------------------------
    %   %-- Read the heart image and make it gray
    %   I = imread('heart.jpg');
    %   I = rgb2gray(I);
    %   [nl,nc] = size(I);
    %   %-- Parameters of the cardiac phased array
    %   param = getparam('P4-2v');
    %   %-- Pseudorandom distribution of scatterers (depth is 15 cm)
    %   [xs,~,zs,RC] = genscat([NaN 15e-2],param,I);
    %   %-- Display the scatterers in a dB scale
    %   scatter(xs*1e2,zs*1e2,2,20*log10(RC/max(RC(:))),'filled')
    %   caxis([-40 0])
    %   colormap hot
    %   axis equal ij tight
    %   set(gca,'XColor','none','box','off')
    %   title([int2str(numel(RC)) ' cardiac scatterers'])
    %   ylabel('[cm]')
    %   
    %   
    %   This function is part of MUST (Matlab UltraSound Toolbox).
    %   MUST (c) 2020 Damien Garcia, LGPL-3.0-or-later
    %
    %   See also SIMUS, GETPARAM.
    %
    %   -- Damien Garcia -- 2021/12, last update 2022/05/10
    %   website: <a
    %   href="matlab:web('http://www.biomecardio.com')">www.BiomeCardio.com</a>
    """

    #%-- Check the input arguments
    assert utils.isnumeric(roidim) and isinstance(roidim, np.ndarray), 'The 1st argument must be a numeric vector.'
    assert len(roidim)==2 or len(roidim)==3, 'The 1st argument must be a vector of length 2 or 3.'
    if isinstance(meandist, utils.Param):
        #%-- The input is PARAM
        #% Check the PARAM structure and calculate the default MEANDIST
        #% (= minimal wavelength at -6 dB)
        param = meandist
        #%--
        #% 1) Center frequency (in Hz)
        assert utils.isfield(param,'fc'), 'A center frequency value (PARAM.fc) is required.'
        #%--
        #% 2) Fractional bandwidth at -6dB (in %)
        if not utils.isfield(param,'bandwidth'):
            param.bandwidth = 75

        assert param.bandwidth>0 and  param.bandwidth<200, 'The fractional bandwidth at -6 dB (PARAM.bandwidth, in %) must be in ]0,200['
       
        #%--
        #% 3) Longitudinal velocity (in m/s)
        if not utils.isfield(param,'c'):
            param.c = 1540 # % default value

        #%--
        meandist = param.c/(param.fc*(1+param.bandwidth/200))
    else:
        assert isinstance(meandist, float) and meandist>0, 'MEANDIST must be a positive scalar.'
    
    if I is not None:
        assert len(I.shape) in [2, 3] and np.all(I>=0), 'I must be 2-D or 3-D with non-negative elements.'
        assert len(roidim)== len(I.shape), 'The number of dimensions of I does not match the length of the 1st argument.'
    else:
        assert np.all(np.isfinite(roidim)), 'The 1st argument must contain only finite elements if I is not given.'
        assert g is None, 'The 4th argument g cannot be used if I is not given.' #Note GB: actually a warning should be enough

    ##%-- Calculate xmin, xmax, ymin, ymax, and zmax (we have zmin = 0)
    width,height  = roidim
    #%
    if len(roidim)==2: #% 2-D
        assert np.any((np.isfinite(roidim))), 'The vector [WIDTH HEIGHT] must contain at least one finite element.'
        if I is not None:
            m,n = I.shape
            if not np.isfinite(width):
                width = n*height/m
            if not np.isfinite(height):
                height = m*width/n
    else:  # 3-D
        tmp = np.sum(np.isfinite(roidim))
        assert tmp==1 or tmp==3, 'The vector [WIDTH HEIGHT DEPTH] must contain one or three finite elements.'
        depth = roidim[2] 
        if I is not None:
            m,n,p = I.shape
            if not np.all(np.isfinite(roidim)): #% [WIDTH HEIGHT DEPTH] contains NaN or Inf
                if np.isfinite(height):
                    width = n*height/m
                    depth = p*height/m
                elif np.isfinite(width):
                    height = m*width/n
                    depth = p*width/n
                else:
                    width = n*depth/p
                    height = m*depth/p
        ymin = -depth/2
        ymax = depth/2

    
    xmin = -width/2
    xmax = width/2
    zmin = 0
    zmax = height


    #%-- Pseudorandom distribution of the scatterers
    if len(roidim)==2: #% 2-D
        #%-- 2-D pseudorandom distribution --
        
        xz_inc = meandist/np.sqrt(2/5)
        #% note: sqrt(2/5) was determined numerically (theory = ?...)
        
        xs,zs =np.meshgrid( np.arange(xmin, xmax, xz_inc), np.arange(zmin, zmax, xz_inc))
        xs = xs + np.random.rand(*xs.shape)*xz_inc-xz_inc/2
        zs = zs + np.random.rand(*zs.shape)*xz_inc-xz_inc/2
        
        idx = np.logical_and(np.logical_and(xs>xmin, xs<xmax), np.logical_and(zs>zmin, zs<zmax))
        xs = xs[idx]
        zs = zs[idx]
        
        ys = np.zeros(xs.shape)

    else: #% 3-D
        #%-- 3-D pseudorandom distribution --
        
        xyz_inc = meandist/np.sqrt(16/39)
        #% note: sqrt(16/39) was determined numerically (theory = ?...)
        
        xs,ys,zs = np.meshgrid(np.arange(xmin, xmax, xz_inc), 
                               np.arange(ymin, ymax, xz_inc),
                               np.arange(zmin, zmax, xz_inc))
        xs = xs + np.random.rand(*xs.shape)*xz_inc-xz_inc/2
        zs = zs + np.random.rand(*zs.shape)*xz_inc-xz_inc/2
        ys = ys + np.random.rand(*ys.shape)*xz_inc-xz_inc/2

        
        idx = np.logical_and.reduce((xs>xmin, xs<xmax, ys>ymin, ys<ymax, zs>zmin, zs<zmax))
        xs = xs[idx]
        ys = ys[idx]
        zs = zs[idx]
        


    #% Random reordering
    idx = np.random.permutation(len(xs))
    xs = xs[idx]
    ys = ys[idx]
    zs = zs[idx]


    #%-- If no I: the reflection coefficients follow a Rayleigh
    #%            distribution of mean 1
    if I is None:
        RC = np.random.rayleigh(1,xs.shape)/np.sqrt(np.pi/2)

    else:

        I = I.astype(float)
        I = I/np.max(I)


        #%-- Image grid
        if len(roidim)==2:
            #%-- 2-D image grid
            assert len(I.shape) == 2,'Number of dimensions of I is not consistent with the first input vector.'
            nl,nc = I.shape
            dxi = (xmax-xmin)/nc
            dzi = (zmax-zmin)/nl
            xi,zi = np.linspace(xmin-dxi/2,xmax-dxi/2,nc), np.linspace(zmin+dzi/2,zmax-dzi/2,nl)
            
        elif  len(roidim)==3:
            #%-- 3-D image grid    
            assert len(I.shape)==3, 'Number of dimensions of I is not consistent with the first input vector.'
            nl,nc,nr = I.shape
            dxi = (xmax-xmin)/nc
            dyi = (ymax-ymin)/nr
            dzi = (zmax-zmin)/nl
            xi,zi,yi = np.linspace(xmin+dxi/2,xmax-dxi/2,nc), \
                np.linspace(zmin+dzi/2,zmax-dzi/2,nl), \
                np.linspace(ymin+dyi/2,ymax-dyi/2,np)
        else:
            raise ValueError('Incorrect roidim size')


        #%-- Reflection coefficients calculated from the image
        if g is None:
            g = 40  #% default value for log compression
        
        if len(roidim)==2:
            interp =  scipy.interpolate.RegularGridInterpolator([xi,zi], I, method = 'linear', fill_value=0, bounds_error = False)
            RC = interp(np.stack([xs,zs], axis = -1))
        elif len(roidim)==3:
            interp =  scipy.interpolate.RegularGridInterpolator([xi, yi, zi], I, method = 'linear', fill_value=0, bounds_error = False)
            RC = interp(np.stack([xs,ys,zs], axis = -1))
        
        if g>1:
            #% log compression
            RC = np.power(10,(g/20*(RC-1)))
        else:
            #% gamma compression
            RC = np.power(RC, 1/g)

        #% add some randomness in the reflection coefficients
        #% RC = RC.*raylrnd(1,1,length(xs))'/sqrt(pi/2);
        RC = RC*np.hypot(np.random.rand(*xs.shape),np.random.rand(*xs.shape))/np.sqrt(np.pi/2)

    return xs,ys,zs,RC