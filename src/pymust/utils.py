import numpy as np, scipy, scipy.interpolate, multiprocessing, multiprocessing.pool
from abc import ABC
import inspect, matplotlib, pickle, os, matplotlib.pyplot as plt, copy
from collections import deque
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import plotly.graph_objects as go

BackendName = Literal["numpy", "jax"]


class dotdict(dict, ABC):
    """Copied from https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary"""
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    def ignoreCaseInFieldNames(self):
        """Convert all field names to lower case"""
        names = self.names
        todelete =[]
        for k, v in self.items():
            if k.lower() in names and k in names:
                if k.lower() == k:
                    continue
                elif names[k] in self:
                    raise ValueError(f'Repeated key {k}')
                else:
                    self[names[k]] = v
                    todelete.append(k)
        for k in todelete:
            del self[k]
        return self
    def copy(self):
        return copy.deepcopy(self)
    def __getstate__(self):
        d = {k : v for k,v in self.items()}
        return d
    def __setstate__(self, d):
        for k, v in self.items():
            self[k] = v
        
class Options(dotdict):
    default_Number_Workers = multiprocessing.cpu_count()
    @property 
    def names(self):
        names = {'dBThresh','ElementSplitting',
                'FullFrequencyDirectivity','FrequencyStep','ParPool',
                'WaitBar', 'Backend'}
        return {n.lower(): n for n in names}
    
    def setParPool(self, workers, mode = 'process'):
        if mode not in ['process', 'thread']:
            raise ValueError('ParPoolMode must be either "process" or "thread"')
        self.ParPool_NumWorkers = workers
        self.ParPoolMode = mode
    
    def getParallelPool(self):
        workers = self.get('ParPool_NumWorkers', self.default_Number_Workers)
        mode = self.get('ParPoolMode', 'thread')
        if mode == 'process':
            pool = multiprocessing.Pool(workers)
        elif mode == 'thread':
            pool = multiprocessing.pool.ThreadPool(workers)
        else:
             raise ValueError('ParPoolMode must be either "process" or "thread"')
        return pool
    
    def getParallelSplitIndices(self, N,n_threads = None):
        if hasattr(N, '__len__'):
            N = len(N)
        assert isinstance(N, int), 'N must be an integer'

        n_threads = self.get('ParPool_NumWorkers', self.default_Number_Workers) if n_threads is None else n_threads
        #Create indices for parallel processing, split in workers
        idx = np.arange(0, N, N//n_threads)

        #Repeat along new axis
        idx = np.stack([idx, np.roll(idx, -1)], axis = 1)
        idx[-1, 1] = N
        return idx

class Param(dotdict):
    @property 
    def names(self):
        names = {'attenuation','baffle','bandwidth','c','fc',
            'fnumber','focus','fs','height','kerf','movie','Nelements',
            'passive','pitch','radius','RXangle','RXdelay'
            'TXapodization','TXfreqsweep','TXnow','t0','width'}
        return {n.lower(): n for n in names}
    
    def read_elements(self) -> np.ndarray:
        """PARAM.elements as a (rows, E) array of coordinates, copied and never written back."""
        elements = np.atleast_2d(np.array(self.elements, dtype = float))
        if elements.shape[0] not in (1, 2, 3):
            elements = elements.T
        assert elements.shape[0] in (1, 2, 3), 'PARAM.elements must have 1, 2 or 3 rows (or columns) of coordinates.'
        assert self.radius is None or np.isinf(self.radius), \
            'PARAM.radius must be Inf (or absent) when PARAM.elements is given.'
        assert self.Nelements is None or self.Nelements == elements.shape[1], \
            'PARAM.Nelements must match the number of elements in PARAM.elements.'
        return elements

    def getElementPositions(self):
        """
        Returns the (x, z) position, normal angle and apothem of each element, all of size (1, E).

        If PARAM.elements is given, it is used as is: one row (x), two rows (x, z) or
        three rows (x, y, z) with y = 0; the element normals point along +z.
        Otherwise the positions of the uniform linear/convex array are computed from
        PARAM.pitch, PARAM.Nelements and PARAM.radius.
        """
        if self.elements is not None:
            elements = self.read_elements()
            assert elements.shape[0] < 3 or not elements[1].any(), \
                'Elements off the y = 0 plane need the 3-D functions (PFIELD3, SIMUS3, TXDELAY3, DASMTX3).'
            xe = elements[0]
            ze = elements[-1] if elements.shape[0] > 1 else np.zeros_like(xe)
            THe = np.zeros_like(xe)
            return xe.reshape((1,-1)), ze.reshape((1,-1)), THe.reshape((1,-1)), np.zeros_like(THe).reshape((1,-1))

        assert self.pitch is not None and self.Nelements is not None, \
            'PARAM.pitch and PARAM.Nelements (or PARAM.elements) are required.'
        RadiusOfCurvature = np.inf if self.radius is None else self.radius # default = linear array
        NumberOfElements = self.Nelements

        if np.isinf(RadiusOfCurvature):
            #% Linear array
            xe =  (np.arange(NumberOfElements)-(NumberOfElements-1)/2)*self.pitch
            ze = np.zeros((1,NumberOfElements))
            THe = np.zeros_like(ze)
            h = np.zeros_like(ze)
        else:
            #% Convex array
            chord = 2*RadiusOfCurvature*np.sin(np.arcsin(self.pitch/2/RadiusOfCurvature)*(NumberOfElements-1))
            h = np.sqrt(RadiusOfCurvature**2-chord**2/4); #% apothem
            #% https://en.wikipedia.org/wiki/Circular_segment
            #% THe = angle of the normal to element #e with respect to the z-axis
            THe = np.linspace(np.arctan2(-chord/2,h),np.arctan2(chord/2,h),NumberOfElements)
            ze = RadiusOfCurvature*np.cos(THe)
            xe = RadiusOfCurvature*np.sin(THe)
            ze = ze-h
        return xe.reshape((1,-1)), ze.reshape((1,-1)), THe.reshape((1,-1)), h.reshape((1,-1))

    def get_element_coordinates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the (x, y, z) centres of the elements for the 3-D functions, each of size (E,).

        PARAM.elements is used as is: one row (x), two rows (x, y) with z = 0, or three
        rows (x, y, z). Without PARAM.elements, a uniform linear array lies along x.
        """
        if self.elements is None:
            assert self.radius is None or np.isinf(self.radius), \
                'Convex arrays are not supported by the 3-D functions: give the element positions in PARAM.elements.'
            element_x, element_z, _, _ = self.getElementPositions()
            return element_x.ravel(), np.zeros(element_x.size), element_z.ravel()
        elements = self.read_elements()
        zeros = np.zeros(elements.shape[1])
        element_y = elements[1] if elements.shape[0] > 1 else zeros
        element_z = elements[2] if elements.shape[0] > 2 else zeros
        return elements[0], element_y, element_z

    def getPulseSpectrumFunction(self, FreqSweep = None):
        if 'TXnow' not in self:
            self.TXnow = 1

        #-- FREQUENCY SPECTRUM of the transmitted pulse
        if FreqSweep is None:
            # We want a windowed sine of width PARAM.TXnow
            T = self.TXnow /self.fc
            wc = 2 * np.pi * self.fc
            pulseSpectrum = lambda w = None: 1j * (mysinc(T * (w - wc) / 2) - mysinc(T * (w + wc) / 2))
        else:
            # We want a linear chirp of width PARAM.TXnow
            # (https://en.wikipedia.org/wiki/Chirp_spectrum#Linear_chirp)
            T = self.TXnow / self.fc
            wc = 2 * np.pi * self.fc
            dw = 2 * np.pi * FreqSweep
            s2 = lambda w = None: np.multiply(np.sqrt(np.pi * T / dw) * np.exp(- 1j * (w - wc) ** 2 * T / 2 / dw),(fresnelint((dw / 2 + w - wc) / np.sqrt(np.pi * dw / T)) + fresnelint((dw / 2 - w + wc) / np.sqrt(np.pi * dw / T))))
            pulseSpectrum = lambda w = None: (1j * s2(w) - 1j * s2(- w)) / T
        return pulseSpectrum

    def getProbeFunction(self):
        #%-- FREQUENCY RESPONSE of the ensemble PZT + probe
        #% We want a generalized normal window (6dB-bandwidth = PARAM.bandwidth)
        #% (https://en.wikipedia.org/wiki/Window_function#Generalized_normal_window)
        #-- FREQUENCY RESPONSE of the ensemble PZT + probe
        # We want a generalized normal window (6dB-bandwidth = PARAM.bandwidth)
        # (https://en.wikipedia.org/wiki/Window_function#Generalized_normal_window)
        wc = 2 * np.pi * self.fc
        wB = self.bandwidth * wc / 100
        p = np.log(126) / np.log(2 * wc / wB)
        probeSpectrum_sqr = lambda w: np.exp(- np.power(np.abs(w - wc) / (wB / 2 / np.power(np.log(2), 1 / p)), p))
        # The frequency response is a pulse-echo (transmit + receive) response. A
        # square root is thus required when calculating the pressure field:
        probeSpectrum = lambda w: np.sqrt(probeSpectrum_sqr(w))
        return probeSpectrum
    
# To maintain same notation as matlab
def interp1(y, xNew, kind):
    if kind == 'spline':
        kind = 'cubic' #3rd order spline
    interpolator = scipy.interpolate.interp1d(np.arange(len(y)), y, kind = kind) 
    return interpolator(xNew)    

def isnumeric(x):
    return isinstance(x, np.ndarray) or isinstance(x, int) or isinstance(x, float) or isinstance(x, np.number)

def iscomplex(x):
    return (isinstance(x, np.ndarray) and np.iscomplexobj(x)) or isinstance(x, complex)

def islogical(v):
    return isinstance(v, bool)

def isfield(d, k ):
    return k in d

mysinc = lambda x = None: np.sinc(x / np.pi) # [note: In MATLAB/numpy, sinc is sin(pi*x)/(pi*x)]


def shiftdim(array, n=None):
    """
    From stack overflow https://stackoverflow.com/questions/67584148/python-equivalent-of-matlab-shiftdim
    """
    if n is not None:
        if n >= 0:
            axes = tuple(range(len(array.shape)))
            new_axes = deque(axes)
            new_axes.rotate(n)
            return np.moveaxis(array, axes, tuple(new_axes))
        return np.expand_dims(array, axis=tuple(range(-n)))
    else:
        idx = 0
        for dim in array.shape:
            if dim == 1:
                idx += 1
            else:
                break
        axes = tuple(range(idx))
        # Note that this returns a tuple of 2 results
        return np.squeeze(array, axis=axes), len(axes)

def isEmpty(x):
    return  x is None or (isinstance(x, list) and len(x) == 0) or (isinstance(x, np.ndarray) and len(x) == 0)

def emptyArrayIfNone(x):
    if isEmpty(x):
        x =  np.array([])
    return x

def eps(s = 'single'):
    if s == 'single':
        return 1.1921e-07 
    else:
        raise ValueError()

def nextpow2(n):
    i = 1
    while (1 << i) < n:
        i += 1
    return i

def fresnelint(x): 
    # FRESNELINT Fresnel integral.
    
    # J = FRESNELINT(X) returns the Fresnel integral J = C + 1i*S.
    
    # We use the approximation introduced by Mielenz in
#       Klaus D. Mielenz, Computation of Fresnel Integrals. II
#       J. Res. Natl. Inst. Stand. Technol. 105, 589 (2000), pp 589-590
    
    siz0 = x.shape
    x = x.flatten()

    issmall = np.abs(x) <= 1.6
    c = np.zeros(x.shape)
    s = np.zeros(x.shape)
    # When |x| < 1.6, a Taylor series is used (see Mielenz's paper)
    if np.any(issmall):
        n = np.arange(0,11)
        cn = np.concatenate([[1], np.cumprod(- np.pi ** 2 * (4 * n + 1) / (4 * (2 * n + 1) *(2 * n + 2)*(4 * n + 5)))])
        sn = np.concatenate([[1],np.cumprod(- np.pi ** 2 * (4 * n + 3) / (4 * (2 * n + 2)*(2 * n + 3)*(4 * n + 7)))]) * np.pi / 6
        n = np.concatenate([n,[11]]).reshape((1,-1))
        c[issmall] = np.sum(cn.reshape((1,-1))*x[issmall].reshape((-1, 1))  ** (4 * n + 1), 1)
        s[issmall] = np.sum(sn.reshape((1,-1))*x[issmall].reshape((-1, 1)) ** (4 * n + 3), 1)
    
    # When |x| > 1.6, we use the following:
    if not np.all(issmall ):
        n = np.arange(0,11+1)
        fn = np.array([0.318309844,9.34626e-08,- 0.09676631,0.000606222,0.325539361,0.325206461,- 7.450551455,32.20380908,- 78.8035274,118.5343352,- 102.4339798,39.06207702])
        fn = fn.reshape((1, fn.shape[0]))
        gn = np.array([0,0.101321519,- 4.07292e-05,- 0.152068115,- 0.046292605,1.622793598,- 5.199186089,7.477942354,- 0.695291507,- 15.10996796,22.28401942,- 10.89968491])
        gn = gn.reshape((1, gn.shape[0]))

        fx = np.sum(np.multiply(fn,x[not issmall ] ** (- 2 * n - 1)), 1)
        gx = np.sum(np.multiply(gn,x[not issmall ] ** (- 2 * n - 1)), 1)
        c[not issmall ] = 0.5 * np.sign(x[not issmall ]) + np.multiply(fx,np.sin(np.pi / 2 * x[not issmall ] ** 2)) - np.multiply(gx,np.cos(np.pi / 2 * x[not issmall ] ** 2))
        s[not issmall ] = 0.5 * np.sign(x[not issmall ]) - np.multiply(fx,np.cos(np.pi / 2 * x[not issmall ] ** 2)) - np.multiply(gx,np.sin(np.pi / 2 * x[not issmall ] ** 2))
    
    f = np.reshape(c, siz0) + 1j * np.reshape(s, siz0)
    return f


# Plotting
def polarplot(x, z, v, cmap = 'gray',background = 'black', probeUpward = True, **kwargs):
    plt.pcolormesh(x, z, v, cmap = cmap, shading='gouraud', **kwargs)
    plt.axis('equal')
    ax = plt.gca()
    ax.set_facecolor(background)
    if probeUpward:
        ax.invert_yaxis()


DEFAULT_ELEMENT_THICKNESS_M = 500e-6
MM_PER_M = 1e3
# Unit box with corner index = ix + 2*iy + 4*iz: x and y centred on the element,
# z from the back (-1) to the front face (0), which lies at the element position
UNIT_BOX_CORNERS = np.array([[ix-0.5, iy-0.5, iz-1] for iz in (0, 1) for iy in (0, 1) for ix in (0, 1)])
UNIT_BOX_TRIANGLES = np.array([
    [0, 1, 3], [0, 3, 2], [4, 5, 7], [4, 7, 6],  # back, front
    [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],  # y sides
    [0, 2, 6], [0, 6, 4], [1, 3, 7], [1, 7, 5],  # x sides
])


def plot_transducer_array(param: Param, thickness: float = DEFAULT_ELEMENT_THICKNESS_M) -> "go.Figure":
    """
    Interactive 3-D view of the transducer elements, drawn as boxes of size
    width (x) x height (y) x thickness (z) in mm.

    Each box's front face is centred on the element position and the box extends
    behind it, along the element normal (tilted for convex arrays). The positions
    are those used by the simulation and beamforming functions: PARAM.elements,
    read as (x, y) or (x, y, z) rows as in PFIELD3, or else the uniform
    linear/convex array. If PARAM.height is missing or infinite, the boxes are
    square (height = width).

    Requires plotly (pip install pymust[plot]); the returned figure renders
    interactively as a Jupyter cell output.
    """
    import plotly.graph_objects as go

    if param.elements is None:
        element_x, element_z, element_tilt, _ = (positions.ravel() for positions in param.getElementPositions())
        element_y = np.zeros_like(element_x)
    else:
        element_x, element_y, element_z = param.get_element_coordinates()
        element_tilt = np.zeros_like(element_x)

    element_width = param.width if param.width is not None else param.pitch - param.kerf
    element_height = param.height if param.height is not None and np.isfinite(param.height) else element_width
    box_corners = UNIT_BOX_CORNERS * np.array([element_width, element_height, thickness])

    # Element frame: in-plane axis (cos, 0, -sin) and normal (sin, 0, cos), as in PFIELD
    zeros = np.zeros_like(element_tilt)
    in_plane_axis = np.stack([np.cos(element_tilt), zeros, -np.sin(element_tilt)], axis = 1)
    normal_axis = np.stack([np.sin(element_tilt), zeros, np.cos(element_tilt)], axis = 1)
    elevation_axis = np.array([0.0, 1.0, 0.0])
    centers = np.stack([element_x, element_y, element_z], axis = 1)
    vertices = (centers[:, None, :]
                + box_corners[None, :, 0, None] * in_plane_axis[:, None, :]
                + box_corners[None, :, 1, None] * elevation_axis
                + box_corners[None, :, 2, None] * normal_axis[:, None, :]).reshape((-1, 3)) * MM_PER_M
    box_offsets = len(UNIT_BOX_CORNERS) * np.arange(len(centers))[:, None, None]
    triangles = (UNIT_BOX_TRIANGLES[None] + box_offsets).reshape((-1, 3))

    element_index = np.repeat(np.arange(len(centers)), len(UNIT_BOX_CORNERS))
    figure = go.Figure(go.Mesh3d(
        x = vertices[:, 0], y = vertices[:, 1], z = vertices[:, 2],
        i = triangles[:, 0], j = triangles[:, 1], k = triangles[:, 2],
        hovertext = [f'element {index}' for index in element_index], hoverinfo = 'text',
        flatshading = True, color = 'lightsteelblue',
    ))
    figure.update_layout(scene = dict(
        xaxis_title = 'x (mm)', yaxis_title = 'y (mm)', zaxis_title = 'z (mm)',
        zaxis_autorange = 'reversed', aspectmode = 'data',
    ))
    return figure


def getDopplerColorMap():
    source_file_path = inspect.getfile(inspect.currentframe())
    with open( os.path.join(os.path.dirname(source_file_path), 'Data', 'colorMap.pkl'), 'rb') as f:
        dMap = pickle.load(f)
    new_cmap = matplotlib.colors.LinearSegmentedColormap('doppler', dMap)
    dopplerCM = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(),cmap=new_cmap)
    return dopplerCM

def get_array_module(backend: BackendName = "numpy"):
    if backend == "numpy":
        return np
    if backend == "jax":
        import jax.numpy as jnp
        return jnp
    raise ValueError(f'Unknown backend "{backend}": use "numpy" or "jax".')

def applyDasMTX(M, IQ, imageShape, backend: BackendName = "numpy"):
    signal = IQ.flatten(order = 'F')
    if get_array_module(backend) is np:
        beamformed = M @ signal
    else:
        from . import jax_backend
        beamformed = jax_backend.apply_das_matrix(M, signal)
    return beamformed.reshape(imageShape, order = 'F')
