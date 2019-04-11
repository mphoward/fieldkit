""" Routines for reading files into NumPy arrays.

"""
from __future__ import division
import numpy as np
import struct

def read_polyfts(filename):
    """ Read the binary field file produced by PolyFTS.

    The returned coordinates and fields can be used to create a
    :py:class:`~fieldkit.mesh.Mesh` and :py:class:`~fieldkit.mesh.Field`
    using their `from_array` constructors.

    Parameters
    ----------
    filename : str
        File to parse.

    Returns
    -------
    coords : ndarray
        Coordinates for the mesh.
    fields : ndarray
        Fields on the mesh.

    Notes
    -----
    This function is based on code from `iotools` shared by Josh
    Lequieu (UCSB).

    Todo
    ----
    This function needs testing, using as supplied.

    """
    # read binary file contents
    f = open(filename,"rb")
    contents = f.read()
    f.close()

    # check header and file version number
    header = struct.unpack_from("@8s",contents)
    if header[0] != b"FieldBin":
        raise RuntimeError("Not an unformatted Field file")
    pos = 9
    version = struct.unpack_from("@I",contents,offset=pos)
    pos = pos + 4
    if version[0] != 51:
        raise RuntimeError("Only version 51 is currently supported")

    # number of fields
    nfields = struct.unpack_from("@I",contents,offset=pos)[0]
    pos = pos + 4

    # number of dimensions
    ndim = struct.unpack_from("@I",contents,offset=pos)[0]
    pos = pos + 4
    if ndim < 1 or ndim > 3:
        raise RuntimeError("Only 1, 2, or 3D is supported")

    # mesh size
    Nx = struct.unpack_from("@{}L".format(ndim),contents,offset=pos)
    pos = pos + 8*ndim
    M = np.prod(Nx)

    # check for format of data
    (kspacedata,complexdata) = struct.unpack_from("@2?",contents,offset=pos)
    pos = pos + 2
    if kspacedata:
        raise RuntimeError("k-space data is not supported")

    # cell matrix
    harray = struct.unpack_from("@{}d".format(ndim*ndim),contents,offset=pos)
    pos = pos + 8*ndim*ndim
    h = np.reshape(harray,(ndim,ndim))

    # element size
    elsize = struct.unpack_from("@L",contents,offset=pos)[0]
    pos = pos + 8
    if elsize == 4 and not complexdata:
        fielddata = struct.unpack_from("@{}f".format(M*nfields),contents,offset=pos)
    elif elsize == 8 and complexdata:
        fielddata = struct.unpack_from("@{}f".format(2*M*nfields),contents,offset=pos)
    elif elsize == 8 and not complexdata:
        fielddata = struct.unpack_from("@{}d".format(M*nfields),contents,offset=pos)
    elif elsize == 16 and complexdata:
        fielddata = struct.unpack_from("@{}d".format(2*M*nfields),contents,offset=pos)
    else:
        raise RuntimeError("Unknown element size")

    # Now we need to return a numpy array with two indices:
    #  1 = fieldidx (with imaginary part as a distinct field index)
    #  2 = PW indx
    # For complex fields, currently the real/imaginary part is the fastest index,
    # so we will have to do a selective transpose
    # before reshaping the re/im sequence into the field indices
    if complexdata:
        fields = np.array(fielddata).reshape([nfields,M,2]).transpose((0,2,1)).reshape([nfields*2,M])
        nfields = nfields*2
    else:
        fields = np.array(fielddata).reshape([nfields,M])

    # calculate coordinates of mesh points
    coords = np.zeros((M,ndim))
    loop_size = np.ones(3,dtype=int)
    loop_size[:ndim] = Nx
    m = 0
    for ix in range(0,loop_size[0]):
        for iy in range(0,loop_size[1]):
            for iz in range(0,loop_size[2]):
                xfrac = np.array([ix,iy,iz],dtype=float)
                xfrac = xfrac[:ndim] / Nx

                xcart = [0.]*ndim
                for i in range(ndim):
                    for j in range(ndim):
                        xcart[j] += h[i][j]*xfrac[i]
                coords[m,:] = xcart
                m += 1

    coords = np.reshape(coords ,list(Nx) + [ndim])
    fields = np.reshape(fields.T ,list(Nx) + [nfields])
    return coords,fields
