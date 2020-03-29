import OpenEXR, Imath, numpy


def info(filename):
    img = OpenEXR.InputFile(filename)
    for k, v in img.header().items():
        print(k, v)

def read(filename):
    img = OpenEXR.InputFile(filename)
    header = img.header()
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    
    def chan(c):
        s = img.channel(c, pt)
        arr = numpy.fromstring(s, dtype=numpy.float32)
        arr.shape = size[1], size[0]
        return arr
    
    # single-channel file
    channels = header['channels'].keys()
    if len(channels) == 1:
        chan_name = list(channels)[0]
        print("[EXR]: Read from a single channel ('%s') exr file." % chan_name)
        return chan(chan_name)

    # standard RGB file
    print("[EXR]: Read from a 3 channels ('R','G','B') exr file.")
    return numpy.dstack([chan('R'), chan('G'), chan('B')])

def write(img, filename):
    assert img.dtype == numpy.float32
    assert len(img.shape) == 2 or len(img.shape) == 3
    if len(img.shape) == 2:
        h, w = img.shape
    else:
        h, w, d = img.shape
        assert d == 3

    header = OpenEXR.Header(w, h)
    header['compression'] = Imath.Compression(Imath.Compression.PIZ_COMPRESSION)
    if len(img.shape) == 2:
        float_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        header['channels'] = dict([('Y',float_chan)])      
    out = OpenEXR.OutputFile(filename, header)
        
    if len(img.shape) == 2:
        y = img.tostring()
        out.writePixels({'Y': y})
        print("[EXR]: Saved  to a single channel ('Y') exr file.")
    else:
        r = img[:,:,0].tostring()
        g = img[:,:,1].tostring()
        b = img[:,:,2].tostring()
        out.writePixels({'R': r, 'G': g, 'B': b})
        print("[EXR]: Saved  to a 3 channels ('R','G','B') exr file.")

def write16(img, filename):
    assert img.dtype == numpy.float32
    assert len(img.shape) == 2 or len(img.shape) == 3
    if len(img.shape) == 2:
        h, w = img.shape
    else:
        h, w, d = img.shape
        assert d == 3

    header = OpenEXR.Header(w, h)
    header['compression'] = Imath.Compression(Imath.Compression.PIZ_COMPRESSION)
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))
    if len(img.shape) == 2:
        header['channels'] = dict([('Y',half_chan)])   
    else:
        header['channels'] = dict([(c, half_chan) for c in "RGB"])
    out = OpenEXR.OutputFile(filename, header)
   
    if len(img.shape) == 2:
        y = img.tostring()
        out.writePixels({'Y': y})
        print("[EXR]: Saved  to single channel ('Y') exr file.")
    else:
        r = img[:,:,0].tostring()
        g = img[:,:,1].tostring()
        b = img[:,:,2].tostring()
        out.writePixels({'R': r, 'G': g, 'B': b})
        print("[EXR]: Saved  to 3 channels ('R','G','B') exr file.")
