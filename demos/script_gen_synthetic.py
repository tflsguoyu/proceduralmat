import sys
sys.path.insert(1, 'src/')
import render as renpt
from util import *

def generateLightCameraPosition(p, angle, colocated=True, addNoise=True):
    theta = (np.pi/180 * np.array([0] + [angle]*8)).astype(np.float32)
    phi   = (np.pi/4   * np.array([0,1,5,3,7,2,6,4,8])).astype(np.float32)

    light_pos = np.stack((p * np.sin(theta) * np.cos(phi),
                          p * np.sin(theta) * np.sin(phi),
                          p * np.cos(theta))).transpose()
    if addNoise:
        light_pos[:,0:2] += np.random.randn(9,2).astype(np.float32)

    if colocated:
        camera_pos = light_pos.copy()
    else:
        camera_pos = np.array([[0,0,p]]).astype(np.float32).repeat(9,axis=0)
        if addNoise:
            camera_pos[:,0:2] += np.random.randn(9,2).astype(np.float32)

    return light_pos, camera_pos

def render_image_pytorch(in_dir, out_dir, res, size, lp, cp, light):

    im = None
    for i in range(1):
        im_this = renpt.renderTex(in_dir+'tex.png', res, size, lp[i,:], cp[i,:], light, fn_im=out_dir + '%02d.png' % i)
        # gyCreateThumbnail(out_dir + '%02d.png' % i)
        im = gyConcatPIL_h(im, im_this)
    im.save(out_dir + 'rendered.png')
    # gyCreateThumbnail(out_dir + 'rendered.png', w=128*9, h=128)

def render_image_tensorflow(in_dir, out_dir, res, size, lp, cp, light):

    im = None
    for i in range(lp.shape[0]):
        im_this = rentf.renderTex(in_dir+'tex.png', res, size, lp[i,:], cp[i,:], light, fn_im=None)
        im = gyConcatPIL_h(im, im_this)
    im.save(out_dir + 'rendered_tf.png')


if __name__ == '__main__':
    in_dir = 'attila/leather_4/'
    out_dir_this = 'attila/leather_4/'
    res = 256
    size = 25
    lp, cp = generateLightCameraPosition(15, 20, True, False)
    light = np.array([1000,1000,1000]).astype(np.float32)
    np.savetxt(out_dir_this+'light_pos.txt',   lp,   fmt='%.4f', delimiter=',')
    np.savetxt(out_dir_this+'camera_pos.txt',  cp,   fmt='%.4f', delimiter=',')
    np.savetxt(out_dir_this+'image_size.txt',  np.array([size]).astype(np.float32), fmt='%.4f', delimiter=',')
    np.savetxt(out_dir_this+'light_power.txt', light.reshape([1,3]),fmt='%.4f', delimiter=',')
    render_image_pytorch(in_dir, out_dir_this, res, size, lp, cp, light)
