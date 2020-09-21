from util import *

fn = 'leather_4'
im = Image.open(fn+'/tex.png')

res = 256*4

albedo = im.crop((res*0,0,res*1,res)).resize((256,256)).rotate(90)
normal = im.crop((res*1,0,res*2,res)).resize((256,256)).rotate(90)
rough = im.crop((res*2,0,res*3,res)).resize((256,256)).rotate(90)
specular = im.crop((res*3,0,res*4,res)).resize((256,256)).rotate(90)

tex = gyConcatPIL_v(gyConcatPIL_h(albedo,normal), gyConcatPIL_h(rough,specular))
tex.save(fn+'/tex2x2.png')

# Image.open(fn+'/00.png').rotate(90).save(fn+'/00.png')