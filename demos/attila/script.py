from util import *

fn = 'leather_4'
im = Image.open(fn+'.png')
N = 4

albedo_tile_tmp = None
albedo_tile = None
albedo = im.crop((32,80,32+256,80+256))
for i in range(N):
	albedo_tile_tmp = gyConcatPIL_h(albedo_tile_tmp,albedo)
for i in range(N):
	albedo_tile = gyConcatPIL_v(albedo_tile,albedo_tile_tmp)
# albedo_tile.save(fn+'_albedo.png')
# exit()
# albedo_tile = gyApplyGammaPIL(albedo_tile, 1/2.2)

normal_tile_tmp = None
normal_tile = None
normal = im.crop((320,400,320+256,400+256))
for i in range(N):
	normal_tile_tmp = gyConcatPIL_h(normal_tile_tmp,normal)
for i in range(N):
	normal_tile = gyConcatPIL_v(normal_tile,normal_tile_tmp)
# normal_tile = gyPIL2Array(normal_tile)*2-1
# normal_tile = normal_tile * 0.5
# normal_tile = gyArray2PIL((normal_tile+1)/2)


rough_tile_tmp = None
rough_tile = None
rough = im.crop((32,400,32+256,400+256))
for i in range(N):
	rough_tile_tmp = gyConcatPIL_h(rough_tile_tmp,rough)
for i in range(N):
	rough_tile = gyConcatPIL_v(rough_tile,rough_tile_tmp)
rough_tile = gyArray2PIL(1-gyPIL2Array(rough_tile))
# rough_tile = gyApplyGammaPIL(rough_tile, 1/2.2)

specular_tile_tmp = None
specular_tile = None
specular = im.crop((320,80,320+256,80+256))
for i in range(N):
	specular_tile_tmp = gyConcatPIL_h(specular_tile_tmp,specular)
for i in range(N):
	specular_tile = gyConcatPIL_v(specular_tile,specular_tile_tmp)
# specular_tile = gyApplyGammaPIL(specular_tile, 1/2.2)

tex = gyConcatPIL_h(gyConcatPIL_h(gyConcatPIL_h(albedo_tile,normal_tile),rough_tile),specular_tile)
tex.save(fn+'/tex.png')
