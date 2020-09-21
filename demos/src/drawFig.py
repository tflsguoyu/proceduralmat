from util import *
from matplotlib.colors import LogNorm
plt.rc('font', family='serif')
plt.rc('text', usetex=True)

flag = True

# flag = False

if flag:

	fn = '../out/1_bump_syn0_direct_sample/sample.csv'
	fn_fig = '../out/1_bump_syn0_direct_sample/sample1.pdf'
	xs = np.loadtxt(fn, delimiter=',', usecols=(2, 3))
	xs = xs[:200,:]
	print(xs.shape)
	# exit()

	fig = plt.figure(figsize=(4,4))
	# print('i,j,k',i,j,k)
	plt.hist2d(xs[:,0], xs[:,1], bins=(40,10), norm=LogNorm())
	plt.plot(4, 0.15, 'r.', markersize=20, label='Random init.')
	plt.xlim(1.3,4.1); plt.ylim(0.03,0.2)
	plt.yticks(np.arange(0.03, 0.2, step=0.07))
	plt.xlabel('$\sigma_f$', fontSize=24)
	plt.ylabel('$c_f$', fontSize=24)
	plt.legend(prop={'size': 16})
	# plt.title('Accept %d, reject %d, time %ds' % (len(xs), num_reject, time))
	plt.tight_layout()
	plt.savefig(fn_fig)
	plt.close()

# else:
	fn = '../out/1_bump_syn0_hu_sample/sample.csv'
	fn_fig = '../out/1_bump_syn0_hu_sample/sample2.pdf'
	xs = np.loadtxt(fn, delimiter=',', usecols=(2, 3))
	xs = xs[:200,:]
	print(xs.shape)
	# exit()

	fig = plt.figure(figsize=(4,4))
	# print('i,j,k',i,j,k)
	plt.hist2d(xs[:,0], xs[:,1], bins=15, norm=LogNorm())
	plt.plot(2.0553, 0.1138, 'r.', markersize=20, label='Init. from [HDR19]')
	plt.xlim(1.3,4.1); plt.ylim(0.03,0.2)
	plt.yticks(np.arange(0.03, 0.2, step=0.07))
	plt.xlabel('$\sigma_f$', fontSize=24)
	plt.ylabel('$c_f$', fontSize=24)
	plt.legend(prop={'size': 16})
	# plt.title('Accept %d, reject %d, time %ds' % (len(xs), num_reject, time))
	plt.tight_layout()
	plt.savefig(fn_fig)
	plt.close()