import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




def drawChairs(chair, chair2):
	"draw chair"
	a = chair[:,0];
	b = chair[:,1];
	c = chair[:,2];
	d = chair[:,3];
	e = chair[:,4];
	f = chair[:,5];
	g = chair[:,6];
	h = chair[:,7];
	i = chair[:,8];
	j = chair[:,9];

	ax = plt.gca(projection="3d")
	#ax.axis([-200,200,-200,200])
	ax.set_xlim3d(-200, 200)
	ax.set_ylim3d(-200, 200)
	ax.set_zlim3d(-200, 200)
	ax.plot([h[0],i[0],j[0],g[0],h[0]],[h[1],i[1],j[1],g[1],h[1]],[h[2],i[2],j[2],g[2],h[2]], color='r')
	ax.plot([h[0],e[0],f[0],g[0]],[h[1],e[1],f[1],g[1]],[h[2],e[2],f[2],g[2]], color='r')
	ax.plot([d[0],h[0]],[d[1],h[1]],[d[2],h[2]], color='r')
	ax.plot([a[0],e[0]],[a[1],e[1]],[a[2],e[2]], color='r')
	ax.plot([c[0],g[0]],[c[1],g[1]],[c[2],g[2]], color='r')
	ax.plot([b[0],f[0]],[b[1],f[1]],[b[2],f[2]], color='r')

	a = chair2[:,0];
	b = chair2[:,1];
	c = chair2[:,2];
	d = chair2[:,3];
	e = chair2[:,4];
	f = chair2[:,5];
	g = chair2[:,6];
	h = chair2[:,7];
	i = chair2[:,8];
	j = chair2[:,9];

	ax.plot([h[0],i[0],j[0],g[0],h[0]],[h[1],i[1],j[1],g[1],h[1]],[h[2],i[2],j[2],g[2],h[2]], color='g')
	ax.plot([h[0],e[0],f[0],g[0]],[h[1],e[1],f[1],g[1]],[h[2],e[2],f[2],g[2]], color='g')
	ax.plot([d[0],h[0]],[d[1],h[1]],[d[2],h[2]], color='g')
	ax.plot([a[0],e[0]],[a[1],e[1]],[a[2],e[2]], color='g')
	ax.plot([c[0],g[0]],[c[1],g[1]],[c[2],g[2]], color='g')
	ax.plot([b[0],f[0]],[b[1],f[1]],[b[2],f[2]], color='g')
	plt.show()
	return;


def drawChair(chair):
	"draw chair"
	a = chair[:,0];
	b = chair[:,1];
	c = chair[:,2];
	d = chair[:,3];
	e = chair[:,4];
	f = chair[:,5];
	g = chair[:,6];
	h = chair[:,7];
	i = chair[:,8];
	j = chair[:,9];

	ax = plt.gca(projection="3d")
	#ax.axis([-200,200,-200,200])
	ax.set_xlim3d(-200, 200)
	ax.set_ylim3d(-200, 200)
	ax.set_zlim3d(-200, 200)

	ax.plot([h[0],i[0],j[0],g[0],h[0]],[h[1],i[1],j[1],g[1],h[1]],[h[2],i[2],j[2],g[2],h[2]], color='r')
	ax.plot([h[0],e[0],f[0],g[0]],[h[1],e[1],f[1],g[1]],[h[2],e[2],f[2],g[2]], color='b')
	ax.plot([d[0],h[0]],[d[1],h[1]],[d[2],h[2]], color='g')
	ax.plot([a[0],e[0]],[a[1],e[1]],[a[2],e[2]], color='y')
	ax.plot([c[0],g[0]],[c[1],g[1]],[c[2],g[2]], color='c')
	ax.plot([b[0],f[0]],[b[1],f[1]],[b[2],f[2]], color='m')



	plt.show()
	return;
'''
def drawProj(chair):
	"draw chair"

	plt.axis([-1000,1000,-1000,1000])
	plt.plot(chair[0,:],chair[1,:], color='r')

	plt.show()
	return;'''


def drawProj(chair):
	"draw chair"
	a = chair[:,0];
	b = chair[:,1];
	c = chair[:,2];
	d = chair[:,3];
	e = chair[:,4];
	f = chair[:,5];
	g = chair[:,6];
	h = chair[:,7];
	i = chair[:,8];
	j = chair[:,9];

	#ax = plt.gca(projection="2d")
	plt.axis([-1000,1000,-1000,1000])
	plt.plot([h[0],i[0],j[0],g[0],h[0]],[h[1],i[1],j[1],g[1],h[1]], color='r')
	plt.plot([h[0],e[0],f[0],g[0]],[h[1],e[1],f[1],g[1]], color='g')
	plt.plot([d[0],h[0]],[d[1],h[1]], color='b')
	plt.plot([a[0],e[0]],[a[1],e[1]], color='c')
	plt.plot([c[0],g[0]],[c[1],g[1]], color='m')
	plt.plot([b[0],f[0]],[b[1],f[1]], color='y')
	plt.show()
	return;

