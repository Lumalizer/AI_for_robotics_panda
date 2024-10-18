
import numpy as np
import matplotlib.pyplot as plt

a = np.load('episode_1.npy', allow_pickle=True)


plt.plot( np.diff(np.array([ k['franka_pose'][:,3][:3] for k in a]),axis=0), '--' )

plt.plot( a[0]['action'][:,:3] )

plt.show()