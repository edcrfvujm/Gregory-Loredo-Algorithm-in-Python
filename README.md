# Gregory-Loredo-Algorithm-in-Python
This is a python implementation of the Gregory-Loredo Algorithm with a good behavior solving the observation gaps.
If we generate the moke data (a time series) with rate function f =  0.5+0.5*np.sin(2*np.pi/25*x)
```
gl = GL(obs_t,T, obs_cover=(t_start,t_stop))
gl.compute(w_range=(2*np.pi/50,2*np.pi/5))
gl.diagram(bins_fow_show=50)
```
<img width="957" alt="image" src="https://user-images.githubusercontent.com/89062673/173188046-c1157dc6-fa4f-4e37-b70d-4679746d0737.png">
