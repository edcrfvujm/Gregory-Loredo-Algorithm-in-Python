from astropy.io import fits
import numpy as np
import os
from GL_test import GL

path = os.walk(r"./cdfs_axbary/")
kk=0
for path, dir_lst, file_lst in path:
    for dir_name in dir_lst:
        kk=kk+1
        print(str(kk)+"/1055 : "+ dir_name)
        dir_path = os.path.join(path, dir_name)

        for path2, dir2_lst, file2_lst in os.walk(dir_path):
            for file2_name in file2_lst:
                if file2_name.split(".")[-1] == "fits":
                    a = fits.open(os.path.join(path2, file2_name))
                    t_start = np.array([a[1].header["TSTART"]])
                    t_stop = np.array([a[1].header["TSTOP"]])

                    obs_t = a[1].data.TIME
                    if len(obs_t)>10:
                        print(file2_name)
                        gl = GL(obs_t, obs_cover=(t_start,t_stop))
                        gl.compute(w_range=(2.*np.pi/20000.,2.*np.pi/200.))
                        gl.diagram(save_path=os.path.join(path2, file2_name.split("_ax")[0])+".jpg")
                        np.save(os.path.join(path2, file2_name.split("_ax")[0])+"lgOm1_w.npy", gl.lgOm1_w)
                        np.save(os.path.join(path2, file2_name.split("_ax")[0])+"lgOm1.npy", gl.lgOm1)
                        with open("./P_period.log",'a') as log:
                            log.write(str(gl.P_period)+"\n")