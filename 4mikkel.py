
from py_wake import NOJ
from py_wake.site import UniformWeibullSite
import numpy as np, numpy.random
from py_wake.examples.data.iea37 import IEA37Site, IEA37_WindTurbines
from py_wake import IEA37SimpleBastankhahGaussian
from py_wake.examples.data import example_data_path

f = np.random.dirichlet(np.ones(12),size=1)

a,k=[], []
for l in range(12):
    a.append(np.random.randint(5,25))
    k.append(np.random.randint(2,32))

hub_height = 150


site2 = UniformWeibullSite(p_wd = f.flatten(),# sector frequencies
                          a = np.array(a).flatten(), # Weibull scale parameter
                          k = np.array(k).flatten(), # Weibull shape parameter
                          ti = 0.1 # turbulence intensity, optional (not needed in all cases)
                        )


site = IEA37Site(16)
x, y = site.initial_position.T
windTurbines = IEA37_WindTurbines()


d = np.load(example_data_path + "/time_series.npz")
n_days=30
wd, ws, ws_std = [d[k][:6*24*n_days] for k in ['wd', 'ws', 'ws_std']]
ti = np.minimum(ws_std/ws,.5)
time_stamp = np.arange(len(wd))/6/24


wf_model = NOJ(site, windTurbines, k = 0.04)

lw = site.local_wind(x,y)

sim_res = wf_model(x, y)

total_aep_with_wake_loss = sim_res.aep().sum().data


sim_res_time = wf_model(x, y, # wind turbine positions
                        wd=wd, # Wind direction time series
                        ws=ws, # Wind speed time series
                        time=time_stamp, # time stamps
                        TI=ti, # turbulence intensity time series
                  )


total_AEP_time = sim_res_time.aep(with_wake_loss=True).sum().data
total_AEP_nowake_time = sim_res_time.aep(with_wake_loss=False).sum().data
sim_res_time.Power.sel(wt=0).plot(figsize=(16,3))
