import numpy as np
from timeit import repeat

N = 5000
d = 4 
rng = np.random.default_rng()
x = rng.uniform(size=(N,d))
rep = 5


def connectivity_threshold(ps):
    n = ps.shape[0]
    s = rng.choice(n)
    component = np.zeros(n,dtype=bool)
    component[s]=True
    dm =  np.linalg.norm(ps[:,None,:] - ps[None,:,:],axis=-1)
    np.fill_diagonal(dm,np.inf)
    dm_ma = np.ma.array(dm,mask=np.tile(component,(n,1)))
    r = dm.min(-1).max()
    while True:
        # collect vertices within distance r from component
        t =  np.max(dm_ma[component]<=r,0)
        if np.any(t): # if discover new things: add them to component; mask more cols
            component[t]=True
            if np.all(component): break
            dm_ma[:,t]=np.ma.masked # see https://numpy.org/doc/stable/reference/maskedarray.generic.html#modifying-the-mask
        else: # bfs done, component is a cluster; compute its distance to others 
            r=dm_ma[component].min()
    return r

def connectivity_threshold2(ps):
    n = ps.shape[0]
    s = rng.choice(n)
    component = np.zeros(n,dtype=bool)
    component[s]=True
    dm =  np.linalg.norm(ps[:,None,:] - ps[None,:,:],axis=-1)
    idx = np.argpartition(dm,1,-1)[:,1]
    r = dm[np.arange(n),idx].max()
    while True:
        # collect vertices within distance r from component
        t =  np.max(dm[component]<=r,axis=0)
        if np.any(t>component): # if discover new things: add them to component
            component[t]=True
            if np.all(component): break
        else: # bfs done, component is a cluster; compute its distance to others 
            r = dm[component][:,~component].min()
    return r

print("with ma")
print(connectivity_threshold(x))
print("without ma")
print(connectivity_threshold2(x))

res1 = repeat("connectivity_threshold(x)", globals=globals(), number=1, repeat = rep)
res2 = repeat("connectivity_threshold2(x)", globals=globals(), number=1, repeat = rep)

with open("bench.txt", "w") as f:
    f.write(f"config: {N=}, {d=}, {rep=}\n")
    f.write("with masked array\n")
    f.write("avg time elapsed:"+str(sum(res1)/rep)+"\n")
    f.write("just numpy array, without mask\n")
    f.write("avg time elapsed:"+str(sum(res2)/rep)+"\n")
