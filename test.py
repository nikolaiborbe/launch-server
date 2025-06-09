import xarray as xr
ds = xr.open_dataset("inputs/MC_env.nc")
print(ds)
print("coords:", list(ds.coords))
print("data_vars:", list(ds.data_vars))