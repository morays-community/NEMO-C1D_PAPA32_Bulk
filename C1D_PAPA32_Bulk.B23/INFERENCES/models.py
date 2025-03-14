"""
Contains User Inference/Analytic Models or Functions.

A model must fit the following requisites and structure :
-------------------------------------------------------
    1. must be a callable function that takes N numpy arrays as inputs
    2. /!\ returns N None for the N awaited outputs if at least one of the input is None /!\
    3. inputs may be freely formatted and transformed into what you want BUT...
    4. ...outputs must be formatted as numpy array for sending back
"""
import numpy as np
from AirSeaFluxCode import AirSeaFluxCode

#             Utils            #
# ++++++++++++++++++++++++++++ #
def Is_None(*inputs):
    """ Test presence of at least one None in inputs """
    return any(item is None for item in inputs)


def airseaflux(wnd,tair,sst,hum,slp):

    if Is_None(wnd,tair,sst,hum,slp):
        return None, None
    else:
        # inputs: from 3D to 1D
        wnd = max(wnd.reshape(1) , np.array([0.05]) )
        tair = tair.reshape(1)
        sst = sst.reshape(1)
        hum = hum.reshape(1)
        slp = slp.reshape(1)

        # compute Air Sea flux
        res = AirSeaFluxCode(wnd, tair, sst, "bulk", meth="UA", lat=np.array([50]), hum=['q', hum], P=slp,
                             hin=10.0, maxiter=10, out_var = ("tau", "sensible", "latent", "lv"))

        # outputs: from 1D to 3D
        tau = np.array([res["tau"]]).reshape(1,1,1)
        latent = np.array([res["latent"]]).reshape(1,1,1)
        sensible = np.array([res["sensible"]]).reshape(1,1,1)
        evap = np.array([res["lv"]]).reshape(1,1,1)
    return tau, latent, sensible, evap


if __name__ == '__main__' :

    # Testing inputs
    # --------------
    wnd = np.array([[[6.0]]])     # wind speed [m/s]
    tair = np.array([[[277.0]]])  # Potential air temperature [K]
    sst = np.array([[[280.0]]])   # Sea surface temperature [K]
    hum = np.array([[[6.0]]])     # specific humidity [g/kg]
    slp = np.array([[[1036.0]]])  # Sea level pressure [hPa]
    print(f'wind: {wnd[0,0,0]}, tair: {tair[0,0,0]}, sst: {sst[0,0,0]}, hum: {hum[0,0,0]}, P: {slp[0,0,0]}')

    # Get latent, sensible, vaporization heat, and wind stress
    tau, latent, sensible, evap = airseaflux(wnd,tair,sst,hum,slp)

    print(f'Returned shape : {tau.shape}')
    print(f'tau: {tau[0,0,0]}, latent: {latent[0,0,0]}, sensible: {sensible[0,0,0]}, evap: {evap[0,0,0]}')
    print(f'Test successful')
