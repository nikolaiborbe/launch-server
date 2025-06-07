from rocketpy import (
    Function,
    LiquidMotor,
    UllageBasedTank,
    MassBasedTank,
    MassFlowRateBasedTank,
    Rocket,
    Flight,
    Environment,
    CylindricalTank,
    Accelerometer,
    Gyroscope,
    Barometer,
    GnssReceiver
)
import pandas as pd

df = pd.read_excel("input.xlsx", index_col=1)
header = df.iloc[1]

headers = df.iloc[0].values
df.columns = headers

def get_rocket() -> Flight:
    