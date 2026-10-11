"""Synthetic fixtures with the same schema/calendar structure as the real file.

Used ONLY for the data-blind pipeline check. Event days are Fri/Sat, holidays Sundays.
"""
import numpy as np
import pandas as pd


def make_synthetic(seed=0, n=30, start="2026-06-01", base=100.0, sd=5.0,
                   event_effect=0.0, holiday_effect=0.0, weekly_amp=0.0, trend=0.0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n, freq="D")
    dow = dates.dayofweek.to_numpy()
    event = np.isin(dow, [4, 5]).astype(int)
    holiday = (dow == 6).astype(int)
    t = np.arange(n)
    y = (base + trend * t + weekly_amp * np.sin(2 * np.pi * dow / 7)
         + event_effect * event + holiday_effect * holiday + rng.normal(0, sd, n))
    return pd.DataFrame({"date": dates, "demand": np.round(y).astype(int),
                         "marketing_event": event, "holiday": holiday})
