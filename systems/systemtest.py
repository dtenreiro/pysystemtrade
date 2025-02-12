from sysdata.sim.csv_futures_sim_data import csvFuturesSimData
from sysdata.config.configdata import Config
from systems.forecasting import Rules
from systems.basesystem import System
from systems.forecast_combine import ForecastCombine
from systems.forecast_scale_cap import ForecastScaleCap
from systems.positionsizing import PositionSizing
from systems.portfolio import Portfolios
from systems.accounts.accounts_stage import Account
from systems.rawdata import RawData
from systems.trading_rules import TradingRule
from systems.provided.rules.carry import carry, relative_carry
from systems.provided.rules.ewmac import robust_vol_calc
import pandas as pd
import numpy as np

def calc_ewmac_forecast(price, Lfast, Lslow=None):
    """
    Calculate the ewmac trading rule forecast, given a price and EWMA speeds
    Lfast, Lslow and vol_lookback

    Args:
        price (pd.Series): Price series
        Lfast (int): Fast ewma window
        Lslow (int, optional): Slow ewma window. If None, will be 4 * Lfast

    Returns:
        pd.Series: EWMAC forecast
    """
    # Convert price to Series if it's not already
    if isinstance(price, pd.DataFrame):
        price = price.iloc[:, 0]
    elif not isinstance(price, pd.Series):
        if hasattr(price, 'flatten'):
            price = pd.Series(price.flatten())
        else:
            price = pd.Series(price)

    # Resample to business days
    price = price.resample("1B").last()

    # Default Lslow
    if Lslow is None:
        Lslow = 4 * Lfast

    # Calculate EWMA
    fast_ewma = price.ewm(span=Lfast).mean()
    slow_ewma = price.ewm(span=Lslow).mean()
    raw_ewmac = fast_ewma - slow_ewma

    # Volatility adjustment
    vol = robust_vol_calc(price.diff())
    
    # Avoid division by zero
    vol = vol.replace(0, np.nan)
    forecast = raw_ewmac / vol
    
    # Return as Series, not DataFrame
    return pd.Series(forecast.fillna(0))

def calculate_carry_forecast(raw_carry):
    """
    Calculate the carry forecast using both standard and relative carry

    Args:
        raw_carry (pd.Series): Raw carry data

    Returns:
        pd.Series: Carry forecast
    """
    # Convert raw_carry to Series if it's not already
    if isinstance(raw_carry, pd.DataFrame):
        raw_carry = raw_carry.iloc[:, 0]
    elif not isinstance(raw_carry, pd.Series):
        if hasattr(raw_carry, 'flatten'):
            raw_carry = pd.Series(raw_carry.flatten())
        else:
            raw_carry = pd.Series(raw_carry)
    
    try:
        # Calculate basic smoothed carry
        smoothed_carry = carry(raw_carry)
        
        # Create zero benchmark
        zero_benchmark = pd.Series(0, index=smoothed_carry.index)
        
        # Calculate relative carry
        carry_forecast = relative_carry(smoothed_carry, zero_benchmark)
        
        # Return as Series, not DataFrame
        return pd.Series(carry_forecast)
        
    except Exception as e:
        print(f"Error calculating carry forecast: {str(e)}")
        return pd.Series(0, index=raw_carry.index)


def systemtest(data=None, config=None, instrument="SOFR"):
    """
    Example test system combining EWMAC and carry strategies
    
    Args:
        data: Optional data source, defaults to csvFuturesSimData
        config: Optional system configuration
        instrument: Instrument code to trade (default: "SOFR")
    """
    if data is None:
        data = csvFuturesSimData()
        
    if config is None:
        # Create trading rules
        ewmac_rule = TradingRule(
            dict(
                function=calc_ewmac_forecast,
                other_args=dict(Lfast=32, Lslow=128)
            )
        )
        carry_rule = TradingRule(calculate_carry_forecast)
        
        # Define config with dynamic instrument
        config = Config(
            dict(
                trading_rules=dict(ewmac=ewmac_rule, carry=carry_rule),
                instruments=[instrument],
                forecast_scalars=dict(ewmac=1.0, carry=1.0),
                forecast_weights=dict(ewmac=0.5, carry=0.5),
                forecast_div_multiplier=1.1,
                instrument_weights={instrument: 1.0},
                instrument_div_multiplier=1.0,
                percentage_vol_target=20.0,
                notional_trading_capital=100000,
                base_currency="USD"
            )
        )

    # Create and wire up system
    my_system = System(
        [
            Account(),
            Portfolios(),
            PositionSizing(),
            ForecastCombine(),
            ForecastScaleCap(),
            Rules(),
            RawData(),
        ],
        data,
        config,
    )

    return my_system

def analyze_strategies(data=None, instrument="SOFR"):
    """
    Analyze EWMAC and carry strategies for a given instrument
    
    Args:
        data: Optional data source, defaults to csvFuturesSimData
        instrument: Instrument code to analyze (default: "SOFR")
    """
    if data is None:
        data = csvFuturesSimData()

    price = data.daily_prices(instrument)
    
    # Calculate EWMAC forecast
    ewmac = calc_ewmac_forecast(price, 32, 128)
    ewmac = pd.DataFrame(ewmac, columns=["forecast"])
    
    print("\nEWMAC Forecast:")
    print(ewmac.tail(5))
    
    # Calculate EWMAC P&L
    account = Account()
    ewmac_account = account.pandl_for_instrument_forecast(
        instrument,
        "ewmac",
        ewmac,
        price
    )
    
    print("\nEWMAC Strategy Stats:")
    print(ewmac_account.percent.stats())
    
    # Calculate Carry forecast
    raw_carry = data.get_instrument_raw_carry_data(instrument)
    carry_forecast = calculate_carry_forecast(raw_carry)
    
    # Align carry forecast with price data
    carry_forecast = carry_forecast.reindex(price.index, method='ffill')
    carry_forecast = carry_forecast.fillna(0)
    
    # Calculate Carry P&L
    carry_account = account.pandl_for_instrument_forecast(
        instrument,
        "carry",
        carry_forecast,
        price
    )
    
    print("\nCarry Strategy Stats:")
    print(carry_account.percent.stats())

    return data