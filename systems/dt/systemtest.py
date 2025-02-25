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
from systems.provided.rules.ewmac import ewmac_forecast_with_defaults as ewmac
import pandas as pd
import numpy as np

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

def systemtest(data=None, config=None, instruments=None, start_date=None, end_date=None, 
               forecast_weights=None):
    """
    Example test system using carry and EWMAC strategies
    
    Args:
        data: Optional data source, defaults to csvFuturesSimData
        config: Optional config object
        instruments: Optional list of instruments to override config
        start_date: Optional start date for analysis (str or datetime)
        end_date: Optional end date for analysis (str or datetime)
        forecast_weights: Optional dictionary of forecast weights to override config
                         e.g. {"carry": 0.5, "ewmac8": 0.25, "ewmac32": 0.25}
    """
    if data is None:
        data = csvFuturesSimData()
        
    # Convert dates to pandas datetime if provided
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
    
    if config is None:
        config = Config("systems.dt.systemtestconfig.yaml")
        
        # Override config with any provided parameters
        if instruments is not None:
            config.instruments = instruments
            # Update instrument weights to be equal
            instrument_weights = {instrument: 1.0/len(instruments) for instrument in instruments}
            config.instrument_weights = instrument_weights
            
        if start_date is not None:
            config.start_date = start_date
        if end_date is not None:
            config.end_date = end_date
    else:
        # If instruments were provided but we're using an existing config,
        # make sure the instruments list is updated
        if instruments is not None:
            config.instruments = instruments

    # Create trading rules if not in config
    if "carry" not in config.trading_rules:
        carry_rule = TradingRule(calculate_carry_forecast)
        config.trading_rules["carry"] = carry_rule
    
    # Add EWMAC rules if not in config
    if "ewmac8" not in config.trading_rules:
        ewmac8_rule = TradingRule((ewmac, [], dict(Lfast=8, Lslow=32)))
        config.trading_rules["ewmac8"] = ewmac8_rule
        
    if "ewmac32" not in config.trading_rules:
        ewmac32_rule = TradingRule(dict(function=ewmac, other_args=dict(Lfast=32, Lslow=128)))
        config.trading_rules["ewmac32"] = ewmac32_rule
    
    # Set forecast weights if provided as parameter (overrides config)
    if forecast_weights is not None:
        config.forecast_weights = forecast_weights
        # Make sure we're not using estimated weights
        config.use_forecast_weight_estimates = False
    # Set default forecast weights if not already set
    elif not hasattr(config, 'forecast_weights') or len(config.forecast_weights) < 2:
        config.forecast_weights = {
            "carry": 0.30,
            "ewmac8": 0.35,
            "ewmac32": 0.35
        }
    
    # Set forecast scalars if not already set
    if not hasattr(config, 'forecast_scalars') or len(config.forecast_scalars) < 2:
        config.forecast_scalars = {
            "carry": 1.0,
            "ewmac8": 5.3,
            "ewmac32": 2.65
        }

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

def analyze_strategies(data=None, instruments=["SOFR"], start_date=None, end_date=None):
    """
    Analyze carry and EWMAC strategies for given instruments
    
    Args:
        data: Optional data source, defaults to csvFuturesSimData
        instruments: List of instrument codes to analyze (default: ["SOFR"])
        start_date: Optional start date for analysis (str or datetime)
        end_date: Optional end date for analysis (str or datetime)
    """
    if data is None:
        data = csvFuturesSimData()

    # Convert dates to pandas datetime if provided
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
    if end_date is not None:
        end_date = pd.to_datetime(end_date)

    results = {}
    for instrument in instruments:
        print(f"\nAnalyzing {instrument}:")
        price = data.daily_prices(instrument)
        
        # Filter price data based on date range if provided
        if start_date is not None:
            price = price[price.index >= start_date]
        if end_date is not None:
            price = price[price.index <= end_date]
        
        # Calculate Carry forecast
        raw_carry = data.get_instrument_raw_carry_data(instrument)
        carry_forecast = calculate_carry_forecast(raw_carry)
        
        # Calculate EWMAC forecasts
        ewmac8_forecast = ewmac(price, Lfast=8, Lslow=32)
        ewmac32_forecast = ewmac(price, Lfast=32, Lslow=128)
        
        # Filter and align forecasts with price data
        carry_forecast = carry_forecast.reindex(price.index, method='ffill')
        carry_forecast = carry_forecast.fillna(0)
        
        ewmac8_forecast = ewmac8_forecast.reindex(price.index, method='ffill')
        ewmac8_forecast = ewmac8_forecast.fillna(0)
        
        ewmac32_forecast = ewmac32_forecast.reindex(price.index, method='ffill')
        ewmac32_forecast = ewmac32_forecast.fillna(0)
        
        # Calculate P&L for each strategy
        account = Account()
        carry_account = account.pandl_for_instrument_forecast(
            instrument,
            "carry",
            carry_forecast,
            price
        )
        
        ewmac8_account = account.pandl_for_instrument_forecast(
            instrument,
            "ewmac8",
            ewmac8_forecast,
            price
        )
        
        ewmac32_account = account.pandl_for_instrument_forecast(
            instrument,
            "ewmac32",
            ewmac32_forecast,
            price
        )
        
        print(f"\n{instrument} Carry Strategy Stats:")
        print(carry_account.percent.stats())
        
        print(f"\n{instrument} EWMAC8 Strategy Stats:")
        print(ewmac8_account.percent.stats())
        
        print(f"\n{instrument} EWMAC32 Strategy Stats:")
        print(ewmac32_account.percent.stats())
        
        results[instrument] = {
            "carry": carry_account,
            "ewmac8": ewmac8_account,
            "ewmac32": ewmac32_account
        }

    return results