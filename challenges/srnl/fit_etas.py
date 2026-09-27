# CHAT: Fit mu and K to raw seismic records using the existing instantaneous ETAS equation.
"""Fit constant background mu and triggering K; keep a,c,p,d,q fixed.

Requires: numpy, scipy, and the updated fuckassequation.py beside this script.
Run from challenges/srnl (dates/region must match your observed TRAIN catalogue):
  python fit_etas.py Data/earthquakeq_train.csv --start 1991-01-01 --end 2022-01-01 --bounds 25 50 -90 -65 --output Data/etas_fit.json

The existing ETAS CSV column is NOT used as a training label. Instead maximize
  sum_i log(mu + K*h_i) - mu*area*duration - K*integral(h),
where h is the existing triggering sum with K=1. Targets and parents both
require magnitude > mag_min. History is strictly earlier, age < dt_max, and
distance < dist_max. Targets are in [start,end) inside the study rectangle.
Earlier earthquakes warm up the history; later earthquakes never enter the fit.
All supplied parents, including those outside the rectangle, can contribute.

Time integration is analytic. Spatial integration uses four reproducible,
scrambled Sobol sequences in source-centred polar coordinates, importance
sampled from the spatial kernel, with spherical area and boundary corrections.
Successive resolutions AND replicate uncertainty must satisfy spatial_rtol.
These checks estimate numerical error; they are not rigorous error bounds.

The fit assumes complete, continuously observed data above mag_min throughout
the rectangle/time interval. Missing offshore regions, changing detection, and
missing parents outside the catalogue can bias it. Uniform mu is a baseline;
this is not a full ETAS shape/spatial-background fit or a hazard validation.
The fitted K is for this UNNORMALIZED kernel, not another package's K convention.

The original etas_score remains a frozen-history 30-day approximation: its cell
integration does not enforce moving pointwise cutoffs across the forecast.
This fitter matches etas_intensity, not that forecast approximation's integral.

Use the result with the EXISTING equation API:
  import json
  import fuckassequation as etas
  fit = json.load(open('Data/etas_fit.json', encoding='utf-8'))
  result = etas.etas_from_csv_row('Data/earthquakeq_test.csv', row_index=73,
      parameters=fit['parameters'], **fit['model_options'])
For separate shared history, use etas.probability_from_csv with the same options.
Merely creating the JSON does not change add_etas_to_csv's defaults: pass the
saved parameters and model_options into its _calculate_etas call to use the fit.

Reference for point-process likelihood: Zhuang et al., CORSSA (2011), Eq. 19,
https://www.corssa.org/export/sites/corssa/.galleries/articles-pdf/Zhuang-et-al-2011-CORSSA-Spatiotemporal-models.pdf
"""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy.optimize import brentq
from scipy.stats import qmc


# CHAT: Import your existing constants and CSV/time helpers without executing its CLI.
def load_equations(filename=None):
    path = Path(filename) if filename else Path(__file__).with_name('fuckassequation.py')
    spec = importlib.util.spec_from_file_location('etas_equations_for_fit', path)
    if spec is None or spec.loader is None:
        raise ValueError(f'Cannot import equations: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# CHAT: Fail on malformed seismic rows instead of silently losing training observations.
def read_catalog(path, equations):
    records, seen = [], set()
    for index, row in enumerate(equations._csv_rows(path)):
        if not equations._is_seismic(row):
            continue
        try:
            lat, lon = equations._coordinates(row)
            day, mag = equations._to_day(row['time']), float(row['magnitude'])
            if not math.isfinite(mag):
                raise ValueError('nonfinite magnitude')
        except (ValueError, TypeError, AttributeError, OverflowError) as exc:
            raise ValueError(f'Invalid seismic data row {index}: {exc}') from exc
        record = (day, lat, lon, mag)
        if record in seen:
            raise ValueError(f'Duplicate seismic time/location/magnitude at row {index}; resolve duplicates before fitting.')
        seen.add(record)
        records.append(record)
    if not records:
        raise ValueError('No valid seismic records.')
    return np.asarray(sorted(records), dtype=float)


# CHAT: Evaluate precisely the K=1 triggering term at each observed earthquake.
def triggering_at_events(parents, targets, coefficients, options, radius):
    days = parents[:, 0]
    lat, lon = np.radians(parents[:, 1]), np.radians(parents[:, 2])
    h = np.zeros(len(targets))
    for i, (day, latitude, longitude, _) in enumerate(targets):
        lo = np.searchsorted(days, day - options['dt_max'], side='right')
        hi = np.searchsorted(days, day, side='left')
        phi, lam = math.radians(latitude), math.radians(longitude)
        hav = (np.sin((lat[lo:hi] - phi) / 2)**2
               + np.cos(lat[lo:hi])*math.cos(phi)*np.sin((lon[lo:hi] - lam) / 2)**2)
        distance = 2*radius*np.arcsin(np.sqrt(np.clip(hav, 0, 1)))
        keep = distance < options['dist_max']
        age = day - days[lo:hi][keep]
        scale = np.exp(-coefficients['a']*(parents[lo:hi, 3][keep] - options['magnitude_reference']))
        kernel = ((age + coefficients['c'])**(-coefficients['p'])
                  * (distance[keep]**2*scale + coefficients['d'])**(-coefficients['q']))
        h[i] = math.fsum(kernel.tolist())
    if not np.isfinite(h).all():
        raise ArithmeticError('Nonfinite triggering kernel; check magnitudes and shape parameters.')
    return h


# CHAT: Integrate spatial kernels on the sphere, including the 200-km cutoff and study boundaries.
def spatial_masses(parents, uniform, bounds, coefficients, options, radius):
    result = np.zeros(len(parents))
    south, north, west, east = np.radians(bounds)
    d, exponent = coefficients['d'], 1 - coefficients['q']
    bearing = 2*np.pi*uniform[:, 1]
    for begin in range(0, len(parents), 32):
        block = parents[begin:begin+32]
        lat, lon = np.radians(block[:, 1:2]), np.radians(block[:, 2:3])
        scale = np.exp(-coefficients['a']*(block[:, 3:4] - options['magnitude_reference']))
        log_span = np.log1p(scale*options['dist_max']**2/d)
        # CHAT: Invert the radial CDF of r*(scale*r^2+d)^(-q), avoiding missed narrow peaks.
        if abs(exponent) < 1e-10:
            log_ratio = uniform[:, 0]*log_span
            planar_mass = np.pi*log_span/scale
        else:
            span = np.expm1(exponent*log_span)
            log_ratio = np.log1p(uniform[:, 0]*span)/exponent
            planar_mass = np.pi*d**exponent*span/(scale*exponent)
        distance = np.sqrt(np.maximum(0, d/scale*np.expm1(log_ratio)))
        angle = distance/radius
        sin_lat2 = np.sin(lat)*np.cos(angle) + np.cos(lat)*np.sin(angle)*np.cos(bearing)
        lat2 = np.arcsin(np.clip(sin_lat2, -1, 1))
        lon2 = lon + np.arctan2(np.sin(bearing)*np.sin(angle)*np.cos(lat),
                               np.cos(angle) - np.sin(lat)*np.sin(lat2))
        lon2 = (lon2 + np.pi) % (2*np.pi) - np.pi
        inside = (lat2 >= south) & (lat2 <= north) & (lon2 >= west) & (lon2 <= east)
        # CHAT: Correct planar r*dr*dtheta to spherical R*sin(r/R)*dr*dtheta.
        jacobian = np.sinc(angle/np.pi)
        result[begin:begin+len(block)] = planar_mass[:, 0]*np.mean(inside*jacobian, axis=1)
    if not np.isfinite(result).all():
        raise ArithmeticError('Nonfinite spatial integral; check magnitude/shape parameters.')
    return result


# CHAT: Optimize the two nonnegative amplitudes without unstable mu/K scaling.
def fit_amplitudes(h, area_days, trigger_exposure):
    h = np.asarray(h, dtype=float)
    n = len(h)
    if (not n or not np.isfinite(h).all() or np.any(h < 0)
            or not math.isfinite(area_days) or area_days <= 0
            or not math.isfinite(trigger_exposure) or trigger_exposure < 0):
        raise ValueError('Invalid likelihood inputs.')
    if trigger_exposure == 0:
        if np.any(h > 0):
            raise ArithmeticError('Positive event kernel with zero integrated exposure.')
        fraction, identifiable = 0.0, False
    else:
        z = area_days*h/trigger_exposure
        identifiable = not np.allclose(z, 1.0, rtol=1e-10, atol=1e-12)
        def derivative(fraction):
            denominator = (1 - fraction) + fraction*z
            if np.any(denominator == 0):
                return -np.inf
            return float(np.sum((z - 1)/denominator))
        # CHAT: At the optimum mu*area_days+K*trigger_exposure=N; profile their mixture fraction.
        if not identifiable or derivative(0.0) <= 0:
            fraction = 0.0
        elif derivative(1.0) >= 0:
            fraction = 1.0
        else:
            fraction = float(brentq(derivative, 0.0, 1.0, xtol=1e-14))
    mu = n*(1 - fraction)/area_days
    K = n*fraction/trigger_exposure if trigger_exposure > 0 else 0.0
    intensity = mu + K*h
    return {'mu': mu, 'K': K, 'triggered_exposure_fraction': fraction,
            'amplitudes_identifiable': identifiable,
            'log_likelihood': float(np.log(intensity).sum() - mu*area_days - K*trigger_exposure)}


# CHAT: Fit only within the declared training interval/region; prehistory supplies parents only.
def fit_etas(csv_filename, start, end, *, equations_file=None,
             bounds=(25.0, 50.0, -90.0, -65.0), mag_min=1.75,
             dt_max=180.0, dist_max=200.0, magnitude_reference=5.0,
             fixed_parameters=None, spatial_rtol=0.002, min_power=10,
             max_power=16, seed=2026, progress=None):
    equations = load_equations(equations_file)
    coefficients = dict(equations.PAPER_ZONE0_PARAMETERS)
    if fixed_parameters:
        if set(fixed_parameters) - {'a', 'c', 'p', 'd', 'q'}:
            raise ValueError('Only a,c,p,d,q may be supplied as fixed_parameters.')
        coefficients.update(fixed_parameters)
    options = dict(dt_max=dt_max, dist_max=dist_max, mag_min=mag_min,
                   magnitude_reference=magnitude_reference)
    t0, t1 = equations._to_day(start), equations._to_day(end)
    south, north, west, east = bounds
    if (not all(math.isfinite(v) for v in [*bounds, *options.values(), *coefficients.values(), spatial_rtol])
            or not -90 <= south < north <= 90 or not -180 <= west < east <= 180
            or not t1 > t0 or dt_max <= 0 or not 0 < dist_max < np.pi*equations.EARTH_RADIUS_KM
            or not 0 < spatial_rtol < 1 or coefficients['a'] < 0
            or min(coefficients[k] for k in ('c', 'p', 'd', 'q')) <= 0
            or not 4 <= min_power < max_power <= 22):
        raise ValueError('Invalid bounds, time window, kernel parameters, or integration settings.')
    data = read_catalog(csv_filename, equations)
    notes = []
    if data[0, 0] > t0 - dt_max:
        notes.append('Less than dt_max of supplied prehistory: early triggering can be underestimated.')
    # CHAT: Apply the same magnitude population to children being fitted and their parents.
    parents = data[(data[:, 3] > mag_min) & (data[:, 0] > t0-dt_max) & (data[:, 0] < t1)]
    targets = parents[(parents[:, 0] >= t0) & (parents[:, 1] >= south)
                      & (parents[:, 1] <= north) & (parents[:, 2] >= west) & (parents[:, 2] <= east)]
    if len(targets) == 0:
        raise ValueError('No above-threshold training earthquakes inside the chosen interval and region.')
    if progress:
        progress(f'{len(targets):,} training earthquakes; {len(parents):,} possible historical parents.')
    radius = equations.EARTH_RADIUS_KM
    area = radius**2*math.radians(east-west)*(math.sin(math.radians(north))-math.sin(math.radians(south)))
    exposure = area*(t1-t0)
    h = triggering_at_events(parents, targets, coefficients, options, radius)
    # CHAT: Integrate each parent's contribution only while it is alive inside the fitting period.
    lower = np.maximum(0.0, t0 - parents[:, 0])
    upper = np.minimum(dt_max, t1 - parents[:, 0])
    time_mass = np.array([equations._time_integral(lo, coefficients['c'], coefficients['p'], hi-lo)
                          for lo, hi in zip(lower, upper)])
    previous = None
    converged = False
    for power in range(min_power, max_power+1):
        totals = []
        for replicate in range(4):
            uniform = qmc.Sobol(d=2, scramble=True, seed=seed+replicate).random_base2(power)
            masses = spatial_masses(parents, uniform, bounds, coefficients, options, radius)
            totals.append(float(np.dot(time_mass, masses)))
        G = float(np.mean(totals))
        replicate_error = 3*float(np.std(totals, ddof=1))/math.sqrt(len(totals))
        change = abs(G-previous) if previous is not None else None
        error = max(replicate_error, change) if change is not None else None
        if progress:
            progress(f'Spatial integration: {4*2**power:,} points per parent; exposure={G:.8g}.')
        if error is not None and G > 0 and error <= spatial_rtol*G:
            converged = True
            break
        previous = G
    if not converged:
        raise ArithmeticError('Spatial integral did not converge; increase --max-power or check the study bounds.')
    fitted = fit_amplitudes(h, exposure, G)
    if fitted['K'] == 0 or fitted['mu'] == 0:
        notes.append('An amplitude fitted to zero (a boundary solution); inspect held-out performance.')
    if not fitted['amplitudes_identifiable']:
        notes.append('The two amplitudes cannot be distinguished from these likelihood inputs.')
    # CHAT: Sensitivity to integration uncertainty is separate from statistical parameter uncertainty.
    sensitivity = [fit_amplitudes(h, exposure, max(np.finfo(float).tiny, G+direction*error))
                   for direction in (-1, 1)]
    original_ll = float(np.log(coefficients['mu']+coefficients['K']*h).sum()
                        - coefficients['mu']*exposure - coefficients['K']*G)
    result_parameters = {**coefficients, 'mu': fitted['mu'], 'K': fitted['K']}
    return {
        'parameters': result_parameters, 'model_options': options,
        'fitted_parameters': ['mu', 'K'], 'fixed_parameters': ['a', 'c', 'p', 'd', 'q'],
        'training': {'csv': str(Path(csv_filename).resolve()), 'start_inclusive': str(start),
                     'end_exclusive': str(end), 'bounds_south_north_west_east': list(bounds),
                     'area_km2': area, 'duration_days': t1-t0, 'event_count': len(targets),
                     'parent_count': len(parents), 'area_days': exposure},
        'fit': {**fitted, 'original_log_likelihood': original_ll,
                'log_likelihood_gain_over_original': fitted['log_likelihood']-original_ll,
                'expected_background_count': fitted['mu']*exposure,
                'expected_triggered_count': fitted['K']*G,
                'poisson_only_mu': len(targets)/exposure},
        'integration': {'trigger_exposure': G, 'estimated_absolute_error': error,
                        'estimated_relative_error': error/G, 'power': power, 'replicates': 4,
                        'seed': seed, 'amplitude_sensitivity_not_confidence_intervals':
                        {k: [min(x[k] for x in sensitivity), max(x[k] for x in sensitivity)]
                         for k in ('mu', 'K')}},
        'notes': notes + ['mu units: events/day/km^2; K uses the existing unnormalized kernel.',
                          'No confidence intervals or out-of-sample validation computed.',
                          'Constant mu and fixed shape parameters; catalogue completeness is assumed.',
                          'Matches instantaneous etas_intensity; the existing ETAS CSV score remains a forecast approximation.'],
    }


# CHAT: Save a reusable parameter file atomically without changing the original equations.
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv_filename', help='Raw TRAIN catalogue, including prehistory; mixed sources are allowed')
    parser.add_argument('--start', required=True, help='Observed fitting interval start, inclusive, ISO UTC')
    parser.add_argument('--end', required=True, help='Observed fitting interval end, exclusive, ISO UTC')
    parser.add_argument('--bounds', nargs=4, type=float, default=(25, 50, -90, -65), metavar=('SOUTH', 'NORTH', 'WEST', 'EAST'))
    parser.add_argument('--output', default='etas_fit.json')
    parser.add_argument('--equations', help='Default: fuckassequation.py beside this script')
    parser.add_argument('--mag-min', type=float, default=1.75)
    parser.add_argument('--dt-max', type=float, default=180.0)
    parser.add_argument('--dist-max', type=float, default=200.0)
    parser.add_argument('--magnitude-reference', type=float, default=5.0)
    for name in ('a', 'c', 'p', 'd', 'q'):
        parser.add_argument('--'+name, type=float, help='Override fixed shape coefficient; otherwise use equations file')
    parser.add_argument('--spatial-rtol', type=float, default=0.002)
    parser.add_argument('--min-power', type=int, default=10)
    parser.add_argument('--max-power', type=int, default=16)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    output = Path(args.output)
    temporary = None
    try:
        equations_path = Path(args.equations) if args.equations else Path(__file__).with_name('fuckassequation.py')
        if output.resolve() in {Path(args.csv_filename).resolve(), equations_path.resolve(), Path(__file__).resolve()}:
            raise ValueError('Output must differ from input CSV and Python files.')
        result = fit_etas(args.csv_filename, args.start, args.end, equations_file=args.equations,
            bounds=args.bounds, mag_min=args.mag_min, dt_max=args.dt_max, dist_max=args.dist_max,
            magnitude_reference=args.magnitude_reference, spatial_rtol=args.spatial_rtol,
            min_power=args.min_power, max_power=args.max_power, seed=args.seed,
            fixed_parameters={k: getattr(args, k) for k in ('a','c','p','d','q') if getattr(args, k) is not None},
            progress=lambda message: print(message, file=sys.stderr, flush=True))
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=output.parent, delete=False) as f:
            temporary = Path(f.name)
            json.dump(result, f, indent=2, allow_nan=False)
            f.write('\n')
        os.replace(temporary, output)
        temporary = None
    except (OSError, ValueError, ArithmeticError) as exc:
        parser.error(str(exc))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"mu = {result['parameters']['mu']:.12g} events/day/km^2")
    print(f"K  = {result['parameters']['K']:.12g}")
    print(f"Saved {output}")
    for note in result['notes']:
        print(note)


if __name__ == '__main__':
    main()
