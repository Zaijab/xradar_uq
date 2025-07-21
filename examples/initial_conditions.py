import requests, pandas as pd, numpy as np
import time, pickle, os
from pathlib import Path

# Cache management: To clear cache and re-download, delete cache/earth_moon_orbits_cache.pkl
# or answer 'n' when prompted about using cached data

# Create cache directory
cache_dir = Path("cache")
cache_dir.mkdir(exist_ok=True)
cache_file = cache_dir / "earth_moon_orbits_cache.pkl"

def query_orbits(family, libr=None, branch=None):
    """Query orbits and return data"""
    params = {'sys': 'earth-moon', 'family': family}
    if libr: params['libr'] = libr
    if branch: params['branch'] = branch
    
    try:
        response = requests.get("https://ssd-api.jpl.nasa.gov/periodic_orbits.api", params=params)
        if response.status_code == 200:
            data = response.json()
            return data.get('data', [])
    except:
        pass
    return []

# Check if we have cached data
use_cached = False
if cache_file.exists():
    print(f"Loading cached orbit data from {cache_file}")
    print(f"Cache file size: {cache_file.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Check if user wants to use cache or re-download
    use_cache = input("Use cached data? (y/n, default=y): ").lower()
    if use_cache != 'n':
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
            all_orbits = cached_data['all_orbits']
            valid_configs = cached_data['valid_configs']
            timestamp = cached_data.get('download_timestamp', 0)
        
        print(f"Loaded {len(all_orbits)} orbits from {len(valid_configs)} configurations")
        if timestamp:
            cache_age_hours = (time.time() - timestamp) / 3600
            print(f"Cache age: {cache_age_hours:.1f} hours")
        use_cached = True
    else:
        print("Will re-download data...")

if not use_cached:
    print("No cache found. Downloading from NASA JPL API...")
    print("This will take a few minutes and will be cached for future use.")
    
    # Comprehensive family configurations based on API documentation
    all_families = ['halo', 'vertical', 'axial', 'lyapunov', 'longp', 'short', 
                   'butterfly', 'dragonfly', 'resonant', 'dro', 'dpo', 'lpo']
    libr_points = [1, 2, 3, 4, 5]
    branches = ['N', 'S', 'E', 'W'] + [f'{i}{j}' for i in range(1,4) for j in range(1,4) if i!=j]

    print("Systematically querying ALL families in Earth-Moon database...")

    all_orbits = []
    valid_configs = []

    for family in all_families:
        print(f"Family: {family}")
        family_orbits = 0
        
        # Try family without libr/branch first
        data = query_orbits(family)
        if data:
            family_orbits += len(data)
            valid_configs.append((family, None, None))
            print(f"  No params: {len(data)} orbits")
            for orbit in data:
                x, y, z, vx, vy, vz, jacobi, period, stability = orbit
                all_orbits.append([family, None, None, x, y, z, vx, vy, vz, 
                                 float(jacobi), float(period), float(stability)])
        
        # Try with libration points
        for libr in libr_points:
            # Without branch
            data = query_orbits(family, libr)
            if data:
                family_orbits += len(data)
                valid_configs.append((family, libr, None))
                print(f"  L{libr}: {len(data)} orbits")
                for orbit in data:
                    x, y, z, vx, vy, vz, jacobi, period, stability = orbit
                    all_orbits.append([family, libr, None, x, y, z, vx, vy, vz,
                                     float(jacobi), float(period), float(stability)])
            
            # With branches
            for branch in branches:
                data = query_orbits(family, libr, branch)
                if data:
                    family_orbits += len(data)
                    valid_configs.append((family, libr, branch))
                    print(f"  L{libr} {branch}: {len(data)} orbits")
                    for orbit in data:
                        x, y, z, vx, vy, vz, jacobi, period, stability = orbit
                        all_orbits.append([family, libr, branch, x, y, z, vx, vy, vz,
                                         float(jacobi), float(period), float(stability)])
                time.sleep(0.05)  # Be nice to server
        
        # Try branches without libr (for families that don't need libr)
        if family_orbits == 0:
            for branch in branches:
                data = query_orbits(family, None, branch)
                if data:
                    family_orbits += len(data)
                    valid_configs.append((family, None, branch))
                    print(f"  {branch}: {len(data)} orbits")
                    for orbit in data:
                        x, y, z, vx, vy, vz, jacobi, period, stability = orbit
                        all_orbits.append([family, None, branch, x, y, z, vx, vy, vz,
                                         float(jacobi), float(period), float(stability)])
                time.sleep(0.05)
        
        print(f"  Total for {family}: {family_orbits} orbits")
        time.sleep(0.1)

    # Cache the results
    print(f"Caching orbit data to {cache_file}")
    cache_data = {
        'all_orbits': all_orbits,
        'valid_configs': valid_configs,
        'download_timestamp': time.time()
    }
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

print(f"\n=== DATABASE SUMMARY ===")
print(f"Total orbits found: {len(all_orbits)}")
print(f"Valid family configurations: {len(valid_configs)}")
print(f"Families with data: {len(set(config[0] for config in valid_configs))}")

if all_orbits:
    # Convert to DataFrame
    df = pd.DataFrame(all_orbits, columns=[
        'family', 'libr', 'branch', 'x', 'y', 'z', 'vx', 'vy', 'vz',
        'jacobi', 'period', 'stability'
    ])
    
    # Analysis
    jacobi_min, jacobi_max = df['jacobi'].min(), df['jacobi'].max()
    print(f"Jacobi range: {jacobi_min:.3f} - {jacobi_max:.3f}")
    print(f"Families found: {sorted(df['family'].unique())}")
    
    # Select 10 evenly spaced orbits from EACH valid family configuration
    selected_orbits = []
    
    print(f"\nSelecting 10 evenly-spaced Jacobi orbits from each family configuration...")
    
    for config in valid_configs:
        family, libr, branch = config
        
        # Handle None values properly for pandas filtering
        if libr is None and branch is None:
            config_df = df[(df['family'] == family) & 
                          (df['libr'].isna()) & 
                          (df['branch'].isna())].copy()
        elif libr is None:
            config_df = df[(df['family'] == family) & 
                          (df['libr'].isna()) & 
                          (df['branch'] == branch)].copy()
        elif branch is None:
            config_df = df[(df['family'] == family) & 
                          (df['libr'] == libr) & 
                          (df['branch'].isna())].copy()
        else:
            config_df = df[(df['family'] == family) & 
                          (df['libr'] == libr) & 
                          (df['branch'] == branch)].copy()
        
        print(f"Debug: {family} L{libr if libr else 'X'} {branch if branch else 'None'} found {len(config_df)} orbits")
        
        if len(config_df) >= 10:
            # Find Jacobi range for this configuration
            config_jacobi_min, config_jacobi_max = config_df['jacobi'].min(), config_df['jacobi'].max()
            target_jacobis = np.linspace(config_jacobi_min, config_jacobi_max, 10)
            
            for target_jacobi in target_jacobis:
                config_df['jacobi_diff'] = abs(config_df['jacobi'] - target_jacobi)
                best_orbit = config_df.loc[config_df['jacobi_diff'].idxmin()]
                
                selected_orbits.append([
                    family, libr, branch,
                    best_orbit['x'], best_orbit['y'], best_orbit['z'],
                    best_orbit['vx'], best_orbit['vy'], best_orbit['vz'],
                    best_orbit['jacobi'], best_orbit['period'], best_orbit['stability'],
                    target_jacobi, best_orbit['jacobi_diff']
                ])
                
                # Remove selected orbit to avoid duplicates
                config_df = config_df.drop(best_orbit.name)
            
            print(f"{family:10s} L{str(libr) if libr else 'X':1s} {str(branch) if branch else 'None':4s}: 10 orbits, "
                  f"Jacobi range {config_jacobi_min:.3f}-{config_jacobi_max:.3f}")
        
        elif len(config_df) > 0:
            # Take all available orbits if less than 10
            for _, orbit in config_df.iterrows():
                selected_orbits.append([
                    family, libr, branch,
                    orbit['x'], orbit['y'], orbit['z'],
                    orbit['vx'], orbit['vy'], orbit['vz'],
                    orbit['jacobi'], orbit['period'], orbit['stability'],
                    orbit['jacobi'], 0.0  # target = actual for small sets
                ])
            
            print(f"{family:10s} L{str(libr) if libr else 'X':1s} {str(branch) if branch else 'None':4s}: {len(config_df)} orbits (all available)")
        else:
            print(f"{family:10s} L{str(libr) if libr else 'X':1s} {str(branch) if branch else 'None':4s}: 0 orbits found - check data")
    
    # Save results
    if selected_orbits:
        result_df = pd.DataFrame(selected_orbits, columns=[
            'family', 'libr', 'branch', 'x', 'y', 'z', 'vx', 'vy', 'vz',
            'jacobi', 'period', 'stability', 'target_jacobi', 'jacobi_diff'
        ])
        result_df.to_csv('cr3bp_all_families_10_per_config.csv', index=False)
        
        print(f"\n=== FINAL RESULTS ===")
        print(f"Selected {len(selected_orbits)} orbits total from {len(valid_configs)} family configurations")
        configs_with_10_plus = sum(1 for config in valid_configs 
                                  if len(df[(df['family']==config[0]) & (df['libr']==config[1]) & (df['branch']==config[2])]) >= 10)
        print(f"Configurations with ≥10 orbits: {configs_with_10_plus}")
        print(f"Families represented: {sorted(result_df['family'].unique())}")
        print(f"Max Jacobi deviation: {result_df['jacobi_diff'].max():.4f}")
        print(f"Mean Jacobi deviation: {result_df['jacobi_diff'].mean():.4f}")
        print(f"Overall Jacobi span: {result_df['jacobi'].min():.3f} - {result_df['jacobi'].max():.3f}")
        print(f"CSV file size estimate: ~{len(selected_orbits) * 0.2:.0f} KB")
        print(f"Data cached in: {cache_file}")
        print(f"To re-download: delete cache file or answer 'n' when prompted")
    else:
        print("\n=== NO ORBITS SELECTED ===")
        print("This might be because:")
        print("1. No family configurations have ≥10 orbits")
        print("2. All configurations have <10 orbits but selection failed")
        print(f"Debug: Found {len(valid_configs)} configurations with {len(all_orbits)} total orbits")
        print(f"Data cached in: {cache_file}")
        
        # Debug: show what we actually found
        if len(all_orbits) > 0:
            print("\nConfiguration sizes:")
            for family, libr, branch in valid_configs[:10]:  # Show first 10
                config_size = len(df[(df['family']==family) & 
                                    (df['libr']==libr) & 
                                    (df['branch']==branch)])
                print(f"  {family} L{libr if libr else 'X'} {branch if branch else 'None'}: {config_size} orbits")
else:
    print("No orbits found in database!")
    print(f"Check if the API is accessible and try deleting {cache_file} to re-download")
