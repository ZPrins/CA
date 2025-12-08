import simpy
import random
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    # Kiln Production (Railton)
    KILN_MEAN_RATE = 150.0  # tons/hour (when running)
    KILN_STD_DEV = 39.0     # standard deviation for production variability
    
    # Mill Production (CM4 + CM5)
    MILL_MEAN_RATE = 95.0   # tons/hour (when running)
    MILL_STD_DEV = 10.0     # standard deviation for mill variability
    MILL_CONVERSION_RATE = 0.88  # 88% of clinker becomes GP, 12% waste
    
    # Downtime as percentage of production days
    KILN_PLANNED_DOWNTIME_PCT = 15.0    # 15% of year for planned maintenance
    KILN_UNPLANNED_DOWNTIME_PCT = 10.0  # 10% of year for unplanned breakdowns
    MILL_PLANNED_DOWNTIME_PCT = 12.0    # 12% of year for planned maintenance
    MILL_UNPLANNED_DOWNTIME_PCT = 8.0   # 8% of year for unplanned breakdowns
    
    # Calculate downtime hours for 1 year (8760 hours)
    YEAR_HOURS = 8760.0
    KILN_PLANNED_HOURS = YEAR_HOURS * (KILN_PLANNED_DOWNTIME_PCT / 100.0)
    KILN_UNPLANNED_HOURS = YEAR_HOURS * (KILN_UNPLANNED_DOWNTIME_PCT / 100.0)
    MILL_PLANNED_HOURS = YEAR_HOURS * (MILL_PLANNED_DOWNTIME_PCT / 100.0)
    MILL_UNPLANNED_HOURS = YEAR_HOURS * (MILL_UNPLANNED_DOWNTIME_PCT / 100.0)
    
    # Railton CL Store (Clinker Storage)
    CL_CAPACITY = 30000.0    # tons
    CL_INITIAL = 11000.0     # tons
    
    # Railton GP Store (Gypsum Product Storage)
    GP_R_CAPACITY = 25000.0  # tons
    GP_R_INITIAL = 15000.0   # tons
    
    # Rail Transport (Railton to Davenport)
    TRAIN_CAPACITY = 652.0   # tons per trip
    TRAINS_PER_DAY = 6       # number of trips per day
    TRAIN_TRAVEL_TIME = 40.0 / 60.0  # 40 minutes = 0.667 hours (one way)
    TRAIN_LOAD_TIME = 1.0    # hours to load
    TRAIN_UNLOAD_TIME = 1.0  # hours to unload
    
    # Davenport GP Store
    GP_D_CAPACITY = 30000.0  # tons
    GP_D_INITIAL = 11000.0   # tons
    
    # Ship Routes from Davenport
    SHIPS = {
        'Osborne': {
            'max_capacity': 5000.0,      # tons
            'loading_rate': 500.0,       # tons/hour
            'travel_time': 24.0,         # hours one way
            'unloading_rate': 400.0,     # tons/hour
            'frequency_days': 7.0,       # Ship departs every 7 days
        },
        'Newcastle': {
            'max_capacity': 8000.0,      # tons
            'loading_rate': 600.0,       # tons/hour
            'travel_time': 36.0,         # hours one way
            'unloading_rate': 500.0,     # tons/hour
            'frequency_days': 10.0,      # Ship departs every 10 days
        },
        'MCF': {
            'max_capacity': 6000.0,      # tons
            'loading_rate': 450.0,       # tons/hour
            'travel_time': 20.0,         # hours one way
            'unloading_rate': 350.0,     # tons/hour
            'frequency_days': 5.0,       # Ship departs every 5 days
        }
    }
    
    # Destination GP Stores
    DESTINATIONS = {
        'Osborne': {
            'capacity': 15000.0,         # tons
            'initial': 5000.0,           # tons
            'daily_demand': 600.0,       # tons/day
        },
        'Newcastle': {
            'capacity': 20000.0,         # tons
            'initial': 8000.0,           # tons
            'daily_demand': 800.0,       # tons/day
        },
        'MCF': {
            'capacity': 12000.0,         # tons
            'initial': 4000.0,           # tons
            'daily_demand': 700.0,       # tons/day
        }
    }

# ==========================================
# DOWNTIME SCHEDULER
# ==========================================
class DowntimeScheduler:
    """Generates random downtime events spread throughout the year"""
    
    @staticmethod
    def generate_downtime_schedule(total_hours, total_downtime_hours, avg_event_duration, year_hours=8760):
        """Generate random downtime events"""
        events = []
        remaining_downtime = total_downtime_hours
        current_time = random.uniform(0, 100)
        
        while remaining_downtime > 0 and current_time < year_hours:
            duration = max(1.0, random.normalvariate(avg_event_duration, avg_event_duration * 0.3))
            duration = min(duration, remaining_downtime)
            
            events.append({'start': current_time, 'duration': duration})
            
            remaining_downtime -= duration
            time_gap = random.uniform(50, 200)
            current_time += duration + time_gap
        
        return sorted(events, key=lambda x: x['start'])

# ==========================================
# COMPONENT PROCESSES
# ==========================================

def kiln_process(env, kiln_resource, cl_store, stats, planned_schedule, unplanned_schedule):
    """Produces clinker continuously when not under maintenance"""
    planned_idx = 0
    unplanned_idx = 0
    
    while True:
        current_time = env.now
        
        # Check if we should be in downtime
        in_planned = False
        in_unplanned = False
        
        if planned_idx < len(planned_schedule):
            event = planned_schedule[planned_idx]
            if event['start'] <= current_time < event['start'] + event['duration']:
                in_planned = True
            elif current_time >= event['start'] + event['duration']:
                planned_idx += 1
        
        if unplanned_idx < len(unplanned_schedule):
            event = unplanned_schedule[unplanned_idx]
            if event['start'] <= current_time < event['start'] + event['duration']:
                in_unplanned = True
            elif current_time >= event['start'] + event['duration']:
                unplanned_idx += 1
        
        if in_planned or in_unplanned:
            yield env.timeout(1.0)
            continue
        
        # Normal production
        rate = max(10.0, random.normalvariate(Config.KILN_MEAN_RATE, Config.KILN_STD_DEV))
        yield env.timeout(1.0)
        
        production = rate * 1.0
        yield cl_store.put(production)
        stats['kiln_production'] += production

def mill_process(env, cl_store, gp_railton, stats, planned_schedule, unplanned_schedule):
    """Grinds clinker from CL Store into gypsum product for GP Railton Store
    88% of clinker becomes GP, 12% is waste"""
    planned_idx = 0
    unplanned_idx = 0
    
    while True:
        current_time = env.now
        
        in_planned = False
        in_unplanned = False
        
        if planned_idx < len(planned_schedule):
            event = planned_schedule[planned_idx]
            if event['start'] <= current_time < event['start'] + event['duration']:
                in_planned = True
            elif current_time >= event['start'] + event['duration']:
                planned_idx += 1
        
        if unplanned_idx < len(unplanned_schedule):
            event = unplanned_schedule[unplanned_idx]
            if event['start'] <= current_time < event['start'] + event['duration']:
                in_unplanned = True
            elif current_time >= event['start'] + event['duration']:
                unplanned_idx += 1
        
        if in_planned or in_unplanned:
            yield env.timeout(1.0)
            continue
        
        rate = max(5.0, random.normalvariate(Config.MILL_MEAN_RATE, Config.MILL_STD_DEV))
        batch_size = rate * 1.0
        
        # Get clinker from CL Store
        yield cl_store.get(batch_size)
        yield env.timeout(1.0)
        
        # Apply conversion rate: only 88% becomes GP, 12% is waste
        gp_produced = batch_size * Config.MILL_CONVERSION_RATE
        waste = batch_size * (1 - Config.MILL_CONVERSION_RATE)
        
        # Put GP into GP Railton
        yield gp_railton.put(gp_produced)
        stats['mill_production'] += gp_produced
        stats['mill_waste'] += waste
        stats['mill_clinker_consumed'] += batch_size

def train_process(env, gp_railton, gp_davenport, stats):
    """Transports product from GP Railton to GP Davenport - 6 trips per day"""
    trip_interval = 24.0 / Config.TRAINS_PER_DAY
    
    while True:
        yield env.timeout(trip_interval)
        
        yield gp_railton.get(Config.TRAIN_CAPACITY)
        yield env.timeout(Config.TRAIN_LOAD_TIME)
        yield env.timeout(Config.TRAIN_TRAVEL_TIME)
        yield env.timeout(Config.TRAIN_UNLOAD_TIME)
        yield gp_davenport.put(Config.TRAIN_CAPACITY)
        stats['train_deliveries'] += 1
        yield env.timeout(Config.TRAIN_TRAVEL_TIME)

def ship_process(env, destination_name, gp_davenport, dest_store, berth, stats):
    """Ship process for a specific destination"""
    ship_config = Config.SHIPS[destination_name]
    frequency_hours = ship_config['frequency_days'] * 24.0
    
    while True:
        # Wait for scheduled departure
        yield env.timeout(frequency_hours)
        
        # Request berth (only one ship can load at a time)
        with berth.request() as req:
            yield req
            
            # Load from Davenport GP Store
            load_quantity = min(ship_config['max_capacity'], gp_davenport.level)
            if load_quantity > 0:
                yield gp_davenport.get(load_quantity)
                
                # Loading time
                loading_time = load_quantity / ship_config['loading_rate']
                yield env.timeout(loading_time)
                
                stats['ships'][destination_name]['departures'] += 1
                stats['ships'][destination_name]['total_loaded'] += load_quantity
        
        # Travel to destination
        yield env.timeout(ship_config['travel_time'])
        
        # Unload at destination
        unloading_time = load_quantity / ship_config['unloading_rate']
        yield env.timeout(unloading_time)
        yield dest_store.put(load_quantity)
        stats['ships'][destination_name]['total_delivered'] += load_quantity
        
        # Return journey
        yield env.timeout(ship_config['travel_time'])

def destination_demand(env, destination_name, dest_store, stats):
    """Customer demand at each destination"""
    daily_demand = Config.DESTINATIONS[destination_name]['daily_demand']
    hourly_demand = daily_demand / 24.0
    
    while True:
        yield env.timeout(1.0)
        
        available = dest_store.level
        if available >= hourly_demand:
            yield dest_store.get(hourly_demand)
            stats['destinations'][destination_name]['demand_met'] += hourly_demand
        elif available > 0:
            yield dest_store.get(available)
            stats['destinations'][destination_name]['demand_met'] += available
            stats['destinations'][destination_name]['stockouts'] += 1
            stats['destinations'][destination_name]['unmet_demand'] += (hourly_demand - available)
        else:
            stats['destinations'][destination_name]['stockouts'] += 1
            stats['destinations'][destination_name]['unmet_demand'] += hourly_demand

# ==========================================
# DATA COLLECTION
# ==========================================
class DataCollector:
    def __init__(self, env, stores_dict):
        self.env = env
        self.stores = stores_dict
        self.data = {name: {'times': [], 'levels': []} for name in stores_dict.keys()}
        
    def collect(self):
        """Periodically collect data"""
        while True:
            for name, store in self.stores.items():
                self.data[name]['times'].append(self.env.now)
                self.data[name]['levels'].append(store.level)
            yield self.env.timeout(1)

# ==========================================
# MAIN SIMULATION
# ==========================================

def run_simulation(duration=8760, show_plot=True):
    env = simpy.Environment()
    
    # Create stores
    cl_store = simpy.Container(env, capacity=Config.CL_CAPACITY, init=Config.CL_INITIAL)
    gp_railton = simpy.Container(env, capacity=Config.GP_R_CAPACITY, init=Config.GP_R_INITIAL)
    gp_davenport = simpy.Container(env, capacity=Config.GP_D_CAPACITY, init=Config.GP_D_INITIAL)
    
    # Create destination stores
    dest_stores = {}
    for dest_name, dest_config in Config.DESTINATIONS.items():
        dest_stores[dest_name] = simpy.Container(
            env, 
            capacity=dest_config['capacity'], 
            init=dest_config['initial']
        )
    
    # Create berth resource (only 1 ship can load at a time)
    berth = simpy.Resource(env, capacity=1)
    
    # Generate downtime schedules
    print("Generating downtime schedules...")
    kiln_planned = DowntimeScheduler.generate_downtime_schedule(
        duration, Config.KILN_PLANNED_HOURS, avg_event_duration=8.0
    )
    kiln_unplanned = DowntimeScheduler.generate_downtime_schedule(
        duration, Config.KILN_UNPLANNED_HOURS, avg_event_duration=4.0
    )
    mill_planned = DowntimeScheduler.generate_downtime_schedule(
        duration, Config.MILL_PLANNED_HOURS, avg_event_duration=6.0
    )
    mill_unplanned = DowntimeScheduler.generate_downtime_schedule(
        duration, Config.MILL_UNPLANNED_HOURS, avg_event_duration=3.0
    )
    
    print(f"  Kiln: {len(kiln_planned)} planned, {len(kiln_unplanned)} unplanned events")
    print(f"  Mill: {len(mill_planned)} planned, {len(mill_unplanned)} unplanned events\n")
    
    # Statistics tracking
    stats = {
        'kiln_production': 0,
        'mill_production': 0,
        'mill_clinker_consumed': 0,
        'mill_waste': 0,
        'train_deliveries': 0,
        'ships': {name: {'departures': 0, 'total_loaded': 0, 'total_delivered': 0} 
                  for name in Config.SHIPS.keys()},
        'destinations': {name: {'demand_met': 0, 'unmet_demand': 0, 'stockouts': 0} 
                        for name in Config.DESTINATIONS.keys()}
    }
    
    # Start processes
    env.process(kiln_process(env, None, cl_store, stats, kiln_planned, kiln_unplanned))
    env.process(mill_process(env, cl_store, gp_railton, stats, mill_planned, mill_unplanned))
    env.process(train_process(env, gp_railton, gp_davenport, stats))
    
    # Start ship and destination processes
    for dest_name in Config.SHIPS.keys():
        env.process(ship_process(env, dest_name, gp_davenport, dest_stores[dest_name], berth, stats))
        env.process(destination_demand(env, dest_name, dest_stores[dest_name], stats))
    
    # Start data collection
    all_stores = {
        'CL_Railton': cl_store,
        'GP_Railton': gp_railton,
        'GP_Davenport': gp_davenport,
        **{f'GP_{name}': store for name, store in dest_stores.items()}
    }
    collector = DataCollector(env, all_stores)
    env.process(collector.collect())
    
    # Run simulation
    print(f"{'='*70}")
    print(f"EXTENDED SUPPLY CHAIN SIMULATION")
    print(f"{'='*70}\n")
    print(f"Simulating {duration} hours ({duration/24:.1f} days)...\n")
    
    env.run(until=duration)
    
    # Print detailed statistics
    print_statistics(stats, duration, all_stores)
    
    if show_plot:
        plot_results(collector)
    
    return collector, stats

def print_statistics(stats, duration, stores):
    """Print comprehensive statistics"""
    print(f"\n{'='*70}")
    print(f"SIMULATION RESULTS (1 YEAR)")
    print(f"{'='*70}")
    
    print(f"\n--- PRODUCTION ---")
    print(f"  Kiln Production:            {stats['kiln_production']:10,.0f} tons (clinker)")
    print(f"  Mill Clinker Consumed:      {stats['mill_clinker_consumed']:10,.0f} tons")
    print(f"  Mill GP Production:         {stats['mill_production']:10,.0f} tons (88% conversion)")
    print(f"  Mill Waste:                 {stats['mill_waste']:10,.0f} tons (12% waste)")
    print(f"  Conversion Efficiency:      {(stats['mill_production']/stats['mill_clinker_consumed']*100) if stats['mill_clinker_consumed'] > 0 else 0:8.1f}%")
    
    print(f"\n--- RAIL TRANSPORT ---")
    print(f"  Train Deliveries:           {stats['train_deliveries']:10.0f} trips")
    print(f"  Total Delivered:            {stats['train_deliveries']*Config.TRAIN_CAPACITY:10,.0f} tons")
    
    print(f"\n--- SHIP OPERATIONS ---")
    for ship_name, ship_stats in stats['ships'].items():
        ship_config = Config.SHIPS[ship_name]
        print(f"  {ship_name}:")
        print(f"    Departures:               {ship_stats['departures']:10.0f} trips")
        print(f"    Total Loaded:             {ship_stats['total_loaded']:10,.0f} tons")
        print(f"    Total Delivered:          {ship_stats['total_delivered']:10,.0f} tons")
        print(f"    Avg per Trip:             {ship_stats['total_loaded']/max(1,ship_stats['departures']):10,.0f} tons")
    
    print(f"\n--- DESTINATION DEMAND ---")
    for dest_name, dest_stats in stats['destinations'].items():
        dest_config = Config.DESTINATIONS[dest_name]
        total_demand = dest_config['daily_demand'] * (duration / 24.0)
        service_level = (dest_stats['demand_met'] / total_demand * 100) if total_demand > 0 else 0
        print(f"  {dest_name}:")
        print(f"    Total Demand:             {total_demand:10,.0f} tons")
        print(f"    Demand Met:               {dest_stats['demand_met']:10,.0f} tons")
        print(f"    Unmet Demand:             {dest_stats['unmet_demand']:10,.0f} tons")
        print(f"    Service Level:            {service_level:8.1f}%")
        print(f"    Stockout Events:          {dest_stats['stockouts']:10.0f}")
    
    print(f"\n--- FINAL INVENTORY LEVELS ---")
    for store_name, store in stores.items():
        print(f"  {store_name:20s}  {store.level:10,.0f} tons")
    
    print(f"{'='*70}\n")

def plot_results(collector):
    """Create visualization of all store levels"""
    num_stores = len(collector.data)
    fig, axes = plt.subplots(num_stores, 1, figsize=(14, 4*num_stores))
    if num_stores == 1:
        axes = [axes]
    
    fig.suptitle('Extended Supply Chain - All Inventory Levels Over 1 Year', 
                 fontsize=16, fontweight='bold')
    
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']
    
    for idx, (store_name, data) in enumerate(collector.data.items()):
        times_days = [t/24 for t in data['times']]
        color = colors[idx % len(colors)]
        
        axes[idx].plot(times_days, data['levels'], color=color, linewidth=1.5, label='Inventory Level')
        axes[idx].fill_between(times_days, 0, data['levels'], alpha=0.3, color=color)
        axes[idx].set_ylabel('Level (tons)', fontsize=11, fontweight='bold')
        axes[idx].set_title(f'{store_name}', fontsize=12)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend(loc='best')
    
    axes[-1].set_xlabel('Time (days)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":
    collector, stats = run_simulation(duration=8760, show_plot=True)