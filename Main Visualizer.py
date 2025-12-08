import simpy
import random
import time
import tkinter as tk
import math

# ==========================================
# CONFIGURATION (Your Exact Config)
# ==========================================
class Config:
    # Kiln Production (Railton)
    KILN_MEAN_RATE = 150.0 
    KILN_STD_DEV = 39.0     
    
    # Mill Production (CM4 + CM5)
    MILL_MEAN_RATE = 95.0 
    MILL_STD_DEV = 10.0    
    
    # Railton CL Store
    CL_CAPACITY = 30000.0   
    CL_INITIAL = 11000.0    
    
    # Railton GP Store
    GP_R_CAPACITY = 25000.0  
    GP_R_INITIAL = 15000.0   
    
    # Rail Transport
    TRAIN_CAPACITY = 652.0   
    TRAINS_PER_DAY = 6       
    TRAIN_TRAVEL_TIME = 40.0 / 60.0 
    TRAIN_LOAD_TIME = 1.0    
    TRAIN_UNLOAD_TIME = 1.0  
    
    # Davenport GP Store
    GP_D_CAPACITY = 30000.0  
    GP_D_INITIAL = 11000.0   
    
    # Ship Routes
    SHIPS = {
        'Osborne':   {'max': 5000.0, 'load_rate': 500.0, 'travel': 24.0, 'unload': 400.0, 'freq': 7.0},
        'Newcastle': {'max': 8000.0, 'load_rate': 600.0, 'travel': 36.0, 'unload': 500.0, 'freq': 10.0},
        'MCF':       {'max': 6000.0, 'load_rate': 450.0, 'travel': 20.0, 'unload': 350.0, 'freq': 5.0}
    }
    
    DESTINATIONS = {
        'Osborne':   {'cap': 15000.0, 'init': 5000.0, 'demand': 600.0},
        'Newcastle': {'cap': 20000.0, 'init': 8000.0, 'demand': 800.0},
        'MCF':       {'cap': 12000.0, 'init': 4000.0, 'demand': 700.0}
    }

# ==========================================
# VISUALIZATION ENGINE (New Addition)
# ==========================================
class Visualizer:
    def __init__(self, root, stores):
        self.root = root
        self.canvas = tk.Canvas(root, width=1000, height=600, bg="#1e1e1e")
        self.canvas.pack()
        self.stores = stores
        self.vehicles = {} # Track vehicle positions
        
        # Coordinates
        self.coords = {
            'Railton': (100, 300),
            'Davenport': (400, 300),
            'Osborne': (800, 150),
            'Newcastle': (800, 300),
            'MCF': (800, 450)
        }
        
        self.draw_map()
        self.tank_rects = {}
        self.init_tanks()

    def draw_map(self):
        # Draw connections
        c = self.coords
        # Rail
        self.canvas.create_line(c['Railton'], c['Davenport'], fill="white", width=4)
        self.canvas.create_text(250, 280, text="Rail Line", fill="white", font=("Arial", 8))
        
        # Sea
        for dest in ['Osborne', 'Newcastle', 'MCF']:
            self.canvas.create_line(c['Davenport'], c[dest], fill="cyan", dash=(4, 4))

    def init_tanks(self):
        # Create tank graphics for each store
        offsets = {
            'CL_Store': (-40, -40), 'GP_Railton': (40, -40), 
            'GP_Davenport': (0, -40),
            'GP_Osborne': (40, 0), 'GP_Newcastle': (40, 0), 'GP_MCF': (40, 0)
        }
        
        base_nodes = {
            'CL_Store': 'Railton', 'GP_Railton': 'Railton',
            'GP_Davenport': 'Davenport',
            'GP_Osborne': 'Osborne', 'GP_Newcastle': 'Newcastle', 'GP_MCF': 'MCF'
        }
        
        for name, store in self.stores.items():
            node = base_nodes.get(name)
            if not node: continue
            
            cx, cy = self.coords[node]
            ox, oy = offsets[name]
            x, y = cx + ox, cy + oy
            
            # Tank Border
            self.canvas.create_rectangle(x-15, y-30, x+15, y+30, outline="white")
            # Tank Fill (Dynamic)
            fill = self.canvas.create_rectangle(x-14, y+29, x+14, y+29, fill="orange", outline="")
            # Label
            self.canvas.create_text(x, y-40, text=name.split('_')[-1], fill="white", font=("Arial", 8))
            
            self.tank_rects[name] = (fill, x, y, store.capacity)

    def update_vehicle(self, id, type, start_node, end_node, progress):
        """ Draw or update a vehicle based on progress (0.0 to 1.0) """
        start = self.coords[start_node]
        end = self.coords[end_node]
        
        # Linear Interpolation
        cur_x = start[0] + (end[0] - start[0]) * progress
        cur_y = start[1] + (end[1] - start[1]) * progress
        
        color = "red" if type == "train" else "blue"
        
        if id not in self.vehicles:
            # Create new vehicle shape
            shape = self.canvas.create_oval(cur_x-6, cur_y-6, cur_x+6, cur_y+6, fill=color, outline="white")
            self.vehicles[id] = shape
        else:
            # Move existing
            self.canvas.coords(self.vehicles[id], cur_x-6, cur_y-6, cur_x+6, cur_y+6)

    def update_tanks(self):
        for name, (rect_id, x, y, cap) in self.tank_rects.items():
            store = self.stores[name]
            level_pct = store.level / cap
            pixel_height = 60 * level_pct
            # Update coordinate of the fill rectangle
            self.canvas.coords(rect_id, x-14, (y+29) - pixel_height, x+14, y+29)
            
            # Change color if empty or full
            color = "orange"
            if level_pct < 0.1: color = "red"
            if level_pct > 0.9: color = "green"
            self.canvas.itemconfig(rect_id, fill=color)

    def update(self):
        self.update_tanks()
        self.root.update()

# ==========================================
# TRACKING CLASSES (Wrappers to track state)
# ==========================================
class VehicleTracker:
    def __init__(self, id, type, vis):
        self.id = id
        self.type = type
        self.vis = vis
        # Initialize as None so we know they haven't started
        self.start_loc = None 
        self.end_loc = None
        self.start_time = 0
        self.arrival_time = 0
        
    def start_trip(self, env, start, end, duration):
        self.start_loc = start
        self.end_loc = end
        self.start_time = env.now
        self.arrival_time = env.now + duration
        
    def update_vis(self, env):
        # FIX: If trip hasn't started (locations are None), do not draw anything
        if self.start_loc is None or self.end_loc is None:
            return

        if env.now < self.arrival_time:
            total_dur = self.arrival_time - self.start_time
            elapsed = env.now - self.start_time
            # Avoid division by zero if duration is 0
            pct = elapsed / total_dur if total_dur > 0 else 1.0
            self.vis.update_vehicle(self.id, self.type, self.start_loc, self.end_loc, pct)
        else:
            # Snap to end
            self.vis.update_vehicle(self.id, self.type, self.end_loc, self.end_loc, 1.0)
# ==========================================
# MODIFIED COMPONENT PROCESSES
# ==========================================
# We only add tracking lines. Logic remains identical.

def kiln_process(env, cl_store):
    while True:
        rate = max(10.0, random.normalvariate(Config.KILN_MEAN_RATE, Config.KILN_STD_DEV))
        yield env.timeout(1.0)
        yield cl_store.put(rate)

def mill_process(env, cl_store, gp_railton):
    while True:
        rate = max(5.0, random.normalvariate(Config.MILL_MEAN_RATE, Config.MILL_STD_DEV))
        yield cl_store.get(rate)
        yield env.timeout(1.0)
        yield gp_railton.put(rate)

def train_process(env, gp_railton, gp_davenport, tracker):
    trip_interval = 24.0 / Config.TRAINS_PER_DAY
    
    while True:
        yield env.timeout(trip_interval)
        
        # Load
        yield gp_railton.get(Config.TRAIN_CAPACITY)
        tracker.start_trip(env, 'Railton', 'Railton', Config.TRAIN_LOAD_TIME)
        yield env.timeout(Config.TRAIN_LOAD_TIME)
        
        # Travel
        tracker.start_trip(env, 'Railton', 'Davenport', Config.TRAIN_TRAVEL_TIME)
        yield env.timeout(Config.TRAIN_TRAVEL_TIME)
        
        # Unload
        tracker.start_trip(env, 'Davenport', 'Davenport', Config.TRAIN_UNLOAD_TIME)
        yield env.timeout(Config.TRAIN_UNLOAD_TIME)
        yield gp_davenport.put(Config.TRAIN_CAPACITY)
        
        # Return
        tracker.start_trip(env, 'Davenport', 'Railton', Config.TRAIN_TRAVEL_TIME)
        yield env.timeout(Config.TRAIN_TRAVEL_TIME)

def ship_process(env, dest_name, gp_davenport, dest_store, berth, tracker):
    ship_cfg = Config.SHIPS[dest_name]
    freq = ship_cfg['freq'] * 24.0
    
    while True:
        yield env.timeout(freq)
        
        # Travel to Davenport
        tracker.start_trip(env, dest_name, 'Davenport', ship_cfg['travel'])
        yield env.timeout(ship_cfg['travel'])
        
        with berth.request() as req:
            yield req
            # Load
            load_qty = min(ship_cfg['max'], gp_davenport.level)
            load_time = load_qty / ship_cfg['load_rate']
            
            tracker.start_trip(env, 'Davenport', 'Davenport', load_time)
            if load_qty > 0:
                yield gp_davenport.get(load_qty)
            yield env.timeout(load_time)
            
        # Return
        tracker.start_trip(env, 'Davenport', dest_name, ship_cfg['travel'])
        yield env.timeout(ship_cfg['travel'])
        
        # Unload
        unload_time = load_qty / ship_cfg['unload']
        tracker.start_trip(env, dest_name, dest_name, unload_time)
        yield env.timeout(unload_time)
        yield dest_store.put(load_qty)

def demand_process(env, dest_store, demand_daily):
    hourly = demand_daily / 24.0
    while True:
        yield env.timeout(1.0)
        if dest_store.level >= hourly:
            yield dest_store.get(hourly)

# ==========================================
# VISUALIZATION CONTROLLER
# ==========================================
def visual_clock(env, vis, trackers):
    """Refreshes the screen every 0.1 simulation hours"""
    while True:
        yield env.timeout(0.5) # Update every 30 sim-minutes
        
        # Update trackers
        for t in trackers:
            t.update_vis(env)
            
        vis.update()
        time.sleep(0.02) # REAL TIME DELAY (Control speed of animation here)

# ==========================================
# MAIN
# ==========================================
def run_visual_simulation():
    # 1. Setup Tkinter
    root = tk.Tk()
    root.title("Supply Chain Digital Twin")
    
    env = simpy.Environment()
    
    # 2. Create Stores
    cl_store = simpy.Container(env, Config.CL_CAPACITY, init=Config.CL_INITIAL)
    gp_railton = simpy.Container(env, Config.GP_R_CAPACITY, init=Config.GP_R_INITIAL)
    gp_davenport = simpy.Container(env, Config.GP_D_CAPACITY, init=Config.GP_D_INITIAL)
    
    dest_stores = {}
    for name, cfg in Config.DESTINATIONS.items():
        dest_stores[name] = simpy.Container(env, cfg['cap'], init=cfg['init'])
        
    # 3. Setup Visualization
    all_stores = {
        'CL_Store': cl_store, 'GP_Railton': gp_railton, 'GP_Davenport': gp_davenport,
        'GP_Osborne': dest_stores['Osborne'], 
        'GP_Newcastle': dest_stores['Newcastle'], 
        'GP_MCF': dest_stores['MCF']
    }
    vis = Visualizer(root, all_stores)
    
    # 4. Start Processes with Trackers
    env.process(kiln_process(env, cl_store))
    env.process(mill_process(env, cl_store, gp_railton))
    
    # Train
    train_tracker = VehicleTracker("Train1", "train", vis)
    env.process(train_process(env, gp_railton, gp_davenport, train_tracker))
    
    # Ships
    berth = simpy.Resource(env, capacity=1)
    trackers = [train_tracker]
    
    for name in Config.SHIPS:
        t = VehicleTracker(f"Ship_{name}", "ship", vis)
        trackers.append(t)
        env.process(ship_process(env, name, gp_davenport, dest_stores[name], berth, t))
        env.process(demand_process(env, dest_stores[name], Config.DESTINATIONS[name]['demand']))
    
    # 5. Start Visual Clock
    env.process(visual_clock(env, vis, trackers))
    
    # 6. Run
    # Note: We run a bit differently to keep Tkinter happy
    print("Simulation Running... Close window to stop.")
    try:
        env.run(until=8760)
    except tk.TclError:
        print("Window closed.")

if __name__ == "__main__":
    run_visual_simulation()