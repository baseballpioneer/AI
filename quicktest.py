import cityflow

print("Testing CityFlow...")
eng = cityflow.Engine("config.json", thread_num=1)
print("✓ Engine created")

for i in range(10):
    eng.next_step()
print(f"✓ Ran 10 simulation steps")

vehicles = eng.get_lane_vehicles()
print(f"✓ Lanes tracked: {len(vehicles)}")
print("CityFlow is working correctly!")