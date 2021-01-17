from .path_map import PathMapCost, STARTS, STEPS

def connect_carla(host, port, map="Town01"):
    import carla
    client = carla.Client(host, port)
    world = client.load_world(map)
    return client, world

