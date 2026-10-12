"""Generate route file for SUMO single-intersection scenario.

Creates vehicle flows with realistic demand patterns for all
turning movements through the intersection.
"""

import xml.etree.ElementTree as ET
import random
import argparse


def generate_routes(
    output_path: str = "sumo_files/single_intersection.rou.xml",
    sim_duration: int = 3600,
    seed: int = 42,
    demand_scale: float = 1.0,
):
    """Generate a route file with vehicle flows.

    Args:
        output_path: Where to write the route XML.
        sim_duration: Simulation duration in seconds.
        seed: Random seed for reproducibility.
        demand_scale: Multiplier for vehicle demand (1.0 = baseline).
    """
    random.seed(seed)

    root = ET.Element("routes")

    # Vehicle type
    vtype = ET.SubElement(root, "vType")
    vtype.set("id", "car")
    vtype.set("accel", "2.6")
    vtype.set("decel", "4.5")
    vtype.set("sigma", "0.5")
    vtype.set("length", "5.0")
    vtype.set("maxSpeed", "13.89")

    # Define all OD pairs (origin_edge -> destination_edge)
    # Incoming edges: N_C, S_C, E_C, W_C
    # Outgoing edges: C_N, C_S, C_E, C_W
    od_pairs = {
        # North approach
        ("N_C", "C_S"): 300,  # through
        ("N_C", "C_E"): 100,  # right
        ("N_C", "C_W"): 100,  # left
        # South approach
        ("S_C", "C_N"): 300,
        ("S_C", "C_W"): 100,
        ("S_C", "C_E"): 100,
        # East approach
        ("E_C", "C_W"): 250,
        ("E_C", "C_S"): 80,
        ("E_C", "C_N"): 80,
        # West approach
        ("W_C", "C_E"): 250,
        ("W_C", "C_N"): 80,
        ("W_C", "C_S"): 80,
    }

    flow_id = 0
    for (from_edge, to_edge), vph in od_pairs.items():
        scaled_vph = int(vph * demand_scale)
        if scaled_vph <= 0:
            continue

        # Calculate period (seconds between vehicles)
        period = max(1.0, 3600.0 / scaled_vph)

        flow = ET.SubElement(root, "flow")
        flow.set("id", f"f_{flow_id}")
        flow.set("type", "car")
        flow.set("from", from_edge)
        flow.set("to", to_edge)
        flow.set("begin", "0")
        flow.set("end", str(sim_duration))
        flow.set("period", f"{period:.1f}")
        flow.set("departLane", "best")
        flow.set("departSpeed", "max")

        flow_id += 1

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(output_path, encoding="unicode", xml_declaration=True)
    print(f"Generated {flow_id} flows -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SUMO routes")
    parser.add_argument("--output", default="sumo_files/single_intersection.rou.xml")
    parser.add_argument("--duration", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demand-scale", type=float, default=1.0)
    args = parser.parse_args()

    generate_routes(args.output, args.duration, args.seed, args.demand_scale)
