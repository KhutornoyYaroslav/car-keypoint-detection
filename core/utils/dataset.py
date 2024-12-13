def parse_point(node: dict, dtype = int):
    if (node is not None) and ("x" in node) and ("y" in node):
        return dtype(node["x"]), dtype(node["y"])
    else:
        return None


def parse_anno_object(obj: dict):
    brect = []
    keypoints = {}

    for shape in obj.get("shapes", []):
        stype = shape.get("type", None)

        if stype == "BoundingRect":
            pts = shape.get("points", {})
            tl = parse_point(pts.get("top-left", None))
            br = parse_point(pts.get("bottom-right", None))
            if (tl is not None) and (br is not None):
                brect = [*tl, *br]

        if stype == "Keypoints":
            pts = shape.get("points", {})
            for key, val in pts.items():
                pt = parse_point(val)
                if pt is not None:
                    keypoints[key] = pt

    return brect, keypoints
