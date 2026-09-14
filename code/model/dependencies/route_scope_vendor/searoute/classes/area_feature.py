from geojson import Feature, FeatureCollection, Polygon
from typing import Union, List, Tuple
from .ports_props import PortProps
from ..utils import distance, pnpoly


class AreaFeature(Feature):

    @staticmethod
    def create(areas:list) ->List:
        """
        Create a feature collection of geojson type from a list of AreaFeature

        Parameters:
        ==========
        - areas: a list of AreaFeature

        Returns:
        ========
        FeatureCollection
        
        """
        if areas is None:
            raise Exception('Areas can not be empty')
        
        features = []
        for area in areas:
            features.append(area)

        return FeatureCollection(features)
        
        
    def __init__(self, coordinates, preferred_ports:list=None, name=None, **kwargs):
        geometry = Polygon(coordinates)
        super().__init__(geometry=geometry, properties={})

        self.properties['preferred_ports'] = self.norm_preferred_ports(preferred_ports)
        self.properties['name'] = name
        self.properties['area'] = self.calculate_geometry_area(coordinates[0])
        self.properties.update(kwargs)
        self.type = 'Feature'

    
    
    def norm_preferred_ports(self, preferred_ports: Union[None, PortProps, str, int, Tuple, List]) -> List[PortProps]:
        if preferred_ports is None:
            return []
        elif isinstance(preferred_ports, PortProps):
            return [preferred_ports]
        elif isinstance(preferred_ports, str):
            return [PortProps(preferred_ports)]
        elif isinstance(preferred_ports, int):
            return [PortProps(str(preferred_ports))]
        elif isinstance(preferred_ports, tuple):
            return [PortProps(*preferred_ports)]
        elif isinstance(preferred_ports, list):
            # Flatten the list by iterating over each item
            result = []
            for p in preferred_ports:
                result.extend(self.norm_preferred_ports(p))  # Use extend to avoid nested lists
            return result
        else:
            return [PortProps(preferred_ports)]


    def calculate_geometry_area(self, polyline_coordinates):
        """
        Calculate the area of a polygon defined by a set of coordinates using the shoelace formula.
    
        Parameters:
        - polyline_coordinates (list): List of coordinates defining the polygon.
    
        Returns:
        - float: The calculated area in square units.
        """
    
        # Ensure the polyline is closed to form a polygon
        if polyline_coordinates[0] != polyline_coordinates[-1]:
            polyline_coordinates.append(polyline_coordinates[0])
    
        # Calculate the area using the shoelace formula
        n = len(polyline_coordinates)
        area = 0.5 * sum(
            (polyline_coordinates[i][0] * polyline_coordinates[i + 1][1] -
             polyline_coordinates[i + 1][0] * polyline_coordinates[i][1])
            for i in range(n - 1)
        )
    
        # Return the absolute value of the calculated area
        return abs(area)
        

    def contains(self, x: float, y: float) -> bool:
        # will ignore other group of coordinates
        # assuming a closed and solid polygon
        poly = self.geometry.coordinates[0]
        n = len(poly)
        vx, vy  = list(zip(*poly))
        return pnpoly(n, vx, vy, x, y) 
    

    def distance(self, x: float, y: float) -> float:
        """
        Compute the shortest distance from the point (x, y) to the polygon boundary.
        This function finds the minimum Euclidean distance from the point to any edge of the polygon.

        Parameters:
        - x (float): X-coordinate (longitude) of the point.
        - y (float): Y-coordinate (latitude) of the point.

        Returns:
        - float: The minimum distance between the point and the polygon.
        """
        def euclidean_distance(p1, p2):
            import math
            """Compute Euclidean distance between two points."""
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        def point_to_segment_distance(px, py, x1, y1, x2, y2):
            """
            Compute the shortest distance from a point (px, py) to a line segment (x1, y1) - (x2, y2).
            Uses projection to find perpendicular distance when inside segment range.
            """
            ABx, ABy = x2 - x1, y2 - y1  # Vector AB
            APx, APy = px - x1, py - y1  # Vector AP
            dot_product = APx * ABx + APy * ABy
            ab_squared = ABx ** 2 + ABy ** 2  # Length squared of segment

            if ab_squared == 0:  # Segment is a point
                return distance((px, py), (x1, y1))

            t = dot_product / ab_squared

            if t < 0:  # Closest to point A
                closest_x, closest_y = x1, y1
            elif t > 1:  # Closest to point B
                closest_x, closest_y = x2, y2
            else:  # Closest to segment itself
                closest_x = x1 + t * ABx
                closest_y = y1 + t * ABy

            return euclidean_distance((px, py), (closest_x, closest_y))

        # Get polygon boundary (first set of coordinates)
        polygon = self.geometry.coordinates[0]
        min_distance = float('inf')

        for i in range(len(polygon) - 1):  # Loop through each edge
            x1, y1 = polygon[i]
            x2, y2 = polygon[i + 1]
            dist = point_to_segment_distance(x, y, x1, y1, x2, y2)
            min_distance = min(min_distance, dist)

        return min_distance