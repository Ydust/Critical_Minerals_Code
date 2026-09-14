import searoute as sr
from searoute.classes.area_feature import AreaFeature
from searoute.classes.ports_props import PortProps
from searoute.tests.test_utils import get_be_poly
import geojson
import pytest

def test_passages():
    traj = sr.searoute([52.99, 25.01], [-61.87, 17.15], append_orig_dest=True, restrictions=['northwest', 'chili'], return_passages=True)
    result_expected = ['suez', 'ormuz', 'babalmandab', 'gibraltar'] 
    true_result = traj['properties']['traversed_passages']
    assert sorted(result_expected)==sorted(true_result)

def test_restriction_passage():
    traj = sr.searoute([52.99, 25.01], [-61.87, 17.15], append_orig_dest=True, restrictions=['suez'], return_passages=True)
    result_expected = [ 'ormuz', 'south_africa'] 
    true_result = traj['properties']['traversed_passages']
    assert sorted(result_expected)==sorted(true_result)


def test_rev_lat_lon_passages():

    traj = sr.searoute([140.02, 35.51], [-97.36, 27.81], append_orig_dest=True, return_passages=True)
    true_result = traj['properties']['traversed_passages']
    assert ['panama'] == true_result

def test_area_searoute():

    first_port, second_port = PortProps('FRLEH', 200), PortProps('BEANR',250 ), 
    
    areaBE = AreaFeature(coordinates=get_be_poly(), name='BE', preferred_ports=[first_port, second_port])
    
    port_param = {
        'ports_in_areas_from' : AreaFeature.create([areaBE])
    }

    paris_point = (2.333333, 48.866667) # Paris
    tokyo_point = (139.67917395748216, 35.77846652689662) # Tokyo

    result = sr.searoute(paris_point, tokyo_point, append_orig_dest=True, include_ports=True, port_params=port_param)
    assert type(result) == geojson.feature.Feature

    brusx_point = (4.352066635732303, 50.85097556499499) # Bruxelles 

    result = sr.searoute(brusx_point, tokyo_point, append_orig_dest=True, include_ports=True, port_params={})
    assert type(result) == geojson.feature.Feature

from searoute.classes import area_feature 
def test_smallest_area():
    first_port, second_port = PortProps('FRLEH', 200), PortProps('BEANR',250 ), 
    
    areaBE = AreaFeature(coordinates=get_be_poly(), name='BE', preferred_ports=[first_port, second_port])
    
    ft = AreaFeature.create([areaBE])

    brusx_point = (4.352066635732303, 50.85097556499499) # Bruxelles 

    smallest_area = min(
            [feature.properties for feature in ft.features 
            if isinstance(feature, area_feature.AreaFeature) and feature.contains(*brusx_point)],
            key=lambda prop: prop.get('area', float('inf')),
            default=None)
    
    print(smallest_area)
    


def test_2_area_searoute():

    first_port, second_port = PortProps('FRLEH', 200), PortProps('BEANR',250 ), 
    
    areaBE = AreaFeature(coordinates=get_be_poly(), name='BE', preferred_ports=[first_port, second_port])
    
    port_param = {
        'ports_in_areas_from' : AreaFeature.create([areaBE])
    }

    brusx_point = (4.352066635732303, 50.85097556499499) # Bruxelles 
    tokyo_point = (139.67917395748216, 35.77846652689662) # Tokyo

    result = sr.searoute(brusx_point, tokyo_point, append_orig_dest=False, include_ports=True, port_params=port_param)
    assert areaBE.contains(*brusx_point) == True
    assert type(result) == list
    assert len(result) == 2
    #print(result)

@pytest.mark.filterwarnings("ignore")
def test_restricted_paths():
    # in this example all routes are restricted 
    # it should return empty route with a warning


    routes =  sr.searoute(
        origin=(103.85457, 1.25760), # Singapore - SGSIN
        destination=(23.62904, 37.94056), # Piraeus - GRPIR
        restrictions=['suez', 'gibraltar'], # 'gibraltar', 'babalmandab'
        return_passages=True
    )
    print(routes)
    assert routes['geometry']['coordinates'] == []
    assert routes['properties']['length'] == 0
    assert routes['properties']['duration_hours'] == 0.0



    